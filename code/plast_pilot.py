"""可塑性 kill-test: PTC-recondition vs Adam(病因) vs shrink-perturb(强baseline) vs SGD.
正向 gate: PTC 能否同时打过 Adam 和 shrink-perturb 的后任务可塑性?
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plast_core import make_target, Learner, effective_rank


def hidden_units_utility(model):
    """每个隐单元的效用 = 出边权重范数 (低=可重置)."""
    lins = [m for m in model.net if isinstance(m, nn.Linear)]
    utils = []
    for i in range(len(lins) - 1):
        w_out = lins[i + 1].weight  # (next, this)
        utils.append(w_out.abs().mean(0))  # 每个本层单元的出边均范数
    return lins, utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="adam", choices=["adam", "sgd", "shrink_perturb", "ptc", "cbp"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_tasks", type=int, default=100)
    ap.add_argument("--steps_per_task", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--in_dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    # shrink-perturb / ptc 超参
    ap.add_argument("--sp_shrink", type=float, default=0.01)
    ap.add_argument("--sp_noise", type=float, default=0.01)
    ap.add_argument("--ptc_vreset", type=float, default=0.1)    # vₜ 软重置系数
    ap.add_argument("--ptc_reperturb_frac", type=float, default=0.1)
    ap.add_argument("--rank_target", type=float, default=25.0)
    ap.add_argument("--no_vreset", action="store_true")
    ap.add_argument("--no_reperturb", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    learner = Learner(args.in_dim, args.hidden, depth=2, act="relu").to(dev)
    params = list(learner.parameters())
    if args.method == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr * 10)
    else:
        opt = torch.optim.Adam(params, lr=args.lr)

    xprobe = torch.randn(512, args.in_dim, device=dev)
    task_loss, task_rank = [], []

    for t in range(args.n_tasks):
        # ---- task boundary re-conditioning ----
        if t > 0:
            if args.method == "shrink_perturb":
                with torch.no_grad():
                    for p in params:
                        p.mul_(1 - args.sp_shrink).add_(torch.randn_like(p) * args.sp_noise)
            elif args.method == "ptc":
                with torch.no_grad():
                    # ① 软重置 Adam vₜ (重新条件化预条件器)
                    for p in (params if not args.no_vreset else []):
                        st = opt.state.get(p, {})
                        if "exp_avg_sq" in st:
                            st["exp_avg_sq"].mul_(args.ptc_vreset)
                        if "exp_avg" in st:
                            st["exp_avg"].mul_(args.ptc_vreset)
                    # ② rank缺口驱动: 重扰动低效用单元
                    er = effective_rank(learner, xprobe)
                    deficit = max(0.0, (args.rank_target - er) / args.rank_target)
                    if deficit > 0 and not args.no_reperturb:
                        lins, utils = hidden_units_utility(learner)
                        for li, u in enumerate(utils):
                            k = max(1, int(len(u) * args.ptc_reperturb_frac * deficit))
                            idx = torch.argsort(u)[:k]  # 最低效用
                            # 重置该单元的入边 (恢复可塑性), 噪声幅度随deficit
                            lins[li].weight.data[idx] = torch.randn_like(
                                lins[li].weight.data[idx]) * (2.0 / lins[li].weight.shape[1] ** 0.5)
                            lins[li].bias.data[idx] = 0.0

        if args.method == "cbp" and t > 0:
            # CBP: 重置最低效用的成熟单元 (替换率 reperturb_frac)
            with torch.no_grad():
                lins, utils = hidden_units_utility(learner)
                for li, u in enumerate(utils):
                    k = max(1, int(len(u) * args.ptc_reperturb_frac))
                    idx = torch.argsort(u)[:k]
                    lins[li].weight.data[idx] = torch.randn_like(
                        lins[li].weight.data[idx]) * (2.0/lins[li].weight.shape[1]**0.5)
                    lins[li].bias.data[idx] = 0.0
                    lins[li+1].weight.data[:, idx] *= 0.0  # 出边清零(CBP特征)
        tgt = make_target(args.in_dim, 32, seed=1000 + t, dev=dev)
        for s in range(args.steps_per_task):
            x = torch.randn(args.bs, args.in_dim, device=dev)
            with torch.no_grad():
                y = tgt(x)
            opt.zero_grad(); l = ((learner(x) - y) ** 2).mean(); l.backward(); opt.step()
        with torch.no_grad():
            yv = tgt(xprobe); fl = ((learner(xprobe) - yv) ** 2).mean().item()
        task_loss.append(fl)
        task_rank.append(effective_rank(learner, xprobe))

    early = float(np.mean(task_loss[:5]))
    late = float(np.mean(task_loss[-10:]))
    res = {"method": args.method, "seed": args.seed,
           "early_loss": early, "late_loss": late,
           "plasticity_drop_pct": 100 * (late - early) / early,
           "late_rank": float(np.mean(task_rank[-10:])),
           "mean_loss": float(np.mean(task_loss)),
           "task_loss": task_loss, "task_rank": task_rank}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"DONE {args.method} s{args.seed}: late_loss={late:.4f} (early {early:.4f}, "
          f"drop {res['plasticity_drop_pct']:+.0f}%) late_rank={res['late_rank']:.1f} "
          f"mean={res['mean_loss']:.4f}")


if __name__ == "__main__":
    main()
