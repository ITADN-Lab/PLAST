"""真 benchmark: 持续 Permuted-MNIST 可塑性 (Dohare Nature2024 标准设定).
每任务一个固定随机像素置换; 在线训练; 可塑性 = 后任务的准确率(高=好).
方法: adam/sgd/shrink_perturb/cbp/ptc. 指标: late-task 准确率.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


def load_cifar(root):
    xs, ys = [], []
    for i in range(1, 6):
        with open(f"{root}/data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"]); ys += d[b"labels"]
    import numpy as _np
    X = torch.from_numpy(_np.concatenate(xs).astype("float32") / 255.0)
    Y = torch.tensor(ys, dtype=torch.long)
    return X, Y


class MLP(nn.Module):
    def __init__(self, hidden=256, depth=2):
        super().__init__()
        L = [nn.Linear(3072, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            L += [nn.Linear(hidden, hidden), nn.ReLU()]
        L += [nn.Linear(hidden, 10)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def utils_of(model):
    lins = [m for m in model.net if isinstance(m, nn.Linear)]
    return lins, [lins[i + 1].weight.abs().mean(0) for i in range(len(lins) - 1)]


@torch.no_grad()
def eff_rank(model, x):
    h = x
    feats = None
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            feats = h
    s = torch.linalg.svdvals(feats - feats.mean(0, keepdim=True))
    s = s / (s.sum() + 1e-12)
    return float(torch.exp(-(s * (s + 1e-12).log()).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="adam", choices=["adam", "sgd", "shrink_perturb", "cbp", "ptc"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_tasks", type=int, default=80)
    ap.add_argument("--epochs_per_task", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--subset", type=int, default=2000)   # 每任务样本数(在线)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--sp_shrink", type=float, default=0.1)
    ap.add_argument("--sp_noise", type=float, default=0.05)
    ap.add_argument("--ptc_vreset", type=float, default=0.1)
    ap.add_argument("--reperturb_frac", type=float, default=0.1)
    ap.add_argument("--rank_target", type=float, default=120.0)
    ap.add_argument("--no_vreset", action="store_true")
    ap.add_argument("--no_reperturb", action="store_true")
    ap.add_argument("--data", default="/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/dataset/cifar-10-batches-py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    X, Y = load_cifar(args.data); X, Y = X.to(dev), Y.to(dev)
    N = X.shape[0]
    model = MLP(args.hidden).to(dev)
    params = list(model.parameters())
    opt = (torch.optim.SGD(params, lr=args.lr * 10) if args.method == "sgd"
           else torch.optim.Adam(params, lr=args.lr))
    rng = np.random.RandomState(args.seed)
    selfix = torch.from_numpy(np.random.RandomState(999).choice(N, args.subset, replace=False)).to(dev)
    Xfix = X[selfix]
    xprobe = Xfix[:512]
    task_acc, task_rank = [], []

    for t in range(args.n_tasks):
        # task boundary re-conditioning
        if t > 0:
            with torch.no_grad():
                if args.method == "shrink_perturb":
                    for p in params:
                        p.mul_(1 - args.sp_shrink).add_(torch.randn_like(p) * args.sp_noise)
                elif args.method == "cbp":
                    lins, us = utils_of(model)
                    for li, u in enumerate(us):
                        k = max(1, int(len(u) * args.reperturb_frac))
                        idx = torch.argsort(u)[:k]
                        lins[li].weight.data[idx] = torch.randn_like(lins[li].weight.data[idx]) * (2.0 / lins[li].weight.shape[1] ** 0.5)
                        lins[li].bias.data[idx] = 0.0
                        lins[li + 1].weight.data[:, idx] *= 0.0
                elif args.method == "ptc":
                    if not args.no_vreset:
                        for p in params:
                            st = opt.state.get(p, {})
                            if "exp_avg_sq" in st: st["exp_avg_sq"].mul_(args.ptc_vreset)
                            if "exp_avg" in st: st["exp_avg"].mul_(args.ptc_vreset)
                    if not args.no_reperturb:
                        er = eff_rank(model, xprobe)
                        deficit = max(0.0, (args.rank_target - er) / args.rank_target)
                        if deficit > 0:
                            lins, us = utils_of(model)
                            for li, u in enumerate(us):
                                k = max(1, int(len(u) * args.reperturb_frac * deficit))
                                idx = torch.argsort(u)[:k]
                                lins[li].weight.data[idx] = torch.randn_like(lins[li].weight.data[idx]) * (2.0 / lins[li].weight.shape[1] ** 0.5)
                                lins[li].bias.data[idx] = 0.0

        # 随机标签记忆: 固定子集, 每任务随机标签
        Yt = torch.from_numpy(rng.randint(0, 10, args.subset)).long().to(dev)
        Xt = Xfix
        correct = tot = 0
        for ep in range(args.epochs_per_task):
            order = torch.randperm(args.subset, device=dev)
            for i in range(0, args.subset, args.bs):
                idx = order[i:i + args.bs]
                xb, yb = Xt[idx], Yt[idx]
                logits = model(xb)
                # online accuracy (训练时即测可塑性)
                correct += (logits.argmax(1) == yb).sum().item(); tot += len(yb)
                opt.zero_grad(); F.cross_entropy(logits, yb).backward(); opt.step()
        task_acc.append(100.0 * correct / tot)
        task_rank.append(eff_rank(model, Xt[:512]))

    early = float(np.mean(task_acc[:5])); late = float(np.mean(task_acc[-10:]))
    res = {"method": args.method, "seed": args.seed, "early_acc": early, "late_acc": late,
           "plasticity_drop": early - late, "late_rank": float(np.mean(task_rank[-10:])),
           "mean_acc": float(np.mean(task_acc)), "task_acc": task_acc}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"DONE {args.method} s{args.seed}: late_acc={late:.2f}% (early {early:.2f}, "
          f"drop {early-late:+.2f}) rank={res['late_rank']:.0f} mean={res['mean_acc']:.2f}")


if __name__ == "__main__":
    main()
