"""LLM 规模可塑性验证: 小 GPT 持续学习 (permuted-token 任务序列, LLM版permuted-MNIST).
每任务一个固定随机 token-ID 置换, 强制重学; 可塑性 = 每任务达到的 val loss(低=好).
测 PTC(vₜ重置,优化器状态机制)在 transformer 上是否泛化. 方法: adam/cbp/ptc/adamo.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
for _p in ["/home/hadoop/workstation/md/AAAI/code", os.path.expanduser("~/rapm_code")]:
    if os.path.isdir(os.path.join(_p, "rapm")):
        sys.path.insert(0, _p); break
from rapm.nanogpt import NanoGPT, NanoGPTConfig


def get_batch(data, bs, block, dev, perm):
    ix = torch.randint(len(data) - block - 1, (bs,))
    x = torch.stack([torch.from_numpy(data[i:i+block].astype(np.int64)) for i in ix]).to(dev)
    return perm[x]  # 应用 token 置换


def linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear) and min(m.weight.shape) > 1]


@torch.no_grad()
def reset_low_util(model, frac):
    ls = linears(model)
    for i in range(len(ls) - 1):
        w = ls[i].weight; no = w.shape[0]
        util = w.detach().abs().mean(1)
        k = max(1, int(no * frac)); idx = torch.argsort(util)[:k]
        w.data[idx] = torch.randn_like(w.data[idx]) * 0.02
        if ls[i].bias is not None: ls[i].bias.data[idx] = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="adam", choices=["adam", "cbp", "ptc", "adamo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_tasks", type=int, default=20)
    ap.add_argument("--steps_per_task", type=int, default=200)
    ap.add_argument("--n_layer", type=int, default=4); ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4); ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--ptc_vreset", type=float, default=0.1)
    ap.add_argument("--reperturb_frac", type=float, default=0.05)
    ap.add_argument("--adamo_lambda", type=float, default=1e-4)
    ap.add_argument("--data", default="/home/hadoop/workstation/md/AAAI/data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    train = np.memmap(os.path.join(args.data, "owt_train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data, "owt_val.bin"), dtype=np.uint16, mode="r")
    V = 50257
    cfg = NanoGPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                        block_size=args.block, vocab_size=V)
    model = NanoGPT(cfg).to(dev); params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.RandomState(args.seed)
    task_loss = []

    for t in range(args.n_tasks):
        if t > 0:
            with torch.no_grad():
                if args.method == "cbp":
                    reset_low_util(model, args.reperturb_frac)
                elif args.method == "ptc":
                    for p in params:
                        st = opt.state.get(p, {})
                        if "exp_avg_sq" in st: st["exp_avg_sq"].mul_(args.ptc_vreset)
                        if "exp_avg" in st: st["exp_avg"].mul_(args.ptc_vreset)
                    reset_low_util(model, args.reperturb_frac)
        perm = torch.from_numpy(rng.permutation(V).astype(np.int64)).to(dev)  # 固定token置换
        model.train()
        for s in range(args.steps_per_task):
            x = get_batch(train, args.bs, args.block, dev, perm)
            loss = model(x, labels=x).loss
            opt.zero_grad(); loss.backward()
            if args.method == "adamo":
                with torch.no_grad():
                    for p in params:
                        if p.dim() == 2 and min(p.shape) > 1:
                            W = p.data
                            if W.shape[0] <= W.shape[1]:
                                G = W @ W.T - torch.eye(W.shape[0], device=W.device)
                                p.data.add_(G @ W, alpha=-args.adamo_lambda * 4)
            opt.step()
        # 该任务 val loss
        model.eval()
        with torch.no_grad():
            vl = np.mean([model(get_batch(val, args.bs, args.block, dev, perm),
                                labels=get_batch(val, args.bs, args.block, dev, perm)).loss.item()
                          for _ in range(10)])
        task_loss.append(float(vl))

    early = float(np.mean(task_loss[:3])); late = float(np.mean(task_loss[-5:]))
    res = {"method": args.method, "seed": args.seed, "early_loss": early, "late_loss": late,
           "plasticity_drop": late - early, "mean_loss": float(np.mean(task_loss)), "task_loss": task_loss}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"DONE {args.method} s{args.seed}: late_loss={late:.3f} (early {early:.3f}, drop {late-early:+.3f})")


if __name__ == "__main__":
    main()
