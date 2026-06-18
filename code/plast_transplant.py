"""State transplant: 终极 causal 实验 (GPT-5.5推荐).
训练任务A得到 Adam state (m_A,v_A); 切到任务B, 用不同方式初始化优化器状态:
  inherited : 保留 stale (m_A,v_A)  ← naive Adam continual
  reset     : 清零 (PTC式)
  shuffled  : 打乱 v_A 的坐标 (保magnitude, 破per-param对应)
  random    : 随机magnitude(匹配v_A分布)
测任务B的学习速度+最终acc. 若 inherited(stale)最慢、reset最快 → stale v_t per-param对应是causal病因.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_mnist(root):
    raw = os.path.join(root, "MNIST", "raw")
    with open(os.path.join(raw, "train-images-idx3-ubyte"), "rb") as f:
        f.read(16); img = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)
    with open(os.path.join(raw, "train-labels-idx1-ubyte"), "rb") as f:
        f.read(8); lab = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.from_numpy(img.astype("float32") / 255.0), torch.from_numpy(lab.astype("int64"))


def mlp(h=256):
    return nn.Sequential(nn.Linear(784, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transplant", default="inherited",
                    choices=["inherited", "reset", "shuffled", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_tasks", type=int, default=20)   # 先在A...序列上"老化"
    ap.add_argument("--stepsA", type=int, default=400)
    ap.add_argument("--stepsB", type=int, default=400)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--data", default="/home/hadoop/workstation/md/plasticity/mnist_data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    X, Y = load_mnist(args.data); X, Y = X.to(dev), Y.to(dev); N = X.shape[0]
    rng = np.random.RandomState(args.seed)
    model = mlp().to(dev); params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    def run_task(perm, steps, log=False):
        accs = []
        for s in range(steps):
            ix = torch.from_numpy(rng.choice(N, args.bs)).to(dev)
            xb = X[ix][:, perm]; yb = Y[ix]
            logits = model(xb)
            if log: accs.append((logits.argmax(1) == yb).float().mean().item())
            opt.zero_grad(); F.cross_entropy(logits, yb).backward(); opt.step()
        return accs

    # 老化: 在 warmup_tasks 个置换上训练, 让 v_t 充分stale
    for t in range(args.warmup_tasks):
        run_task(torch.from_numpy(rng.permutation(784)).to(dev), args.stepsA)

    # 任务A (最后一个) → 得到 stale state
    permA = torch.from_numpy(rng.permutation(784)).to(dev)
    run_task(permA, args.stepsA)

    # ---- transplant: 改优化器状态 ----
    with torch.no_grad():
        for p in params:
            st = opt.state.get(p, {})
            if "exp_avg_sq" not in st: continue
            if args.transplant == "reset":
                st["exp_avg_sq"].zero_(); st["exp_avg"].zero_()
            elif args.transplant == "shuffled":
                flat = st["exp_avg_sq"].flatten()
                st["exp_avg_sq"].copy_(flat[torch.randperm(flat.numel(), device=dev)].reshape(st["exp_avg_sq"].shape))
            elif args.transplant == "random":
                m = st["exp_avg_sq"].mean()
                st["exp_avg_sq"].copy_(torch.rand_like(st["exp_avg_sq"]) * 2 * m)
            # inherited: 不动

    # ---- 任务B: 测学习速度+最终acc ----
    permB = torch.from_numpy(rng.permutation(784)).to(dev)
    accsB = run_task(permB, args.stepsB, log=True)
    early_auc = float(np.mean(accsB[:100]))      # 前100步学习速度
    final = float(np.mean(accsB[-50:]))
    res = {"transplant": args.transplant, "seed": args.seed,
           "taskB_early_auc": early_auc, "taskB_final": final, "accsB": accsB[::10]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"DONE {args.transplant} s{args.seed}: taskB early_auc={early_auc:.3f} final={final:.3f}")


if __name__ == "__main__":
    main()
