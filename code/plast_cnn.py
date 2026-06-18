"""CNN 持续学习可塑性 (泛化性关键测试): class-incremental CIFAR-10 + 小CNN.
证明 PTC 不只对 MLP/MNIST 有效, 能泛化到卷积架构+更难数据.
每任务随机二分10类→二分类, 长序列看可塑性. 方法: adam/cbp/ptc/adamo/ptc_adamo.
"""
import os, sys, json, argparse, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_cifar100(root):
    import pickle as _p
    d = _p.load(open(f"{root}/train", "rb"), encoding="bytes")
    X = np.array(d[b"data"]).astype("float32") / 255.0
    X = X.reshape(-1, 3, 32, 32)
    return torch.from_numpy(X), torch.tensor(d[b"fine_labels"], dtype=torch.long)


def load_cifar(root):
    xs, ys = [], []
    for i in range(1, 6):
        with open(f"{root}/data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"]); ys += d[b"labels"]
    X = np.concatenate(xs).astype("float32") / 255.0
    X = X.reshape(-1, 3, 32, 32)
    return torch.from_numpy(X), torch.tensor(ys, dtype=torch.long)


class SmallCNN(nn.Module):
    def __init__(self, n_out=2):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, n_out)
        self.acts = {}

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def layers_seq(model):
    return [model.c1, model.c2, model.c3, model.fc1, model.fc2]


def reset_low_util(model, frac, deficit_scale=1.0):
    """重置低效用单元/通道(Conv+Linear通用)."""
    L = layers_seq(model)
    for i in range(len(L) - 1):
        layer, nxt = L[i], L[i + 1]
        w = layer.weight
        n_units = w.shape[0]
        util = w.detach().abs().reshape(n_units, -1).mean(1)  # 每个输出单元/通道
        k = max(1, int(n_units * frac * deficit_scale))
        idx = torch.argsort(util)[:k]
        # 重置入边
        std = (2.0 / w[0].numel()) ** 0.5
        w.data[idx] = torch.randn_like(w.data[idx]) * std
        if layer.bias is not None:
            layer.bias.data[idx] = 0.0
        # 清零出边 (CBP风格): nxt 对应该单元的输入
        if isinstance(nxt, nn.Conv2d):
            nxt.weight.data[:, idx] *= 0.0
        elif isinstance(nxt, nn.Linear) and isinstance(layer, nn.Conv2d):
            # conv->flatten->linear: 每通道对应 4*4 个输入
            spatial = nxt.weight.shape[1] // n_units
            for u in idx:
                nxt.weight.data[:, u * spatial:(u + 1) * spatial] *= 0.0
        elif isinstance(nxt, nn.Linear):
            nxt.weight.data[:, idx] *= 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="adam", choices=["adam", "cbp", "ptc", "adamo", "ptc_adamo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_tasks", type=int, default=40)
    ap.add_argument("--epochs_per_task", type=int, default=3)
    ap.add_argument("--n_per_task", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--ptc_vreset", type=float, default=0.1)
    ap.add_argument("--reperturb_frac", type=float, default=0.05)
    ap.add_argument("--adamo_lambda", type=float, default=0.001)
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100"])
    ap.add_argument("--data", default="/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/dataset/cifar-10-batches-py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    if args.dataset == "cifar100":
        X, Y = load_cifar100(args.data)
        NC = 100
    else:
        X, Y = load_cifar(args.data); NC = 10
    X, Y = X.to(dev), Y.to(dev); N = X.shape[0]
    model = SmallCNN(2).to(dev); params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.RandomState(args.seed)
    task_acc = []

    for t in range(args.n_tasks):
        if t > 0:
            with torch.no_grad():
                if args.method == "cbp":
                    reset_low_util(model, args.reperturb_frac)
                elif args.method in ("ptc", "ptc_adamo"):
                    for p in params:
                        st = opt.state.get(p, {})
                        if "exp_avg_sq" in st: st["exp_avg_sq"].mul_(args.ptc_vreset)
                        if "exp_avg" in st: st["exp_avg"].mul_(args.ptc_vreset)
                    reset_low_util(model, args.reperturb_frac)
        # class-incremental: 随机二分
        groupA = set(rng.choice(NC, NC // 2, replace=False).tolist())
        sel = torch.from_numpy(rng.choice(N, args.n_per_task, replace=False)).to(dev)
        Xt = X[sel]; Yt = torch.tensor([0 if int(y) in groupA else 1 for y in Y[sel]], device=dev)
        correct = tot = 0
        for ep in range(args.epochs_per_task):
            order = torch.randperm(args.n_per_task, device=dev)
            for i in range(0, args.n_per_task, args.bs):
                idx = order[i:i + args.bs]; xb, yb = Xt[idx], Yt[idx]
                logits = model(xb)
                correct += (logits.argmax(1) == yb).sum().item(); tot += len(yb)
                opt.zero_grad(); F.cross_entropy(logits, yb).backward(); opt.step()
                if args.method in ("adamo", "ptc_adamo"):
                    with torch.no_grad():
                        for p in params:
                            if p.dim() >= 2:
                                W = p.data.reshape(p.shape[0], -1)
                                if W.shape[0] <= W.shape[1]:
                                    G = W @ W.T - torch.eye(W.shape[0], device=W.device)
                                    upd = (G @ W).reshape(p.shape)
                                    p.data.add_(upd, alpha=-args.adamo_lambda * 4)
        task_acc.append(100.0 * correct / tot)

    early = float(np.mean(task_acc[:5])); late = float(np.mean(task_acc[-10:]))
    res = {"method": args.method, "seed": args.seed, "early_acc": early, "late_acc": late,
           "plasticity_drop": early - late, "mean_acc": float(np.mean(task_acc)), "task_acc": task_acc}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"DONE {args.method} s{args.seed}: late_acc={late:.2f}% (early {early:.2f}, drop {early-late:+.2f})")


if __name__ == "__main__":
    main()
