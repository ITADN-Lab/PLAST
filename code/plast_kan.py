"""Pilot: KAN-MLP vs 标准 MLP 的 permuted-MNIST 可塑性对比.
测 KAN 的样条局部更新是否(a)抗可塑性丧失(late-task acc高)(b)维持更高有效秩.
KAN 层 = efficient-kan 风格(base SiLU + B-spline). 都用 Adam, 公平比.
"""
import os, json, argparse, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


def load_mnist(root):
    raw = os.path.join(root, "MNIST", "raw")
    with open(os.path.join(raw, "train-images-idx3-ubyte"), "rb") as f:
        f.read(16); img = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)
    with open(os.path.join(raw, "train-labels-idx1-ubyte"), "rb") as f:
        f.read(8); lab = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.from_numpy(img.astype("float32") / 255.0), torch.from_numpy(lab.astype("int64"))


class KANLinear(nn.Module):
    """efficient-kan 风格: y_j = Σ_i [base_w·SiLU(x_i) + spline_w·Bspline(x_i)]."""
    def __init__(self, in_f, out_f, grid_size=5, spline_order=3, grid_range=(-2, 2)):
        super().__init__()
        self.in_f, self.out_f, self.grid_size, self.spline_order = in_f, out_f, grid_size, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
        self.register_buffer("grid", grid.expand(in_f, -1).contiguous())
        self.base_w = nn.Parameter(torch.empty(out_f, in_f))
        self.spline_w = nn.Parameter(torch.empty(out_f, in_f, grid_size + spline_order))
        nn.init.kaiming_uniform_(self.base_w, a=math.sqrt(5))
        nn.init.normal_(self.spline_w, 0, 0.1)

    def b_splines(self, x):                      # x:(B,in) -> (B,in,G+k)
        g = self.grid; x = x.unsqueeze(-1)
        b = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            b = ((x - g[:, :-(k+1)]) / (g[:, k:-1] - g[:, :-(k+1)]) * b[:, :, :-1]
                 + (g[:, k+1:] - x) / (g[:, k+1:] - g[:, 1:-k]) * b[:, :, 1:])
        return b

    def forward(self, x):
        base = F.silu(x) @ self.base_w.T
        spline = torch.einsum("bic,oic->bo", self.b_splines(x), self.spline_w)
        return base + spline


class KANMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.l1 = KANLinear(784, hidden); self.l2 = KANLinear(hidden, 10)
        self.feat = None
    def forward(self, x):
        h = self.l1(x); self.feat = h; return self.l2(h)


class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.l1 = nn.Linear(784, hidden); self.l2 = nn.Linear(hidden, 10)
        self.feat = None
    def forward(self, x):
        h = F.relu(self.l1(x)); self.feat = h; return self.l2(h)


@torch.no_grad()
def eff_rank(feat):
    s = torch.linalg.svdvals(feat - feat.mean(0, keepdim=True)); s = s / (s.sum() + 1e-12)
    return float(torch.exp(-(s * (s + 1e-12).log()).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["kan", "mlp"], default="mlp")
    ap.add_argument("--hidden", type=int, default=0)   # 0=auto
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_tasks", type=int, default=40)
    ap.add_argument("--n_per_task", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--data", default="/home/hadoop/workstation/md/plasticity/mnist_data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    X, Y = load_mnist(args.data); X, Y = X.to(dev), Y.to(dev); N = X.shape[0]
    rng = np.random.RandomState(args.seed)
    H = args.hidden or (64 if args.arch == "kan" else 256)
    model = (KANMLP(H) if args.arch == "kan" else MLP(H)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    nparam = sum(p.numel() for p in model.parameters())
    task_acc, rank_traj = [], []
    for t in range(args.n_tasks):
        perm = torch.from_numpy(rng.permutation(784)).to(dev)
        correct = tot = 0
        for i in range(0, args.n_per_task, args.bs):
            ix = torch.from_numpy(rng.choice(N, args.bs)).to(dev)
            xb = X[ix][:, perm]; yb = Y[ix]
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item(); tot += len(yb)
            opt.zero_grad(); F.cross_entropy(logits, yb).backward(); opt.step()
        task_acc.append(100.0 * correct / tot)
        with torch.no_grad():
            model(X[:512][:, perm]); rank_traj.append(eff_rank(model.feat))
    early = float(np.mean(task_acc[:5])); late = float(np.mean(task_acc[-10:]))
    res = {"arch": args.arch, "hidden": H, "nparam": nparam, "seed": args.seed,
           "early_acc": early, "late_acc": late, "late_rank": float(np.mean(rank_traj[-10:])),
           "task_acc": task_acc, "rank_traj": rank_traj}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"DONE {args.arch}(H{H},{nparam//1000}k) s{args.seed}: late_acc={late:.1f} (early {early:.1f}) late_rank={res['late_rank']:.1f}")


if __name__ == "__main__":
    main()
