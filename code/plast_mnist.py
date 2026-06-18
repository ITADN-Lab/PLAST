"""真 benchmark: 持续 Permuted-MNIST 可塑性 (Dohare Nature2024 标准设定).
每任务一个固定随机像素置换; 在线训练; 可塑性 = 后任务的准确率(高=好).
方法: adam/sgd/shrink_perturb/cbp/ptc. 指标: late-task 准确率.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_mnist(root):
    import numpy as _np
    raw = os.path.join(root, "MNIST", "raw")
    with open(os.path.join(raw, "train-images-idx3-ubyte"), "rb") as f:
        f.read(16); img = _np.frombuffer(f.read(), dtype=_np.uint8).reshape(-1, 784)
    with open(os.path.join(raw, "train-labels-idx1-ubyte"), "rb") as f:
        f.read(8); lab = _np.frombuffer(f.read(), dtype=_np.uint8)
    X = torch.from_numpy(img.astype("float32") / 255.0)
    Y = torch.from_numpy(lab.astype("int64"))
    return X, Y



@torch.no_grad()
def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float() / (G.norm() + 1e-7)
    tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * (A @ A); X = a * X + B @ X
    if tr: X = X.T
    return X


class MuonOpt:
    """2D权重 Muon(动量+NS正交化), 其余 Adam."""
    def __init__(self, params, lr, beta=0.9):
        self.lr = lr; self.beta = beta
        self.params = list(params); self.state = {}
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad = None
    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            st = self.state.setdefault(id(p), {})
            if "m" not in st: st["m"] = torch.zeros_like(p)
            st["m"].mul_(self.beta).add_(p.grad)
            if p.dim() == 2 and min(p.shape) > 1:
                O = newton_schulz(st["m"]); O = O / (O.pow(2).mean().sqrt() + 1e-8)
                p.add_(O, alpha=-self.lr * 3)
            else:
                p.add_(st["m"], alpha=-self.lr)

class MLP(nn.Module):
    def __init__(self, hidden=256, depth=2):
        super().__init__()
        L = [nn.Linear(784, hidden), nn.ReLU()]
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
    ap.add_argument("--method", default="adam", choices=["adam", "sgd", "shrink_perturb", "cbp", "ptc", "l2init", "redo", "trac", "ptc_l2", "trac_official", "muon", "ptc_muon", "adamo", "ptc_adamo", "ptc_all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_tasks", type=int, default=80)
    ap.add_argument("--epochs_per_task", type=int, default=1)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n_per_task", type=int, default=6000)   # 每任务样本数(在线)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--sp_shrink", type=float, default=0.1)
    ap.add_argument("--sp_noise", type=float, default=0.05)
    ap.add_argument("--ptc_vreset", type=float, default=0.1)
    ap.add_argument("--reperturb_frac", type=float, default=0.1)
    ap.add_argument("--rank_target", type=float, default=120.0)
    ap.add_argument("--no_vreset", action="store_true")
    ap.add_argument("--no_reperturb", action="store_true")
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--reset_m_only", action="store_true")
    ap.add_argument("--reset_v_only", action="store_true")
    ap.add_argument("--hard_reset", action="store_true")
    ap.add_argument("--recon_prob", type=float, default=1.0)
    ap.add_argument("--periodic_recon", type=int, default=0)
    ap.add_argument("--eval_forget", action="store_true")
    ap.add_argument("--l2init_lambda", type=float, default=0.01)
    ap.add_argument("--redo_tau", type=float, default=0.025)
    ap.add_argument("--adamo_lambda", type=float, default=0.001)
    ap.add_argument("--mode", default="permute", choices=["permute","class_inc"])
    ap.add_argument("--data", default="/home/hadoop/workstation/md/plasticity/mnist_data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    X, Y = load_mnist(args.data); X, Y = X.to(dev), Y.to(dev)
    N = X.shape[0]
    model = MLP(args.hidden, args.depth).to(dev)
    params = list(model.parameters())
    if args.method == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr * 10)
    elif args.method == "trac_official":
        from trac_optimizer import start_trac
        import tempfile
        _lf = tempfile.mktemp()
        opt = start_trac(_lf, torch.optim.Adam)(params, lr=args.lr)
    elif args.method in ("muon", "ptc_muon"):
        opt = MuonOpt(params, lr=args.lr)
    else:
        opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, args.beta2))
    theta_init = [p.detach().clone() for p in params]  # l2init
    # TRAC 状态 (参数无关自适应尺度, 多时间尺度coin-betting)
    trac_betas = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
    trac_ref = [p.detach().clone() for p in params]
    trac_s, trac_sigma, trac_var = 1.0, [0.0]*len(trac_betas), [0.0]*len(trac_betas)
    rng = np.random.RandomState(args.seed)
    recon_rng = np.random.RandomState(args.seed + 99991)
    saved_perms = []
    xprobe = X[:512]
    task_acc, task_rank = [], []

    for t in range(args.n_tasks):
        # task boundary re-conditioning
        if t > 0 and (recon_rng.random() < args.recon_prob):
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
                elif args.method == "redo":
                    # ReDo: 重置近零激活(dormant)的神经元
                    h = xprobe
                    acts = []
                    for layer in model.net:
                        h = layer(h)
                        if isinstance(layer, nn.ReLU): acts.append(h.abs().mean(0))
                    lins, _ = utils_of(model)
                    for li, a in enumerate(acts):
                        score = a / (a.mean() + 1e-9)
                        idx = (score < args.redo_tau).nonzero(as_tuple=True)[0]
                        if len(idx) > 0:
                            lins[li].weight.data[idx] = torch.randn_like(lins[li].weight.data[idx]) * (2.0/lins[li].weight.shape[1]**0.5)
                            lins[li].bias.data[idx] = 0.0
                            lins[li+1].weight.data[:, idx] *= 0.0
                elif args.method == "ptc_muon":
                    for p in params:
                        st = opt.state.get(id(p), {})
                        if "m" in st: st["m"].mul_(args.ptc_vreset)
                    er = eff_rank(model, xprobe); deficit = max(0.0, (args.rank_target - er)/args.rank_target)
                    if deficit > 0:
                        lins, us = utils_of(model)
                        for li, u in enumerate(us):
                            k = max(1, int(len(u) * args.reperturb_frac * deficit)); idx = torch.argsort(u)[:k]
                            lins[li].weight.data[idx] = torch.randn_like(lins[li].weight.data[idx]) * (2.0/lins[li].weight.shape[1]**0.5)
                            lins[li].bias.data[idx] = 0.0
                elif args.method in ("ptc", "ptc_l2", "ptc_adamo", "ptc_all"):
                    if not args.no_vreset:
                        g = 0.0 if args.hard_reset else args.ptc_vreset
                        for p in params:
                            st = opt.state.get(p, {})
                            if "exp_avg_sq" in st and not args.reset_m_only: st["exp_avg_sq"].mul_(g)
                            if "exp_avg" in st and not args.reset_v_only: st["exp_avg"].mul_(g)
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

        if args.mode == "class_inc":
            # 随机二分10类 → 二分类 (class-incremental可塑性)
            groupA = set(rng.choice(10, 5, replace=False).tolist())
            sel = torch.from_numpy(rng.choice(N, args.n_per_task, replace=False)).to(dev)
            Xt = X[sel]
            Yt = torch.tensor([0 if int(y) in groupA else 1 for y in Y[sel]], device=dev)
        else:
            perm = torch.from_numpy(rng.permutation(784)).to(dev)
            if args.eval_forget: saved_perms.append(perm)
            sel = torch.from_numpy(rng.choice(N, args.n_per_task, replace=False)).to(dev)
            Xt, Yt = X[sel][:, perm], Y[sel]
        correct = tot = 0
        for ep in range(args.epochs_per_task):
            order = torch.randperm(args.n_per_task, device=dev)
            for i in range(0, args.n_per_task, args.bs):
                idx = order[i:i + args.bs]
                xb, yb = Xt[idx], Yt[idx]
                logits = model(xb)
                # online accuracy (训练时即测可塑性)
                correct += (logits.argmax(1) == yb).sum().item(); tot += len(yb)
                opt.zero_grad(); F.cross_entropy(logits, yb).backward()
                if args.method in ("l2init", "ptc_l2", "ptc_all"):
                    for p, p0 in zip(params, theta_init):
                        if p.grad is not None: p.grad.add_(p - p0, alpha=args.l2init_lambda)
                if args.method == "trac":
                    # TRAC: 参数无关自适应尺度, 缩放距初始的偏移
                    with torch.no_grad():
                        # reward h = -<g, Δ>, Δ=θ-ref
                        hdot = sum((p.grad * (p - r)).sum().item() for p, r in zip(params, trac_ref) if p.grad is not None)
                        opt.step()
                        import math as _m
                        snew = 0.0
                        for i, b in enumerate(trac_betas):
                            trac_var[i] = b*trac_var[i] + (hdot/ max(trac_s,1e-8))**2
                            trac_sigma[i] = b*trac_sigma[i] - hdot/ max(trac_s,1e-8)
                            snew += (1.0/len(trac_betas)) * _m.tanh(trac_sigma[i] / (_m.sqrt(trac_var[i])+1e-8))
                        trac_s = max(0.01, 1.0 + snew)
                        for p, r in zip(params, trac_ref):
                            p.data.copy_(r + trac_s * (p.data - r))
                else:
                    opt.step()
                if args.method in ("adamo", "ptc_adamo", "ptc_all"):
                    with torch.no_grad():
                        for p in params:
                            if p.dim() == 2 and min(p.shape) > 1:
                                W = p.data
                                if W.shape[0] >= W.shape[1]:
                                    G = W.T @ W - torch.eye(W.shape[1], device=W.device)
                                    p.data.add_(W @ G, alpha=-args.adamo_lambda * 4)
                                else:
                                    G = W @ W.T - torch.eye(W.shape[0], device=W.device)
                                    p.data.add_(G @ W, alpha=-args.adamo_lambda * 4)
        task_acc.append(100.0 * correct / tot)
        task_rank.append(eff_rank(model, Xt[:512]))

    forget_avg = None
    if args.eval_forget and saved_perms:
        model.eval()
        with torch.no_grad():
            retained = []
            for pm in saved_perms:
                c=t_=0
                for _ in range(5):
                    ix=torch.from_numpy(rng.choice(N,256)).to(dev)
                    pred=model(X[ix][:,pm]).argmax(1); c+=(pred==Y[ix]).sum().item(); t_+=256
                retained.append(100.0*c/t_)
            forget_avg = float(np.mean(retained))  # 最终模型在所有任务上的平均保持准确率
    early = float(np.mean(task_acc[:5])); late = float(np.mean(task_acc[-10:]))
    res = {"method": args.method, "seed": args.seed, "early_acc": early, "late_acc": late,
           "plasticity_drop": early - late, "late_rank": float(np.mean(task_rank[-10:])),
           "mean_acc": float(np.mean(task_acc)), "task_acc": task_acc, "task_rank": task_rank, "forget_avg": forget_avg}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"DONE {args.method} s{args.seed}: late_acc={late:.2f}% (early {early:.2f}, "
          f"drop {early-late:+.2f}) rank={res['late_rank']:.0f} mean={res['mean_acc']:.2f}")


if __name__ == "__main__":
    main()
