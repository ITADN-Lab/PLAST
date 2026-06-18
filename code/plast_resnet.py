"""ResNet 规模可塑性验证: ResNet-18(CIFAR适配, 带BN+残差) class-incremental CIFAR-10.
测 PTC(vₜ重置, 优化器状态机制)是否泛化到深度残差网络+BatchNorm.
方法: adam/cbp/ptc/adamo/ptc_adamo. BN是当年坑per-channel曲率法的地方,但PTC不依赖曲率语义.
"""
import os, sys, json, argparse, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_cifar(root, n100=False):
    if n100:
        d = pickle.load(open(f"{root}/train", "rb"), encoding="bytes")
        X = np.array(d[b"data"]).astype("float32") / 255.0
        return torch.from_numpy(X.reshape(-1, 3, 32, 32)), torch.tensor(d[b"fine_labels"])
    xs, ys = [], []
    for i in range(1, 6):
        d = pickle.load(open(f"{root}/data_batch_{i}", "rb"), encoding="bytes")
        xs.append(d[b"data"]); ys += d[b"labels"]
    X = np.concatenate(xs).astype("float32") / 255.0
    return torch.from_numpy(X.reshape(-1, 3, 32, 32)), torch.tensor(ys, dtype=torch.long)


class BasicBlock(nn.Module):
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(out)
        self.c2 = nn.Conv2d(out, out, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(out)
        self.sc = nn.Sequential()
        if stride != 1 or inp != out:
            self.sc = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.sc(x))


class ResNet18(nn.Module):
    def __init__(self, n_out=2, w=64, bottleneck=0, mid_bottleneck=0):
        super().__init__()
        self.c1 = nn.Conv2d(3, w, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(w)
        self.l1 = self._layer(w, w, 2, 1)
        self.l2 = self._layer(w, 2 * w, 2, 2)
        self.l3 = self._layer(2 * w, 4 * w, 2, 2)
        self.l4 = self._layer(4 * w, 8 * w, 2, 2)
        self.bottleneck = bottleneck
        self.mid_bottleneck = mid_bottleneck
        if mid_bottleneck > 0:    # 因果: l3后插1x1 conv channel瓶颈, 硬限中间层特征秩≤k
            self.mid_proj = nn.Conv2d(4 * w, mid_bottleneck, 1, bias=False)
            self.mid_up = nn.Conv2d(mid_bottleneck, 4 * w, 1, bias=False)
        if bottleneck > 0:                          # 线性瓶颈: 硬限head所见特征秩≤k (因果干预)
            self.bn_proj = nn.Linear(8 * w, bottleneck, bias=False)
            self.fc = nn.Linear(bottleneck, n_out)
        else:
            self.fc = nn.Linear(8 * w, n_out)

    def _layer(self, inp, out, n, stride):
        layers = [BasicBlock(inp, out, stride)] + [BasicBlock(out, out, 1) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        x = F.relu(self.b1(self.c1(x)))
        x = self.l3(self.l2(self.l1(x)))
        if self.mid_bottleneck > 0:                  # 中间层channel瓶颈(因果)
            x = self.mid_up(self.mid_proj(x))
        x = self.l4(x)
        h = F.adaptive_avg_pool2d(x, 1).flatten(1)
        feat = self.bn_proj(h) if self.bottleneck > 0 else h   # head所见的特征(瓶颈则k维)
        out = self.fc(feat)
        return (out, feat) if return_feat else out


def ntk_diagnostics(model, xprobe, n=16):
    """正规经验NTK: K_ij = Σ_c <∂f_c(x_i)/∂θ, ∂f_c(x_j)/∂θ> (对所有类别logit求和, 非sum-of-logits).
    返回 NTK 条件数(↓好)、NTK有效秩(↑好)、权重谱条件数中位数."""
    model.eval()
    xb = xprobe[:n]
    params = [p for p in model.parameters() if p.requires_grad]
    C = model(xb[:1]).shape[1]
    # 每样本每类: g[i,c] = ∂f_c(x_i)/∂θ (展平); 堆成 (n*C) x P
    rows = []
    for i in range(xb.shape[0]):
        out = model(xb[i:i+1])[0]    # C logits
        for c in range(C):
            g = torch.autograd.grad(out[c], params, retain_graph=(c < C - 1))
            rows.append(torch.cat([gi.reshape(-1) for gi in g]))
    J = torch.stack(rows)                       # (n*C) x P
    K = (J @ J.T).double()                       # (nC)x(nC) 经验NTK
    ev = torch.linalg.eigvalsh(K).clamp_min(1e-12)
    p_ = ev / ev.sum()
    ntk_erank = float(torch.exp(-(p_ * p_.log()).sum()))
    ntk_cond = float(ev[-1] / ev[ev.shape[0] // 20])   # 鲁棒条件数: max / 5%分位 (避开数值零)
    conds = []
    with torch.no_grad():
        for p in params:
            if p.dim() >= 2:
                W = p.reshape(p.shape[0], -1)
                s = torch.linalg.svdvals(W.float())
                if s[-1] > 1e-9: conds.append(float(s[0] / s[-1]))
    model.train()
    import numpy as _np
    return {"ntk_erank": ntk_erank, "ntk_cond": ntk_cond, "wcond_median": float(_np.median(conds)) if conds else None}


@torch.no_grad()
def block_ranks(model, xprobe):
    """各 block (l1-l4) 输出的 channel-wise 有效秩 (盲区: 中间层特征秩, 非penultimate)."""
    model.eval()
    outs = {}; hooks = []
    for nm in ["l1", "l2", "l3", "l4"]:
        m = getattr(model, nm)
        hooks.append(m.register_forward_hook(lambda mod, i, o, k=nm: outs.__setitem__(k, o)))
    model(xprobe[:256]);
    for h in hooks: h.remove()
    res = {}
    for nm, o in outs.items():
        f = o.permute(0, 2, 3, 1).reshape(-1, o.shape[1])   # [B*H*W, C]
        f = f[torch.randperm(f.shape[0])[:2048]]
        s = torch.linalg.svdvals(f - f.mean(0, keepdim=True)); s = s / (s.sum() + 1e-12)
        res[nm] = float(torch.exp(-(s * (s + 1e-12).log()).sum()))
    model.train()
    return res


@torch.no_grad()
def eff_rank_resnet(model, xprobe):
    """penultimate 特征的谱熵有效秩 (机理: Muon 是否维持更高 rank)."""
    model.eval()
    _, feat = model(xprobe, return_feat=True)
    model.train()
    s = torch.linalg.svdvals(feat - feat.mean(0, keepdim=True))
    s = s / (s.sum() + 1e-12)
    return float(torch.exp(-(s * (s + 1e-12).log()).sum()))


def reset_low_util(model, frac):
    """重置低效用conv输出通道的入边+对应BN affine (残差结构, 不清出边)."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.weight.shape[1] > 3]
    for c in convs:
        w = c.weight; no = w.shape[0]
        util = w.detach().abs().reshape(no, -1).mean(1)
        k = max(1, int(no * frac))
        idx = torch.argsort(util)[:k]
        w.data[idx] = torch.randn_like(w.data[idx]) * (2.0 / w[0].numel()) ** 0.5


@torch.no_grad()
def reset_dormant(model, xprobe, tau):
    """ReDo: 重置激活长期接近0的'休眠'conv通道 (入边重置, 出边清零)."""
    acts = {}
    hooks = []
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.weight.shape[1] > 3]
    convset = set(id(c) for c in convs)
    def mk(name):
        def hook(mod, inp, out): acts[name] = out.abs().mean(dim=(0, 2, 3))  # per-channel
        return hook
    for i, c in enumerate(convs):
        hooks.append(c.register_forward_hook(mk(i)))
    model.eval(); model(xprobe); model.train()
    for h in hooks: h.remove()
    for i, c in enumerate(convs):
        a = acts.get(i)
        if a is None: continue
        score = a / (a.mean() + 1e-9)
        idx = (score < tau).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            c.weight.data[idx] = torch.randn_like(c.weight.data[idx]) * (2.0 / c.weight[0].numel()) ** 0.5


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
    """Conv/Linear 权重走 Muon(动量+NS正交化, 4D conv reshape 成 2D), 其余(BN/bias/stem)走动量SGD."""
    def __init__(self, params, lr, beta=0.9, rank_frac=0.0):
        self.lr = lr; self.beta = beta; self.rank_frac = rank_frac  # 因果: 截断更新到秩 rank_frac*min(shape)
        self.params = list(params); self.state = {}
    def zero_grad(self):
        for p in self.params:
            p.grad = None
    def _orth(self, W):
        if self.rank_frac > 0:        # 因果干预: SVD截断更新到低秩(破坏Muon满秩更新)
            U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
            r = max(1, int(self.rank_frac * S.shape[0]))
            O = U[:, :r] @ Vt[:r, :]
        else:
            O = newton_schulz(W)
        return O / (O.pow(2).mean().sqrt() + 1e-8)
    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            st = self.state.setdefault(id(p), {})
            if "m" not in st: st["m"] = torch.zeros_like(p)
            st["m"].mul_(self.beta).add_(p.grad)
            m = st["m"]
            if p.dim() == 4 and p.shape[1] > 3:      # conv: [out,in,kh,kw] -> [out, in*kh*kw]
                O = self._orth(m.reshape(m.shape[0], -1))
                p.add_(O.reshape(p.shape), alpha=-self.lr * 3)
            elif p.dim() == 2 and min(p.shape) > 1:   # linear
                O = self._orth(m)
                p.add_(O, alpha=-self.lr * 3)
            else:                                      # BN affine, bias, 3-ch stem
                p.add_(m, alpha=-self.lr)


class HybridOpt:
    """层定位实验: muon_ids 里的参数走 Muon, 其余走 Adam. 测可塑性增益来自哪些层."""
    def __init__(self, params, muon_ids, lr, beta=0.9):
        self.lr = lr; self.beta = beta; self.params = list(params)
        self.muon_ids = muon_ids; self.state = {}; self.t = 0
    def zero_grad(self):
        for p in self.params: p.grad = None
    @torch.no_grad()
    def step(self):
        self.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8
        for p in self.params:
            if p.grad is None: continue
            st = self.state.setdefault(id(p), {})
            if id(p) in self.muon_ids and (p.dim() == 4 and p.shape[1] > 3 or p.dim() == 2 and min(p.shape) > 1):
                if "m" not in st: st["m"] = torch.zeros_like(p)
                st["m"].mul_(self.beta).add_(p.grad)
                W = st["m"].reshape(st["m"].shape[0], -1) if p.dim() == 4 else st["m"]
                O = newton_schulz(W); O = O / (O.pow(2).mean().sqrt() + 1e-8)
                p.add_(O.reshape(p.shape), alpha=-self.lr * 3)
            else:                       # Adam
                if "ea" not in st: st["ea"] = torch.zeros_like(p); st["eas"] = torch.zeros_like(p)
                st["ea"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["eas"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                mh = st["ea"] / (1 - b1 ** self.t); vh = st["eas"] / (1 - b2 ** self.t)
                p.addcdiv_(mh, vh.sqrt().add_(eps), value=-self.lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="adam", choices=["adam", "cbp", "ptc", "adamo", "ptc_adamo", "muon", "ptc_muon",
                                                          "sgd", "adamw", "l2init", "redo", "shrink_perturb"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_tasks", type=int, default=30)
    ap.add_argument("--epochs_per_task", type=int, default=2)
    ap.add_argument("--n_per_task", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--ptc_vreset", type=float, default=0.3)
    ap.add_argument("--reperturb_frac", type=float, default=0.05)
    ap.add_argument("--adamo_lambda", type=float, default=0.0005)
    ap.add_argument("--l2init_lambda", type=float, default=0.001)
    ap.add_argument("--sp_shrink", type=float, default=0.4)   # shrink-perturb: 收缩系数
    ap.add_argument("--sp_noise", type=float, default=0.01)
    ap.add_argument("--redo_tau", type=float, default=0.1)    # ReDo: 休眠阈值
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--classes_per_task", type=int, default=2)
    ap.add_argument("--no_vreset", action="store_true")     # 消融: 关掉优化器状态(动量/v)软重置
    ap.add_argument("--no_reperturb", action="store_true")  # 消融: 关掉 rank-deficit 重扰动
    ap.add_argument("--log_rank", action="store_true")      # 机理: 逐task记penultimate有效秩
    ap.add_argument("--log_ntk", action="store_true")       # 机理: 末态测NTK条件数/有效秩+权重谱(直接dynamical isometry)
    ap.add_argument("--bottleneck", type=int, default=0)    # 因果: 线性瓶颈硬限(最终)特征秩≤k
    ap.add_argument("--mid_bottleneck", type=int, default=0)  # 因果: l3后channel瓶颈硬限中间层秩≤k
    ap.add_argument("--log_adapt", action="store_true")  # 测每epoch适应速度
    ap.add_argument("--muon_rank_frac", type=float, default=0.0)  # 因果: 截断Muon更新到秩 frac*min(shape)
    ap.add_argument("--disjoint", action="store_true")  # 标准class-incremental: 不相交类划分(无复用)
    ap.add_argument("--muon_blocks", default="", choices=["", "early", "mid", "late", "all", "none"])  # 层定位: Muon用在哪些块
    ap.add_argument("--data", default="/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/dataset/cifar-10-batches-py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    X, Y = load_cifar(args.data, args.dataset == "cifar100")
    NC = 100 if args.dataset == "cifar100" else 10
    X, Y = X.to(dev), Y.to(dev); N = X.shape[0]
    model = ResNet18(args.classes_per_task, args.width, args.bottleneck, args.mid_bottleneck).to(dev); params = list(model.parameters())
    theta_init = [p.detach().clone() for p in params]   # for l2init
    if args.muon_blocks:   # 层定位: 按块划分 Muon/Adam (RP仍施加, 对比 muon+RP 增益来自哪)
        block_map = {"early": ("c1", "b1", "l1", "l2"), "mid": ("l2", "l3"),
                     "late": ("l3", "l4", "fc"), "all": None, "none": ()}
        pref = block_map[args.muon_blocks]
        muon_ids = set()
        for name, p in model.named_parameters():
            top = name.split(".")[0]
            if pref is None or top in pref: muon_ids.add(id(p))
        opt = HybridOpt(params, muon_ids, lr=args.lr)
    elif args.method in ("muon", "ptc_muon"):
        opt = MuonOpt(params, lr=args.lr, rank_frac=args.muon_rank_frac)
    elif args.method == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr * 10, momentum=0.9)
    elif args.method == "adamw":
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    else:
        opt = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.RandomState(args.seed)
    xprobe = X[torch.from_numpy(rng.choice(N, 512, replace=False)).to(dev)]
    task_acc = []; rank_traj = []; test_acc = []; adapt_traj = []

    for t in range(args.n_tasks):
        if t > 0:
            with torch.no_grad():
                if args.muon_blocks:   # 层定位: RP施加(对齐muon+RP)
                    reset_low_util(model, args.reperturb_frac)
                elif args.method == "cbp":
                    reset_low_util(model, args.reperturb_frac)
                elif args.method in ("ptc", "ptc_adamo"):
                    if not args.no_vreset:
                        for p in params:
                            st = opt.state.get(p, {})
                            if "exp_avg_sq" in st: st["exp_avg_sq"].mul_(args.ptc_vreset)
                            if "exp_avg" in st: st["exp_avg"].mul_(args.ptc_vreset)
                    if not args.no_reperturb:
                        reset_low_util(model, args.reperturb_frac)
                elif args.method == "ptc_muon":      # 软重置 Muon 动量缓冲 + rank-deficit 重扰动
                    if not args.no_vreset:
                        for p in params:
                            st = opt.state.get(id(p), {})
                            if "m" in st: st["m"].mul_(args.ptc_vreset)
                    if not args.no_reperturb:
                        reset_low_util(model, args.reperturb_frac)
                elif args.method == "shrink_perturb":   # 全权重收缩+加噪
                    for p in params:
                        p.mul_(args.sp_shrink).add_(torch.randn_like(p) * args.sp_noise)
                elif args.method == "redo":             # 重置休眠(低激活)conv通道
                    reset_dormant(model, xprobe, args.redo_tau)
        k = args.classes_per_task
        if k == 2:
            groupA = set(rng.choice(NC, NC // 2, replace=False).tolist())
            sel = torch.from_numpy(rng.choice(N, args.n_per_task, replace=False)).to(dev)
            Xt = X[sel]; Yt = torch.tensor([0 if int(y) in groupA else 1 for y in Y[sel]], device=dev)
        else:
            # k-way: 选k类, relabel 0..k-1; 留出 held-out 测试集(同类不同样本)
            if args.disjoint:   # 标准 class-incremental: 不相交划分(task t 用第t组)
                if t == 0: disj = rng.permutation(NC)
                cls = disj[t * k:(t + 1) * k]
            else:               # 默认: 随机选k类(跨任务复用, 长流可塑性)
                cls = rng.choice(NC, k, replace=False)
            cmap = {c: i for i, c in enumerate(cls)}
            mask = torch.zeros(N, dtype=torch.bool, device=dev)
            for c in cls: mask |= (Y == c)
            idx_all = mask.nonzero(as_tuple=True)[0]
            perm = torch.from_numpy(rng.permutation(len(idx_all))).to(dev)
            idx_all = idx_all[perm]
            ntr = min(args.n_per_task, len(idx_all) - 500)
            sel = idx_all[:ntr]; sel_te = idx_all[ntr:ntr + 1000]
            Xt = X[sel]; Yt = torch.tensor([cmap[int(y)] for y in Y[sel]], device=dev)
            Xte = X[sel_te]; Yte = torch.tensor([cmap[int(y)] for y in Y[sel_te]], device=dev)
        correct = tot = 0; ntr_act = Xt.shape[0]
        ep_test = []   # 每epoch后的held-out测试acc(适应速度)
        model.train()
        for ep in range(args.epochs_per_task):
            order = torch.randperm(ntr_act, device=dev)
            for i in range(0, ntr_act, args.bs):
                idx = order[i:i + args.bs]; xb, yb = Xt[idx], Yt[idx]
                logits = model(xb)
                correct += (logits.argmax(1) == yb).sum().item(); tot += len(yb)
                opt.zero_grad(); F.cross_entropy(logits, yb).backward()
                if args.method == "l2init":
                    for p, p0 in zip(params, theta_init):
                        if p.grad is not None: p.grad.add_(p - p0, alpha=args.l2init_lambda)
                opt.step()
                if args.method in ("adamo", "ptc_adamo"):
                    with torch.no_grad():
                        for p in params:
                            if p.dim() >= 2:
                                W = p.data.reshape(p.shape[0], -1)
                                if W.shape[0] <= W.shape[1]:
                                    G = W @ W.T - torch.eye(W.shape[0], device=W.device)
                                    p.data.add_((G @ W).reshape(p.shape), alpha=-args.adamo_lambda * 4)
            if args.log_adapt and k > 2:   # 每epoch后held-out测试acc = 适应速度
                model.eval()
                with torch.no_grad():
                    tc = sum((model(Xte[j:j+256]).argmax(1) == Yte[j:j+256]).sum().item() for j in range(0, Xte.shape[0], 256))
                ep_test.append(100.0 * tc / Xte.shape[0]); model.train()
        if args.log_adapt and ep_test: adapt_traj.append(ep_test)
        task_acc.append(100.0 * correct / tot)
        if k > 2:   # held-out 测试精度(同任务新样本上的泛化, 非online拟合)
            model.eval()
            with torch.no_grad():
                tc = sum((model(Xte[j:j+256]).argmax(1) == Yte[j:j+256]).sum().item() for j in range(0, Xte.shape[0], 256))
            test_acc.append(100.0 * tc / Xte.shape[0]); model.train()
        if args.log_rank:
            rank_traj.append(eff_rank_resnet(model, Xt[:512]))

    ntk = ntk_diagnostics(model, xprobe, n=48) if args.log_ntk else None
    blockrank = block_ranks(model, xprobe) if args.log_ntk else None
    early = float(np.mean(task_acc[:5])); late = float(np.mean(task_acc[-10:]))
    late_test = float(np.mean(test_acc[-10:])) if test_acc else None
    res = {"method": args.method, "seed": args.seed, "early_acc": early, "late_acc": late,
           "late_test_acc": late_test, "test_acc": test_acc, "ntk": ntk, "blockrank": blockrank, "adapt_traj": adapt_traj,
           "plasticity_drop": early - late, "mean_acc": float(np.mean(task_acc)), "task_acc": task_acc,
           "rank_traj": rank_traj}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"DONE {args.method} s{args.seed}: late_acc={late:.2f}% (early {early:.2f}, drop {early-late:+.2f})")


if __name__ == "__main__":
    main()
