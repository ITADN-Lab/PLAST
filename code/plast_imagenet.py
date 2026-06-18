"""ImageNet-100 class-incremental plasticity (规模化验证, 回应 reviewer 对 scale 的攻击).
真 ResNet-18 (torchvision, 7x7 stem+maxpool, 128x128 输入). 方法: adam/muon/adamo/ptc_muon/cbp/redo/l2init.
late-task online acc = 可塑性. 数据: ImageFolder train/.
"""
import os, json, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image


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
    def __init__(self, params, lr, beta=0.9):
        self.lr = lr; self.beta = beta; self.params = list(params); self.state = {}
    def zero_grad(self):
        for p in self.params: p.grad = None
    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            st = self.state.setdefault(id(p), {})
            if "m" not in st: st["m"] = torch.zeros_like(p)
            st["m"].mul_(self.beta).add_(p.grad); m = st["m"]
            if p.dim() == 4 and p.shape[1] > 3:
                O = newton_schulz(m.reshape(m.shape[0], -1)); O = O / (O.pow(2).mean().sqrt() + 1e-8)
                p.add_(O.reshape(p.shape), alpha=-self.lr * 3)
            elif p.dim() == 2 and min(p.shape) > 1:
                O = newton_schulz(m); O = O / (O.pow(2).mean().sqrt() + 1e-8)
                p.add_(O, alpha=-self.lr * 3)
            else:
                p.add_(m, alpha=-self.lr)


@torch.no_grad()
def reset_low_util(model, frac):
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.weight.shape[1] > 3]
    for c in convs:
        w = c.weight; no = w.shape[0]
        util = w.detach().abs().reshape(no, -1).mean(1)
        k = max(1, int(no * frac)); idx = torch.argsort(util)[:k]
        w.data[idx] = torch.randn_like(w.data[idx]) * (2.0 / w[0].numel()) ** 0.5


@torch.no_grad()
def reset_dormant(model, xprobe, tau):
    acts = {}; hooks = []
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.weight.shape[1] > 3]
    def mk(name):
        def hook(mod, inp, out): acts[name] = out.abs().mean(dim=(0, 2, 3))
        return hook
    for i, c in enumerate(convs): hooks.append(c.register_forward_hook(mk(i)))
    model.eval(); model(xprobe); model.train()
    for h in hooks: h.remove()
    for i, c in enumerate(convs):
        a = acts.get(i)
        if a is None: continue
        idx = (a / (a.mean() + 1e-9) < tau).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            c.weight.data[idx] = torch.randn_like(c.weight.data[idx]) * (2.0 / c.weight[0].numel()) ** 0.5


def ntk_diagnostics(model, head, xprobe, n=32):
    model.eval()
    xb = xprobe[:n]
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(xb.shape[0]):
        out = head(model(xb[i:i+1])).sum()
        g = torch.autograd.grad(out, params, retain_graph=False)
        rows.append(torch.cat([gi.reshape(-1) for gi in g]))
    J = torch.stack(rows); K = (J @ J.T).double()
    ev = torch.linalg.eigvalsh(K).clamp_min(1e-12); p_ = ev / ev.sum()
    model.train()
    return {"ntk_erank": float(torch.exp(-(p_ * p_.log()).sum())), "ntk_cond": float(ev[-1] / ev[0])}


def load_class_index(root, res):
    """返回 {class_idx: [图片路径...]} 和 transform."""
    classes = sorted(os.listdir(root))
    tf = transforms.Compose([transforms.Resize((res, res)), transforms.ToTensor()])
    cls_paths = {}
    for ci, c in enumerate(classes):
        d = os.path.join(root, c)
        fs = [os.path.join(d, f) for f in os.listdir(d)][:600]   # 每类≤600加速
        cls_paths[ci] = fs
    return cls_paths, tf


def load_batch(paths, tf, dev):
    imgs = []
    for p in paths:
        try: imgs.append(tf(Image.open(p).convert("RGB")))
        except: imgs.append(torch.zeros(3, tf.transforms[0].size[0], tf.transforms[0].size[0]))
    return torch.stack(imgs).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="adam", choices=["adam", "muon", "adamo", "ptc_muon", "cbp", "redo", "l2init"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_tasks", type=int, default=20)
    ap.add_argument("--classes_per_task", type=int, default=5)
    ap.add_argument("--epochs_per_task", type=int, default=2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--ptc_vreset", type=float, default=0.5)
    ap.add_argument("--reperturb_frac", type=float, default=0.05)
    ap.add_argument("--adamo_lambda", type=float, default=3e-2)
    ap.add_argument("--l2init_lambda", type=float, default=1e-3)
    ap.add_argument("--redo_tau", type=float, default=0.1)
    ap.add_argument("--log_ntk", action="store_true")
    ap.add_argument("--data", default="/home/hadoop/imagenet100/train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = "cuda"
    cls_paths, tf = load_class_index(args.data, args.res)
    NC = len(cls_paths); rng = np.random.RandomState(args.seed)
    backbone = resnet18(num_classes=args.classes_per_task)
    feat_dim = backbone.fc.in_features
    head = backbone.fc; backbone.fc = nn.Identity()
    backbone = backbone.to(dev); head = head.to(dev)
    model = nn.Sequential(); model.backbone = backbone; model.head = head  # for module iteration
    params = list(backbone.parameters()) + list(head.parameters())
    theta_init = [p.detach().clone() for p in params]
    def fwd(x): return head(backbone(x))

    if args.method in ("muon", "ptc_muon"): opt = MuonOpt(params, lr=args.lr)
    else: opt = torch.optim.Adam(params, lr=args.lr)

    # 固定探针(第一个任务的图)
    xprobe = None
    task_acc = []
    for t in range(args.n_tasks):
        if t > 0:
            with torch.no_grad():
                if args.method == "cbp": reset_low_util(backbone, args.reperturb_frac)
                elif args.method == "redo": reset_dormant(backbone, xprobe, args.redo_tau)
                elif args.method == "ptc_muon":
                    for p in params:
                        st = opt.state.get(id(p), {})
                        if "m" in st: st["m"].mul_(args.ptc_vreset)
                    reset_low_util(backbone, args.reperturb_frac)
        cls = rng.choice(NC, args.classes_per_task, replace=False)
        paths = []; labels = []
        for li, c in enumerate(cls):
            for p in cls_paths[c]: paths.append(p); labels.append(li)
        labels = torch.tensor(labels, device=dev)
        idx_all = np.arange(len(paths))
        correct = tot = 0
        backbone.train()
        for ep in range(args.epochs_per_task):
            rng.shuffle(idx_all)
            for i in range(0, len(idx_all), args.bs):
                bidx = idx_all[i:i + args.bs]
                xb = load_batch([paths[j] for j in bidx], tf, dev); yb = labels[bidx]
                if xprobe is None: xprobe = xb[:32].clone()
                logits = fwd(xb)
                correct += (logits.argmax(1) == yb).sum().item(); tot += len(yb)
                opt.zero_grad(); loss = F.cross_entropy(logits, yb); loss.backward()
                if args.method == "l2init":
                    for p, p0 in zip(params, theta_init):
                        if p.grad is not None: p.grad.add_(p - p0, alpha=args.l2init_lambda)
                opt.step()
                if args.method == "adamo":
                    with torch.no_grad():
                        for p in params:
                            if p.dim() >= 2:
                                W = p.data.reshape(p.shape[0], -1)
                                if W.shape[0] <= W.shape[1]:
                                    G = W @ W.T - torch.eye(W.shape[0], device=W.device)
                                    p.data.add_((G @ W).reshape(p.shape), alpha=-args.adamo_lambda * 4)
        task_acc.append(100.0 * correct / tot)
        print(f"  task {t}: {task_acc[-1]:.1f}", flush=True)

    ntk = ntk_diagnostics(backbone, head, xprobe) if args.log_ntk else None
    early = float(np.mean(task_acc[:5])); late = float(np.mean(task_acc[-10:]))
    res = {"method": args.method, "seed": args.seed, "early_acc": early, "late_acc": late,
           "ntk": ntk, "task_acc": task_acc}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"DONE {args.method} s{args.seed}: late_acc={late:.2f}% (early {early:.2f})", flush=True)


if __name__ == "__main__":
    main()
