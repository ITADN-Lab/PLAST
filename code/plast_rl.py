"""RL 可塑性 benchmark (Fast TRAC 主场): DQN on CartPole 任务序列.
每任务一个固定随机观测变换(RL版permuted), 逼迫重学; 可塑性 = 每任务达到的return.
方法: adam/cbp/ptc/ptc_l2/trac_official. 公平测 PTC 在 TRAC 主场是否仍竞争.
"""
import os, sys, json, argparse, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


class QNet(nn.Module):
    def __init__(self, obs, act, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, act))

    def forward(self, x):
        return self.net(x)


def utils_of(model):
    lins = [m for m in model.net if isinstance(m, nn.Linear)]
    return lins, [lins[i + 1].weight.abs().mean(0) for i in range(len(lins) - 1)]


@torch.no_grad()
def eff_rank(model, x):
    h = x; feats = None
    for l in model.net:
        h = l(h)
        if isinstance(l, nn.ReLU): feats = h
    s = torch.linalg.svdvals(feats - feats.mean(0, keepdim=True)); s = s / (s.sum() + 1e-12)
    return float(torch.exp(-(s * (s + 1e-12).log()).sum()))


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
    """Q-net 的 2D 权重走 Muon(动量+NS正交化), bias 走动量SGD."""
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
            st["m"].mul_(self.beta).add_(p.grad)
            if p.dim() == 2 and min(p.shape) > 1:
                O = newton_schulz(st["m"]); O = O / (O.pow(2).mean().sqrt() + 1e-8)
                p.add_(O, alpha=-self.lr * 3)
            else:
                p.add_(st["m"], alpha=-self.lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="adam", choices=["adam", "cbp", "ptc", "ptc_l2", "trac_official", "muon", "ptc_muon", "adamo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_tasks", type=int, default=30)
    ap.add_argument("--episodes_per_task", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ptc_vreset", type=float, default=0.1)
    ap.add_argument("--reperturb_frac", type=float, default=0.1)
    ap.add_argument("--rank_target", type=float, default=100.0)
    ap.add_argument("--l2init_lambda", type=float, default=0.01)
    ap.add_argument("--adamo_lambda", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = "cuda"
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]; n_act = env.action_space.n
    q = QNet(obs_dim, n_act, args.hidden).to(dev)
    qt = QNet(obs_dim, n_act, args.hidden).to(dev); qt.load_state_dict(q.state_dict())
    params = list(q.parameters())
    theta_init = [p.detach().clone() for p in params]
    if args.method == "trac_official":
        from trac_optimizer import start_trac; import tempfile
        opt = start_trac(tempfile.mktemp(), torch.optim.Adam)(params, lr=args.lr)
    elif args.method in ("muon", "ptc_muon"):
        opt = MuonOpt(params, lr=args.lr)
    else:
        opt = torch.optim.Adam(params, lr=args.lr)

    buf = deque(maxlen=10000)
    rng = np.random.RandomState(args.seed)
    xprobe = torch.randn(256, obs_dim, device=dev)
    task_ret = []
    gstep = 0

    for t in range(args.n_tasks):
        # 任务边界重条件化
        if t > 0:
            with torch.no_grad():
                if args.method == "cbp":
                    lins, us = utils_of(q)
                    for li, u in enumerate(us):
                        k = max(1, int(len(u) * args.reperturb_frac)); idx = torch.argsort(u)[:k]
                        lins[li].weight.data[idx] = torch.randn_like(lins[li].weight.data[idx]) * (2.0 / lins[li].weight.shape[1] ** 0.5)
                        lins[li].bias.data[idx] = 0.0; lins[li + 1].weight.data[:, idx] *= 0.0
                elif args.method in ("ptc", "ptc_l2", "ptc_muon"):
                    for p in params:
                        st = opt.state.get(id(p) if args.method == "ptc_muon" else p, {})
                        if "exp_avg_sq" in st: st["exp_avg_sq"].mul_(args.ptc_vreset)
                        if "exp_avg" in st: st["exp_avg"].mul_(args.ptc_vreset)
                        if "m" in st: st["m"].mul_(args.ptc_vreset)
                    er = eff_rank(q, xprobe); deficit = max(0.0, (args.rank_target - er) / args.rank_target)
                    if deficit > 0:
                        lins, us = utils_of(q)
                        for li, u in enumerate(us):
                            k = max(1, int(len(u) * args.reperturb_frac * deficit)); idx = torch.argsort(u)[:k]
                            lins[li].weight.data[idx] = torch.randn_like(lins[li].weight.data[idx]) * (2.0 / lins[li].weight.shape[1] ** 0.5)
                            lins[li].bias.data[idx] = 0.0

        # RL版permuted: 固定随机正交变换观测 (逼迫重学)
        W = torch.from_numpy(np.linalg.qr(rng.randn(obs_dim, obs_dim))[0]).float().to(dev)
        eps_greedy = 1.0
        rets = []
        for ep in range(args.episodes_per_task):
            s, _ = env.reset(seed=int(rng.randint(1e6)))
            s = torch.from_numpy(s).float().to(dev) @ W
            done = False; ep_ret = 0; eps_greedy = max(0.05, eps_greedy * 0.95)
            while not done:
                if random.random() < eps_greedy:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad(): a = int(q(s.unsqueeze(0)).argmax(1).item())
                ns, r, term, trunc, _ = env.step(a); done = term or trunc
                ns = torch.from_numpy(ns).float().to(dev) @ W
                buf.append((s, a, r, ns, float(done))); s = ns; ep_ret += r
                # DQN update
                if len(buf) >= 128:
                    batch = random.sample(buf, 128)
                    bs = torch.stack([b[0] for b in batch]); ba = torch.tensor([b[1] for b in batch], device=dev)
                    br = torch.tensor([b[2] for b in batch], device=dev); bns = torch.stack([b[3] for b in batch])
                    bd = torch.tensor([b[4] for b in batch], device=dev)
                    with torch.no_grad():
                        tgt = br + args.gamma * (1 - bd) * qt(bns).max(1)[0]
                    qv = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                    loss = F.smooth_l1_loss(qv, tgt)
                    opt.zero_grad(); loss.backward()
                    if args.method == "ptc_l2":
                        for p, p0 in zip(params, theta_init):
                            if p.grad is not None: p.grad.add_(p - p0, alpha=args.l2init_lambda)
                    opt.step(); gstep += 1
                    if args.method == "adamo":
                        with torch.no_grad():
                            for p in params:
                                if p.dim() == 2:
                                    Wp = p.data
                                    if Wp.shape[0] <= Wp.shape[1]:
                                        G = Wp @ Wp.T - torch.eye(Wp.shape[0], device=Wp.device)
                                        p.data.add_((G @ Wp), alpha=-args.adamo_lambda * 4)
                    if gstep % 200 == 0: qt.load_state_dict(q.state_dict())
            rets.append(ep_ret)
        task_ret.append(float(np.mean(rets[-10:])))  # 该任务后10ep均return

    early = float(np.mean(task_ret[:5])); late = float(np.mean(task_ret[-10:]))
    res = {"method": args.method, "seed": args.seed, "early_ret": early, "late_ret": late,
           "mean_ret": float(np.mean(task_ret)), "task_ret": task_ret}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"DONE {args.method} s{args.seed}: late_ret={late:.1f} (early {early:.1f}) mean={res['mean_ret']:.1f}")


if __name__ == "__main__":
    main()
