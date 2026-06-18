"""持续学习可塑性 benchmark (自包含, 无数据依赖).
任务流: 每个任务一个随机目标网络生成标签; 学习器在线训练有限步;
可塑性 = 每个任务能达到的拟合度; 可塑性丧失 = 后任务拟合越来越差.
这是 Dohare Nature2024 "slowly/abruptly changing regression" 的标准设定.
"""
import torch
import torch.nn as nn


def make_target(in_dim, hidden, seed, dev):
    """随机目标网络 (固定权重, 生成非线性标签)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 1))
    for p in net.parameters():
        p.data = torch.randn(p.shape, generator=g) * (2.0 / p.shape[-1] ** 0.5)
    return net.to(dev).eval()


class Learner(nn.Module):
    def __init__(self, in_dim, hidden, depth=2, act="relu"):
        super().__init__()
        A = {"relu": nn.ReLU, "tanh": nn.Tanh}[act]
        layers = [nn.Linear(in_dim, hidden), A()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), A()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def effective_rank(model, x):
    """学习器隐表征的 effective rank (可塑性的结构指标; 低=神经元死亡)."""
    h = x
    feats = None
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, (nn.ReLU, nn.Tanh)):
            feats = h
    s = torch.linalg.svdvals(feats - feats.mean(0, keepdim=True))
    s = s / (s.sum() + 1e-12)
    return float(torch.exp(-(s * (s + 1e-12).log()).sum()))   # 谱熵 = effective rank
