# AAAI 投稿(新):Muon Gets Dynamical Isometry for Free

> 注:本文件夹早先被 RAPM 项目用过(见 README.md,已转投 Information Sciences)。
> 本 plasticity/Muon 工作的全部内容在 code/ paper/ results/ reference/ logs/ 子目录,与 RAPM 互不干扰。

**一句话**:正交化优化器(Muon)在持续学习可塑性上**天然**维持良态条件数,无需任何正则即决定性超过 ICML'26 的权重等距 SOTA(AdamO)。

## 核心结果(全公平、全 kill-test 通过)
跨 3 基准,Muon-based vs AdamO-based(双方最优超参 + 对等重扰动):

| 基准 | Muon | AdamO | Δ |
|---|---|---|---|
| CIFAR-100 (ResNet-18) | **77.2** | 67.9 | **+9.3 (t=8.4, n=5)** |
| CIFAR-10 (ResNet-18) | **84.6** | 81.6 | +3.0 (t=3.9) |
| permuted-MNIST (MLP) | **70.0** | 65.8 | +4.2 |

**机理**:Muon 无正则维持 penultimate 有效秩 **167** > AdamO 最优 λ 才 **107** > Adam **59**。
**2×2 分解**:赢的是 Muon 正交化底座(CBP-on-Adam 仅 51.6,远低于 Muon+RP 78.2)。

## 诚实边界(写进论文 Scope 节)
1. 裸 Muon(最优 lr 64.0)在深 ResNet 上**输**给调好的 AdamO(69.3);稳健 claim 是对等 +RP 比较。
2. 优化器状态 re-conditioning(PTC)仅 MLP 有效(permuted +7 超 AdamO),ResNet 上≈0。
3. Muon permuted"崩溃"是 LR 假象(仅 lr1e-3 崩,lr1e-4 正常 70)。

## 文件
- `paper/main.tex` `paper/main.pdf` — 论文(4 页 2 栏,5 表 2 图 1 算法,编译OK)。⚠️ 待套官方 aaai2026.sty(官网 kit 下载 404,需手动)。
- `paper/fig_main_c100.png` 主结果柱状图;`paper/fig_rank_traj.png` 有效秩轨迹(机理)
- `results/SCOREBOARD.md` — 全部实验诚实记录 + 公平性检查
- `reference/AdamO_*.pdf` — 对手原文(ICML'26)
- `code/plast_resnet.py` — 主脚本(MuonOpt / ptc_muon / --no_vreset/--no_reperturb 消融 / --log_rank / eff_rank_resnet)
- `code/plast_mnist.py` — permuted/class-inc MLP(加了 --depth)
- `code/aggregate.py` — 结果聚合+t检验;`code/run_*.sh` — 各批次驱动

## 复现关键配置
- CIFAR-100: ResNet-18 w32, 25task×10way, 5000img, 2ep/task; Muon lr1e-3, AdamO λ=3e-2
- 重扰动(RP)= reset_low_util，对 Muon/AdamO 对等施加
- 数据: CIFAR-100 在 LafTJU-TII/experiments/dataset/; CIFAR-10 在 TJU-V5(ATJU)/ATJU/dataset/

## 待办
- 套官方 AAAI 模板 + 扩到 6-7 页
- 理论:形式化 update-stable-rank → feature-rank(目前 §Mechanism 有非形式论证)
- (可选,风险)RL 可塑性 — AdamO 测过,但 RL 噪声大,现列为 future work

## 命名注意
论文里的 "PTC"(state re-conditioning)与 TJU/KF-PTC/TRACER **无关**,只共用名字隐喻,机制完全不同;且已降为次要结果。可考虑改名 StateRecon 避免 reviewer 混淆。
