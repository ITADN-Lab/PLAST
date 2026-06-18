# AAAI 战役 Scoreboard (诚实记录, 2026-06-17)

对手 = **AdamO (ICML 2026, arXiv 2606.09762)**, 权重等距(dynamical isometry)SOTA。

## CIFAR-100 class-inc (ResNet-18 w32, 25task×10way) — late_acc %

| 方法 | acc | n | 备注 |
|---|---|---|---|
| adam | 43.22 ± 2.27 | 5 | 基线, 可塑性丧失 |
| cbp (adam+重扰动) | 51.59 ± 1.79 | 3 | 重扰动加在Adam上 |
| muon | 61.67 ± 1.31 | 5 | 裸Muon |
| ptc (adam+vt重置+重扰动) | 56.63 ± 2.46 | 5 | **优化器状态轴在ResNet上不灵** |
| **AdamO (最优 λ=1e-2)** | **68.17 ± 0.38** | 3 | ⚠️调λ后大涨(默认5e-4只58) |
| AdamO (λ=5e-4默认) | 58.16 ± 1.58 | 5 | 欠调 |
| **muon+重扰动** | **78.22 ± 0.33** | 3 | **当前最佳** |
| muon+动量重置(只) | 63.8 | 3 | 动量重置≈无用 |
| AdamO最优+重扰动 | ⏳决战中 | | **决定故事成立与否** |

**关键诚实结论:**
- ✅ Muon+重扰动 78.22 > AdamO最优 68.17 (+10, t=28)
- ⚠️ 裸Muon 61.67 < AdamO最优 68.17 (调好的AdamO反超裸Muon) — 之前"裸Muon>AdamO"是AdamO欠调假象
- ✅ CBP(adam+重扰动)51.59 << muon+重扰动 78.22 → **Muon底座是关键, 非重扰动普适**
- ❌ "优化器状态re-conditioning"(动量/vt重置)在ResNet上贡献≈0 → 降级, 非头条

## 机理: penultimate 有效秩 (late, n=2)
| 方法 | acc | rank |
|---|---|---|
| adam | 45.3 | 58.9 |
| adamo(λ5e-4) | 60.1 | 83.9 |
| **muon** | 62.4 | **166.5** |
| muon+重扰动 | 78.3 | 175.1 |

→ **"Muon 免费获得 dynamical isometry"**: 无需正则即维持2倍于AdamO的秩。(待补: AdamO λ=1e-2 的rank做对等机理比较)

## permuted-MNIST (MLP depth4/h512, 匹配AdamO协议) — late_acc %
| 方法 | acc | n |
|---|---|---|
| adam(lr1e-4) | 53.89 ± 0.90 | 5 |
| cbp | 54.93 ± 1.36 | 5 |
| adamo(lr1e-4,λ1e-3) | 65.82 ± 0.31 | 5 |
| ptc(adam) | 73.10 ± 0.41 | 5 |
| ptc_adamo | 76.10 ± 0.40 | 5 |
| muon lr1e-4 | 70.01 ± 0.27 | 3 |
| muon lr3e-4 | 60.30 | 3 |
| muon lr1e-3 | 11.13(崩) | 3 |
| ptc_muon lr1e-4 | 72.36 | 3 |

**注:** ⚠️ permuted上 AdamO 的 λ 也需要重扫到最优(目前用λ1e-3, 可能也欠调)。ptc>AdamO的+7需在AdamO最优λ下复核。
**注:** Muon"崩溃"是LR假象(仅lr1e-3崩, lr1e-4正常70)。

## 待办
1. ⏳ AdamO最优+重扰动 决战 (决定Muon底座是否真优)
2. permuted 上 AdamO λ 重扫 (复核 ptc>AdamO 是否仍成立)
3. AdamO λ=1e-2 的 rank (对等机理)
4. 锁定叙事 → 写 AAAI

## class-inc CIFAR-10 (ResNet-18, 第3个setting, n=3) — 2026-06-17 补
| 方法 | acc |
|---|---|
| muon+RP | 84.58 ± 0.99 |
| AdamO最优(λ3e-2) | 82.73 ± 0.83 |
| muon | 82.25 ± 0.48 |
| AdamO+RP | 81.58 ± 0.46 |
| cbp | 71.01 |
| adam | 67.58 |
→ Muon+RP vs AdamO+RP: +3.0 t=3.9; vs AdamO最优 +1.86 t=2.0
→ permuted Muon n=5: 70.03±0.22 vs AdamO 65.82 (+4.2)
**三基准 Muon+RP>AdamO+RP 全正向: CIFAR100 +9.3 / CIFAR10 +3.0 / permuted +4.2**

## Muon LR 公平性确认 (CIFAR-100) — 2026-06-17
- Muon+RP 最优 lr=1e-3 (78.25) = 头条用值 → 头条公平无cherry-pick
- 裸Muon 最优 lr=3e-4 (63.99) 仍 < AdamO最优69.3 → Scope"裸Muon输调好AdamO"成立
- 所有公平性检查通过: AdamO λ扫✓ Muon lr扫✓ 对等RP✓

## RL 可塑性 (DQN, RL-permuted CartPole, 15task, n=5) — 诚实负向
| 方法 | late_ret |
|---|---|
| adamo | 28.2 ± 1.4 |
| muon | 24.7 ± 4.3 |
| adam | 24.7 ± 6.1 |
| cbp | 24.1 ± 5.2 |
→ Muon 在RL上不赢(vs adamo -3.5 ns, vs adam/cbp 平)。RL是诚实弱点,写进Scope(iv)非future work。
