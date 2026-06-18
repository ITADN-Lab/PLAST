# AAAI 2026 Reproducibility Checklist

**Paper:** Muon Gets Dynamical Isometry for Free: Orthogonalizing Optimizers Preserve Plasticity in Continual Learning

This document records our answers to the AAAI reproducibility checklist with brief justifications. (Format: **Yes / Partial / No / NA**.)

---

## 1. General

- **Includes a conceptual outline and/or pseudocode description of AI methods introduced** — **Yes.** Algorithm 1 gives the Muon-base-for-plasticity procedure; §Method describes it in prose.
- **Clearly delineates statements that are opinions, hypothesis, and speculation from objective, verifiable claims** — **Yes.** §Mechanism explicitly labels which propositions are exact and which element ("an adaptive preconditioner concentrates over a stream") is empirical; §Scope marks bounding/negative claims.
- **Provides well-marked pedagogical references for less-familiar readers** — **Yes.** Background section reviews the NTK/dynamical-isometry framing with citations.

## 2. Theoretical Contributions

**Does this paper make theoretical contributions?** **Yes.**

- **All assumptions and restrictions are stated clearly and formally** — **Yes.** Definition 1 (stable rank); Prop. 1 states the full-rank-momentum and exact-orthogonalization-limit assumptions.
- **All novel claims are stated formally** — **Yes.** Prop. 1, Lemma 1, Prop. 2, Cor. 1.
- **Proofs of all novel claims are included** — **Yes** for the exact claims (Prop. 1, Lemma 1 have proofs). Prop. 2's adaptive-side statement is explicitly flagged as an empirical regularity, not a theorem.
- **Proof sketches or intuitions are given for complex/informal proofs** — **Yes.** The increment-to-feature-rank argument and the connection to AdamO's NTK decomposition (Cor. 1) are explained intuitively.
- **Appropriate citations to theoretical tools are given** — **Yes.** NTK decomposition cited to AdamO (Rosseau et al. 2026); Newton–Schulz/Muon cited.
- **All theoretical claims are demonstrated empirically to hold** — **Yes.** Fig. (rank trajectory) verifies the rank prediction; Fig. (rank–accuracy) shows r=0.77.
- **All experimental code used to eliminate or disprove claims is included** — **Yes.** Ablation/decomposition code is released.

## 3. Datasets

**Does this paper rely on one or more datasets?** **Yes.**

- **A motivation is given for the selected datasets** — **Yes.** CIFAR-100/CIFAR-10/permuted-MNIST are the standard plasticity benchmarks; permuted-MNIST matches AdamO's reported MLP setting.
- **All novel datasets introduced are included in a data appendix** — **NA.** No novel datasets; all are standard.
- **All novel datasets will be made publicly available** — **NA.**
- **All datasets from existing literature are accompanied by appropriate citations** — **Yes.** CIFAR (Krizhevsky), MNIST, and the continual protocols are cited.
- **All datasets from existing literature are publicly available** — **Yes.** CIFAR-10/100, MNIST, CartPole are all public.
- **Datasets not publicly available are described in detail** — **NA.**

## 4. Computational Experiments

**Does this paper include computational experiments?** **Yes.**

- **Code required for pre-processing data is included** — **Yes.** Data loaders are in the released code.
- **All source code for conducting and analyzing experiments is included in a code appendix** — **Yes.** `plast_resnet.py`, `plast_mnist.py`, `plast_rl.py`, `aggregate.py`, and driver scripts are released.
- **All source code will be made publicly available upon publication** — **Yes.** (Anonymized repository during review.)
- **Source code implementing new methods has comments describing the implementation** — **Yes.**
- **If an algorithm depends on randomness, the method for setting seeds is described** — **Yes.** Seeds are set for `torch`, `numpy`, and `random`; we report n=5 seeds (n=3 for secondary sweeps).
- **Specifies the computing infrastructure used** — **Yes.** Single GPU; the full study (>200 runs) completes in a few hours (§Implementation details).
- **Formally describes evaluation metrics and motivates them** — **Yes.** Late-task online accuracy (mean over the final 10 tasks) is defined and motivated as the plasticity measure; effective rank is defined formally.
- **States the number of algorithm runs used to compute each reported result** — **Yes.** n=5 (n=3 for sweeps), stated in captions/tables.
- **Analysis goes beyond single-dimensional summaries (e.g., reports variation)** — **Yes.** All headline results report mean ± std; trajectories and scatter plots are shown.
- **Significance of improvements judged using appropriate statistical tests** — **Yes.** Welch's t-test reported for the main comparisons (e.g., t=8.4 for the headline).
- **Lists all final hyperparameters for each model/algorithm** — **Yes.** §Implementation details + §Hyperparameter sensitivity (Muon lr, AdamO λ, β, RP fraction ρ, NS steps, etc.).
- **States the number and range of hyperparameter values tried during development** — **Yes.** AdamO λ ∈ {5e-4 … 1e-1} and Muon lr ∈ {3e-4, 1e-3, 3e-3} are reported with a sensitivity figure.

---

**Summary:** Yes/NA on all applicable items. The only "Partial"-flavored nuance is that one component of Prop. 2 (adaptive-preconditioner concentration) is established empirically rather than proven, which we state explicitly in the paper.
