# Cover Letter / Submission Statement — AAAI 2026

**Title:** Muon Gets Dynamical Isometry for Free: Orthogonalizing Optimizers Preserve Plasticity in Continual Learning
**Track:** Main Technical Track — Machine Learning (continual / lifelong learning; optimization)
**Submission type:** Full paper (7 pages), double-blind.

---

Dear Program Committee,

We submit our paper for consideration at AAAI 2026. Below we summarize the contribution, its significance, and the steps we took to make the claims trustworthy.

## What the paper shows

Loss of plasticity—the progressive inability of a continually trained network to fit new tasks—has been attacked almost exclusively by adding machinery *around* a fixed Adam optimizer: resetting units (Continual Backprop, ReDo) or regularizing weights toward isometry. The strongest current method, **AdamO (ICML 2026)**, frames plasticity through the empirical NTK and adds an explicit weight-space dynamical-isometry penalty to Adam.

Our paper makes a single, consequential observation that this line of work has not tested: **the modern Muon optimizer, which orthogonalizes its update at every step, attains the dynamical isometry that AdamO regularizes toward—for free.** Concretely:

- **Decisive empirical result.** Under a fully matched comparison (both optimizers tuned to their optima, both given identical unit reperturbation), a Muon base beats AdamO by **+9.3 points (t=8.4, n=5)** on class-incremental CIFAR-100, with the advantage holding across three supervised benchmarks (CIFAR-100, CIFAR-10, permuted-MNIST).
- **Transparent mechanism.** Muon sustains a penultimate effective rank of ~167 with no plasticity regularizer, versus ~107 for AdamO at its best-tuned penalty and ~59 for Adam; across methods, effective rank predicts plasticity (r=0.77).
- **Theory.** We prove that Muon's increments are exactly isometric (maximal stable rank) and that this upper-bounds the attainable representational rank, connecting directly to AdamO's own layerwise-NTK decomposition.

## Why it is a good fit for AAAI

The result *reframes* an active problem: the choice of optimizer—not an added regularizer—may be the primary lever for plasticity. It engages the very recent state of the art (AdamO) on its own theoretical terms and beats it, while the prescription is a one-line change to the training loop. The work is squarely of interest to the continual-learning and optimization communities at AAAI.

## Commitment to rigor and honesty

We were deliberate about not overclaiming, and the paper reports several negative or bounding results in the main text rather than hiding them:

1. We sweep **AdamO's regularization strength to its peak** on every benchmark (using its default would understate it by up to 11 points) and sweep Muon's learning rate likewise, so the gap is not a tuning artifact.
2. We show plain Muon **loses** to a strongly-regularized AdamO on deep ResNets unless paired with reperturbation; the robust claim is the matched comparison.
3. We show a competing optimizer-state re-conditioning lever helps **only on MLPs**, and that an apparent Muon "collapse" is a learning-rate artifact.
4. We report that the advantage **does not transfer to value-based RL**, as an honest negative boundary rather than a win.

A 2×2 optimizer×reperturbation decomposition isolates the orthogonalizing optimizer as the cause of the gain. All results are multi-seed with Welch's t-tests; code and per-seed logs are released for reproducibility.

## Reproducibility

The paper includes full architectural and hyperparameter details and a reproducibility statement. The complete study (>200 runs) runs on a single GPU in a few hours.

We thank the committee and reviewers for their time.

— The Authors (anonymized for double-blind review)
