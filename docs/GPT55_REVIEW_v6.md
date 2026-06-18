1. New AAAI 2026 acceptance probability: 31% range: 26–36%.

Change vs prior 23–32%: up about +4 points in midpoint. Previously borderline/weak-positive ceiling; now credible borderline-positive, but not safely above the AAAI bar.

2. Yes, both changes materially help.

- Rank correlation fix: strong help. The old `r=0.77 over 5 method means` was close to non-evidence mechanistically. `250 task-level points` plus within-method `+0.44` removes the most obvious “between-method artifact” objection.
- Split CIFAR-100 control: strong help. This directly addresses the concern that the recurring random-task protocol engineered the effect. The smaller but consistent win on disjoint class-incremental CIFAR-100 is important.
- Theory softening: also helps. You removed the claim most likely to get attacked as mathematically overreaching.
- Held-out test retention: necessary and good. Without it, online plasticity-only would still be a major weakness.

3. Remaining high-severity issue that caps it: the mechanism is still overclaimed relative to causality.

The paper’s title and framing say “Rank, Not Kernel Conditioning, Predicts Plasticity,” but the strongest causal test — the bottleneck — partially undercuts the simple rank story. You now say rank is “best predictor, not proven cause,” which is good, but the paper still rhetorically leans on rank as the explanation while admitting head-visible rank is not causal and backbone mechanism remains open.

Other serious caps:

- `Muon+RP` is the real method, not plain Muon, on deep ResNets. That weakens the “orthogonalizing optimizers preserve plasticity” thesis.
- AdamO is ICML 2026 / future-paper framing; if reviewers dislike or distrust that anchor, the comparison may feel odd unless the venue timing makes sense.
- n=3 on the new Split CIFAR-100 control is acceptable as a control, but not fully satisfying.
- Theory remains mostly explanatory intuition: full-rank increments do not imply feature-rank preservation, useful plasticity, or generalization.

4. Is it now at the realistic AAAI ceiling? Mostly yes.

For this version, I’d call the realistic AAAI ceiling about 35–38% unless you add a genuinely stronger causal mechanism or broader independent validation. The empirical package is now strong enough for acceptance, but still not the kind of airtight theory+experiments paper that clears AAAI comfortably.

TMLR still has better odds: yes.

Estimated TMLR acceptance probability: 45–55%.

Reason: TMLR is more tolerant of a strong empirical/mechanistic correction with careful limitations and less dependent on novelty-packaging under a compressed conference review. AAAI reviewers may punish the causal ambiguity, Muon+RP dependence, and theory looseness more harshly.
