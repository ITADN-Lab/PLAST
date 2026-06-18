**Verdict**
Borderline-to-weak reject as written: interesting empirical observation and strong narrative, but several setup inconsistencies, overclaims, weak causal/theoretical support, and insufficient benchmark rigor would make AAAI reviewers attack it hard. Estimated AAAI 2026 acceptance probability: 15–22% current; around 30–40% if repaired aggressively.

**Scores**
- Novelty: 7/10 — using Muon/orthogonalized updates as a plasticity method and falsifying AdamO’s mechanism is a fresh angle.
- Soundness: 5/10 — empirical claims may be real, but current design/reporting has inconsistencies and mechanistic conclusions are overstated.
- Significance: 7/10 — if true, this is a useful optimizer-level baseline and a meaningful correction to a recent SOTA mechanism.
- Rigor: 5/10 — tuning effort is appreciated, but n is small, benchmarks are narrow, NTK measurement is underspecified, and key controls are missing.
- Clarity: 7/10 — the paper is readable and unusually honest, but the story currently says too many slightly different things.
- Theory: 3/10 — Proposition 1 is basically a restatement of Muon’s polar update; the link to feature rank/plasticity is weak and partly incorrect.

**Top 3 Strengths**
- Strong empirical hook: Muon+RP beating AdamO+RP by +9.3 on CIFAR-100, with matched reperturbation, is potentially publishable if the protocol is correct.
- Honest mechanistic negative result: directly measuring NTK conditioning and finding AdamO best-conditioned but worse-performing is a valuable falsification of an appealing explanation.
- Good self-calibration: the paper admits plain Muon does not beat AdamO on deep ResNets, rank is not proven causal, and RL transfer fails. This helps credibility.

**Top 5 Weaknesses / Reviewer Attacks**
- Fatal-looking benchmark inconsistencies and protocol ambiguity. “CIFAR-100: 25 tasks of 10 random classes each” is impossible without class reuse, since CIFAR-100 has 100 classes. If classes repeat, it is not standard class-incremental CIFAR-100. “5000 images, 2 epochs/task” is also ambiguous. This alone could trigger rejection unless fixed.
- The headline claim overstates Muon. Plain Muon loses to best AdamO on ResNet-18, and the decisive method is Muon+RP. The abstract/introduction repeatedly imply “Muon beats AdamO” or “orthogonalizing optimizer preserves plasticity,” but the strongest supervised result depends on reperturbation. Reviewers will call this bait-and-switch.
- Mechanistic evidence is correlational and underpowered. Rank correlation is reported across only five optimizer configurations, not many independent points; r=0.77 over five points is fragile. NTK is measured on a 48-sample probe, condition numbers are likely noisy, and no confidence intervals/error bars are given for rank/NTK metrics.
- Theory does not substantiate the main mechanism. Proposition 1 only says exact polar/orthogonalized updates have equal singular values. Lemma 1 says representation rank is bounded by weight rank, which is obvious and not enough. Proposition 2 uses “generically rank r” and vague adaptive-update concentration; it does not prove Muon sustains feature rank or plasticity. The phrase “upper-bounds the attainable representational rank” is also conceptually backwards as a positive mechanism.
- Baseline and fairness questions remain. AdamO is sensitive to λ, Muon to LR, RP has its own thresholds/ρ/rank target/utility definition, but only limited sweeps are shown. AdamO+RP “does not help” may be because the RP gate is rank-deficit-based and therefore interacts differently with AdamO. Reviewers may argue the matched RP is not actually neutral.

**Must-Do Before Submission**
- Fix and fully specify the benchmark protocols.  
  Expected gain: +5–8 acceptance points.  
  State exact CIFAR-100 task construction: number of unique classes, whether classes repeat, samples per task, train/test split, relabeling, evaluation metric, and whether “late-task online accuracy” is train accuracy or held-out test accuracy. If it is 10 classes/task, use 10 tasks, not 25, or explain class reuse and rename the benchmark.

- Reframe the central claim around “Muon+RP” rather than “Muon” where appropriate.  
  Expected gain: +3–5 points.  
  The honest headline should be: “orthogonalized optimization is a strong base optimizer under matched reperturbation; plain Muon alone is not uniformly SOTA.” Keep the plain-Muon result as an important ablation, not the headline.

- Add confidence intervals/error bars for all mechanistic metrics.  
  Expected gain: +3–5 points.  
  Report per-seed representational rank, NTK condition number, NTK effective rank, and late accuracy. Include uncertainty for Table 5/6 and Figure 5. Show whether AdamO’s better NTK conditioning and Muon’s higher representation rank hold seed-wise.

- Strengthen the NTK falsification.  
  Expected gain: +4–7 points.  
  Measure empirical NTK on multiple probe sizes, e.g. 32/48/96/128 samples, and show rank/condition ordering is stable. Report spectrum plots, not only condition number. Use regularized condition number or effective dimension because raw condition numbers on tiny probes are unstable.

- Expand the rank-predicts-plasticity analysis beyond five method-level points.  
  Expected gain: +4–6 points.  
  Correlate rank and future-task learning across seeds, tasks, hyperparameters, and checkpoints. A convincing analysis would have dozens/hundreds of points: late accuracy at task t+1 predicted by rank at task t, controlling for optimizer and current loss.

- Add a proper held-out test metric or clearly justify training accuracy.  
  Expected gain: +3–6 points.  
  Plasticity often uses ability to fit new tasks, but AAAI reviewers will still ask whether the method merely improves online training fit. Report both online train accuracy and held-out task test accuracy for final 10 tasks.

- Remove or substantially soften the theory claims.  
  Expected gain: +2–4 points.  
  Present Proposition 1 as a formal property of Muon updates, not as a proof of the mechanism. Delete or rewrite unsupported statements like “full-rank increment… upper-bounds the attainable representational rank” and “explains why Muon attains higher rank.” Say it “is consistent with” or “suggests.”

**High-Leverage If Time Allows**
- Add one stronger supervised benchmark beyond CIFAR/permuted-MNIST.  
  Expected gain: +4–8 points.  
  Good choices: Split TinyImageNet, Split CIFAR-100 with standard 10 or 20 tasks, or continual mini-ImageNet. If Muon+RP beats AdamO there, the empirical contribution becomes much harder to dismiss.

- Compare against additional optimizer baselines.  
  Expected gain: +2–4 points.  
  Include Shampoo/K-FAC-like preconditioners if feasible, OrthoGrad/orthogonal gradient variants, Lion, SGD+orthogonalized momentum, and Muon without Newton-Schulz approximation variants. This clarifies whether the effect is Muon-specific or orthogonalized updates generally.

- Decouple RP from rank-based gating.  
  Expected gain: +2–5 points.  
  Since RP is triggered by rank deficit, it may favor Muon/Adam differently. Add fixed-rate RP, random-unit RP, utility-only RP, and same-number-of-reinitializations controls. Report actual number of units reset per method.

- Add compute/runtime and stability costs.  
  Expected gain: +1–2 points.  
  Muon’s orthogonalization has overhead. AAAI reviewers will ask whether +9 points costs 2× compute or memory. Report wall-clock, memory, and Newton-Schulz iteration count sensitivity.

- Clarify AdamO implementation and reproduction.  
  Expected gain: +2–4 points.  
  Give exact AdamO update, λ sweep, ηiso, whether Gram penalty uses WWᵀ or WᵀW depending on shape, treatment of convs/BN/biases, and whether numbers reproduce the original paper on its benchmark.

- Add intervention experiments on feature rank that are not just a classifier bottleneck.  
  Expected gain: +2–5 points.  
  Try explicit feature-rank regularization, decorrelation/Barlow-style penalties, low-rank projection of hidden activations, or orthogonalization only in selected layers. The current bottleneck test refutes only a narrow “head-visible dimension” story.

**Specific Text Edits Needed**
- Replace “Muon beats AdamO by +9.3” with “Muon+RP beats AdamO+RP by +9.3 under matched reperturbation.”
- Replace “the orthogonalizing optimizer, not reperturbation, carries the gain” with a more precise claim: “the optimizer base strongly modulates the benefit of RP; plain Muon alone does not beat best AdamO on ResNet.”
- Remove “fully matched comparison” unless RP hyperparameters, reset counts, optimizer sweeps, and evaluation budgets are truly matched and reported.
- Reword “NTK-conditioning account is falsified” to “NTK conditioning is not sufficient to explain these results.” The data do not prove NTK conditioning is irrelevant.
- Make the limitations visible earlier, not only in §7. Reviewers reward honesty more when it is not hidden after the sales pitch.

**Realistic Max Probability**
- Current paper: 15–22% AAAI acceptance probability.
- After fixing protocol inconsistencies, softening claims, adding uncertainty, and strengthening NTK/rank analyses: 28–35%.
- With one additional strong benchmark, robust seed-level mechanism analysis, and clean RP controls: 35–45%.
- Realistic ceiling at AAAI: about 45%. I would not expect much higher because the theory is weak, the mechanism remains open, and the main method is “use Muon plus a known reset heuristic.”

**Venue Fit**
- AAAI: plausible but risky. The empirical story is interesting, but AAAI reviewers may be conservative about optimizer papers with limited benchmarks and weak theory.
- TMLR: likely better odds if the authors want a careful empirical/mechanistic article and can iterate with reviewers. Estimated eventual acceptance after revisions: 45–60%.
- NeurIPS/ICML main track: possible only with stronger scale, cleaner mechanism, and broader benchmarks; current odds would be lower than AAAI.
- Workshop: very strong fit immediately, especially continual learning, optimization, or mechanistic understanding workshops.

[usage] 13477
