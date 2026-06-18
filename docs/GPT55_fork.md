**Bottom Line**
- Your paper is still viable at AAAI, but the story has changed: it is no longer a “mechanism paper.” It is an empirical + negative-mechanism paper about optimizer-induced plasticity in continual supervised learning.
- Current realistic AAAI probability: ~25–30%, down from ~35%.
- With the right reframing, stronger ablations, and one good positive diagnostic short of a full mechanism: max ~40%.
- Without a positive explanatory hook, the paper can still get in, but it must be extremely careful, well-controlled, and useful to the community.

My honest AC read: the result is interesting enough that I would not desk-reject intellectually. But reviewers will ask: “Why should we care beyond this optimizer working better in your setup?” Your job is to make the answer unavoidable.

---

**1. Best Honest Framing**

The strongest framing is:

“Muon with reset-and-perturbation is a simple, highly effective optimizer-level intervention for maintaining plasticity in supervised continual learning. Contrary to prominent hypotheses, its advantage is not explained by NTK conditioning or penultimate feature-rank preservation. The mechanism appears to lie deeper in backbone update dynamics. We provide strong controlled empirical evidence and falsify two plausible mechanistic accounts.”

Do not frame it as:

- “Muon preserves rank”
- “Muon improves conditioning”
- “We explain plasticity”
- “We discover the mechanism”
- “Muon solves continual learning”

Frame it as:

- “A surprisingly strong optimizer baseline”
- “A challenge to existing mechanistic explanations”
- “A benchmarked empirical finding under matched tuning”
- “A negative result that narrows the mechanism search”
- “Evidence that plasticity failures are not fully captured by head-level representation rank or NTK conditioning”

This is viable at AAAI if the empirical section is airtight.

The key contribution stack should be:

- Muon+RP consistently improves late-stage continual learning accuracy across multiple supervised benchmarks.
- The gains persist under matched reset-and-perturbation, tuning budget, architecture, and width variations.
- Muon+RP beats specialized plasticity methods: CBP, ReDo, L2-init, shrink-perturb, SGD, AdamW, AdamO.
- AdamO improves or has superior NTK conditioning but underperforms, falsifying the “conditioning explains plasticity” hypothesis in this setting.
- Hard bottlenecking the penultimate representation does not eliminate the Muon advantage, arguing against a simple “more penultimate rank causes better plasticity” explanation.
- Therefore, the cause likely lies in backbone optimization/update dynamics rather than output-layer-visible representational dimensionality.

That is a legitimate paper. But it is a harder sell than “we identify the mechanism.”

Realistic probability:

- Current paper if written honestly but mostly empirical: ~25–30%.
- With a clear positive diagnostic of backbone dynamics: ~35–40%.
- With a full causal mechanism: ~45–55%, but that seems unlikely within a short cycle.
- If reviewers perceive it as “we tried mechanisms and failed, but Muon works”: ~15–20%.

So the writing matters a lot.

---

**2. Is The Bottleneck Experiment A Refutation?**

It is a genuine refutation of a specific strong hypothesis, but not of all rank-related hypotheses.

It refutes:

“Muon’s advantage comes primarily from maintaining high effective penultimate feature rank available to the classifier.”

Because if that were true, forcing both methods through rank ~8 should substantially hurt Muon and/or close the gap. It does neither.

It does not refute:

- Rank in earlier backbone layers matters.
- Class-conditional feature geometry matters more than global rank.
- Rank within convolutional feature maps matters.
- Rank of gradients or updates matters.
- Penultimate rank above ~8 may be unnecessary for 10-way incremental classification.
- Muon’s benefit may be rank-related but not visible at the final feature layer.
- The bottleneck may be learnable enough to reparameterize useful information into k dimensions.

So include it, but phrase it carefully.

Recommended wording:

“Because each task has only 10 classes, this experiment does not rule out all rank-based explanations: a low-dimensional bottleneck may still be sufficient for task discrimination. However, it does rule against the simple hypothesis that Muon’s advantage is caused by preserving high penultimate feature dimensionality. Even when the classifier observes only a rank-8 bottleneck, the Muon+RP advantage remains essentially unchanged.”

Do include it. It is valuable because negative mechanism results are rare and it strengthens your honesty. But do not oversell it as “rank is irrelevant.”

Put it in the main paper if space permits. If not, put the core table in main and details in appendix.

I would title it something like:

“Penultimate Feature Rank Is Not Sufficient to Explain the Gap”

Not:

“Rank Does Not Matter”

Also, add one sanity check if cheap:

- Evaluate whether k=4 or k=2 collapses performance.
- If k=2 hurts both substantially and k=8 does not, then you can show the bottleneck is real but k=8 is enough.
- If even k=2 works well, reviewers will conclude the experiment is too weak for CIFAR-100 10-way tasks.
- If k=4 starts hurting but the Muon gap remains, that is stronger.

Right now k=8 may indeed be too loose. A k sweep down to 1, 2, 4 would make this much more convincing.

---

**3. Mechanism Probes Most Likely To Find The Real Cause**

You need cheap diagnostics that target backbone update dynamics. I would not chase exotic theory now. You need one concrete positive empirical signature.

Here are the top three.

---

**Experiment A: Backbone Update Effective Rank / Spectrum**

Hypothesis:

Muon’s advantage comes from producing higher-rank or better-distributed updates in backbone layers, especially conv layers, preventing plasticity collapse even when final representation rank is capped.

What to measure:

For each convolutional or linear weight tensor, reshape update ΔW into matrix form:

- Conv: `out_channels x (in_channels * kernel_h * kernel_w)`
- Linear: `out_features x in_features`

At each task boundary or training window, compute:

- Effective rank of ΔW
- Stable rank: `||ΔW||_F^2 / ||ΔW||_2^2`
- Singular value entropy
- Top singular value concentration: `σ1 / sum σi`
- Cosine similarity of updates across tasks
- Fraction of update energy in top r singular directions

Compare:

- Muon+RP
- AdamO+RP
- AdamW+RP
- SGD+RP if possible

Predicted useful outcomes:

- If Muon updates are less spectrally collapsed, more isotropic, or higher stable-rank in backbone layers, that gives you a positive mechanism candidate.
- If AdamO has good NTK conditioning but low-rank/backbone-concentrated updates, you get a very clean story: conditioning is not enough; update geometry matters.

This is probably the single best cheap experiment.

Why it is strong:

It connects directly to Muon. Muon orthogonalizes/whitens-ish update matrices. If its advantage comes from anything natural, update spectrum is the first place to look.

Causal variant if you have time:

- Take AdamO updates and project/normalize them to match Muon-like spectral shape.
- Or take Muon updates and truncate to low rank.
- But even observational update spectra would help.

---

**Experiment B: Layerwise Plasticity / Feature Change After Task Switch**

Hypothesis:

AdamO preserves or conditions something globally, but its backbone features stop moving usefully after many tasks. Muon maintains trainability in intermediate layers.

What to measure:

At each task switch, freeze checkpoints and train on the new task for a short fixed budget, e.g. first 100–500 minibatches. Track:

- New-task loss decrease rate.
- Layerwise parameter movement: `||ΔW_l|| / ||W_l||`.
- Layerwise feature movement: CKA distance between features before and after adaptation.
- Gradient norm per layer.
- Fraction of dead/inactive ReLUs if applicable.
- Linear probe accuracy from each block before/after adaptation.

The key metric:

“Plasticity slope”: how quickly the model can reduce loss on a fresh task late in the stream.

Compare early tasks vs late tasks.

Predicted useful outcome:

- Muon+RP should retain fast loss descent and feature movement in earlier/mid backbone blocks.
- AdamO may show head adaptation but weak backbone adaptation, or gradient concentration in late layers.

Why it is strong:

It directly supports your current claim: “the gap lives in backbone optimization, not penultimate dimensionality.”

This is also intuitive to reviewers. Plots of late-task learning curves and layerwise movement are easy to understand.

---

**Experiment C: Activation Diversity / Unit Utilization Across Backbone Layers**

Hypothesis:

Muon maintains diverse, non-saturated convolutional features, while AdamO accumulates inactive or redundant features.

Measure per block:

- Activation covariance effective rank.
- Mean pairwise channel correlation.
- Fraction of near-dead channels.
- Spatial/channel entropy.
- Class-conditional feature separation.
- CKA similarity across tasks.

Important: do not only measure penultimate rank. Measure block1/block2/block3/block4 feature maps.

For conv activations, reshape as:

- channels x `(batch * spatial positions)`

Then compute effective rank/channel covariance rank.

Predicted useful outcome:

- Penultimate rank may not matter, but earlier/mid-layer channel diversity may differ strongly.
- Muon may preserve broader feature dictionaries in the backbone even if the final bottleneck is low-dimensional.

Why it is useful:

This rescues a refined “representation” story without contradicting your bottleneck result. The right claim becomes:

“Muon’s advantage is not explained by the dimensionality of the final representation observed by the classifier; instead, it may reflect healthier update/activation geometry throughout the backbone.”

But if you only have time for one, do Experiment A.

---

**What I Would Not Prioritize**

I would deprioritize:

- Full Hessian spectra: expensive, noisy, hard to interpret.
- NTK variants: you already have a strong negative result.
- More benchmark proliferation: useful but lower marginal value now.
- Trying many bottleneck dimensions before measuring update spectra, except maybe k=2/4 sanity.
- Sophisticated mutual information or intrinsic dimension estimators: reviewers often distrust them.
- Loss landscape visualizations: pretty but rarely decisive.

---

**4. Max Acceptance Probability And Single Best Next Move**

Maximum realistic acceptance probability for this paper this cycle:

- Without positive mechanism: ~30%.
- With strong update-spectrum evidence: ~38–42%.
- With update-spectrum evidence plus one causal update intervention: ~45%.
- Above ~50% would require either a very broad benchmark suite, a clean causal mechanism, or adoption-level practical impact.

The single best next move:

Run the backbone update-spectrum experiment.

Specifically:

For each optimizer, log per-layer update matrices during late tasks and compare stable rank / spectral entropy / top singular concentration. Then show whether Muon produces broader, less collapsed backbone updates than AdamO.

Why this is the best move:

- It is cheap.
- It is directly tied to Muon’s algorithmic bias.
- It targets the backbone, where your current evidence points.
- It can produce a positive explanatory signature even without full causality.
- It gives reviewers something mechanistic to hold onto.

Minimum version:

- CIFAR-100 ResNet-18 only.
- Muon+RP vs AdamO+RP vs AdamW+RP.
- 3 seeds if possible, 1 seed acceptable for exploratory appendix but not main claim.
- Measure last 3–5 tasks.
- Plot per-block stable rank and spectral entropy of updates.
- Correlate update stable rank with late-task accuracy across optimizer/seed/configs.

Even better:

Add a causal ablation:

- Low-rank truncate Muon updates to top r singular components in conv/linear layers.
- If truncating Muon updates degrades plasticity and narrows the gap, you have a real mechanism candidate.
- Conversely, if spectral-shaping AdamO updates helps, even stronger.

But the observational update-spectrum result alone is likely enough to improve the paper.

---

**How To Write The Paper Now**

I would structure it as:

1. **Problem:** Plasticity loss in supervised continual learning remains poorly understood.
2. **Empirical finding:** Muon+RP is a strong, simple optimizer-level baseline.
3. **Controls:** Matched tuning, matched RP, multiple baselines, multiple datasets, width variation.
4. **Negative mechanisms:** Neither NTK conditioning nor penultimate representation rank explains the gap.
5. **Backbone dynamics:** New diagnostics suggest the advantage lies in update/feature dynamics inside the backbone.
6. **Limitations:** Mechanism not fully proven; rank intervention limited by low-dimensional task structure; results are supervised CL, not task-free/general CL.
7. **Takeaway:** Future plasticity methods should benchmark against Muon+RP and study backbone update geometry, not only head-level features or kernels.

This is the honest and strongest version.

---

**Reviewer Risk Assessment**

Likely positive reviewer comments:

- “Strong empirical baseline.”
- “Good controls against common confounds.”
- “Useful negative result challenging existing explanations.”
- “Practical and reproducible.”
- “The bottleneck experiment is interesting.”

Likely negative reviewer comments:

- “Mechanism remains unclear.”
- “Is this just optimizer benchmarking?”
- “Only supervised/task-incremental setting?”
- “Does Muon require extra compute or memory?”
- “Are hyperparameters equally tuned?”
- “Are results robust to task order, augmentation, batch size, reset schedule?”
- “Why AAAI rather than an empirical workshop paper?”

You must preempt those.

Especially include:

- Equal or larger tuning budget for AdamO/AdamW.
- Wall-clock or per-step overhead.
- Task-order variance if feasible.
- Same RP schedule for all optimizers.
- Width-64 result prominently.
- At least one dataset beyond CIFAR-100 that is not trivially similar.
- Clear failure/limitation section.

---

**My Concrete Recommendation**

Include the bottleneck result, but as a bounded negative result.

Then run exactly one more mechanistic probe:

1. Measure backbone update spectra for Muon+RP vs AdamO+RP.
2. If Muon has higher update stable rank / spectral entropy, make that the positive diagnostic.
3. If feasible, truncate Muon updates to test whether collapsing update rank hurts plasticity.

If the update-spectrum story works, your new title could be something like:

“Optimizer-Induced Plasticity in Continual Learning: Muon Beats AdamO Beyond Kernels and Feature Rank”

Better:

“Muon as a Strong Plasticity Baseline for Supervised Continual Learning”

Even better if update spectra work:

“Backbone Update Geometry Predicts Optimizer-Induced Plasticity in Continual Learning”

Avoid titles that overclaim “explains.”

Final calibrated assessment:

- Right now: ~25–30% AAAI.
- With clean update-spectrum diagnostics: ~35–40%.
- With causal update truncation/intervention: ~40–45%.
- If no positive diagnostic emerges and mechanism remains fully open: submit only if the empirical suite is very strong and the negative results are framed as a contribution. Still viable, but borderline.
