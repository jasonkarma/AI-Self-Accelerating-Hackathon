# Problem Statement

**Project:** Self-Accelerating Antibody Lead Optimization (the "Hybrid Virtual Lab", v1.2)
**Context:** Self-Accelerating AI for Drug Development Hackathon

---

## 1. Summary

This project addresses **two** problems, and the second one is the harder one.

1. **The wet-lab problem.** Finding a better antibody lead is a sequential decision problem under a
   hard budget: each measurement costs money and weeks, the objective is two-dimensional
   (affinity *and* developability), and the response surface is epistatic — single-mutation effects
   do not add up. A method earns its keep only if it reaches a better lead using *fewer* assays than
   random search could.

2. **The evidence problem.** "An AI agent self-accelerated drug discovery" is a claim that is easy
   to make and easy to fake. If the model can see the labels, if the baseline was never run, if the
   threshold was chosen after the results, or if the win is a mean over seeds whose spread is larger
   than the gap — the demo works and the science does not. This repository is built to make the
   second failure mode *structurally impossible to hide*, including when the answer is a negative
   result.

---

## 2. The domain problem

Antibody lead optimization is one of the most expensive search problems in drug development.
A development candidate must simultaneously satisfy:

- **Affinity** — tight binding to the target (here, HER2), measured as `y_bind = -log10(Kd [M])`.
- **Developability** — manufacturability and safety liabilities: aggregation-prone hydrophobic
  patches, unpaired cysteines, deamidation (`NG`, `NS`), isomerization (`DG`, `DS`), charge
  clustering.

These objectives **conflict**. The highest-affinity variants in the 1N8Z dataset carry 4–5 aromatic
residues and start `AR...`; the textbook "keep the wild-type anchor residues" rule penalises exactly
those molecules. A method that optimises affinity alone proposes molecules that fail developability
screening — which is why the repo gates every query on `y_dev >= tau_dev` before spending budget.

The search space is a 13-residue CDR loop (`SRWGGDGFYAMDY` for wild-type trastuzumab) over a fixed
candidate pool. The budget is **35 binding assays** — 5 shared seeds plus 6 rounds of 5 queries.
Within 30 post-seed queries the system must decide which 30 of the remaining ~400 candidates are
worth measuring.

### Why the obvious approaches are not enough

| Approach | Why it fails here |
| --- | --- |
| Random / exhaustive | Exhaustive screening of the pool is the thing being avoided; random search reaches the same lead only after spending the whole budget. |
| Greedy on a trained surrogate | Exploits a model that is still cold-started on 10–15 labels; stalls in local optima and never probes the regions where the model is wrong. |
| Static expert prior ("conserve W3/Y11/D12/Y13, avoid aromatics") | On real 1N8Z data this rule *inverts* the frontier — it penalises the best binders. It is implemented as baseline **M4** precisely to demonstrate that failure. |
| Unconstrained LLM in the loop | An agent that can rewrite the objective, filter the candidate pool, or read hidden labels will "win" by corrupting the metric. |

---

## 3. The evidence problem (what makes this repo different)

A benchmark that can be gamed is not evidence. This project therefore treats **the evaluation
harness as the primary deliverable**, not the model. Concretely, it commits to:

- **A real information boundary.** `y_bind` is reachable *only* through `BindingOracle.query()`.
  Nothing under `core/`, `data/`, or `app/` can read a hidden label for an unqueried candidate, and
  agent context builders are constructed so that hidden labels are structurally unreachable.
- **Pre-registration before labels.** The developability threshold `tau_dev` is frozen from the
  sequence column alone, before any binding label is read. `config/frozen_protocol.json` is
  write-once: rewriting it with different bytes raises.
- **A tamper-evident ledger.** Every reveal is appended to a SQLite ledger with a SHA-256 hash chain
  and triggers that reject `UPDATE`/`DELETE`. "We only queried 35 times" is verifiable, not asserted.
- **A negative control you cannot skip.** The lab runs **5 matched methods** (M1–M5) on the *same*
  five initial candidate IDs, so the agent layer's contribution is isolated rather than assumed.
  M5's own batch contains an H0 null slot that tests exactly what the non-agent baseline (M3) would
  have queried next.
- **A pre-registered holdout.** The lockbox target (`aayl51`) was selected by a metadata-only script
  from all 17 AbBiBench targets *before* any M1–M5 number existed for it, and is published even
  though it is a **failure** for the method.
- **Honest metrics.** Four learning metrics beside AUBQC (headroom-normalised AUBQC, top-5 portfolio
  depth, Spearman ρ, top-decile hit rate), with undefined correlations reported as `null`, never
  `0.0`. The README states explicitly that the 100-seed standard error on AUBQC (~0.007) is the same
  order as the gaps between methods — so an AUBQC mean alone invites reading noise as a win.

The point: a self-accelerating system is only credible if the acceleration claim survives an
adversarial audit. That audit is implemented in code and in CI, not written in a slide.

---

## 4. What "solved" means

The project is done when all of the following hold, and each is machine-checked (`tests/`,
`scripts/run_benchmark.py`, `scripts/generate_api_contract.py --check`):

1. `pytest -q` is green, including the API contract test.
2. The benchmark runs 5 methods over 100 seeds and reports AUBQC and Spearman ρ with paired
   significance tests.
3. M5's rank model beats the strongest non-agent baseline on ρ **where a learnability gate says a
   rank signal exists** — and the repo reports a tie or a loss where it does not.
4. Every proposed batch is 5 unique, developability-qualified, unqueried candidates, and the agent's
   influence `Δ_agent` is bounded to `[-0.15, +0.15]` per prior and `±0.06` in aggregate, so it can
   never filter a candidate or alter the metric.
5. Falsified hypotheses are inverted in subsequent rounds — the lab genuinely self-corrects rather
   than merely re-describing its own outputs.
6. The demo path is reproducible: a frozen golden run can be restored in ~12 ms, rewound to any
   round, and replayed live in front of an audience.

---

## 5. Results as of this writing

**Where the method wins (1N8Z, 100 seeds).** M5 beats every baseline on rank correlation at q=35:

| Metric | M1 | M2 | M3 | M4 | **M5** |
| --- | --- | --- | --- | --- | --- |
| AUBQC ± SE | 1.9061 | 1.9240 | **1.9246** | 1.9120 | 1.9237 |
| Spearman ρ @ q=35 | n/a | 0.5081 | 0.3442 | 0.2991 | **0.6578** |

ρ is also the axis comparable to published zero-shot affinity predictors on 1N8Z (AlphaFold3
`-0.16`, ProteinMPNN `-0.17`, ESM-2 `-0.13`). Against the ablated agent layer (M3) the gain is
`+0.3136`, `p = 8.4e-32`, won on **98 of 100 seeds**.

**Where it does not.** M5 does *not* win on AUBQC (Δ = −0.0009 vs M3, p = 0.77), and it trails M2 on
portfolio depth and top-decile hit rate. On the pre-registered lockbox `aayl51` it ranks 3/5 at every
coverage level — a published negative result.

### 5.1 A second modality, and the harness catching its own bugs

The same loop is run on **small molecules against three human kinase targets**, using Talus Bio's
**Ptarmigan-1** as the virtual screener and **ChEMBL measured affinities** as ground truth
(`scripts/run_ptarmigan_loop.py`). M5 beats the cold baseline on share of the possible gain in
**3 of 3** targets — CDK2 +0.078 (p = 3.5e-07, 40/50), BRAF +0.039 (p = 1.5e-03, 32/50), CHK1 +0.056
(p = 6.8e-10, 43/50) — at $2.2450 of live screen spend, every target `ood=[]`.

This section exists in the problem statement rather than only in the README because it is the
clearest available demonstration of the thesis in §1: **the evaluation harness is the deliverable.**
Building it surfaced three defects in the benchmark that no passing test would have caught, and each
one was producing a number that looked like a result:

1. The round loop picked one compound per round while `BATCH_SIZE = 5` sat unused, so the campaign
   spent **11 of 35 wells** while the artifact's own metadata advertised 35. AUBQC was hardcoded
   `None` because the Q=35 checkpoint was unreachable.
2. `Δ_agent` was computed, clamped, and never added to the score — so the "hybrid" method was
   byte-identical to the static one in-loop. Fixing that revealed a subtler variant: because
   engagement and hotspot cosine are both positive, the weighted sum floored *above* the ±0.15 clip,
   so **100 % of compounds clamped to a constant**, and a constant cannot change an `argmax`. The
   prior had to be centred on the pool mean to be signed at all.
3. A regression test written for defect 2 **passed against the broken code**, because it asserted
   that the two methods pick different compounds while they also differ in seed batch — the seeds
   alone satisfied it. It was replaced by one that holds the seed batch fixed.

Defect 3 is the general lesson: a test that cannot fail is worse than no test, because it converts
an unverified claim into a green check. The repo now pins the invariant directly (spread > 0, some
negative values, < 25 % saturation) and measures 0 % saturation on real data against 100 % before.

**The most useful result is a failure.** On `BRAF` the Ptarmigan prior is statistically
indistinguishable from noise (pool ρ = −0.020, p ≈ 0.5). Blindly spending the seed batch on its
top-5 by engagement scores 0.535 against the cold baseline's 0.892 — Δ = −0.357, p = 7.0e-34, **0 of
50 seeds won**, top-decile hit rate exactly 0.000. The centred hybrid survives it (+0.039,
p = 1.5e-03) precisely because a centred Δ can go negative, so the loop can spend against its own
prior's advice. An uncentred prior has no negative region to invert into, and the self-correction
claimed in §4 clause 5 would be untestable. `ERBB2` was removed from the panel because full-length
`P04626` (1255 aa) exceeds Ptarmigan-1's `training_max_residues = 766`; `BRAF` at exactly 766 aa is
the longest in-distribution target available, confirmed by a zero-credit `--dry-run` coverage
inspection before any screen was queued.

---

## 6. Explicitly out of scope / open problems

These are stated up front so the claims above are not read as broader than they are.

- **The bundled 419-row dataset is a deterministic synthetic fixture**, clearly marked as such. It
  reproduces the schema, row count and wild-type sequence of the real AbBiBench 1N8Z export; the real
  export is installed with `python scripts/fetch_real_1n8z.py`. Synthetic-fixture numbers are
  regression pins, not biology.
- **The agent layer is conditional on library density.** `Δ_agent` fires only when the modal best
  residue at a position differs from the pool's own reference. On sparse "wild-type plus independent
  mutations" pools (e.g. `2fjg`, 1.00 substitutions per variant) the fire rate is **0 %**, so an
  inert-agent cell and a working-agent cell look identical in a results table unless the fire rate is
  reported beside ρ — so it is.
- **A tie is not a safety claim.** On `aayl51`, `4d5_her2` and `aayl49` no method correlates with
  truth; the benchmark has no statistical power there. That is an absence of evidence.
- **Single-round targets are structural controls, not evidence.** At `1mlc` the protocol clamps to
  R=1, so M5 and M3 propose the identical batch and AUBQC is bit-identical by construction.
- **The 16 `L=14` AbBiBench targets are unusable** — they collapse to 1–2 distinct CDR3 sequences and
  are not search problems.
- **The small-molecule panel is exploratory, not a holdout.** CDK2, BRAF and CHK1 were selected for
  being *in-distribution* for Ptarmigan-1 (`ood=[]`, ≤ 766 aa) and for having ≥ 250 exact binding
  measurements in ChEMBL — not chosen blind by a metadata-only script before any number existed. The
  antibody half's pre-registered lockbox (`aayl51`) is the only holdout in this repo, and it is
  consumed. Nothing in §5.1 may be described as confirmatory generalisation.
- **No cross-modality metric is comparable with the antibody metrics.** The small-molecule pool is
  300 compounds with a `pchembl_value` gradient, not a 13-residue epistatic CDR loop with a
  developability gate; there is no `y_dev`, no `tau_dev`, no M1–M4 and no six-method comparison.
  AUBQC is a different absolute scale in the two halves and is only ever compared *within* one.
- **Two scope rules the artifact enforces mechanically.** ρ is recorded with `rho_scope`
  (`"queried"`, n=35 / `"pool"`, n=300) and `rho_n` on every record, because a static ranker that
  never enters the lab cannot be scored on the same 35 compounds as an active method. And the
  share-of-gain floor is the **pool median**, never a method's own seed batch — a per-method floor
  rewards starting badly and inverted the BRAF comparison outright (M5 0.167 vs M3 0.686 while M5's
  raw AUBQC was higher). Both are pinned by tests in `tests/test_ptarmigan.py`.
- **Ptarmigan-1 supplies no metric.** It returns `engagement`, `score`, `predicted_residue`,
  per-residue `residue_engagement` and `coverage`. Every score in §5.1 — AUBQC, share of gain, ρ,
  top-decile hit rate, falsification rate — is computed in this repo against ChEMBL measurements.
  Attributing any of them to Talus would be a false claim about a third party's product.

---

## 7. Why this matters beyond the hackathon

Most of the cost of AI-driven discovery is not the model — it is the **evidence that the model
works**. This project contributes a reusable pattern: hidden oracle + hash-chained ledger +
pre-registered threshold + matched baselines + a null slot inside every batch + published
falsifications. That harness applies to any setting where an agent proposes expensive experiments and
someone has to decide whether to believe the improvement.
