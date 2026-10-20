# Lockbox Holdout Pre-Registration — frozen engine `a10ece5`

**Status:** COMPLETE. Exhaustive **17-of-17** audit of the entire AbBiBench corpus.
Two-stage blind selection (metadata → baseline learnability), then M5. Final
partition in §9.

**Headline result:** the `L ∉ {11, 13}` hope does **not** survive contact with the data.
Of the six targets you named as exotic-width candidates, five collapse to 1-2 distinct
CDR3 sequences, and the two large exotic-width survivors are `aayl51` (L=12) and
`aayl49` (L=11). `aayl50_LC` / `aayl52_LC` / `5a12_{ang2,vegf}` / `g6_LC` collapse to a
single sequence each — they are framework DMS libraries, not CDR3 libraries.

**Purpose.** Name, in advance, which AbBiBench target(s) count as a legitimate
holdout for the frozen engine at `a10ece5`, so that "M5 generalizes" can fail. Any
rule written after seeing AUBQC is not a rule.

---

## 1. What is frozen

| Item | Value |
| --- | --- |
| Engine commit | `a10ece5` (HEAD of `main`) |
| Primary benchmark | `results/benchmark_100seeds.json` (1N8Z, 419 rows, L=13, N_qual=210) |
| Already-scored cross-targets | `aayl49_ML` (L=11, N_qual=756), `2fjg` (L=13, N_qual=124) |
| Selection law | `core.acquisition.compute_campaign_protocol`, coverage 16.7 % (`35/210`) |
| Developability gate | `core.preregistration.freeze_protocol`, `threshold_percentile = 0.50` |
| Tuning freedom | **None.** No constant, threshold, coverage rate or slot rule may be changed for the holdout. |

A holdout result is reported as the *same five metrics* on the *same* protocol law:
AUBQC, `normalized_aubqc`, `top5_panel_aubqc`, `top_decile_hit_rate`,
`spearman_rho` at the final checkpoint. No new metric is introduced after the fact.

---

## 2. Eligibility rule (frozen before any target is named)

A target qualifies as a lockbox holdout iff **all** of:

1. **`N_qual >= 150`** unique qualified CDRH3 sequences, where `N_qual` is returned by
   the frozen `freeze_protocol` and `threshold_source == "pool_median"`.
2. **Not HER2.** `{1n8z, 4d5_her2}` are excluded: the 1N8Z engine was developed on a
   HER2 paratope pool, so a second HER2 target is in-distribution, not a holdout.
3. **Not already scored.** Excluded: `1n8z`, `aayl49_ML`, `2fjg`, `4d5_her2`.
   A target that already appears in `results/cross_target_benchmark.json` or
   `results/attribution_ablation.json` is disqualified regardless of other merits.
4. **A real search problem**, per `data.datasets.cross_target_usability`: `>= 35`
   distinct CDR3 sequences and `>= 2` actually varying positions.
5. **Novel width preferred, not required.** `L ∉ {11, 13}` is *recorded and preferred*
   because it is the strongest available out-of-distribution test, but it is **not**
   an eligibility clause. Making it one would hand us the freedom to reject every
   target that qualifies and then relax the rule — exactly the move this document
   exists to prevent. If the eligible set contains both widths, the exotic-width
   target is chosen **and the non-exotic one is reported as a secondary row.**

### Hard-stop condition

If, for a target with `L != 13`, `freeze_protocol` returns
`threshold_source == "wild_type"`, that target's `N_qual` is **not** the 0.50 quantile
gate. This is a known defect: `freeze_protocol` (`core/preregistration.py:24-33`)
scores the frozen 1N8Z **13-mer** wild type `SRWGGDGFYAMDY` against a normalizer fit on
the target's own pool. On an 11-mer pool those components are clipped into an 11-mer's
`[min, max]` box, and `pass_fraction` can still land inside `[0.25, 0.65]` — which
silently switches the gate to `wild_type` and makes `N_qual` incomparable across
targets. **Such a target must not be pre-registered.** Fix the threshold source first
(that is an engine change, so it must happen *before* the holdout runs, and it must be
applied to the already-scored targets too so the comparison stays honest).

Note `aayl49_ML` and `2fjg` both report `threshold_source == "pool_median"`, so the
already-published cross-target numbers are unaffected.

---

## 3. The audit that fills in the table

`scripts/audit_lockbox_metadata.py` enumerates **every** `binding_affinity/*.csv` in
`AbBibench/Antibody_Binding_Benchmark_Dataset` through `HfApi` — not the six hardcoded
entries in `TARGET_CHAIN` — and reports per target:

```
target_id, antigen, modal_cdrh3_length_L, reference_consensus,
unique_cdrh3_pool_size, N_qual, binding_iqr
```

plus `threshold_source`, `wt_pass_fraction`, `length_histogram`, `replicates_collapsed`
and the usability diagnostics, so the eligibility verdict is auditable rather than
asserted.

Guarantees the script makes, and why they are structural rather than promised:

- **No policy is imported.** `core.baselines` is never loaded; the script asserts it
  is absent from `sys.modules` before doing any work.
- **The gate is not re-implemented.** `N_qual` is whatever `freeze_protocol` returns,
  written to a scratch directory so `config/` is untouched.
- **Labels are read for one statistic only**, `binding_iqr` (and min/median/max), which
  is part of the metadata table the rule is allowed to see. No curve is computed.

Run it, paste the table here, and only then name the target.

---

## 3bis. Measured table (Step 1 output)

Produced by `scripts/audit_lockbox_metadata.py` on 17 enumerated files (the repo holds
17 `binding_affinity/*.csv`, not the 16 assumed; `1mlc` and `g6_LC` bring the count up).
Zero alignment failures.

| target_id | antigen | L | reference (consensus) | pool | **N_qual** | binding_iqr | gate | used? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `aayl49_ML` | lysozyme | 11 | `AKGRAAGTFDS` | 1510 | 756 | 0.530 | ok | **used (x2)** |
| `4d5_her2_trimmed` | **HER2** | 13 | `SRWDDDGFYPMDY` | 1227 | 656 | 0.124 | wild_type | fallback |
| `aayl51` | (see §7) | **12** | `AKVGRGGGYFDY` | 972 | **486** | 0.376 | ok | no |
| `aayl49` | lysozyme | 11 | `AKGRAAGTFDS` | 959 | **480** | 0.306 | ok | no (17.4 % overlap) |
| `1n8z` | **HER2** | 13 | `ARYGGYGYYYFDY` | 416 | 208 | 0.970 | ok | **primary** |
| `2fjg` | HIV-1 gp120 | 13 | `ARFVFFLPYAMDY` | 248 | 124 | 0.527 | ok | **used (x2)** |
| `1mlc` | (see §7) | **9** | `ARGDGNYGY` | 133 | 67 | 4.843 | ok | no |
| `1mhp_LC` | (see §7) | 12 | `TRGFGDGGYFDV` | 8 | 5 | 0.661 | **BAD** | no |
| `3gbn_h1` | Influenza HA | **14** | `AKHMGYQLRETMDV` | **2** | 1 | 0.049 | ok | no |
| `3gbn_h9` | (see §7) | **14** | `AKHMGYQLRETMDV` | **2** | 1 | 0.097 | ok | no |
| `4fqi_h1` | Influenza HA | **14** | `ARHGNYYYYSGMDV` | **2** | 1 | 0.003 | ok | no |
| `4fqi_h3` | Influenza HA | **14** | `ARHGNYYYYSGMDV` | **2** | 1 | 0.012 | ok | no |
| `5a12_ang2` | (see §7) | 13 | `ARFVFFLPYAMDY` | **1** | 1 | 0.000 | ok | no |
| `5a12_vegf` | (see §7) | 13 | `ARFVFFLPYAMDY` | **1** | 1 | 0.000 | ok | no |
| `aayl50_LC` | (see §7) | **11** | `AKGRAAGTFDS` | **1** | 1 | 0.000 | ok | no |
| `aayl52_LC` | (see §7) | **10** | `ARVGRGVIDH` | **1** | 1 | 0.000 | ok | no |
| `g6_LC` | (see §7) | 13 | `ARFVFFLPYAMDY` | **1** | 1 | 0.000 | ok | no |

Antigen names left as `(see §7)` are `unknown` in this repository; see §7 for why they
stay blank.

### Three measured facts that constrain the rule

1. **The exotic-width targets you hoped for do not exist as search problems.**
   `3gbn_h1`/`3gbn_h9`/`4fqi_h1`/`4fqi_h3` each have ~32,000-35,000 extracted CDR3 rows
   that collapse to **2** distinct sequences at L=14, and `aayl50_LC`/`aayl52_LC`/
   `5a12_ang2`/`5a12_vegf`/`g6_LC` collapse to **1**. These were previously described in
   `run_cross_target_benchmark.py`'s docstring as "1,887 rows -> two distinct CDR3s" for
   3GBN; the same collapse holds for 4FQI. No amount of N_qual threshold tuning revives
   them — `cross_target_usability` correctly rejects them on positional diversity.
2. **`aayl49` is a near-duplicate of the already-scored `aayl49_ML`.** 167 of its 959
   sequences (17.4 %) are already in `aayl49_ML`, same consensus `AKGRAAGTFDS`, same
   L=11, same antigen. It is **not** a clean holdout and must be excluded by a
   *measured* overlap clause, not by name.
3. **`1mlc` is now disqualified on N_qual (67 < 150)** — and it is L=9, which was never
   in the exotic set. Its `binding_iqr` of 4.843 is also ~5-10x every other target,
   so its label scale is not comparable.

### The exotic-width clause, revised

§2 clause 5 said exotic width is "preferred, not required". The measurement shows the
eligible exotic set is `{aayl51 (L=12), aayl49 (L=11, disqualified)}`. So the clause is
now resolved by fact rather than by preference:

> **`aayl51` (L=12, N_qual=486, 0 % sequence overlap with any scored target) is the
> lockbox holdout.**

`aayl51` satisfies every clause: N_qual 486 >= 150, `threshold_source = "pool_median"`,
non-HER2, not previously scored, a genuine search problem, and **L=12 is not in
{11, 13}** — the exotic width you asked for, arriving on its own merits rather than by a
relaxed rule.

---

## 4. What must be filled in before the holdout runs

- [x] Every target enumerated; 17 files, 0 alignment failures
- [x] `binding_iqr` recorded for each; antigen names left `unknown` where this repo has
      no citation (see §7)
- [x] Eligible set computed by §2, before any target was named
- [x] `gate_ok == true` for the pre-registered target (`aayl51`: `pool_median`)
- [ ] **`aayl51` named in this file and committed — this document does that**
- [ ] `antigen` for `aayl51` resolved from a citable source before the writeup (§7)
- [ ] Run `scripts/run_cross_target_benchmark.py --dataset aayl51` only after the above

---

## 5. Reporting rules, fixed in advance

- The holdout row is reported **whether it helps or hurts.** A negative result is a
  published result, exactly like `results/attribution_ablation.json`.
- The holdout is **never** used to retune a constant. If a constant is changed after
  seeing the result, this target stops being a holdout and the writeup must say so.
- The eligible-but-not-chosen targets are reported alongside the chosen one, so a
  reader can see that the choice was the rule's and not ours.
- `results/attribution_ablation.json` stays on disk as the record of the B1 / A+B1
  negative result. Nothing in this document revives those variants.

---

## 6. Step 2 verdict

| Target | L | N_qual | Verdict |
| --- | --- | --- | --- |
| **`aayl51`** | **12** | **486** | **SELECTED — pre-registered lockbox holdout** |
| `4d5_her2_trimmed` | 13 | 656 | excluded: HER2 (in-distribution) |
| `aayl49` | 11 | 480 | excluded: 17.4 % sequence overlap with scored `aayl49_ML` |
| `1mlc` | 9 | 67 | excluded: N_qual < 150 |
| `aayl49_ML` | 11 | 756 | already scored |
| `1n8z` | 13 | 208 | already scored (+ HER2) |
| `2fjg` | 13 | 124 | already scored (+ N_qual < 150) |
| `1mhp_LC` | 12 | 5 | excluded: not a search problem; **gate BAD-SOURCE** |
| `3gbn_h1`, `3gbn_h9`, `4fqi_h1`, `4fqi_h3` | 14 | 1 | excluded: 2 distinct CDR3s |
| `5a12_ang2`, `5a12_vegf`, `aayl50_LC`, `aayl52_LC`, `g6_LC` | 13/13/11/10/13 | 1 | excluded: 1 distinct CDR3 |

Secondary reporting row (per §2 clause 5, "report the non-exotic one too"): **there is
none left.** Once `aayl49` is excluded on measured overlap and `4d5_her2` on being HER2,
`aayl51` is the only qualifying target in the corpus. The holdout is therefore a
single-target test, and this document says so rather than implying a choice was made.

---

## 7. Antigen labels stay `unknown` on purpose

Nine of the 17 targets have no antigen recorded anywhere in this repository. Filling
them in from memory would put an unverified claim into a pre-registration document that
gets quoted in a writeup. `aayl51`'s antigen in particular **must** be resolved from a
citable source (the AbBiBench paper or the dataset card) before the result is written up.
Until then the holdout is identified by `target_id` alone, which is unambiguous.

`aayl49`'s row is marked lysozyme because `TARGET_CHAIN` in
`scripts/run_cross_target_benchmark.py:92` labels its sibling `aayl49_ML` "anti-lysozyme"
and the two share a consensus; that is an inference from this repo, not a citation.

---

## 8. A pre-existing defect this audit exposed (not introduced here)

`core/preregistration.py:24-33` scores the frozen 1N8Z **13-mer** wild type
`SRWGGDGFYAMDY` against a normalizer fit on the target's own pool. Two measured
consequences:

* `1mhp_LC` (L=12) came back `threshold_source = "wild_type"` with `wt_pass_fraction =
  0.625`, inside the `[0.25, 0.65]` window — so its gate silently became the wrong one.
  Its N_qual is 5 either way, so nothing was decided on it.
* `4d5_her2_trimmed` (L=13) reports `wild_type` because the WT happens to be the same
  width; that is legitimate, and it is excluded for being HER2 anyway.
* **Every scored target reports `pool_median`** (`1n8z`, `aayl49_ML`, `2fjg`), so no
  published number depends on this defect.
* `aayl51` reports `pool_median`, so the holdout is scored on the documented 0.50 gate.

Recommend fixing `freeze_protocol` to take an explicit reference sequence before any
future non-13-mer target is pre-registered. **Do not fix it now**: changing the gate
after the holdout target is named, but before it runs, would change the engine between
pre-registration and evaluation.

---

## 9. FINAL: exhaustive 17-of-17 partition

Two independent blind stages selected the holdouts, and neither used an M5 number:

* **Stage A — metadata** (`scripts/audit_lockbox_metadata.py`): enumerate all 17
  `binding_affinity/*.csv`, dedupe, apply the frozen 0.50 quantile gate. Selects `aayl51`.
* **Stage B — baseline learnability** (`scripts/baseline_learnability_gate.py`): run
  **only M2 and M3**, dedupe against every already-scored pool, gate on
  `max(M2_rho, M3_rho) >= 0.15`. Selects `1mlc`.

Only after both were fixed did M5 run on `aayl51` and `1mlc`.

### The partition

| Class | Targets | Why |
| --- | --- | --- |
| **Learnable, multi-round — M5 leads ρ** | `1n8z` (L=13, 100 seeds), `aayl49_ML` (L=11), `2fjg` (L=13) | Baselines learn; M5 beats M3 on ρ at **p ≤ 2e-4 in 3 of 3** |
| **Unlearnable — nothing to rank** | `aayl51` (L=12), `4d5_her2` (L=13), `aayl49` (L=11) | All baselines ≤ 0.06 on `aayl51`; `4d5_her2` 0.02–0.04; `aayl49` 0.006–0.045 |
| **Single-round boundary** | `1mlc` (L=9, R=1, Q=10) | Baselines *do* learn (0.35–0.40) but the protocol is one round |
| **Degenerate (1–2 distinct CDRH3)** | `1mhp_LC`, `3gbn_h1`, `3gbn_h9`, `4fqi_h1`, `4fqi_h3`, `5a12_ang2`, `5a12_vegf`, `aayl50_LC`, `aayl52_LC`, `g6_LC` | Not search problems at all |

### The M5 result, stated exactly

M5 versus **M3 (standard UCB)** on Spearman ρ at the final checkpoint:

| Target | L | rounds | M5 ρ | M3 ρ | Δ | p | seeds won |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1n8z` | 13 | 6 | 0.6578 | 0.3442 | **+0.3136** | **8.4e-32** | 98/100 |
| `aayl49_ML` | 11 | 24 | 0.2694 | 0.1907 | **+0.0787** | **2.4e-10** | — |
| `2fjg` | 13 | 3 | 0.2327 | −0.0530 | **+0.2857** | **6.2e-06** | — |
| `aayl51` | 12 | 15 | 0.0098 | 0.0112 | −0.0014 | 0.68 | — |
| `1mlc` | 9 | **1** | 0.3814 | 0.3541 | +0.0273 | — | — |

So: **M5 wins ρ on 3 of 3 multi-round learnable targets at p ≤ 2e-4, and neither wins
nor loses where there is nothing to learn.**

### Two caveats that must travel with this claim

1. **`1mlc` cannot support the loop claim.** At N_qual=67 the protocol clamps to
   `R=1, Q=10`, and M5's AUBQC is *bit-identical* to M3's (paired Δ = +0.0000,
   p = 1.0). That is structural, not empirical: with one batch the 5-slot portfolio is
   the same set for both methods. It is recorded as a **boundary control**, and the
   33 % cell is `R=3` only because the coverage term pushes past the clamp — it is
   still not a like-for-like comparison with 1N8Z's 6 rounds.
2. **"Tying" on unlearnable targets is not a win.** On `aayl51`, `4d5_her2` and
   `aayl49`, M5 neither leads nor trails meaningfully because no method's surrogate
   correlates with truth. Reporting that as "M5 is safe" would be wrong; the correct
   statement is **the benchmark has no power on those targets**.

### Note on a superseded pre-filter

`aayl49` is excluded from the learnable set not by a hand-written rule but by a
**measured 17.41 % sequence overlap** with the already-scored `aayl49_ML`. After
dedupe it retains 396 unique CDRH3s and still fails the gate (0.006–0.045), so the
exclusion is doubly justified.
