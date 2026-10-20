# Step 3 — Dynamic Campaign Protocol Sizing

## 1. Verified premises

Measured from the artifacts, not assumed:

| quantity | 1N8Z | aayl49_ML | ratio |
|---|---|---|---|
| pool `N` | 419 | 1,510 | 3.60× |
| qualified `N_qual` | **210** | **756** | 3.60× |
| `tau_dev` source | `pool_median` | `pool_median` | same rule |
| `Q` spent | 35 | 35 | 1.00× |
| `coverage_pct` | **16.67%** | **4.63%** | 3.60× short |

Your coverage argument is correct and its arithmetic is exact: holding `Q = 35` on a
35.99%-larger qualified pool examines 3.6× less of it. The "equivalent to stopping 1N8Z
after q=10" framing is also right in coverage terms (10/210 = 4.76% ≈ 4.63%).

Also verified from `core/acquisition.py:1172-1206` and `995-1000`:

* At `VoI = 0` the objective **is** already exploitation-only: `exploit_weight = 1 − 0.35·0 = 1.0`,
  so `a_null = mu_bind_norm`, `optimistic = a_h1`, and slots 1, 2, 3, 4 all rank on `a_h1`.
* The null control (slot 5, `a_null`) and the VoI probe (slot 3) are therefore **the only**
  remaining non-exploitative slots once `VoI = 0`. Slots 3 and 4 are 40% of the batch, which
  is the "20–40% wasted" you identified.

Two constants also leak 1N8Z into the cross-target path and must be parameterized with the
protocol:

* `core/tool_arsenal.py:551` — `full_pool = 419` is hard-coded in
  `compute_cost_and_effort_saved`, so aayl49's "saved vs exhaustive" is computed against
  1N8Z's pool.
* `core/constants.py:33` — `CHECKPOINT_QUERIES = tuple(range(5, 36, 5))`, plus the literal
  `30.0` AUBQC denominator in `core/baselines.py:630,633,635` and
  `core/campaign.py:152`, plus `6 * SLOT_COUNT` at `core/baselines.py:412`.

---

## 2. The two requirements that cannot both hold

### 2.1 Terminal-harvest All-In (2a) contradicts 1N8Z bit-identity

`horizon_factor` returns `0.0` **iff** `future_queries == 0`, i.e. only on the final batch.
So `VoI == 0` is, by construction, exactly the last round — and in 1N8Z that is **R6**, which
is in `results/golden_seed21.sqlite3` and `results/benchmark_100seeds.json`.

Going All-In at `VoI == 0` therefore changes 1N8Z's R6: slot 5 stops taking the H0 control,
so the five queried ids change, so `curves`, `aubqc`, `m5_warm_start` and `paired_t_tests`
all change. There is no formulation of 2a that leaves 1N8Z bit-identical, because 1N8Z's
final batch is precisely the regime 2a targets.

(Triggering on `sar_maturity == 1.0` instead is worse: 1N8Z reaches 1.0 at **R2**, so 2a
would rewrite R2–R6.)

### 2.2 Confounding-scaled shrinkage (2b) also contradicts it

Shrinking declared priors by a confounding factor changes `Δ_agent`, which changes `A_H1`,
which changes every pick from R1 on. 1N8Z's top revealed candidates are 4-mutants
(`ARWGGYGFYAMDY` vs `SRWGGDGFYAMDY`: positions 1, 5, 8, 11), so any honest
confounding factor is `< 1.0` there and 1N8Z moves.

**There is no version of 2a or 2b that is both "scales with state" and "1N8Z unchanged".**
The only way to have both is to make them part of the *protocol object* and freeze 1N8Z's
protocol at its current value. That is defensible — R is already becoming protocol — but it
must be stated as "different protocol", never as "same constants". §4 proposes exactly that.

---

## 3. `compute_campaign_protocol(n_qualified, seq_len)`

### 3.1 Formula

```python
REFERENCE_N_QUALIFIED = 210          # 1N8Z qualified pool
REFERENCE_COVERAGE    = 35 / 210     # 0.166667, pre-registered
B_0 = 5                              # seed batch, unchanged
B   = 5                              # batch, unchanged (see §3.2)
ROUNDS_MAX = 30                      # guard; binds on no current target

q_target = round(n_qualified * REFERENCE_COVERAGE)
rounds   = clip(round((q_target - B_0) / B), 1, ROUNDS_MAX)
Q_max    = B_0 + B * rounds
checkpoints = tuple(range(B_0, Q_max + 1, B))
aubqc_normalizer = Q_max - B_0       # == 30 for 1N8Z
```

Results:

| target | `N_qual` | `q_target` | `R` | `Q_max` | `coverage_pct` |
|---|---|---|---|---|---|
| 1N8Z | 210 | 35 | **6** | **35** | **16.67%** |
| aayl49_ML | 756 | 126 | 24 | 125 | 16.54% |
| 2FJG | ~124 (to confirm) | 21 | 3 | 20 | 16.13% |

**1N8Z identity is exact by construction, not by tolerance**: `round(210 × 35/210) = 35`,
`round(30/5) = 6`, `Q_max = 35`, `checkpoints = (5,…,35)`, normalizer `= 30`. Every one of
`CHECKPOINT_QUERIES`, the AUBQC denominator and the round count reproduces the frozen value.
This is the property §6 tests first.

### 3.2 Scale `R`, not `B` — and this is the important recommendation

The request says to "scale the batch/round protocol and the slot portfolio". Scaling **B**
is a much larger blast radius than it looks, and every one of these breaks:

| invariant | where | breaks if B ≠ 5 |
|---|---|---|
| H1 arm = slots 1, 2, 4 | `H1_EXPLOIT_SLOT_IDS` (`tool_arsenal.py:117`) | arm membership is positional |
| H0 control = slot 5 | `SLOT5_FALSIFY_LABEL` | control index is hard-coded |
| slot 3 = the VoI probe | `SLOT3_LABEL` | so is the exploration index |
| `slot_id` order 1..5 | `tests/test_stage_readiness.py:142` | asserts `[1,2,3,4,5]` |
| `round_number` 1..6 | `core/schemas.py:36` `Field(ge=1, le=6)` | would reject R7 |

Scaling **R** with `B = 5` reaches the same `coverage_pct` while leaving all five invariants
untouched: the 3-exploit/2-experimental split, the paired audit, the H1/H0 comparison and
the slot labels all keep working, and only the loop length changes. The cost is that `Q_max`
moves in steps of 5 rather than exactly hitting `q_target` (125 vs 126 on aayl49) — a 0.8%
residual, recorded as `coverage_pct` rather than hidden.

`core/schemas.py:36`'s `le=6` must become `le=ROUNDS_MAX` for R=24 to validate at all.

---

## 4. Reconciling 2a/2b with bit-identity

Proposal: put allocation policy in the protocol object.

```json
"allocation": {
  "mode": "fixed_split",          // 1N8Z, frozen protocol
  "terminal_harvest_all_in": false,
  "confounding_shrinkage": false
}
```
Dynamically-sized protocols get `"mode": "adaptive_split"` with both flags `true`.

* `fixed_split` → today's `select_five_slots`, byte-for-byte. 1N8Z is unaffected because its
  **protocol file already exists and is not rewritten**.
* `adaptive_split` → two additional behaviours:
  * **All-In**: when `voi == 0.0` **and** `sar_maturity >= 1.0`, slots 3–5 also rank on
    `a_h1`; the H0 control is retained only while `voi > 0`. Recorded per round in
    `voi_diagnostics.allocation_mode` and in the audit as
    `control_suppressed: true`, so a suppressed control is visible rather than silently
    absent.
  * **Confounding shrinkage**: the declared-prior term is multiplied by
    `1 / (1 + c)` where `c` is the mean number of non-WT mutations among revealed candidates
    sharing the prior's position, normalised by `L`. Label-free (uses sequences + the
    ledger), `c = 0` ⇒ factor `1.0` ⇒ no change on a single-mutant ledger.

I am **not** claiming this is a constants-free change. It is a protocol change, and §5 argues
that the protocol is exactly what should be allowed to vary across targets — but the artifact
and the report must say `allocation_mode` next to every number, and the 1N8Z row must stay
labelled `fixed_split`.

**Alternative, if you would rather keep one code path for everything**: apply 2a/2b
unconditionally to all targets *including* 1N8Z, accept that 1N8Z is no longer
`1.9282` AUBQC, and re-freeze `results/benchmark_100seeds.json`. That is a larger,
irreversible change to the headline artifact and I do not recommend it.

---

## 5. Evidential risk I have to name

The 16.7% coverage rule is derived from 1N8Z, and we are choosing it **after** observing that
M5 lost ground at 4.6%. Choosing the protocol after seeing the ranking is the ordinary way a
benchmark becomes unfalsifiable. Two mitigations, both cheap:

1. **Pre-register the rule.** `REFERENCE_COVERAGE = 35/210` is fixed in code before the run,
   with no tuning against results. It is the only defensible choice: it is the coverage the
   frozen 1N8Z protocol already had.
2. **Report a coverage sensitivity table**: `coverage ∈ {5%, 16.7%, 33%}` for both targets.
   If M5's advantage appears only at one coverage level, that is a property of the coverage
   knob, not of the architecture, and the table says so. If it holds across all three, the
   claim is robust. Either way the reader sees the dependence instead of one cherry-picked
   number.

**I will not promise that M5 wins.** I will run it and report the table. If M5 still trails
M2 on AUBQC at equal coverage, that is the finding, and it is more useful than a tuned win.

### Two outcome risks worth stating in advance

* **AUBQC can still favour M2 at equal coverage.** Greedy `mu_bind` is strong on landscapes
  where the surrogate ranks well; M5's advantage on 1N8Z came partly from exploration that
  this budget now funds for everyone equally.
* **`m5_warm_start` is not a fair comparison.** It picks a better Round 0, so it has an
  information advantage M1–M5 do not; it beat M5 even at 4.6% coverage. It is *expected* to
  win and its win is not evidence about the acquisition architecture. It must stay reported
  separately, as it already is.

---

## 6. Tests

1. `test_campaign_protocol_is_identity_on_1n8z` — `compute_campaign_protocol(210, 13)` returns
   exactly `B_0=5, B=5, R=6, Q_max=35`, `checkpoints == CHECKPOINT_QUERIES`,
   `aubqc_normalizer == 30`. Plus the 100-seed bit-identity re-check against
   `results/benchmark_100seeds.json` (the hard requirement).
2. `test_dynamic_protocol_scales_with_qualified_pool` — monotone in `N_qual`; coverage within
   one batch of 16.67% for 210/756/~124; `ROUNDS_MAX` binds recorded as `cap_bound`.
3. `test_adaptive_split_activates_only_at_terminal_harvest` — with `adaptive_split`, slots
   3–5 rank on `a_h1` iff `voi == 0 and sar_maturity >= 1.0`; with `fixed_split`, the 3/2
   split is unchanged for every `voi`.
4. `test_confounding_shrinkage_is_identity_without_confounding` — factor `== 1.0` on a
   single-mutant ledger, `< 1.0` on a confounded one.
5. `test_cross_target_reports_protocol` — artifact carries `B_0, B, R, Q_max, coverage_pct,
   cap_bound, allocation_mode` per target, and `coverage_pct > 0.10` for both.

---

## 7. Files

| file | change |
|---|---|
| `core/acquisition.py` | `CampaignProtocol` + `compute_campaign_protocol`; `select_five_slots(allocation=...)`; confounding shrink in `adaptive_prior_score` |
| `core/constants.py` | `CHECKPOINT_QUERIES` becomes derived via the protocol (1N8Z value pinned by test) |
| `core/baselines.py` | thread protocol: `6*SLOT_COUNT` → `protocol.rounds`, `/30.0` → `/protocol.aubqc_normalizer`, `range(5,36,5)` → protocol checkpoints |
| `core/campaign.py` | `aubqc_from_curve` normalizer; per-round `allocation_mode` in diagnostics |
| `core/schemas.py` | `round_number` `le=6` → `le=ROUNDS_MAX` |
| `core/tool_arsenal.py` | `full_pool=419` → protocol `pool_size` |
| `config/settings.py` | protocol defaults + `allocation_mode` |
| `scripts/run_cross_target_benchmark.py` | run **both** targets, both coverage sweep, record protocol |
| `tests/test_stage_readiness.py` | tests 1–5 |

Protected artifacts stay read-only; only `results/cross_target_benchmark.json` is rewritten.
