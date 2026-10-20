# Step 2 — Option C Tool Arsenal + Cross-Target Generalization

## Outcome (implemented)

`95 passed` (was 92; +3 new stage tests). `1N8Z` re-run at 100 seeds is **bit-identical**
across all 12 report sections (`curves`, `aubqc`, `normalized_*`, `top5_panel_*`,
`spearman_rho_curves`, `top_decile_hit_rate`, `matched_initial_seeds`, `paired_t_tests`,
`m5_warm_start`) against `results/benchmark_100seeds.json`.

Cross-target ran on **aayl49_ML** (anti-lysozyme), L=11, 1510 distinct CDR3s,
reference `AKGRAAGTFDS`, tau_dev `0.6976`, 50 seeds → `results/cross_target_benchmark.json`.

Deviations from this plan, each forced by a measurement:

1. **`4FQI` and `3GBN` cannot be used at any length.** They are framework DMS libraries:
   65,094 and 1,887 rows collapse to **two** distinct CDR3s, 12 of 14 positions 100%
   conserved. A usability gate (`data.datasets.cross_target_usability`) rejects them with
   the measured reason, which the artifact records. Added `2FJG` to the chain (248 unique
   13-mers, 13/13 positions varying) as the one 13-residue alternative.
2. **The problem was length, not just reference.** The request's Option C paragraph said
   "without changing a single constant", but `FEATURE_DIM = 299`, `range(13)` and
   `le=13` hard-coded 13 in five modules. Length/reference parameterization was approved
   mid-task and applied to `core/features.py` (not in the original list, but the actual
   blocker), `developability`, `acquisition`, `baselines`, `schemas`, `data/datasets`.
3. **`detect_epistasis_clashes` drops the `sar_maturity` clause.** The golden ledger
   reports `sar_maturity = 0.65` in R1, so `or sar_maturity >= 0.30` opened the drill in
   the exact round the request requires it closed. Gate is `queries_spent >= 10`
   (== `round_number >= 2`).
4. **`compute_cost_and_effort_saved` opens in R2**, not R3: R2's VoI is `0.28 <= 0.35`.
   That is the specified OR, so the test asserts the rule rather than a round.

## Original plan

Design decisions established from reading the code, before any edit. Two things in the
request are **not implementable as literally stated**; they are resolved below and
flagged in §3 so they are not discovered at the end.

Nothing in this step reads or writes `results/golden_seed21.sqlite3`,
`results/campaign.sqlite3`, `results/benchmark_100seeds.json` or
`results/benchmark_10seeds.json`.

---

## 0. The contract constraint that shapes everything

`tests/test_core.py:53` asserts **exact key equality**, not a subset:

```python
state=client.get("/api/state").json(); contract=json.load(open("api_contract.json"))
assert state.keys()==contract.keys()
```

That test hits `/api/state` with no `round` parameter, which returns `lab.state` — built by
`Campaign._refresh_state`, **not** by `state_at`. So:

* a new **top-level** key on `state_at(k)` is safe (the stage-readiness test at
  `tests/test_stage_readiness.py:101` uses `CONTRACT_KEYS <= set(state)`);
* a new top-level key on `lab.state` is **not** safe.

The request asks for `active_tools_metadata` on both `GET /api/state?round=k` **and**
`GET /api/rounds`. `/api/rounds` is derived from `state_at`, so the only thing that forces a
decision is whether the *live* view should carry it too.

**Decision:** put it inside the existing `voi_diagnostics` object as
`voi_diagnostics.active_tools_metadata`. `voi_diagnostics` is already additive-only in every
test (membership checks only — `tests/test_api_sync.py:152`, `tests/test_stage_readiness.py:116`),
it is populated by `empty_voi_diagnostics()`, `_voi_diagnostics()` and `state_at()` alike, and
it is exactly where VoI lives — which is what drives half the triggers. This gives the live
`/api/state`, `/api/state?round=k` and `/api/rounds` all the same metadata with **zero** risk to
the frozen key set. The name from the request (`active_tools_metadata`) is preserved.

The alternative — top-level key only on `state_at` — would leave the live view with no tool
state, forcing the UI to call `?round=current` to find out what is running. Rejected.

---

## 1. `select_active_tools` — the two-tier selector

New in `core/tool_arsenal.py`, pure and deterministic:

```python
def select_active_tools(round_number: int, queries_spent: int,
                        voi: float, sar_maturity: float) -> dict
```

Returns `{"round_number", "voi", "queries_spent", "sar_maturity", "active", "tools"}` where
`tools` is six entries in fixed order of `{name, tier, active, trigger_reason, routed_to}`.
`tier` is `"SOP Baseline"` or `"State-Triggered Drill"`.

| tool | tier | active when | routed to |
|---|---|---|---|
| `scan_positional_sar` | SOP Baseline | `round_number >= 1` | Affinity Scientist, Experimental Planner |
| `profile_developability_liabilities` | SOP Baseline | `round_number >= 1` | Developability Scientist, Experimental Planner |
| `inspect_joint_uncertainty_hotspots` | State-Triggered Drill | `voi > 0.0` | Scientific Skeptic, Experimental Planner |
| `audit_past_hypotheses` | State-Triggered Drill | `round_number >= 2` | Scientific Skeptic, Experimental Planner |
| `detect_epistasis_clashes` | State-Triggered Drill | `queries_spent >= 10 or sar_maturity >= 0.30` | Affinity Scientist, Scientific Skeptic |
| `compute_cost_and_effort_saved` | State-Triggered Drill | `round_number >= 3 or voi <= 0.35` | Experimental Planner |

Three decisions inside that table:

1. **Tier 1 gated on `round_number >= 1`, not "always".** Round 0 is the seed batch; no council
   runs and no batch is designed, so claiming the SOP baselines were routed would be false. The
   seed round reports them as `Deactivated: seed round has no council planning step`.
2. **`detect_epistasis_clashes` uses the literal predicate plus the stated early trigger.** The
   request gives `queries_spent >= 10` and then, parenthetically, `round_number >= 2 or
   sar_maturity >= 0.30`. `queries_spent >= 10` and `round_number >= 2` are the same statement
   (the round-`k` planning prefix holds `5 + 5(k-1)` queries), so the two clauses that are *not*
   redundant are kept: `queries_spent >= 10 or sar_maturity >= 0.30`. The second clause lets a
   genuinely converged SAR table activate the drill one batch early rather than being ignored.
3. **`trigger_reason` is written from the values, not templated per round**, so it is checkable:
   `"Active: VoI = 0.5780 > 0"` in R1 versus `"Deactivated: Terminal harvest (VoI = 0.0000)"` in R6.

Golden-ledger arc this produces (VoI comes from the persisted batch design, `queries_spent` from
the ledger prefix **through round `k-1`**):

| round | queries before | VoI | hotspots | audit | epistasis | cost |
|---|---|---|---|---|---|---|
| R1 | 5 | 0.578 | ✔ | ✘ | ✘ | ✘ |
| R2 | 10 | >0 | ✔ | ✔ | ✔ | rule-dependent |
| R3 | 15 | >0 | ✔ | ✔ | ✔ | ✔ |
| R5 | 25 | 0.07 | ✔ | ✔ | ✔ | ✔ |
| R6 | 30 | 0.000 | ✘ | ✔ | ✔ | ✔ |

R2's cost drill is genuinely `voi <= 0.35`-dependent; the test asserts the *rule* there and the
guaranteed anchors elsewhere (see §4).

### 1.1 `queries_spent` must be the planning prefix, not the round's own spend

This is the one off-by-one-batch bug that would silently break the feature, so it is called out.
`state_at(k)` builds `revealed = self._round_reveals(k)` — the prefix **through** `k`, i.e.
`5 + 5k` queries. Feeding that to the selector would activate `detect_epistasis_clashes` in R1
(10 queries) instead of R2. The selector must be fed the prefix the *council* saw:

```python
queries_before = 0 if round_number <= 0 else len(self._round_reveals(round_number - 1))
```

### 1.2 Live view and reconstructed view must be byte-identical

`tests/test_stage_readiness.py:281` compares `live["voi_diagnostics"] == rebuilt["voi_diagnostics"]`
by **exact dict equality**. The metadata is therefore computed by one helper called from
`_voi_diagnostics`, on the **rounded** `voi`/`sar_maturity` that the same dict already exports —
not on raw floats. Otherwise R6's `-1e-17` residual would deactivate the drill live and activate
it in reconstruction, or vice versa.

### 1.3 Routing is enforced, not merely advertised

`VirtualLabRunner.plan_round` currently computes all five tools unconditionally
(`core/virtual_lab_runner.py:293`) and `core/agents.py:245` hands each seat every tool in its
static `tools_invoked` tuple. Changes:

* `plan_round` calls `score_pool` **before** building the tool table (it already returns the
  `voi`/`sar_maturity` context), then computes only the active tools and emits `tool_output` only
  for those. A drill that is off costs no work and produces no event.
* `run_scientists(..., active_tools=None)` filters each seat's `tools_invoked` to the active set
  before building `AgentMemo`. One seat can therefore lose a tool mid-campaign, which is the
  visible behaviour the request asks for (`Affinity Scientist` drops
  `detect_epistasis_clashes` in R1 and regains it in R2).
* `_deterministic_decision(..., tools=...)` and `_reconcile_decision(..., tools=...)` receive the
  active names, so `RoundDecisionRecord.tools_invoked` in the ledger is the real arsenal.
* `agent_council` memos read back from a **pre-Option-C ledger** (the golden file) were recorded
  with static lists. They are left verbatim and annotated additively with `tools_invoked_active`
  and `tools_inactive` at read time in `state_at`. Rewriting the recorded list would make the
  audit trail claim the council saw something it did not.

---

## 2. Cross-target benchmark

`scripts/run_cross_target_benchmark.py` + `GET /api/cross-target`.

Unchanged on purpose: `core/acquisition.py`, `core/baselines.py`, `core/surrogate.py`,
`core/features.py`. The benchmark calls `core.baselines.run_benchmark(frame, normalizer, tau_dev,
seeds=N)`, which already covers `M1..M5` plus `m5_warm_start` — the same function that produced
`results/benchmark_100seeds.json`. No constant moves.

### 2.1 `tau_dev` is label-blind and frozen to its own file

`core.preregistration.freeze_protocol` **writes to `SETTINGS.protocol_path` by default** —
`config/frozen_protocol.json`, the frozen protocol for the 1N8Z campaign. Calling it without an
explicit `output=` would overwrite the primary protocol and invalidate the live campaign and the
golden ledger. The script therefore passes
`output=SETTINGS.root / "config/cross_target_protocol.json"`. `freeze_protocol` already rejects a
frame carrying `y_bind` (`core/preregistration.py:23`), so the label-blindness guarantee is
inherited rather than re-implemented.

### 2.2 Artifact

`results/cross_target_benchmark.json` = the verbatim `run_benchmark` report plus a `cross_target`
block:

```json
{
  "cross_target": {
    "dataset": "4fqi", "target": "CR9114 / Influenza HA",
    "source": "huggingface | data/raw/dev_hcdr3.csv",
    "source_file": "binding_affinity/4fqi_benchmarking_data.csv",
    "reference_sequence": "SRWGGDGFYAMDY",
    "alignment": "same 13-aa HCDR3 window; reference sequence unchanged",
    "rows": 0, "unique_sequences": 0, "replicates_collapsed": 0,
    "tau_dev": 0.0, "qualified_count": 0, "qualification_pass_rate": 0.0,
    "dataset_sequence_sha256": "...", "csv_sha256": "...",
    "seeds": 50, "baselines_unchanged": true
  },
  "...": "curves, aubqc, normalized_aubqc, top5_panel_*, spearman_rho_curves,
          top_decile_hit_rate, matched_initial_seeds, paired_t_tests, m5_warm_start"
}
```

`GET /api/cross-target` serves that file, memoized like `/api/benchmark`, and returns a
structured `404`-free empty payload when the artifact is absent (the stage test skips instead).

---

## 3. Two parts of the request that cannot be taken literally

### 3.1 "A different wild-type/consensus sequence as reference" contradicts "no constant changes"

`WT_SEQUENCE = "SRWGGDGFYAMDY"` is not a parameter of the acquisition module — it is imported
from `core/constants.py` and used for *positional identity* by `sar_effect_table`,
`_residue_matrix`, `loop_prior_score`, `adaptive_prior_score` and `raw_components`
(`ANCHOR_POSITIONS = (3, 11, 12, 13)` likewise). A second dataset whose own wild type is, say,
`ARWG...` would be scored against 1N8Z's reference at every position, which is not "the same
architecture generalizing" — it is a coordinate-system error. Changing those constants is
forbidden by the request, and rightly so: it would invalidate the 100-seed artifact.

So the generalization claim must be stated precisely, in the artifact and in the summary table:

> The M5 + `m5_warm_start` architecture is re-run **without modification** on a second
> experimentally measured antibody–antigen dataset, *aligned onto the same 13-aa HCDR3 reference
> window*. Generalization is demonstrated across **antigen/target and mutation landscape**, not
> across a different reference frame.

Every row of the cross-target dataset is required to be a 13-mer (the existing
`prepare_abbibench_frame(..., length=13)` filter) and positions are compared against
`WT_SEQUENCE`. That is an honest, checkable claim; the alternative phrasing is not.

### 3.2 The second dataset is not on this machine, and there is no network in this mode

`data/raw/` holds exactly two files: `1n8z_hcdr3.csv` (419 rows, the primary) and
`dev_hcdr3.csv` (300 rows). `dev_hcdr3.csv` is **real experimental data** — it is the
AbBiBench `binding_affinity/4d5_her2_benchmarking_data_trimmed.csv` slice (4D5 / HER2), already
length-13, and it is currently unused by every benchmark path. It is a genuine second
antibody–antigen dataset sitting on disk.

It is *not* what the request names first. Implementation therefore:

1. tries `hf_hub_download` for each configured target in order
   (`4fqi`, `3gbn`, `aayl49_ml`, `1mlc`), through the same `HfApi`/`hf_hub_download` path as
   `scripts/fetch_real_1n8z.py`;
2. caches the first one that yields a usable pool (≥ 5 qualified and ≥ 35 total rows after the
   13-mer filter and replicate collapse) to `data/processed/cross_target.csv`;
3. **falls back to the on-disk `dev_hcdr3.csv` slice** when the network is unavailable, and
   records `source` honestly;
4. never silently substitutes — `cross_target.dataset` and `cross_target.source` say which one ran.

**Open question for you:** `4FQI`'s HCDR3 length is unverified from here (no shell, no network),
and AbBiBench targets are reported as HCDR3s of 8–18 residues. If 4FQI's CDR3 is not 13, the
13-mer filter empties it and the fallback fires. If you want a specific target guaranteed, tell me
and I will add an explicit truncation/padding rule — at the cost of introducing residues the
assay never measured, which weakens the claim.

### 3.3 Replicates

`dev_hcdr3.csv` has 300 rows but far fewer unique sequences (many identical sequences with
different measured values — this is real replicate structure). Duplicate sequences with different
labels make the surrogate's training set ambiguous and allow the same molecule to be "discovered"
twice. The script collapses to one row per sequence at the mean label and records
`replicates_collapsed`. This is a preprocessing decision, not a model change, and it is recorded
in the artifact.

---

## 4. Tests to add to `tests/test_stage_readiness.py`

**(a) `test_option_c_tool_activation_across_rounds`** — step R0 → R6 through `GET /api/state?round=k`
and assert, on `state["voi_diagnostics"]["active_tools_metadata"]`:

* six tools, stable order, each with `tier` ∈ {`SOP Baseline`, `State-Triggered Drill`}, `active`,
  `trigger_reason`, `routed_to`;
* Tier 1 active for every `k` in 1..6 and inactive at `k == 0`;
* `audit_past_hypotheses` inactive at R1, active R2..R6 (guaranteed by `round_number` alone);
* `compute_cost_and_effort_saved` inactive at R1, active R3..R6 (guaranteed by `round_number` alone);
* `inspect_joint_uncertainty_hotspots` active R1..R5 and inactive R6 with `trigger_reason` containing
  `Terminal harvest` — the requested deactivation;
* `detect_epistasis_clashes` inactive at R1 and active R2..R6;
* `trigger_reason` string contains the actual VoI value, and the activation boolean is recomputed
  from the tool's own stated predicate so the test fails if the predicate and the flag disagree;
* at least one tool flips **on** and at least one flips **off** between consecutive rounds, so the
  "tools turning on and off" demo is asserted rather than assumed;
* `/api/rounds` entries carry `active_tools` and it equals the same round's metadata.

**(b) `test_cross_target_endpoint`** — `GET /api/cross-target` returns
`seeds`, `curves` (7 checkpoints, 5..35), `aubqc` over all of `METHODS`, `m5_warm_start`,
`paired_t_tests`, and a `cross_target` block whose `source` and `dataset_sequence_sha256` match the
file. Skips when the artifact is absent, mirroring the existing `_require`/`pytest.skip` idiom.

The script prints the cross-target summary table (AUBQC / normalized AUBQC / top-5 panel /
top-decile hit rate / Spearman ρ at q=35 per method, plus the M5−M3 and M5−M2 paired tests), and
the test prints the same table via `scripts.run_cross_target_benchmark.format_summary(...)` so the
pytest log carries the evidence.

---

## 5. Files touched

| file | change |
|---|---|
| `core/tool_arsenal.py` | `select_active_tools`, `TIER1_TOOLS`/`TIER2_TOOLS`, `TOOL_ROUTING`; registry exported for the API |
| `core/campaign.py` | `_tool_activation`, `voi_diagnostics.active_tools_metadata`, `agent_council[*].tools_invoked_active`, `timeline()` gains `active_tools` |
| `core/virtual_lab_runner.py` | compute `voi` before tools; build/emit only active tools; pass active set to `run_scientists` |
| `core/agents.py` | `run_scientists(..., active_tools=...)` filters each seat's `tools_invoked` |
| `core/state.py` | `empty_voi_diagnostics()` gains an empty `active_tools_metadata` |
| `config/settings.py` | one additive field, `cross_target_benchmark_path` (defaulted; frozen dataclass stays frozen) |
| `data/datasets.py` | `align_cross_target_frame(...)`: 13-mer filter, replicate collapse, id assignment |
| `scripts/run_cross_target_benchmark.py` | new: fetch/cache, freeze to a **separate** protocol file, `run_benchmark`, write artifact, print tables |
| `app/server.py` | `GET /api/cross-target`, memoized |
| `tests/test_stage_readiness.py` | tests (a) and (b) |

Untouched: `core/acquisition.py`, `core/baselines.py`, `core/surrogate.py`, `core/features.py`,
`api_contract.json`, and the four protected artifacts.

---

## 6. Verification order

1. `pytest tests/test_core.py tests/test_api_sync.py tests/test_runner.py` — contract and
   runner regressions first (the `state.keys()==contract.keys()` test is the one that will break
   if `active_tools_metadata` is ever promoted to a top-level key).
2. `pytest tests/test_stage_readiness.py` — the new tests plus the existing timeline,
   live-vs-reconstructed and rewind guards.
3. `python scripts/run_cross_target_benchmark.py --seeds 50` — writes
   `results/cross_target_benchmark.json`; report `source`, rows, `tau_dev`.
4. Full suite; confirm `git status` shows the four protected artifacts unmodified.
