# SPEC.md — Self-Accelerating Antibody Lead Optimization (v1.2, Post-Kickoff Edition)

**Hybrid Virtual Lab for the Self-Accelerating AI for Drug Development Hackathon.**
This document is the single source of truth for the Python repository, the agents, and the FastAPI
backend. The React frontend is built separately against `api_contract.json`.

---

## 0. Scope of this session

In scope (this repository):

- `data/` dataset vault + hidden oracles + immutable ledger
- `core/` deterministic tool arsenal, dual surrogates, 5 matched benchmark methods,
  multi-agent orchestration, campaign state
- `app/server.py` FastAPI backend (`/api/state`, `/api/stream-round`, CORS `*`)
- `tests/`, `scripts/`, `config/`

Out of scope (deliberately not built here): `frontend/` React app, `3Dmol.js` panel.

---

## 1. Scientific problem

Retrospective, blinded replay of an existing experimental campaign:

- Candidate pool: **N = 419** fixed-length **L = 13** HCDR3 variants of trastuzumab
  (1N8Z / Absci SPR, VH 97–109).
- **Wild type:** `SRWGGDGFYAMDY` (13 aa).
- Objective 1 — measured HER2 binding `y_bind = -log10(Kd [M])` (higher = tighter). **Hidden.**
- Objective 2 — developability safety `y_dev = 1 - Risk(s) ∈ [0,1]` (higher = safer). Derived from
  sequence-only proxies, available at `t=0`.
- The system must find high-affinity **and** developability-qualified leads while spending as few
  wet-lab queries as possible.

Engineering note: a real 419-row export may not be present in the repo. The repository therefore
ships a **deterministic, regression-pinned fixture generator** that reproduces the exact schema,
row count, and wild-type sequence, with SHA-256 pinning and a data provenance manifest. When the
real AbBiBench export is dropped in `data/raw/`, the loader prefers it and re-freezes the protocol.

---

## 2. Information boundary (hard rules)

| Information | Visibility |
|---|---|
| HCDR3 sequence | `t=0`, free |
| 6 raw developability proxies + normalized composite `Risk(s)` | `t=0`, free |
| `tau_dev` (frozen qualification threshold) | `t=0`, **computed label-blind first** |
| `y_bind` | only via `BindingOracle.query()` |
| `y_dev` (label form of developability) | only via `DevelopabilityOracle.query()` |

Non-negotiable guardrails:

1. **`tau_dev` is frozen before any binding label is read.** `core/preregistration.py` reads *only*
   the sequence column, computes the composite risk over the unlabeled pool, and applies the
   pre-registered rule.
2. **`config/frozen_protocol.json` is write-once.** Rewriting it with different bytes raises.
3. **Every reveal is logged** to an append-only ledger with a SHA-256 hash chain and SQLite
   triggers that reject `UPDATE`/`DELETE`.
4. **Oracles are idempotent**: re-querying a revealed ID returns the cached value, does not append
   a second reveal, and does not spend budget twice.

### 2.1 Frozen developability risk proxy

For a 13-aa sequence `s` (each component min-max normalized across the **unlabeled 419 pool**):

```
R_hydro   = mean_p Kyte-Doolittle hydropathy(s_p)
R_win     = max over 9 windows of mean hydropathy over 5 consecutive residues
R_arom    = fraction of residues in {F, W, Y}
R_charge  = |sum_p q(s_p)| / 13          # K,R=+1.0; H=+0.1; D,E=-1.0 (pH 7.4)
R_cluster = count of adjacent same-sign charged dipeptides (++/--), positions 1..12
R_liab    = count of unpaired Cys + deamidation (NG, NS) + isomerization (DG, DS) beyond WT baseline

Risk(s)   = 0.25*R_hydro~ + 0.25*R_win~ + 0.20*R_arom~ + 0.10*R_charge~ + 0.10*R_cluster~ + 0.10*R_liab~
y_dev(s)  = 1 - Risk(s)
```

Frozen weights: `(0.25, 0.25, 0.20, 0.10, 0.10, 0.10)`.

### 2.2 Pre-registered `tau_dev`

```
tau_WT = Risk(SRWGGDGFYAMDY)
p      = fraction of the 419 unlabeled pool with Risk(s) <= tau_WT

tau_dev = tau_WT                      if 0.25 <= p <= 0.65      ("wild_type" branch)
        = median(Risk over pool)       otherwise                 ("pool_median" fallback)
```

Both branches are recorded in the frozen protocol manifest, together with `p`, the anchor residues,
and the dataset hash. Deterministic tie-break: median is `numpy.median`.

---

## 3. Features

`x_i ∈ R^299` (available for every candidate at `t=0`):

- `13 × 20 = 260` position-specific one-hot amino-acid indicators (canonical 20-letter order).
- `13 × 3 = 39` position-specific physicochemical descriptors:
  Kyte-Doolittle hydropathy, side-chain charge at pH 7.4, normalized van der Waals volume.

Non-canonical residues raise `ValueError` (fail loudly, never silently zero-fill).

---

## 4. Dual surrogates

Two **separate** 50-tree bootstrapped ExtraTrees ensembles, each returning `(mu, sigma)`:

- `BindingSurrogate` → `mu_bind, sigma_bind` (fit on revealed `y_bind`)
- `DevelopabilitySurrogate` → `mu_dev, sigma_dev` (fit on revealed `y_dev`, plus a deterministic
  sequence-descriptor prior at cold start)

`mu` = ensemble mean; `sigma` = ensemble standard deviation, then **min-max normalized to [0,1]
across the unqueried pool** (per SPEC §2.3). Raw values are retained for diagnostics.

```
A_base(s)   = 0.65*(mu_bind + beta0*sigma_bind) + 0.35*(mu_dev + beta0*sigma_dev)
              - lambda_pen * max(0, tau_dev - mu_dev)

sigma_joint = sqrt(0.65*sigma_bind^2 + 0.35*sigma_dev^2)
```

Frozen: `beta0 = 1.0` (M3/M4/M5; `beta0 = 0` for M2), `lambda_pen = 2.0`.

Determinism: `random_state` derives from the campaign seed via a stable hash of
`(seed, objective_name)` so that M3/M4/M5 see identical bootstrap draws.

---

## 5. The 6 deterministic tools (`core/tool_arsenal.py`)

Agents never do math on raw strings; they select tools and read exact tables.

1. `scan_positional_sar(ledger_df)` — per position × mutated residue: mean `y_bind`, `n`, and
   effect size `Δy_bind` vs WT. Suppresses rows with `n < min_n` into a `low_n` section.
2. `detect_epistasis_clashes(ledger_df)` — position pairs `(p_a, p_b)` where observed double
   mutants deviate from additive single-mutant expectation by `> 1.5 σ`.
3. `profile_developability_liabilities(candidate_ids)` — exact 5-mer hydrophobic window, aromatic
   density, net charge @ pH 7.4, adjacent charge clusters, deamidation/isomerization flags.
4. `inspect_joint_uncertainty_hotspots(unqueried_df, top_k=15)` — highest `sigma_bind`,
   `sigma_dev`, `sigma_joint` regions.
5. `audit_past_hypotheses(ledger_df, previous_decisions)` — stamps each past Slot 4 / Slot 5 pick
   `SUPPORTED` / `FALSIFIED` / `INCONCLUSIVE` against the round median.
6. `compute_cost_and_effort_saved(ledger_df, assay_cost_usd=1500)` — cumulative wet-lab dollars and
   assay-days, vs Random and Greedy baselines.

Every tool returns a JSON-serializable dict with a `tool` key and is registered in `TOOL_REGISTRY`
for agent-side invocation by name. All tools are pure functions of the revealed ledger (plus
surrogates for tool 4); no tool can read hidden labels for unqueried IDs.

---

## 6. Batch allocation (`K = 5`) — matched across methods

M5's batch is a **paired H1/H0 design**. Two scores are defined over the qualified unqueried pool:

```
A_base(s) = surrogate UCB                       # M3's objective; the H0 null hypothesis (Δ_agent = 0)
A_H1(s)   = A_base(s) + Δ_agent(s)              # M5, |Δ_agent| <= 0.06
```

| Slot | Rule |
|---|---|
| 1–2 | Top 2 unqueried by `A_H1` |
| 3 | Max `A_H1 + 0.50 · VoI · sigma_joint` |
| 4 | Max `A_H1 + 0.50 · max(0, Δ_agent)` — H1 promotion arm |
| 5 | Max `A_base` — H0 null-baseline control (what M3 would query next) |

`A_null` is `A_base`, not `A_base + loop_prior`: the static WT anchor rule (`loop_prior_score`) is
M4's objective alone. On 1N8Z the best binders start `AR...` and carry 4–5 aromatics, so that rule
penalises the true frontier and M4 trails plain UCB; M5 must not inherit the bias. The H0 control is
the **equality** hypothesis `Δ_agent == 0`, never `Δ_agent <= 0` — banning positively-boosted
candidates would make the lab boycott its own proven substitutions. When slots 1–4 promote a
hypothesis candidate over `A_base`, slot 5 automatically tests the displaced baseline candidate.

For M1–M4, slots 3–5 fall back to the deterministic extension of each method (random / greedy /
UCB / UCB+fixed-prior). Slots are always de-duplicated, and the selector always returns exactly 5
unique IDs while the unqueried pool has ≥ 5 members.

`Δ_agent(s) ∈ [-0.15, +0.15]` is bounded per positional soft prior, capped in aggregate after the
`0.65/0.35` objective weighting, and can **never** filter a candidate or alter the metric.

`VoI = H · (1 − 0.65 · C)` scales slot 3's exploration bonus only; it is not a phase switch.

---

## 7. Benchmark (5 matched methods)

- Budget: `5` shared initial seeds + `6` rounds × `5` queries = **35** total queries.
- Seed selection: one RNG stream per replicate seed; the **same 5 initial candidate IDs** are
  replayed into every method.
- M1 Random Search
- M2 Greedy Dual-Surrogate (`mu` only, `beta0 = 0`)
- M3 Dual-Surrogate Active Learning (joint UCB, no LLM agents)
- M4 M3 + Fixed Expert Loop Priors (static WT anchor bonus on 3, 11, 12, 13; penalty for >3 aromatics)
- M5 Hybrid Virtual Lab (deterministic tool arsenal + bounded agent soft priors + 5-slot quota)

Metrics:

- **AUBQC** = `(1/30) * Σ_{q=5,10,...,35} y*(q)`, where `y*(q)` is the best measured `y_bind`
  among **risk-qualified** candidates (`Risk ≤ tau_dev`) seen by query `q`.
- **`normalized_aubqc`** — the same area with `y*(q)` rescaled between the seed score `y*(5)`
  (0.0) and the best qualified binder in the pool (1.0), so the number is a fraction of the
  available headroom rather than an absolute log-`Kd`.
- **`top5_panel_aubqc`** — mean `y_bind` of the best five qualified candidates discovered so far,
  at each checkpoint. Portfolio depth, not one lucky maximum; exploration slots that surface
  non-binders simply do not enter the panel.
- **`spearman_rho_curves`** — `scipy.stats.spearmanr` between the method's scoring model
  (`mu_bind + Δ_agent`) and true `y_bind` over the qualified pool, at each checkpoint. This is the
  metric comparable to zero-shot affinity predictors on 1N8Z (AF3 `-0.16`, ProteinMPNN `-0.17`,
  ESM-2 `-0.13`). Undefined correlations are reported as `null`, not `0.0`.
- **`top_decile_hit_rate`** — fraction of the 30 post-seed queries that discover a candidate at or
  above the 90th percentile of all qualified binders in the pool.
- Target qualification success rate at `q ∈ {15, 25, 35}`.
- Pareto hypervolume over `(y_bind, -Risk)` with reference `(6.0, 1.0)`.
- Uncertainty coverage of the nominal 80% interval.
- Hypothesis confirmation / falsification rates.

The benchmark runner emits `benchmark_curves` in the exact shape required by `api_contract.json`
(`query_count`, `m1_random`, `m3_ucb`, `m5_hybrid`, plus `m2_greedy`/`m4_fixed_prior` for the UI).

---

## 8. Multi-agent orchestration

`core/virtual_lab_runner.py` expands `zou-group/virtual-lab` with frontier reasoning models
(**Gemini 4 Argon / Claude Opus 5.5 / GPT-6.1 Sol**, Thinking Mode enabled), each agent mapped to a
model and a `deterministic` fallback reasoner.

Per round:

1. **Tools first.** Execute the selected deterministic tools on the revealed ledger.
2. **Parallel science.** `asyncio.gather(Affinity Scientist, Developability Scientist)`.
3. **Then the Skeptic** (sees both memos).
4. **Planner synthesizes** into a validated `RoundDecisionRecord` (Pydantic):
   - `weight_delta` strictly within `[-0.15, +0.15]` (schema-level validation)
   - `tool` names must exist in `TOOL_REGISTRY`
   - slot targets must be **unqueried** IDs (re-validated against the live unqueried set)
   - one repair retry, then a deterministic repair pass, then a deterministic fallback record
5. **Batch execution** behind either:
   - `[Autonomous 6-Round Execution]`, or
   - `[Human-in-the-Loop Batch Approval Gate]` — the runner emits a `batch_pending_approval`
     event and awaits `approve()` / `reject()` before spending oracle budget.

Agents receive **only** revealed ledger rows and unqueried sequence features. Hidden binding
labels for unqueried candidates are structurally unreachable from the agent context builders.

---

## 9. API contract (`api_contract.json`) — exact shapes

`GET /api/state` returns:

```json
{
  "current_round": 3,
  "phase": "Self-Accelerating Payoff",
  "kpis": {
    "best_qualified_binding": 9.38,
    "delta_vs_wt": 0.53,
    "query_budget_spent": 20,
    "total_budget": 35,
    "wet_lab_cost_saved_usd": 24000,
    "hypotheses_supported": 2,
    "hypotheses_falsified": 1
  },
  "hcdr3_ribbon": [{"position": 1, "wt_residue": "S", "soft_prior_weight": 0.0, "status": "neutral"}],
  "agent_council": [{"role": "...", "tools_invoked": ["..."], "thinking_trace": "...", "memo": "..."}],
  "batch_slots": [{"slot_id": 1, "slot_type": "Dual-Surrogate UCB", "candidate_id": "...", "sequence": "...",
                   "mu_bind": 9.12, "sigma_bind": 0.45, "risk_score": 0.38, "risk_passed": true}],
  "pareto_data": [{"candidate_id": "WT", "risk": 0.45, "binding": 8.85, "status": "wild_type"}],
  "benchmark_curves": [{"query_count": 5, "m1_random": 7.2, "m3_ucb": 7.2, "m5_hybrid": 7.2}]
}
```

Contract rules enforced in code (`core/state.py` + `tests/test_api_contract.py`):

- Every field present; no missing keys; no `null` inside `hcdr3_ribbon` / `batch_slots`.
- `phase ∈ {"Cold-Start Tuition", "Self-Accelerating Payoff"}` (UI label form).
- `status ∈ {"neutral", "anchor_conserved", "low_risk_swap", "epistasis_alert"}`.
- `slot_type ∈ {"Dual-Surrogate UCB", "Max Joint Uncertainty", "Agent Hypothesis Test",
  "Skeptic Falsification Probe"}`; optional `hypothesis_id` / `falsification_rationale`.
- `hcdr3_ribbon` is a **sparse, curated** list (anchor / low-risk / epistasis positions + "neutral"
  filler) — not all 13 positions — matching the contract's sample.
- `pareto_data` statuses: `wild_type`, `queried_round_N`, `pareto_frontier`.
- Floats rounded to 2 dp for the wire; `kpis` integers stay integers.

### 9.1 SSE — `GET /api/stream-round`

`text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`, heartbeat comments.
Event types, in order:

1. `round_started` — `{round, phase, budget_spent, total_budget}`
2. `tool_call` — `{tool, role, args_summary, output, duration_ms}`
3. `agent_thinking` — `{role, token/chunk, tools_invoked}` (streamed traces)
4. `agent_memo` — `{role, memo, tools_invoked}`
5. `planner_synthesis` — the full `RoundDecisionRecord`
6. `batch_pending_approval` — only in HITL mode
7. `slot_selected` — one per slot
8. `oracle_reveal` — `{candidate_id, y_bind, y_dev, new_best}`
9. `round_complete` — `{state: <full contract JSON>, round_summary: {...}}`
10. `error` — `{message}`
11. `stream_end` — `{ok: true}`

The final `round_complete.data.state` is byte-identical to what `GET /api/state` would return.

### 9.2 Control endpoints

- `GET /api/health`
- `GET /api/config` — tau_dev provenance, seeds, budget, model routing, HITL mode
- `GET /api/ledger` — revealed rows + hash chain verification
- `GET /api/hypotheses` — hypothesis history with SUPPORTED/FALSIFIED badges
- `GET /api/candidates?...` — pool browsing (sequence + risk only, never hidden labels)
- `POST /api/round/run` — trigger one round (autonomous)
- `POST /api/round/approve` / `POST /api/round/reject` — HITL gate
- `POST /api/campaign/reset`
- `GET /api/benchmark` — cached M1–M5 curves

CORS: `allow_origins=["*"]`, all methods/headers, `expose_headers=["*"]` so localhost:3000/5173
work without a proxy.

---

## 10. Deliverable tree

```
config/            settings.py, frozen_protocol.json (generated)
data/              datasets.py, fixture.py, raw/ (optional real export)
core/              constants, features, developability, preregistration, ledger, oracle,
                   tool_arsenal, surrogate, schemas, prompts, acquisition, baselines,
                   llm, agents, virtual_lab_runner, campaign, state, errors
app/               server.py
scripts/           generate_benchmark_fixture.py, freeze_benchmark.py, run_benchmark.py,
                   run_campaign.py
tests/             unit + contract + async orchestration tests
results/           benchmark + ledger artifacts
```

## 11. Definition of done

- `pytest -q` green.
- `python scripts/run_benchmark.py --seeds 1 --methods m1,m2,m3,m4,m5` prints AUBQC per method and
  shows M5 ≥ M3 ≥ M2 ≥ M1 in expectation on the smoke seed.
- `uvicorn app.server:app` → `curl localhost:8000/api/state` validates against
  `api_contract.json` (key-by-key structural equality), and `curl -N localhost:8000/api/stream-round`
  emits the full event sequence ending in `round_complete` + `stream_end`.
- No file under `core/`, `data/`, or `app/` can read `y_bind` outside `BindingOracle`.
