# Self-Accelerating Antibody Lead Optimization

Python implementation of the v1.2 Hybrid Virtual Lab. The bundled 419-row dataset is a clearly
marked deterministic synthetic fixture; replace it with the real AbBiBench export using
`python scripts/fetch_real_1n8z.py` before scientific evaluation.

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_benchmark_fixture.py
pytest -q
python scripts/run_benchmark.py --seeds 10
uvicorn app.server:app --reload --port 8000
```

Frontend clients on localhost:3000/5173 may call `GET /api/state`, `GET /api/candidates`,
`GET /api/benchmark`, and subscribe to `GET /api/stream-round`. CORS permits all origins.

### Stage operations: golden ledger, rewind, time travel

`results/golden_seed21.sqlite3` is a frozen copy of the verified Seed-21 run (6 rounds,
`8.580` best qualified binding, `1.963` AUBQC). It is **read-only on the stage path**:
`restore` and `rewind` copy *from* it and can never write to it.

| Endpoint | Does |
| --- | --- |
| `POST /api/restore-golden` | Copies the golden ledger over `results/campaign.sqlite3` and re-hydrates the campaign (~12 ms). |
| `POST /api/rewind` `{"round": k}` | Truncates the live ledger to the end of round `k` (`0` = seed batch only), restoring from golden first if the live file is shorter. `{"round": 5}` leaves `30/35` queries spent so Round 6 can be run live. |
| `GET /api/state?round=k` | The complete contract state *as of* round `k` — KPIs, VoI/`sar_maturity`, ribbon, declared priors, council memos, 5 slots, Pareto scatter, and deterministic `tool_outputs`. Read-only; never mutates the database. |
| `GET /api/rounds` | One compact entry per round `0..current` for the `[Seed] -> [R1] -> ... -> [R6]` stepper. |

Rewind is *structural*, not a read-time filter: the file on disk becomes the shorter one, and
`entries_no_delete` is honoured rather than defeated — the surviving prefix is rewritten into a
staging file and swapped in atomically, so the hash chain that results is the chain that already
verified. `GET /api/benchmark` additionally serves `zero_shot_references` (AF3 `-0.16`,
ProteinMPNN `-0.17`, ESM-2 `-0.13`) and `same_seed_comparison` (live M5 vs M1-M5 re-run on Seed
21's identical five initial queries, at `q=5` / `q=35` / AUBQC).

To re-freeze the golden file deliberately after a new verified run:

```bash
python scripts/promote_golden.py --expect-best 8.580 --expect-aubqc 1.963
```

It refuses to overwrite unless the hash chain verifies, the run is complete, the spend matches the
budget and the quoted numbers still hold.

The end-to-end staging contract is covered by `tests/test_stage_readiness.py`.

The full race is `python scripts/run_benchmark.py --seeds 100`; ExtraTrees use all CPU cores.
Set `M5_POLICY_MODE=live_agent` to transparently stamp live-agent M5 runs, or leave the default
`adaptive_tool`. Set `HITL=1` for the human approval gate.

### Cognitive-diversity council

Each agent role is pinned to a distinct model family (see `AGENT_MODELS` in `core/llm.py`) so the
council cannot collapse into a single-model echo chamber:

| Agent | Provider | Model | Thinking | `max_tokens` |
| --- | --- | --- | --- | --- |
| Affinity Scientist | Google | `gemini-4-argon` | provider-native reasoning (no thinking block) | 8192 |
| Developability Scientist | Anthropic | `claude-5.5-sonnet` | `budget_tokens=2048` | 4096 |
| Scientific Skeptic | Anthropic | `claude-5.5-opus` | `budget_tokens=4096` | 8192 |
| Experimental Planner | OpenAI | `gpt-6.1-sol` | `reasoning_effort=high` + `json_object` | 8192 |

Credentials are provider-scoped: `GOOGLE_API_KEY`/`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, and
`OPENAI_API_KEY`. Base URLs can be redirected to a gateway with `GEMINI_BASE_URL`, `ANTHROPIC_BASE_URL`
(e.g. an internal router), and `OPENAI_BASE_URL`; role-specific `AFFINITY_BASE_URL` /
`SKEPTIC_BASE_URL` style overrides win. Missing credentials safely select the deterministic reasoner,
and `ModelRouter().describe_roster()` reports which seats are live.

Anthropic extended thinking is opt-in per agent. Pass `thinking_budget=...` to `AnthropicReasoner`
to inject `{"type": "enabled", "budget_tokens": X}`; leave it `None` for default execution. Tune
without editing code via `SKEPTIC_THINKING_BUDGET=8192` / `DEVELOPABILITY_THINKING_BUDGET=off`, or
at runtime with `ModelRouter(thinking_budgets={"skeptic": 8192})` and `router.configure_thinking(...)`.
`thinking_configuration()` in `core/agents.py` prints the resolved per-agent budgets.
Seats also accept `<SLUG>_TIMEOUT` (seconds); any seat re-homed onto a reasoning model gets a
240s floor automatically, since reasoning tokens bill against the same wall clock.

`max_tokens` is a *completion ceiling*, not an answer length: reasoning-capable models bill their
hidden deliberation against the same budget, so a seat can spend the entire ceiling thinking and
return an empty or truncated memo (`finish_reason="length"`). The usable cap is a property of the
**gateway**, so each seat accepts `<SLUG>_MAX_TOKENS` — e.g. `AFFINITY_MAX_TOKENS=16384` — and
`describe_roster()` / the CLI preflight report the ceiling actually sent on the wire. Prefer raising
the ceiling over disabling reasoning: the deliberation is what produces the separation between
single-position effects and epistatic confounding.

### Running the campaign from the CLI

```bash
python scripts/run_campaign.py --rounds 1
```

Forces `M5_POLICY_MODE=live_agent` before `config.settings` is imported, executes the rounds for
real (instead of dumping t=0 state), and streams each scientist's name, thinking trace and memo to
the terminal as it finishes, then prints the validated `RoundDecisionRecord` JSON.
Flags: `--rounds N`, `--allow-offline`, `--verbose` (full tool payloads), `--no-color`.

The final `CAMPAIGN STATE` block ends with a `BASELINE COMPARISON · q=35` table read from
`results/benchmark_10seeds.json` — which, despite the filename, holds the **100-seed** run (the name
is historical: `run_benchmark.py` writes `benchmark_<n>seeds.json` and copies the result over this
path whenever `n` is at least the published seed count). Label it from the artifact's own `seeds`
field, never from the filename; `/api/benchmark` publishes this as `source.label` for exactly that
reason. The table shows this live run's `best_qualified_binding` and AUBQC next to
`M1`–`M4` **and** `M5 (100-seed)`, so the benchmark M5 and the single live M5 are visible
side by side. `delta vs WT` is measured against the measured WT when the pool contains it, otherwise
against the frozen q=5 seed baseline, so it moves with `best_qualified_binding` instead of staying at
`0.0`. Each round's hypothesis is graded as a paired H1-vs-H0 experiment (Slot 4 against Slot 5) and
printed with its verdict, and the campaign can be resumed from an existing ledger without losing
those counters — delete `results/campaign.sqlite3` for a clean slate.

### How the batch is designed

Every slot is filled from the developability-qualified pool (`y_dev >= tau_dev`), because an
unqualified candidate can never move `best_qualified_binding`. Two acquisition scores decide the
batch:

```
A_base(s) = surrogate UCB                        # M3's objective = the H0 null hypothesis
A_H1(s)   = A_base(s) + Δ_agent(s)               # M5, |Δ_agent| <= 0.06
```

`A_null` is `A_base` itself. The static WT anchor rule (`loop_prior_score`) is **M4's** objective
only: on 1N8Z the best binders start `AR...` and carry 4–5 aromatics, which that rule penalises, so
M4 trails plain UCB and M5 must not inherit the bias.

> **`A_null := A_base` is the production law (commit `e567cd4`).** The VoI anneal that used to move
> `A_null` toward the surrogate prediction was retired, and the `no_anneal` ablation is the evidence:
> it *tied* on AUBQC (p ≈ 0.94 / 0.81) while **losing ρ significantly** (−0.043, p = 2.0e-4 at
> 16.7 % coverage; −0.075, p = 2.4e-16 at 33 %). `exploit_weight` / `exploration_weight` are still
> exported by `/api/state`, but they are **diagnostics of what the anneal would have done**, not
> coefficients of the shipped objective. The payload says so explicitly via
> `voi_diagnostics.steers_null_objective: false` — do not label them as steering `a_null`.

| Slot | Objective |
| --- | --- |
| 1–2 | top 2 by `A_H1` — surrogate exploitation plus the agents' vote |
| 3 | `A_H1 + 0.50 · VoI · σ_joint` — optimistic UCB; the optimism bonus is withdrawn once `slot_graduation` fires |
| 4 | `A_H1 + 0.50 · max(0, Δ_agent)` — the H1 promotion arm |
| 5 | top `A_base` — the H0 null-baseline control (what M3 would query next) |

Slots 4 and 5 form a paired control experiment inside every batch, so a hypothesis is graded by the
candidates measured against it rather than against a cross-round median. The control is the
*equality* hypothesis `Δ_agent == 0`, not its opposite: restricting slot 5 to `Δ_agent <= 0` made the
lab boycott its own proven substitutions for a fifth of every early batch. When slots 1–4 promote a
hypothesis candidate over `A_base`, slot 5 automatically tests the displaced baseline candidate.

`Δ_agent` is the council's summed positional priors plus an empirical-Bayes SAR term over surrogate
**residuals**, shrunk by `n / (n + 3)` with `n < 2` contributing nothing — noisy singletons are damped
during cold start, and confirmed multi-observation SAR steers later rounds without any round switch. A
rule that loses its paired test is inverted in later rounds, so the lab genuinely self-corrects.

### Benchmark metrics

`python scripts/run_benchmark.py --seeds 100` reports four metrics beside AUBQC:

| Metric | What it measures |
| --- | --- |
| `normalized_aubqc` | AUBQC with `y*(q)` rescaled 0.0 (seed score) → 1.0 (pool ceiling), i.e. fraction of available headroom |
| `top5_panel_aubqc` | mean binding of the best five qualified leads found so far — portfolio depth, not one lucky max |
| `spearman_rho` | rank correlation between the method's scoring model and true binding over the qualified pool |
| `top_decile_hit_rate` | fraction of the 30 post-seed queries that land in the pool's top decile |

The Spearman column is the one comparable to zero-shot affinity predictors on 1N8Z (AF3 `-0.16`,
ProteinMPNN `-0.17`, ESM-2 `-0.13`); it is reported as `n/a` for M1, which has no scoring model at all.

### Results: 1N8Z primary benchmark (100 seeds)

From `results/benchmark_100seeds.json`. **On AUBQC and ρ the story is different, and both are true:**

| Metric | M1 | M2 | M3 | M4 | **M5** |
| --- | --- | --- | --- | --- | --- |
| AUBQC ± SE | 1.9061 ± 0.0071 | 1.9240 ± 0.0060 | **1.9246 ± 0.0070** | 1.9120 ± 0.0079 | 1.9237 ± 0.0072 |
| Spearman ρ @ q=35 | n/a | 0.5081 | 0.3442 | 0.2991 | **0.6578** |

M5 beats every baseline on **rank correlation** at q=35 (+0.3136 vs M3, p = 8.4e-32, **98/100
seeds**), which is the axis comparable to the zero-shot predictors. It does **not** beat them on
AUBQC: −0.0009 vs M3 (p = 0.77), and it is *behind* M2 on `top5_panel_aubqc` (−0.0188, p = 4.2e-7)
and `top_decile_hit_rate` (−0.0680, p = 2.9e-11). The 100-seed SE on AUBQC is ~0.007, the same order
as every M3/M4/M5 gap — so a bare AUBQC mean invites reading noise as a win. Report ρ and AUBQC
together or not at all.

### Results: cross-target + pre-registered lockbox (`aayl51`, L=12)

`python scripts/run_cross_target_benchmark.py --dataset aayl51 --seeds 50 --merge`
(stored in `results/cross_target_benchmark.json`; `--merge` appends without touching published cells).

| Target | L | N_qual | Coverage | R | Q | M5 AUBQC ± SE | Best baseline | M5 rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `aayl49_ML` | 11 | 756 | 16.53 % | 24 | 125 | 2.3109 ± 0.0126 | 2.3254 (M3) | 3/5 |
| `2fjg` | 13 | 124 | 16.13 % | 3 | 20 | 0.0782 ± 0.0029 | 0.0799 (M4) | 2/5 |
| **`aayl51`** | **12** | **487** | 16.43 % | 15 | 80 | **1.9726 ± 0.0060** | 1.9774 (M4) | 3/5 |

**The lockbox is a negative result for M5 on AUBQC**, and it is published as one. It was selected by
`scripts/audit_lockbox_metadata.py` from all 17 AbBiBench targets **before** any M1–M5 number existed
for it (N_qual 486, L=12, 0 % sequence overlap with any scored target) — see
`LOCKBOX_PREREGISTRATION.md` and `results/lockbox_metadata.json`. M5 ranks 3/5 at every coverage
level on `aayl51` and never leads ρ. The 16 `L=14` targets are unusable: `3gbn_*`, `4fqi_*` collapse
to **2** distinct CDR3s and `5a12_*`, `aayl50_LC`, `aayl52_LC`, `g6_LC` to **1**.

AUBQC is **not** comparable across these rows: the protocol sizes itself to the qualified pool, so
`2fjg` gets 20 queries and `aayl49_ML` 125. Compare rank-vs-baselines *within* a row.

### Results: exhaustive 17-of-17 AbBiBench partition

Two blind stages selected the holdouts; neither used an M5 number. Stage A
(`scripts/audit_lockbox_metadata.py`) is metadata-only. Stage B
(`scripts/baseline_learnability_gate.py`) runs **only M2/M3** and gates on
`max(M2_rho, M3_rho) >= 0.15`.

| Class | Targets | Evidence |
| --- | --- | --- |
| **Learnable, multi-round — M5 leads ρ** | `1n8z` (L=13) · `aayl49_ML` (L=11) · `2fjg` (L=13) | M5 > M3 on ρ at p ≤ 2e-4 in **3 of 3** |
| **Unlearnable — no rank signal** | `aayl51` (L=12) · `4d5_her2` (L=13) · `aayl49` (L=11) | baselines 0.006–0.045 |
| **Single-round boundary** | `1mlc` (L=9, **R=1**, Q=10) | baselines learn (0.35–0.40) but only 1 round |
| **Degenerate (1–2 distinct CDRH3)** | `1mhp_LC`, `3gbn_h1`, `3gbn_h9`, `4fqi_h1`, `4fqi_h3`, `5a12_ang2`, `5a12_vegf`, `aayl50_LC`, `aayl52_LC`, `g6_LC` | not search problems |

**M5 vs M3 on Spearman ρ (the generalisation claim):**

| Target | L | rounds | M5 ρ | M3 ρ | Δ | p | seeds won |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1n8z` | 13 | 6 | 0.6578 | 0.3442 | **+0.3136** | **8.4e-32** | 98/100 |
| `aayl49_ML` | 11 | 24 | 0.2981 | 0.1760 | **+0.1101** | **1.4e-06** | 39/50 |
| `2fjg` | 13 | 3 | 0.2327 | −0.0530 | **+0.2857** | **6.2e-06** | — |
| `aayl51` | 12 | 15 | 0.0234 | 0.0232 | −0.0036 | 0.66 | 23/50 |
| `1mlc` | 9 | **1** | 0.3814 | 0.3541 | +0.0273 | — | — |

**Gate revision (`aayl49_ML`, `aayl51` re-measured).** `core/developability.py`'s
5-residue hydropathy window previously scanned `range(9)`, which is only correct at
13 residues; it is now `range(len(s) - 4)`. `2fjg` and `1mlc` are **bit-identical**
after the fix, which is the check that it was scoped correctly. The two affected
targets were re-run at 50 seeds per coverage (`--merge`, published cells replaced):

| Target | τ_dev (old → new) | N_qual | M5 ρ (old → new) | M5−M3 Δρ (old → new, p) |
| --- | --- | --- | --- | --- |
| `aayl49_ML` | 0.69764 → 0.69572 | 756 → 756 | 0.2694 → **0.2981** | +0.0785 (4.5e-05) → **+0.1101 (1.4e-06)** |
| `aayl51` | 0.67212 → 0.66208 | 486 → **487** | 0.0098 → 0.0234 | −0.0014 (0.68) → −0.0036 (0.66) |

`N_qual` is pinned to 50 % of the pool by `pool_median`, so it barely moves; the
`aayl51` 486 → 487 shift is a median tie-break, not a gate change. What does move is
*which* sequences qualify (`threshold_source` stays `pool_median` in both). The
headline conclusion is unchanged: `aayl51` remains a **null** for M5 (Δρ = +0.0002,
p = 0.96), and `aayl49_ML`'s lead is marginally larger, not smaller.

**Read this as:** M5's rank model beats standard UCB wherever a rank signal exists, and
neither wins nor loses where none does. Two caveats travel with it:

- **`1mlc` is a boundary control, not evidence.** At N_qual=67 the protocol clamps to
  `R=1, Q=10`, and M5's AUBQC is *bit-identical* to M3's (paired Δ = +0.0000, p = 1.0)
  because one 5-slot batch is the same set for both methods. Structural, not empirical.
- **A tie on `aayl51`/`4d5_her2`/`aayl49` is not a safety claim.** No method correlates
  with truth there, so the benchmark has **no power** on those targets — not proof that
  M5 is robust.

Raw gate output: `results/learnability_gate.json`, also served by
`/api/cross-target` as `learnability_gate`.

### When does the agent layer actually activate?

`Δ_agent` is not always non-zero, and the condition is measurable **before** any
M1–M5 number exists. `positional_priors_from_revealed` declares a position only when

```
mode( best-5 revealed residue at p )  !=  reference[p]
```

and `reference` is the pool's own per-position mode. So a pool that is
"wild-type plus sparse independent mutations" makes the rule a **no-op**: the best
molecules' modal residue *is* the reference at every position, nothing is declared,
`Δ_agent ≡ 0`, and `A_H1 ≡ A_base`. M5 then selects the same five slots as M3.

`scripts/probe_delta_agent_conditions.py` measures the fire rate directly (uncapped,
best-5 of the reveal, 40 draws per size); `scripts/probe_candidate_pool.py` applies the
same test to any candidate CSV. 1N8Z is the positive control and confirms the probe:

| Target | L | N_qual | R | mean subst/variant | **agent fire rate @reveal=5** |
| --- | --- | --- | --- | --- | --- |
| **`1n8z`** | 13 | 210 | 6 | **5.71** / 13 | **100 %** |
| `aayl49_ML` | 11 | 756 | 24 | — | 88 % |
| `adams2017` (FLAb2, L=9) | 9 | 431 | 13 | 1.89 / 9 | 12 % |
| **`2fjg`** | 13 | 124 | 3 | **1.00** / 13 | **0 %** |
| `koenig2017 g6` (FLAb2) | 13 | 124 | 3 | 1.00 / 13 | 0 % |
| `makowski2022` (emibetuzumab) | 8 | 37 / 21 | **1** | 2.24 / 8 | 98 % / 80 % |

**Read this as a boundary condition, not a set of failures.** The agent layer is
conditional on library *density*, and density and pool size are in tension by
construction: combinatorial libraries enumerate a few CDR positions (dense, but capped at
a few hundred distinct sequences → `R` collapses to 1), while whole-loop scans stay near
the modal residue (large, but `Δ_agent = 0`). 1N8Z is the narrow overlap — combinatorial
*and* deep. `2fjg`'s published **+0.2857** ρ lead survives at a 0 % fire rate: with
`Δ_agent ≈ 0`, `A_H1 ≡ A_base`, so M5 can only differ from M3 through the surrogate and
the VoI/control slot rules. That is the sharpest available reminder that a ρ win is not
by itself evidence about the multi-agent hypothesis.

Report the fire rate beside ρ for any new target. Without it, an inert-agent cell and a
working-agent cell are indistinguishable in the table.

### Results: `adams2017` (FLAb2) — exploratory, **not** confirmatory

`python scripts/run_adams2017_ablation.py --seeds 50`. L=9, N_qual=431, R=13 at the
pre-registered 16.7 % rate. Chosen because it is the only real non-13-mer pool in FLAb2
with genuine positional diversity **and** multi-round sizing. It is **not** a holdout —
the sole pre-registered lockbox (`aayl51`) is consumed, so nothing here may be described
as confirmatory generalisation.

| coverage | R | Q | M2 ρ | M3 ρ | M4 ρ | **M5 ρ** | M5 rank | M5−M3 ρ (p, seeds won) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 % | 3 | 20 | 0.2702 | 0.2193 | 0.2677 | 0.2581 | 3/4 | n/a (Q < 35) |
| **16.7 %** | **13** | **70** | 0.4041 | 0.3721 | 0.3822 | **0.4215** | **1/4** | +0.0285 (0.057, 36/50) |
| 33 % | 27 | 140 | 0.3037 | 0.2723 | 0.2509 | **0.3317** | **1/4** | +0.0251 (0.146, 37/50) |

M5 leads ρ in both multi-round cells, directionally consistent (36/50 and 37/50 seeds),
but **neither reaches p < 0.05**. Three things must travel with that number:

- **It is not an agent result.** Fire rate is 12 %, so `Δ_agent ≈ 0` in most rounds and
  M5's AUBQC is nearly bit-identical to M3's (1.7473 vs 1.7464 at 16.7 %). The ρ
  difference comes from pool-level scoring, not from the council's hypotheses.
- **The label needed both a sign and a scale fix.** `fitness` is the *raw* Kd in molar
  (measured: Spearman +1.0000 against the `Kd (Tite-Seq) [M]` column), so it is
  higher-is-worse. Negating it linearly is dimensionally right and numerically useless —
  Kd spans 3.2e-10 … 1.0e-5 M, so `-Kd` is ~1e-5 and AUBQC underflows to `0.0000`. The
  transform used is `-log10(Kd)`; the first, linear attempt returned five tied methods.
- **The pool is partly saturated at the assay ceiling.** 4 of 5 seed labels are exactly
  5.000 (= Kd 1e-5 M); >50 % of the `Kd` column sits at that detection limit, which caps
  the resolvable dynamic range.

Falsification control: flipping the direction moves ρ from +0.3637 to −0.3787, so the
direction handling is load-bearing rather than cosmetic.

### API contract

```bash
python scripts/generate_api_contract.py          # regenerate from live payloads
python scripts/generate_api_contract.py --check  # fail if it is stale
```

`api_contract.json` is auto-generated from `TestClient` payloads for `/api/state`, `/api/benchmark`,
`/api/cross-target` and `/api/attribution`. Do not hand-edit it; `--check` runs in the test suite, so
a stale contract fails CI instead of being discovered by whoever builds the frontend.

### Pre-flight check

```bash
python scripts/check_council.py
```

Prints the pinned model, live/offline transport, effective base URL, and resolved thinking budget
for every seat, and exits non-zero if any seat fell back to the deterministic reasoner — so a
degraded single-model run can't be mistaken for a diverse live council.

## Deploy: Render (backend) + Vercel (frontend)

The two halves are independent. `render.yaml` at the repo root describes the Python
service; `frontend/vercel.json` describes the static bundle. Deploy the backend first — the
frontend bakes its URL in at build time.

### 1. Backend on Render

```bash
# Dashboard → New → Blueprint → select this repo (render.yaml is detected)
```

The blueprint pins `uvicorn app.server:app --host 0.0.0.0 --port $PORT --workers 1`, health
checks `/api/health`, and prompts for `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY`
without storing them in the repo. With no keys the service still boots and serves the frozen
Seed-21 campaign — every council seat falls back to the deterministic reasoner — so a deploy
can be verified before any credential is added.

`--workers 1` is required, not a default. The live campaign is an in-process `Campaign`
object appending to `results/campaign.sqlite3`; two workers would hold two divergent
campaigns and interleave writes into one ledger.

Two free-tier behaviours matter for a live demo:

| behaviour | consequence |
| --- | --- |
| the instance sleeps after ~15 min idle | the first request takes ~30–60 s to wake; open `/api/health` a minute before presenting |
| the filesystem is ephemeral — a deploy or restart resets it | live rounds are lost on redeploy. `POST /api/restore-golden` reinstalls the verified run in ~12 ms, which is exactly what that endpoint is for |

`GET /api/stream-round` sends `X-Accel-Buffering: no` and returns `text/event-stream`, which
Render proxies without buffering. A live round calls four frontier models, so allow several
minutes; the stream keeps the connection open for the duration.

### 2. Frontend on Vercel

```bash
# Dashboard → Add New → Project → import the repo
#   Root Directory : frontend
#   Framework      : Vite (prefilled from vercel.json)
#   Env var        : VITE_API_BASE = https://virtual-lab-api.onrender.com
```

`VITE_API_BASE` is substituted by Vite at build time, so **changing it requires a redeploy**,
not just a settings edit. It must be the backend's HTTPS origin with no path and no trailing
slash; `frontend/src/lib/api.ts` strips a trailing slash defensively.

The bundle then calls the backend directly across origins, which is safe here because
`app/server.py` installs `allow_origins=["*"]`. If you later want a single origin instead, the
Railway path in `frontend/README.md` serves the same bundle behind nginx with an `/api/` proxy.

`vercel.json` caches `/assets/*` for a year (hashed filenames) and marks `index.html`
`no-store`, so a redeploy can never serve a stale bundle hash. Its rewrite sends every other
path to `index.html` for the SPA fallback.

### Verifying a deploy

```bash
curl -s https://<backend>.onrender.com/api/health          # {"status":"ok","version":"1.2"}
curl -s https://<frontend>.vercel.app | grep -c 'assets/index-'   # 1 → bundle is wired
```

Then open the Vercel URL; the provenance pill reads `GET /api/ledger`. If it says it cannot
reach the backend, `VITE_API_BASE` was empty at build time or still points at localhost.

## Install the official AbBiBench 1N8Z table

```bash
python scripts/fetch_real_1n8z.py
cat config/frozen_protocol.json
```

The fetcher uses direct Hugging Face Hub CSV downloads and therefore avoids `datasets`/`dill`
compatibility problems on Python 3.14. It validates and stages all files before an atomic swap.
The official table contains 419 measurements over 416 unique extracted HCDR3 sequences and no
measured `SRWGGDGFYAMDY` row; duplicates are preserved as experimental replicates, while WT remains
the label-blind developability threshold reference. Successful installation must report
`"is_synthetic": false`. When the dataset hash changes, prior campaign/benchmark artifacts are
archived with a `pre-real-<hash>` suffix to prevent mixed-provenance results.
The frozen protocol reports both `wt_pass_fraction` (the pre-fallback WT diagnostic) and
`qualification_pass_rate` / `qualified_count` (the final frozen threshold outcome). A median
fallback records `threshold_percentile: 0.5`; ties can make the realized pass rate slightly above
or below exactly 50%.

2026 Tzu-Shun Chou. All Rights Reserved. Proprietary and confidential.