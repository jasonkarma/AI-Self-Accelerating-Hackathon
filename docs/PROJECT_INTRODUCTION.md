# Project Introduction

**Self-Accelerating Antibody Lead Optimization** — a Hybrid Virtual Lab for the *Self-Accelerating AI
for Drug Development* Hackathon. Version 1.2.

---

## 1. What this is

A closed-loop, multi-agent **virtual lab** that designs the next batch of antibody variants to assay,
learns from the results, and does it again — under a hard budget of 35 binding measurements.

Each round, a council of four frontier-model agents inspects the revealed experimental ledger through
a set of deterministic analysis tools, argues about which substitutions to try, and hands five
specific candidates to an experimental planner. Those five are purchased from a **hidden binding
oracle**, appended to a tamper-evident ledger, and the loop repeats. Six rounds, five slots each, five
shared seed measurements.

The deliverable is not just the loop. It is the loop **plus the apparatus that proves it works**:
a frozen information boundary, a hash-chained ledger, a pre-registered qualification threshold, five
matched baseline methods, a pre-registered holdout target, and a web front-end that shows the
reasoning trace, the batch, the reveals and the benchmark curves live.

---

## 2. At a glance

| | |
| --- | --- |
| **Problem** | Sample-efficient, multi-objective antibody lead optimization under a fixed assay budget |
| **Task** | Retrospective, blinded replay of the 1N8Z / trastuzumab HCDR3 campaign |
| **Search space** | 419 fixed-length **L = 13** HCDR3 variants; wild type `SRWGGDGFYAMDY` |
| **Objectives** | `y_bind = -log10(Kd [M])` (hidden) and `y_dev = 1 - Risk(s)` (free at `t=0`) |
| **Budget** | 5 seeds + 6 rounds × 5 queries = **35** binding assays |
| **Method under test** | **M5** Hybrid Virtual Lab — dual surrogate + deterministic tools + bounded agent priors |
| **Baselines** | M1 Random · M2 Greedy dual-surrogate · M3 Dual-surrogate UCB · M4 UCB + fixed expert priors |
| **Headline result** | M5 ρ = **0.6578** vs M3 ρ = 0.3442 at q=35 (+0.3136, p = 8.4e-32, **98/100 seeds**) |
| **Stack** | Python 3.11+ / FastAPI backend · React + Vite + Tailwind frontend · SQLite ledger |
| **Models** | Gemini 4 Argon, Claude Opus 5.5, Claude Sonnet 5.5, GPT-6.1 Sol — one family per seat |

---

## 3. The scientific setup

**Candidate pool.** 419 HCDR3 variants of trastuzumab, all 13 residues, from the AbBiBench 1N8Z /
Absci SPR dataset. Objectives:

- `y_bind = -log10(Kd [M])` — measured HER2 binding, higher is tighter. **Hidden.**
- `y_dev = 1 - Risk(s)` — a frozen, sequence-only developability proxy in `[0, 1]`. Free at `t=0`.

`Risk(s)` is a weighted composite of six components, computed over the unlabelled pool and min-max
normalised: hydropathy mean, worst 5-residue hydrophobic window, aromatic fraction, net charge at
pH 7.4, adjacent same-sign charge clusters, and liability motifs (unpaired Cys, `NG`/`NS`
deamidation, `DG`/`DS` isomerization).

**The qualification gate.** `tau_dev` is frozen *before any binding label is read*, by a
pre-registered rule with two branches (wild-type anchor, or pool median fallback). Every one of the
five slots in every batch — in the live campaign *and* in all four baselines — is spent on a
developability-qualified candidate, because an unqualified pick can never improve the reported
metrics.

**Metrics.** AUBQC (area under the best-so-far binding curve over the seven 5-query checkpoints),
plus four learning-oriented metrics reported beside it: `normalized_aubqc` (fraction of available
headroom), `top5_panel_aubqc` (portfolio depth, not one lucky max), `spearman_rho` (scoring model vs
true binding over the qualified pool) and `top_decile_hit_rate`.

---

## 4. How it works

### 4.1 The five matched methods

All five start from the **same five candidate IDs** and the same 35-query budget, so the comparison
is matched rather than approximate.

| Method | Objective |
| --- | --- |
| **M1** Random Search | uniform over the qualified pool — the floor |
| **M2** Greedy Dual-Surrogate | `mu` only (`beta0 = 0`), no exploration term |
| **M3** Dual-Surrogate Active Learning | joint UCB over binding *and* developability — the strong non-agent baseline |
| **M4** M3 + Fixed Expert Loop Priors | static WT-anchor conservation and aromatic penalty — the textbook rule, included to show it fails under epistasis |
| **M5** Hybrid Virtual Lab | M3's surrogate **plus** bounded agent priors and a 5-slot portfolio |

M5 is deliberately *not* "M4 plus agents". The static expert rule stays with M4; M5 sits on the
unbiased surrogate. That choice is itself the experiment: a fixed textbook prior loses to plain UCB
here, while empirical-Bayes priors over surrogate **residuals** do not.

### 4.2 The five-slot batch as a paired experiment

```
A_base(s) = dual-surrogate UCB              # M3's objective; the H0 null hypothesis
A_H1(s)   = A_base(s) + Δ_agent(s)          # M5's hypothesis, |Δ_agent| <= 0.06
```

| Slot | Rule |
| --- | --- |
| 1–2 | Top 2 by `A_H1` — exploit the surrogate and the agents' vote |
| 3 | Max `A_H1 + 0.50 · VoI · sigma_joint` — optimistic UCB; the optimism bonus vanishes as SAR converges |
| 4 | Max `A_H1 + 0.50 · max(0, Δ_agent)` — the H1 promotion arm |
| 5 | Max `A_base` — the H0 control: exactly what M3 would query next |

Every batch therefore contains its own control. Slot 4 and Slot 5 are compared on **measured**
bindings, not on a cross-round median: control wins ⇒ `FALSIFIED`, H1 wins ⇒ `SUPPORTED`. Positions a
falsified rule actually exercised are **inverted** in later rounds, so the lab self-corrects.

### 4.3 The agent council

Four seats, each pinned to a different model family so the council cannot collapse into a
single-model echo chamber:

| Seat | Provider | Contribution |
| --- | --- | --- |
| Affinity Scientist | Google (Gemini 4 Argon) | which substitutions should improve binding |
| Developability Scientist | Anthropic (Claude Sonnet 5.5) | which substitutions introduce liabilities |
| Scientific Skeptic | Anthropic (Claude Opus 5.5) | attacks both memos; runs the falsification probe |
| Experimental Planner | OpenAI (GPT-6.1 Sol) | synthesises a validated 5-slot `RoundDecisionRecord` |

Orchestration per round: **tools first**, then Affinity and Developability in parallel, then the
Skeptic (who sees both memos), then the Planner. Output is schema-validated — priors clamped, tool
names checked against the registry, slot targets re-checked against the live unqueried set — with one
repair retry, a deterministic repair pass, and finally a deterministic fallback record. Missing
credentials degrade safely to a deterministic reasoner, and `scripts/check_council.py` exits non-zero
if any seat degraded, so a single-model run cannot be mistaken for a diverse live council.

### 4.4 The deterministic tool arsenal

Agents never do arithmetic on raw strings; they call tools and read exact tables:

1. `scan_positional_sar` — per position × residue: mean binding, `n`, effect vs WT, low-n rows suppressed
2. `detect_epistasis_clashes` — position pairs deviating > 1.5σ from additive single-mutant expectation
3. `profile_developability_liabilities` — hydrophobic windows, aromatic density, charge, liability motifs
4. `inspect_joint_uncertainty_hotspots` — highest `sigma_bind` / `sigma_dev` / `sigma_joint` regions
5. `audit_past_hypotheses` — stamps each past Slot 4/5 pick `SUPPORTED` / `FALSIFIED` / `INCONCLUSIVE`
6. `compute_cost_and_effort_saved` — cumulative wet-lab dollars and assay-days vs Random and Greedy

All are pure functions of the revealed ledger, so none can reach a hidden label.

### 4.5 The guardrails

| Guardrail | Implementation |
| --- | --- |
| Hidden binding labels | only via `BindingOracle.query()`, idempotent, budget-counted |
| Frozen threshold | `core/preregistration.py` reads the sequence column only; `config/frozen_protocol.json` is write-once |
| Tamper-evident history | SQLite append-only ledger, SHA-256 hash chain, triggers reject `UPDATE`/`DELETE` |
| Bounded agent influence | `Δ_agent` capped at `±0.06` in aggregate — can never filter a candidate or alter a metric |
| Reproducible demo | frozen Seed-21 golden ledger; restore in ~12 ms, rewind to any round, time-travel `GET /api/state?round=k` |
| Contract stability | `api_contract.json` auto-generated from live payloads; `--check` runs in the test suite |

---

## 5. What you can see in the UI

The React front-end is built against `api_contract.json` and has five views:

- **Overview** — KPIs (best qualified binding, Δ vs reference, budget spent, dollars saved, hypotheses
  supported/falsified), the HCDR3 ribbon with per-position soft priors, Pareto scatter of binding vs
  risk, and the benchmark curves.
- **Lab** — the live round: agent thinking traces, memos, the planner's decision record, the five
  slots, oracle reveals, and the streaming SSE round runner.
- **Benchmark** — M1–M5 curves, paired significance tests, and the zero-shot reference lines.
- **Cross-target** — per-target results, the learnability gate, and the published lockbox.
- **Attribution** — the ablation stack: what each component of M5 actually contributes.

Endpoints: `GET /api/state`, `/api/candidates`, `/api/benchmark`, `/api/cross-target`,
`/api/attribution`, `/api/ledger`, `/api/hypotheses`, `/api/config`, `/api/rounds`, and
`GET /api/stream-round` (server-sent events, eleven event types, ending in `round_complete` +
`stream_end`). Control: `POST /api/round/run`, `/api/round/approve`, `/api/round/reject`,
`/api/campaign/reset`, `/api/restore-golden`, `/api/rewind`.

---

## 6. Results at a glance

**1N8Z, 100 seeds.** M5 wins on rank correlation at q=35 and ties on AUBQC:

| Metric | M1 | M2 | M3 | M4 | **M5** |
| --- | --- | --- | --- | --- | --- |
| AUBQC ± SE | 1.9061 ± 0.0071 | 1.9240 ± 0.0060 | **1.9246 ± 0.0070** | 1.9120 ± 0.0079 | 1.9237 ± 0.0072 |
| Spearman ρ @ q=35 | n/a | 0.5081 | 0.3442 | 0.2991 | **0.6578** |

**Generalisation (M5 vs M3 on ρ, where a rank signal exists):**

| Target | L | Rounds | M5 ρ | M3 ρ | Δ | p |
| --- | --- | --- | --- | --- | --- | --- |
| `1n8z` | 13 | 6 | 0.6578 | 0.3442 | **+0.3136** | 8.4e-32 |
| `aayl49_ML` | 11 | 24 | 0.2981 | 0.1760 | **+0.1101** | 1.4e-06 |
| `2fjg` | 13 | 3 | 0.2327 | −0.0530 | **+0.2857** | 6.2e-06 |
| `aayl51` (lockbox) | 12 | 15 | 0.0234 | 0.0232 | −0.0036 | 0.66 |

`aayl51` is the pre-registered holdout and it is a **negative result** for the method, published as
one. Three further targets (`4d5_her2`, `aayl49`, `aayl51`) are marked *unlearnable* by a
baseline-only gate — no method correlates with truth there, so the benchmark has no power and a tie
says nothing about robustness.

**Three caveats that travel with every number:** (a) AUBQC's 100-seed standard error is ~0.007, the
same order as the gaps between methods, so it is reported beside ρ or not at all; (b) a ρ win is not
by itself evidence for the multi-agent hypothesis — where the agent fire rate is 0 % (e.g. `2fjg`),
M5's advantage comes from pool-level scoring, not from the council; (c) the bundled dataset is a
deterministic synthetic fixture until the real export is installed.

---

## 7. Repository map

```
data/       dataset loader + synthetic fixture generator + hidden oracles
core/       features, developability, preregistration, ledger, oracle, tool arsenal,
            dual surrogates, acquisition, 5 baselines, LLM routing, agents,
            virtual-lab runner, campaign, contract state
app/        FastAPI server (/api/*, SSE round stream)
frontend/   React + Vite + Tailwind, built against api_contract.json
scripts/    fixture generation, freeze, benchmark, campaign runner, cross-target,
            attribution ablation, lockbox audit, learnability gate, contract generator
tests/      unit, contract, async orchestration, stage-readiness
config/     settings + frozen_protocol.json (write-once)
results/    benchmark artifacts, golden ledger, lockbox metadata
```

---

## 8. Running it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_benchmark_fixture.py   # deterministic fixture (or fetch real data below)
python scripts/fetch_real_1n8z.py              # official AbBiBench 1N8Z export
pytest -q                                       # unit + contract + orchestration
python scripts/check_council.py                 # seat preflight: live vs deterministic
python scripts/run_benchmark.py --seeds 100     # the full race
python scripts/run_campaign.py --rounds 1       # one live agent round, streamed to the terminal
uvicorn app.server:app --reload --port 8000     # API + SSE
python scripts/generate_api_contract.py --check # fail if the contract is stale
```

---

## 9. Where to read next

| Document | Contents |
| --- | --- |
| [`docs/PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) | Why the problem is hard, what "solved" means, what is honestly not solved |
| [`SPEC.md`](../SPEC.md) | The frozen technical specification: features, surrogates, tools, batch rules, contract shapes |
| [`README.md`](../README.md) | Results tables, ablations, cross-target evidence, deployment |
| [`IMPLEMENTATION_DECISIONS.md`](../IMPLEMENTATION_DECISIONS.md) | 20 numbered decisions and the experiment that justified each |
| [`LOCKBOX_PREREGISTRATION.md`](../LOCKBOX_PREREGISTRATION.md) | The holdout protocol, written before any number existed for it |
