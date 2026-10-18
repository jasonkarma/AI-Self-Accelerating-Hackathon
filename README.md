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
`results/benchmark_10seeds.json`, showing this live run's `best_qualified_binding` and AUBQC next to
`M1`–`M4` **and** `M5 (10-seed)`, so the ten-seed benchmark M5 and the single live M5 are visible
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
A_null(s) = A_base(s) + loop_prior_score(s)      # shared with M4
A_H1(s)   = A_null(s) + Δ_agent(s)               # M5, |Δ_agent| <= 0.06
```

| Slot | Objective |
| --- | --- |
| 1–2 | top 2 by `A_H1` — surrogate exploitation plus the agents' vote |
| 3 | `A_null + (1 − spent/budget) · 0.50 · σ_joint` — optimistic UCB that anneals to pure exploitation |
| 4 | top `A_H1` among `Δ_agent > 0` — the H1 positive arm |
| 5 | top `A_null` among `Δ_agent <= 0` — the H0 null control |

Slots 4 and 5 form a paired control experiment inside every batch, so a hypothesis is graded by the
candidates measured against it rather than against a cross-round median. `Δ_agent` is the council's
summed positional priors plus an empirical-Bayes SAR term over surrogate **residuals**, shrunk by
`n / (n + 3)` with `n < 2` contributing nothing — noisy singletons are damped during cold start, and
confirmed multi-observation SAR steers later rounds without any round switch. A rule that loses its
paired test is inverted in later rounds, so the lab genuinely self-corrects.

### Pre-flight check

```bash
python scripts/check_council.py
```

Prints the pinned model, live/offline transport, effective base URL, and resolved thinking budget
for every seat, and exits non-zero if any seat fell back to the deterministic reasoner — so a
degraded single-model run can't be mistaken for a diverse live council.

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
