# Virtual Lab · Antibody CDRH3 Console

A React + TypeScript front end for the Self-Accelerating Multi-Agent Virtual Lab. It reads the
FastAPI backend (`app/server.py`) and nothing else: every number, chart, sequence, p-value and
badge on screen comes from `/api/state`, `/api/benchmark`, `/api/cross-target` or
`/api/attribution`, plus the control routes `/api/rounds`, `/api/config` and `/api/ledger`.

**This directory is additive.** No file under `core/`, `data/`, `results/`, `config/`, `scripts/`,
`tests/`, `app/`, no `*.sqlite3`, and not `api_contract.json` is read, written or modified by the
frontend or by its checks. The two ledgers were verified byte-identical after the work
(`sha256 49fea618572b3836…`).

---

## Quickstart

```bash
# 1. backend (repo root)
. .venv/bin/activate
uvicorn app.server:app --port 8000

# 2. frontend (this directory)
npm install
npm run dev            # http://localhost:5173 — /api is proxied to 127.0.0.1:8000
```

Production build:

```bash
npm run build          # tsc --noEmit && vite build  →  dist/
```

### What the four views are

| tab | subtitle | what it answers |
|---|---|---|
| `01 Live Virtual Lab` | Interactive Replay | what was discovered, in which slot, and what the three scientists argued |
| `02 100-Seed Benchmark` | Foundation vs. Loop Showdown | does the loop out-rank the zero-shot foundation models on 100 seeds |
| `03 Multi-Target Generalization` | AbBiBench Audit | does the same loop hold up on targets it was never tuned on |
| `04 Ablations & Controls` | Mechanism Attribution | which component is actually doing the work, and which deletions do nothing |

The header carries exactly one piece of technical metadata: a
`ⓘ Provenance & Verification` pill in the top-right corner. Opening it shows the ledger hash-chain
verdict, the golden-ledger SHA-256, the backend suite result, the objective law, the API version and
the frozen protocol's `τ_dev`/pool/qualified counts. Nothing else in the chrome names an endpoint.

Above-the-fold on View 1, in order: a hero band (the champion binder and five KPIs), the round
stepper (`Seed` → `R6`, each chip showing that round's best qualified binding, with `★ R6` in gold
once the harvest phase holds the champion), then a 60/40 split — five-slot plate on the left,
tabbed scientist council plus falsification tracker and loop telemetry on the right — and only then
the two discovery charts, side by side. Methodology prose is collapsed into `ⓘ` hover cards and
single-line `Note` disclosures; the plate is the first thing a reader sees, not the third.

`dist/` is a static bundle. Serve it from any origin and point it at the API with
`VITE_API_BASE=https://host:8000 npm run build` — the backend already sends
`Access-Control-Allow-Origin: *`, so no proxy is required. Without that variable the app uses
relative `/api/...`, which is what the dev proxy and a same-origin reverse proxy expect.

---

## Deploy on Railway

The frontend is a static bundle behind nginx, and it **proxies** `/api/*` to the backend rather
than baking an API URL into the build. That means the browser only ever talks to one origin: no
CORS preflight, no mixed content, and the backend's URL can change without rebuilding the image.

`Dockerfile`, `railway.json` and `nginx.conf.template` are in this directory. To deploy:

```bash
# 1. backend (repo root) — deploy it first so you have its public URL
railway up            # or connect the repo with Root Directory = /

# 2. frontend — create a second service from the same repo
railway up            # with Root Directory = frontend
```

Then set one variable on the **frontend** service:

| variable | value | notes |
|---|---|---|
| `API_ORIGIN` | `https://<backend>.up.railway.app` | **required**; the container refuses to start without it, and says so in the logs |
| `PORT` | `8080` | usually injected by Railway; the image defaults to it |

`nginx.conf.template` is rendered by the image's own `envsubst` bootstrap, so `API_ORIGIN` and
`PORT` are substituted at container start — nothing is compiled in. The `/api/` location disables
`proxy_buffering` so `GET /api/stream-round` keeps streaming, and `/assets/` is cached for a year
while `index.html` is never cached, so a redeploy cannot serve a stale bundle hash.

Build details worth knowing:

* The image builds in two stages on `node:22-alpine` and runs `npm ci`, so esbuild and rollup
  resolve for Linux. It never copies the working tree's `node_modules`.
* `npm run build` is `tsc --noEmit && vite build`, so a type error fails the deploy.

### If the backend lives on another host

Set `VITE_API_BASE` instead and skip the proxy: `VITE_API_BASE=https://api.example.com npm run build`
makes the bundle call that origin directly. The backend sends `Access-Control-Allow-Origin: *`, so
this works — the proxy is just the tidier default.

---

## Verification

Six checks, all runnable without a browser and none of which touch the backend. `npm run check`
runs them in order.

| command | what it proves |
|---|---|
| `npm run typecheck` | `tsc --noEmit`, `strict`, no `any`, no unused symbols |
| `npm run contract` | every path the UI dereferences still exists in `../api_contract.json`, with the expected JSON type (145 paths, 13 ablation variants, 12 cross-target rows) |
| `npm run verify` | **541 assertions** that recompute each derived figure from the raw contract and compare it with what the views render — Δρ, ΔAUBQC, ρ ratios, paired p-values, seeds won, funnel counts, coverage ranks |
| `npm run audit:literals` | collects every result value the contract publishes (956 of them) and fails if any appears as a literal under `src/`. A hard-coded `0.6578` cannot survive a re-run |
| `npm run audit:i18n` | **514 assertions** that every CJK council string in the ledger has an English rendering, and that every position index, candidate id, sequence, number and contract field name in the original survives into it |
| `npm run audit:render` | renders `Header` plus all four views with React's server renderer against the real payloads, **for every round of the ledger**, and asserts the markup has no `undefined`/`NaN`, contains the expected figures, and **contains zero CJK characters in View 1** |

Two useful extras:

```bash
npm run i18n:extract   # re-derive src/utils/council-zh-source.json from the read-only ledger
npm run preview        # serve the built bundle on :4173
```

---

## Layout

```
src/
  main.tsx, App.tsx           shell, data loading, tab state
  lib/
    api.ts                    typed fetch + SSE URL, no `any`
    derive.ts                 every derived figure (slot roles, partitions, deltas, ranks)
    format.ts                 number / p-value / method formatting — never invents precision
    pinned.ts                 the three repository facts no endpoint serves
  types/
    state.ts, report.ts       wire types transcribed from api_contract.json
  utils/
    normalizeCouncilText.ts   CJK → English at the presentation boundary
    councilTranslations.ts    the seven hand-authored renderings
    council-zh-source.json    the raw ledger strings, for the "Raw ledger" toggle
    councilHash.ts            FNV-1a key shared by the extractor and the app
  components/                 Panel, Header, charts, SequenceAlignment, Markdown, RoundTimeline
  views/
    lab/                      View 1 — plate, council, telemetry, discovery, live runner
    benchmark/                View 2 — ρ showdown, AUBQC, metric table, same-seed
    cross/                    View 3 — 17-of-17 funnel, partition, coverage matrix
    attribution/              View 4 — causal winner, negative controls, ledger
scripts/
  check-contract.ts           contract shape guard
  verify-numbers.ts           every derived figure, recomputed
  audit-literals.ts           no result value typed into src/
  audit-council-i18n.ts       CJK coverage + token fidelity
  render-smoke.tsx            server-render every view, every round; zero CJK in View 1
  extract-council-zh.ts       re-derive council-zh-source.json from the ledger
```

---

## Two places where the payload and the prose disagree

Both were found while building, and both are enforced by `npm run verify` so they cannot regress.

**`cross_target.targets[].paired` is an AUBQC test, not a ρ test.** The only paired *ρ* test for a
cross-target cell is in `/api/attribution`. View 3's learnable-class table therefore reads its
ρ p-values from the attribution ledger and shows an em dash where no such test exists, rather than
printing an AUBQC p-value under a ρ heading.

**The attribution mechanism rows use a different baseline.** `no_anneal`, `greedy_slots` and the
slot-role family carry `_baseline`, the pre-`e567cd4` AUBQC (which still contained the anneal), and
their published delta is `_baseline − variant`. Read raw, the sign looks like production losing to
its own ablation. `derive.ts` normalises to `variant − baseline` and labels each row's basis, so the
causal panel states the actual finding: removing the anneal **improves** ρ (`+0.0429`, `p = 2.0e-4`;
`+0.0748`, `p = 2.4e-16`) while leaving AUBQC neutral.

A third, smaller one: `1n8z` is not in `targets[]` (it is `reference_targets[0]`, measured by the
primary 35-query protocol, with its ρ test in `/api/benchmark`), and the HER2 control arrives from
the audit under the CSV-derived id `4d5_her2_benchmarking_data_trimmed`. `canonicalTargetId()` joins
those across payloads.

---

## Council language handling

The recorded Seed-21 campaign used live frontier models. Two Anthropic seats — Developability
Scientist and Scientific Skeptic — answered in Chinese on rounds 1, 2, 3, 4 and 6. `GET /api/state`
serves that prose verbatim and the ledger is frozen, so the translation lives in
`src/utils/normalizeCouncilText.ts`, keyed by an FNV-1a hash of the raw string.

Each of the seven renderings follows four rules, all checked by `npm run audit:i18n`:

1. No claim is added and no number is dropped: every candidate id, `HYP-` id, position index
   (`W3`, `G6`, `Y10`), 13-mer sequence, numeric reading and contract field name present in the
   Chinese appears in the English.
2. Where the ledger is truncated, the rendering is truncated at the same point and marked. Three of
   the seven were cut mid-sentence by a completion ceiling; the missing clause is **not** inferred.
3. Structure is preserved — same headings, same tables, same bullet order.
4. Repository terminology (`max_5mer_hydropathy`, `decisive`, `SUPPORTED`) is kept verbatim.

Each translated card carries a small `EN` badge and a collapsed **Raw ledger** disclosure that
renders the original string. The disclosure is not mounted until opened, so neither the page nor the
markup a screenshot captures contains Chinese.

`normalizeCouncilText` also handles a memo the table has never seen — a live round, a future
artifact. Structural tokens are carved out and copied verbatim, surrounding prose goes through a
glossary, and any fragment still containing CJK is replaced with an explicit marker. The result is
always English; it is never a guess. `audit:i18n` exercises that path with a synthetic Chinese memo.

One thing worth knowing when demoing: Round 5's Developability and Skeptic seats recorded an **empty**
memo and an empty thinking trace. The card says so rather than showing a blank box — the batch was
still designed, by the deterministic fallback planner.

### The council is three seats, not four

`/api/state.agent_council` and the ledger's `round_council` payloads both record **three** seats:
Affinity Scientist, Developability Scientist, Scientific Skeptic. The Experimental Planner is the
*consumer* of those memos — it synthesises the `RoundDecisionRecord` — and its reasoning is not
persisted as a memo, so there is nothing for the console to render for it. The panel header reads
`{seats.length}-seat` from the payload rather than asserting a count, so a run that seats a planner
will grow the label on its own.

Seat → provider is pinned in `core/llm.py` and printed verbatim from `provider`/`model`:
Gemini (`gemini-4-argon`), Claude Sonnet (`claude-sonnet-5-5`), Claude Opus (`claude-opus-5-5`).

---

## What could not be verified in the development sandbox

The work above was built and checked in a sandbox with **no network egress** (`npm install` and
`pypi` both return `403 blocked-by-allowlist`). `frontend/node_modules` happened to be present, so
most of the toolchain could run; two things could not, and both need your machine:

| not run | why | command |
|---|---|---|
| `vite build` | `node_modules/@rollup/rollup-darwin-arm64` is the macOS binary; the sandbox is Linux/arm64 | `npm install && npm run build` |
| `pytest` (backend) | the repo `.venv` is macOS Python 3.14 (`/opt/homebrew/...`), not executable on Linux | `pytest -q` |

Everything else in the verification table ran here against the real payloads. `vite build`'s
TypeScript half is already covered by `npm run typecheck`, and the bundler half by an `esbuild`
bundle of `src/main.tsx` (1.64 MB, no unresolved imports) — so a `vite build` failure would point at
the Vite config rather than at application code.

The backend was not touched: `git status` shows zero modified tracked files under `core/`, `data/`,
`config/`, `scripts/`, `tests/`, `app/` or `api_contract.json`, and both ledgers still hash to
`49fea618572b3836…`.

`frontend/.trash/` holds scratch files from the build (`_dbg.tsx`, `_t.tsx`, a superseded
`tsconfig.offline.json`, and a stray root-level `_writetest` file that a sandbox write test left
behind). None is imported by the app; delete the directory whenever convenient.


---

## Notes on the running app

- **Nothing polls on a timer except the round index.** `/api/rounds` is refreshed every 8 s while
  View 1 is open and the tab is visible, so a round triggered from a terminal appears without a
  reload. The state itself is re-fetched only when the selected round changes. Views 2–4 are frozen
  artifacts fetched once.
- **The round scrubber is read-only.** Selecting an earlier round issues `GET /api/state?round=k`,
  which reconstructs that round from the ledger prefix and never writes to the database.
- **Three controls write.** `POST /api/restore-golden` (copy the frozen ledger over the live one),
  `POST /api/rewind`, and the SSE runner `GET /api/stream-round` (spends 5 queries). All three are
  behind explicit, labelled buttons; none fires on load.
- **The theme toggle is a token swap**, not a second stylesheet: `[data-theme]` on `<html>` flips
  `--bg`/`--panel`/`--ink`, and every component reads those tokens. Light mode is there for
  projectors.
- **Charts are hand-rolled SVG** (`src/components/charts.tsx`): line with shaded bands and reference
  lines, horizontal bars, scatter with a threshold line, point-with-whisker error bars. A `null`
  becomes a break in the line or an em dash in a label — never a zero.
