/**
 * Render smoke test: boots the real component tree against the real contract
 * payloads and renders every view to static markup.
 *
 * `tsx scripts/render-smoke.tsx`. This is the offline equivalent of loading the
 * page: it shims the browser APIs the app touches (`fetch`, `matchMedia`,
 * `localStorage`, `ResizeObserver`) so that a payload-shaped bug — a missing
 * field, a `null` dereference, a chart handed an empty array — throws here
 * instead of in front of a judge.
 *
 * What it proves: every view mounts, reads its payload, and produces markup.
 * What it does not prove: pixel layout (no layout engine in Node) and event
 * handlers (never fired).
 */
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

import { CJK_PATTERN, findCjk, normalizeCouncilText, normalizeCouncilSeat } from "../src/utils/normalizeCouncilText";
import { COUNCIL_TRANSLATIONS } from "../src/utils/councilTranslations";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const contractPath = resolve(root, "../api_contract.json");
const ledgerPath = resolve(root, "../results/golden_seed21.sqlite3");
const contract = JSON.parse(readFileSync(contractPath, "utf8")) as Record<string, unknown>;

// ── Browser shims ─────────────────────────────────────────────────────────
const store = new Map<string, string>();
const matchMedia = (query: string) => ({
  matches: false,
  media: query,
  onchange: null,
  addEventListener: () => undefined,
  removeEventListener: () => undefined,
  addListener: () => undefined,
  removeListener: () => undefined,
  dispatchEvent: () => false,
});

class ResizeObserverShim {
  observe(): void {
    /* charts fall back to their minimum width without layout */
  }
  unobserve(): void { }
  disconnect(): void { }
}

const globalScope = globalThis as unknown as Record<string, unknown>;
globalScope.window = {
  localStorage: {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
  },
  matchMedia,
  setInterval: () => 0,
  clearInterval: () => undefined,
  setTimeout: () => 0,
  clearTimeout: () => undefined,
  addEventListener: () => undefined,
  removeEventListener: () => undefined,
  innerWidth: 1600,
  innerHeight: 1000,
  devicePixelRatio: 2,
};
globalScope.document = {
  documentElement: { dataset: {} as Record<string, string> },
  visibilityState: "visible",
  getElementById: () => ({ appendChild: () => undefined }),
  addEventListener: () => undefined,
  removeEventListener: () => undefined,
};
globalScope.localStorage = (globalScope.window as { localStorage: unknown }).localStorage;
globalScope.ResizeObserver = ResizeObserverShim;

// `fetch` serves the contract: `/api/<x>` maps to the captured payload.
const ROUTES: Record<string, unknown> = {
  "/api/state": contract.state,
  "/api/benchmark": contract.benchmark,
  "/api/cross-target": contract.cross_target,
  "/api/attribution": contract.attribution,
  "/api/rounds": {
    seed: 21,
    current_round: (contract.state as { current_round: number }).current_round,
    total_rounds: 6,
    total_budget: (contract.state as { kpis: { total_budget: number } }).kpis.total_budget,
    batch_size: 5,
    rounds: buildTimeline(contract.state as Record<string, unknown>),
  },
  "/api/config": {
    dataset_sequence_sha256: "cae03b6bb7fe77c9f49fbb21ad7796dce09aa35e7960de9f49e0c8eafb6feb67",
    is_synthetic: false,
    normalizer: { maxima: {}, minima: {} },
    pool_size: 419,
    qualification_pass_rate: 0.5011933174224343,
    qualified_count: 210,
    tau_dev: 0.6478721298495645,
    tau_risk: 0.35212787015043545,
    threshold_percentile: 0.5,
    threshold_source: "pool_median",
    version: "1.2-post-kickoff",
    wild_type: "SRWGGDGFYAMDY",
    wt_pass_fraction: 0.6992840095465394,
    wt_risk: 0.3920272028051125,
    total_budget: 35,
    hitl: false,
    m5_policy_mode: "adaptive_tool",
  },
  "/api/ledger": {
    verified: true,
    metadata: { m5_policy_mode: "live_agent", is_synthetic: false },
    entries: Array.from({ length: 88 }, () => ({})),
  },
  "/api/health": { status: "ok", version: "1.2" },
};

/**
 * A minimal `/api/rounds` body derived from the state payload.
 *
 * The real endpoint walks the ledger; the smoke test only needs the *shape* the
 * stepper dereferences, and deriving it from the state keeps the two payloads
 * consistent (a stepper whose `current_round` disagrees with the state's is a bug
 * worth failing on, and this makes that visible).
 */
function buildTimeline(state: Record<string, unknown>): unknown[] {
  const current = state.current_round as number;
  const kpis = state.kpis as Record<string, number>;
  const diagnostics = state.voi_diagnostics as Record<string, unknown>;
  const council = state.agent_council as unknown[];
  const slots = state.batch_slots as Array<{ candidate_id: string }>;
  const activeMetadata = diagnostics.active_tools_metadata as {
    active: string[];
    tools: Array<{ name: string; tier: string; active: boolean }>;
  };
  return Array.from({ length: current + 1 }, (_, round) => ({
    round,
    label: round === 0 ? "Seed" : `R${round}`,
    phase: round <= 2 ? "Cold-Start Tuition" : "Self-Accelerating Payoff",
    query_count: round === 0 ? 5 : 5 + round * 5,
    best_qualified_binding: kpis.best_qualified_binding,
    delta_vs_wt: kpis.delta_vs_wt,
    voi: diagnostics.voi,
    sar_maturity: diagnostics.sar_maturity,
    exploit_weight: diagnostics.exploit_weight,
    hypotheses_supported: kpis.hypotheses_supported,
    hypotheses_falsified: kpis.hypotheses_falsified,
    hypothesis_id: round === 0 ? null : `HYP-R${round}-01`,
    verdict: round === 0 ? null : round % 3 === 0 ? "FALSIFIED" : "SUPPORTED",
    statement: round === 0 ? null : "A minimal construct reaches the predicted binding.",
    council_seats: council.length,
    slots: slots.length,
    slot_candidate_ids: slots.map((slot) => slot.candidate_id),
    active_tools: activeMetadata.active,
    inactive_tools: activeMetadata.tools.filter((tool) => !tool.active).map((tool) => tool.name),
    tool_tiers: Object.fromEntries(activeMetadata.tools.map((tool) => [tool.name, tool.tier])),
  }));
}

globalScope.fetch = async (input: string) => {
  const path = String(input).replace(/^https?:\/\/[^/]+/, "");
  const key = path.split("?")[0] ?? path;
  const body = ROUTES[key];
  if (body === undefined) {
    return { ok: false, status: 404, statusText: "Not Found", json: async () => ({ detail: `no route ${key}` }) };
  }
  return { ok: true, status: 200, statusText: "OK", json: async () => body };
};

// ── Render ────────────────────────────────────────────────────────────────
const { Header } = await import("../src/components/Header");

/**
 * The shell is rendered as `Header` rather than `App`.
 *
 * `App` wraps its views in `framer-motion`'s `<AnimatePresence>`, which mounts a
 * DOM projection node on the client and throws under `renderToStaticMarkup`. That
 * is a property of the animation library, not of this app: in a browser the shell
 * mounts normally. Rendering the header directly still exercises the provenance
 * pill, the tab bar and the theme/localStorage path.
 */
const shellMarkup = renderToStaticMarkup(
  createElement(Header, {
    tab: "lab",
    onTab: () => undefined,
    theme: "dark",
    onTheme: () => undefined,
    onReload: () => undefined,
    loading: false,
    ledger: ROUTES["/api/ledger"] as never,
    ledgerError: false,
    health: ROUTES["/api/health"] as never,
    config: ROUTES["/api/config"] as never,
  }) as never,
);

const failures: string[] = [];

function expect(label: string, condition: boolean, detail: string): void {
  if (!condition) failures.push(`${label}: ${detail}`);
}

function expectIncludes(needle: string, label: string): void {
  if (!shellMarkup.includes(needle)) failures.push(`${label}: "${needle}" not found in markup`);
}

expectIncludes("Self-Accelerating Multi-Agent Virtual Lab", "header");
// The tab bar is product copy, so the assertion names the strings the header
// declares: each section's number, title and subtitle must all reach the markup.
for (const label of [
  "Overview",
  "What we built, in 30 seconds",
  "The recorded run",
  "Watch it decide, round by round",
  "Beats the standard method",
  "100 runs, head to head",
  "Works on other targets",
  "Four more public datasets, same code",
  "Why it works",
  "The ablations, including the nulls",
]) {
  expectIncludes(label, `tab ${label}`);
}
// The header has to say what the product *does*, not only what it is called.
expectIncludes("Designs antibody mutations and decides which five to assay next", "header tagline");
expectIncludes("49fea618", "provenance pill: ledger digest");
expectIncludes("a_null := a_base", "provenance pill: objective law");
expectIncludes("VERIFIED", "provenance pill: chain verification");
if (shellMarkup.length < 5_000) failures.push(`shell markup too small (${shellMarkup.length} chars)`);

// The pages render from `useResource`, which resolves promises after paint; the
// static render therefore exercises the *loading* branch. Render each view again
// with the payloads handed in directly to exercise the data branches.
const { OverviewView } = await import("../src/views/overview/OverviewView");
const { LabView } = await import("../src/views/lab/LabView");
const { BenchmarkView } = await import("../src/views/benchmark/BenchmarkView");
const { CrossTargetView } = await import("../src/views/cross/CrossTargetView");
const { AttributionView } = await import("../src/views/attribution/AttributionView");

const state = contract.state as never;
const benchmark = contract.benchmark as never;
const crossTarget = contract.cross_target as never;
const attribution = contract.attribution as never;

const views: Array<[string, string]> = [];

/**
 * Render one view and assert it produced a plausible amount of markup.
 *
 * The floor is a *smoke* check — "the component mounted and did not bail out to
 * an empty shell" — not a size target, so it is set per view: the report views
 * are long documents (20k+), while the cold open is deliberately short (the
 * whole point is that it fits on one screen).
 */
function renderView(
  label: string,
  element: Parameters<typeof renderToStaticMarkup>[0],
  minChars = 20_000,
): string {
  try {
    const html = renderToStaticMarkup(element);
    if (html.length < minChars) failures.push(`${label}: markup too small (${html.length} chars)`);
    return html;
  } catch (error) {
    failures.push(`${label}: threw ${(error as Error).message}`);
    return "";
  }
}

/**
 * A rendered payload must not leak the *language* of a missing value.
 *
 * The check is on rendered text, not on substrings: a caption that literally says
 * "ρ undefined, not 0" is the console being honest about a null correlation, and
 * failing on it would push the copy toward saying nothing. Only the tokens that
 * mean "a value went missing" — `>undefined<`, `>NaN<`, `>null<`, `[object
 * Object]` — are failures.
 */
function expectNoMissingValues(label: string, html: string): void {
  for (const token of [">undefined<", ">NaN<", ">null<", "[object Object]", ">Infinity<"]) {
    if (html.includes(token)) failures.push(`${label}: markup renders ${token}`);
  }
}

// The cold open renders against the same payloads the other views read, so its
// three headline numbers are checked against data rather than accepted on trust.
views.push([
  "overview",
  renderView(
    "overview",
    createElement(OverviewView, {
      benchmark: { status: "ready", data: benchmark, error: null } as never,
      cross: { status: "ready", data: crossTarget, error: null } as never,
      state: { status: "ready", data: state, error: null } as never,
      config: ROUTES["/api/config"] as never,
      onTab: () => undefined,
    }) as unknown as string,
    // The cold open is one screen by design; 8k still catches an empty shell.
    8_000,
  ),
]);

views.push([
  "lab",
  renderView(
    "lab",
    createElement(LabView, {
      state: { status: "ready", data: state, error: null } as never,
      rounds: ROUTES["/api/rounds"] as never,
      config: ROUTES["/api/config"] as never,
      selectedRound: 6,
      onSelectRound: () => undefined,
      onRestoreGolden: async () => undefined,
      restoreError: null,
      onReloadAll: () => undefined,
      playing: false,
      onPlayToggle: () => undefined,
    }) as unknown as string,
  ),
]);

// ── View 1 across every round, checking for CJK ──────────────────────────
//
// The council is the one part of the UI that renders free-form model prose, and
// the recorded campaign had two seats answer in Chinese. `/api/state?round=k`
// serves that prose verbatim, so the check has to cover **every** round the
// stepper can select — not just the round the contract happens to snapshot.
//
// Per-round payloads are reconstructed from the frozen ledger the same way
// `GET /api/state?round=k` does — that round's own `round_council` memos slotted
// into the contract's shape — so the render path under test is the real one.
const ledgerDb = new DatabaseSync(ledgerPath, { readOnly: true });
const councilByRound = new Map<number, unknown[]>();
for (const row of ledgerDb
  .prepare("SELECT round_number, payload FROM entries WHERE event_type = 'round_council' ORDER BY round_number")
  .all() as Array<{ round_number: number; payload: string }>) {
  const parsed = JSON.parse(row.payload) as { memos?: unknown[] };
  councilByRound.set(row.round_number, parsed.memos ?? []);
}
ledgerDb.close();

const baseState = state as { agent_council: unknown[]; current_round: number };
const roundsRendered: string[] = [];
for (const [round, memos] of [...councilByRound.entries()].sort((a, b) => a[0] - b[0])) {
  const roundState = { ...baseState, current_round: round, agent_council: memos };
  const html = renderView(
    `lab@R${round}`,
    createElement(LabView, {
      state: { status: "ready", data: roundState, error: null } as never,
      rounds: ROUTES["/api/rounds"] as never,
      config: ROUTES["/api/config"] as never,
      selectedRound: round,
      onSelectRound: () => undefined,
      onRestoreGolden: async () => undefined,
      restoreError: null,
      onReloadAll: () => undefined,
      playing: false,
      onPlayToggle: () => undefined,
    }) as unknown as string,
  );
  views.push([`lab@R${round}`, html]);
  roundsRendered.push(`R${round}`);

  // The assertion this feature exists to satisfy: the View 1 markup for this
  // round contains **zero** CJK code points.
  const leaked = findCjk(html);
  expect(
    `lab@R${round}: markup contains zero CJK characters`,
    leaked.length === 0,
    `${leaked.length} line(s), first: ${leaked[0] ?? ""}`,
  );

  // Each seat's prose must still be present in English — a translation that
  // returned an empty string would trivially pass a CJK check. Round 5 is the
  // exception the ledger itself creates: two seats recorded an empty memo *and* an
  // empty thinking trace, so there is nothing to translate and the card says so.
  for (const memo of memos as Array<{ role: string; memo: string; thinking_trace?: string }>) {
    const normalized = normalizeCouncilSeat(memo as never);
    const raw = (memo.memo ?? "").trim();
    if (raw === "") {
      expect(
        `lab@R${round}: ${memo.role} empty memo is handled, not rendered as CJK`,
        normalized.memo === "" && !CJK_PATTERN.test(normalized.memo),
        "an empty ledger memo produced non-empty output",
      );
      continue;
    }
    expect(
      `lab@R${round}: ${memo.role} renders English prose`,
      normalized.memo.trim().length > 40 && !CJK_PATTERN.test(normalized.memo),
      `${normalized.memo.trim().length} chars`,
    );
    if (memo.thinking_trace && memo.thinking_trace !== memo.memo) {
      const trace = normalizeCouncilText(memo.thinking_trace);
      expect(
        `lab@R${round}: ${memo.role} thinking trace is English`,
        !CJK_PATTERN.test(trace.text),
        "still contains CJK",
      );
    }
  }
}

// A translation that silently emptied a memo would satisfy the CJK check while
// destroying the evidence, so each English rendering must keep its substance: the
// seven source memos run 720–3744 characters, and no rendering may be a stub.
for (const [hash, translation] of Object.entries(COUNCIL_TRANSLATIONS)) {
  expect(
    `translation ${hash} is substantial`,
    translation.text.length > 600,
    `only ${translation.text.length} chars`,
  );
}

views.push([
  "benchmark",
  renderView(
    "benchmark",
    createElement(BenchmarkView, {
      benchmark: { status: "ready", data: benchmark, error: null } as never,
    }) as unknown as string,
  ),
]);

views.push([
  "cross",
  renderView(
    "cross",
    createElement(CrossTargetView, {
      cross: { status: "ready", data: crossTarget, error: null } as never,
      attribution: { status: "ready", data: attribution, error: null } as never,
      benchmark: { status: "ready", data: benchmark, error: null } as never,
    }) as unknown as string,
  ),
]);

views.push([
  "attribution",
  renderView(
    "attribution",
    createElement(AttributionView, {
      attribution: { status: "ready", data: attribution, error: null } as never,
    }) as unknown as string,
  ),
]);

for (const [label, html] of views) {
  console.log(`rendered ${label}: ${html.length.toLocaleString("en-US")} chars`);
  expectNoMissingValues(label, html);
}

// Spot-check that the real numbers reached the markup.
const overviewHtml = views.find(([label]) => label === "overview")?.[1] ?? "";
const labHtml = views.find(([label]) => label === "lab")?.[1] ?? "";
const benchHtml = views.find(([label]) => label === "benchmark")?.[1] ?? "";
const crossHtml = views.find(([label]) => label === "cross")?.[1] ?? "";
const attrHtml = views.find(([label]) => label === "attribution")?.[1] ?? "";

// The cold open must carry the same three derived claims as the views they open.
const overviewCurves = (contract.benchmark as { spearman_rho_curves: Array<Record<string, number | null>> })
  .spearman_rho_curves;
const overviewLast = overviewCurves[overviewCurves.length - 1]!;
const overviewGain = `${Math.round(
  ((overviewLast.m5_hybrid! - overviewLast.m3_ucb!) / overviewLast.m3_ucb!) * 100,
)}%`;
if (!overviewHtml.includes(overviewGain))
  failures.push(`overview: headline gain ${overviewGain} not rendered`);
if (!overviewHtml.includes("A virtual lab that designs antibody mutations"))
  failures.push("overview: the one-sentence task statement is missing");
if (!overviewHtml.includes("never needed a test"))
  failures.push("overview: the library-skipped claim is missing");
if (!overviewHtml.includes("learnable public targets won"))
  failures.push("overview: the generalization claim is missing");

const stateKpi = String((contract.state as { kpis: { best_qualified_binding: number } }).kpis.best_qualified_binding);
if (!labHtml.includes(stateKpi)) failures.push(`lab: best qualified binding ${stateKpi} not rendered`);
if (!labHtml.includes("a_null := a_base")) failures.push("lab: objective law not rendered");
if (!labHtml.includes("ALL-IN PHASE") && !labHtml.includes("FULL EXPLORATION UNTIL"))
  failures.push("lab: the exploration/exploitation phase badge is not rendered");
if (!labHtml.includes("SRWGGDGFYAMDY") && !labHtml.includes("ARWPFYGPYNYDY"))
  failures.push("lab: no CDRH3 sequence rendered");

// View 1's plain-language surface: the fold-change hero, the skipped-library
// tile, the per-candidate safety verdict and the "under the hood" disclosure
// that keeps the internals out of the demo path.
const foldHero = Math.pow(10, (contract.state as { kpis: { delta_vs_wt: number } }).kpis.delta_vs_wt).toFixed(1);
if (!labHtml.includes(`${foldHero}×`)) failures.push(`lab: fold-change hero ${foldHero}× not rendered`);
if (!labHtml.includes("Library skipped")) failures.push("lab: the library-skipped tile is missing");
if (!labHtml.includes("passes") && !labHtml.includes("fails"))
  failures.push("lab: no safety-filter verdict rendered");
if (!labHtml.includes("Under the hood: live engine telemetry"))
  failures.push("lab: the telemetry disclosure is missing");
if (!labHtml.includes("The scientists")) failures.push("lab: the scientist council header is missing");
if (!labHtml.includes("Did the lab tell the truth?"))
  failures.push("lab: the falsification tracker header is missing");

const rho = String(
  (contract.benchmark as { spearman_rho_curves: Array<{ m5_hybrid: number }> }).spearman_rho_curves.slice(-1)[0]!
    .m5_hybrid,
);
if (!benchHtml.includes(rho)) failures.push(`benchmark: final M5 ρ (${rho}) not rendered`);
if (!benchHtml.includes("AlphaFold3")) failures.push("benchmark: zero-shot reference not rendered");

// The plain-language surface. These are the strings a judge reads, so a
// refactor that quietly drops the headline, the scale anchor or the honest
// strip is a failure even though every payload field is still present.
if (!benchHtml.includes("more accurate at ranking candidates"))
  failures.push("benchmark: the plain-language headline is missing");
if (!benchHtml.includes("0 · guessing") || !benchHtml.includes("1 · perfect"))
  failures.push("benchmark: the 0→1 accuracy scale is not anchored");
if (!benchHtml.includes("The one we didn&#x27;t win") && !benchHtml.includes("The one we didn't win"))
  failures.push("benchmark: the honest 'did not win' strip is missing");
if (/Δ\s*\/\s*SE/.test(benchHtml)) failures.push("benchmark: the raw Δ/SE tile is back");
if (/Spearman\s*ρ/.test(benchHtml.replace(/&rho;/g, "ρ")))
  failures.push("benchmark: the raw 'Spearman ρ' label leaked back onto the surface");

// The headline must be the *derived* gain, not a typed-in percentage: the smoke
// test recomputes it from the two published scores and requires that string.
const rhoCurves = (contract.benchmark as { spearman_rho_curves: Array<Record<string, number | null>> })
  .spearman_rho_curves;
const lastRho = rhoCurves[rhoCurves.length - 1]!;
const derivedGain = `${Math.round(((lastRho.m5_hybrid! - lastRho.m3_ucb!) / lastRho.m3_ucb!) * 100)}%`;
if (!benchHtml.includes(derivedGain))
  failures.push(`benchmark: derived headline gain ${derivedGain} not rendered`);

if (!crossHtml.includes("aayl51")) failures.push("cross: holdout target not rendered");
if (!crossHtml.includes("4d5_her2")) failures.push("cross: HER2 control row not rendered");
if (!crossHtml.includes("17")) failures.push("cross: corpus size not rendered");

if (!attrHtml.includes("no_anneal")) failures.push("attribution: no_anneal row not rendered");
if (!attrHtml.includes("biochem_pooling")) failures.push("attribution: SAR control not rendered");
if (!attrHtml.includes("slot1_model")) failures.push("attribution: slot-role control not rendered");

if (failures.length) {
  console.error(`\nrender smoke FAILED (${failures.length})`);
  for (const failure of failures) console.error(`  ✗ ${failure}`);
  process.exit(1);
}
console.log(
  `render smoke ok: ${views.length} renders (${roundsRendered.join(", ")} + 3 report views + header), ` +
  `no NaN / undefined, zero CJK in View 1, payload numbers present`,
);
