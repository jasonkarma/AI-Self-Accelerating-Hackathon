/**
 * Number audit: every figure the UI attributes to a comparison is recomputed here
 * **from the raw contract**, independently of the code path the components use,
 * and asserted to agree.
 *
 * `npm run verify`. Exits non-zero on any disagreement, so "the UI shows X" and
 * "the artifact says X" cannot drift apart silently. This is the substitute for
 * eyeballing screenshots: if a derived number changes because a payload changed,
 * this fails here first.
 *
 * The imports are of the *real* modules the views import — not a copy. A bug in
 * `derive.ts` therefore shows up as a failed assertion, not as a green check on
 * a parallel implementation.
 *
 * Three structural facts about the artifacts shape these checks, and each one is
 * a trap the UI had to be written around:
 *
 * 1. **`cross_target.targets[].paired` is an AUBQC test.** The paired *ρ* test for
 *    a cross-target cell exists only in `/api/attribution`. Using the former as a
 *    ρ p-value would print a mislabelled statistic next to a real one.
 * 2. **1N8Z is not in `targets[]`.** It is published under `reference_targets`
 *    because it was measured by the primary 35-query protocol, and its paired ρ
 *    test lives in `/api/benchmark`.
 * 3. **Attribution mechanism rows carry a different baseline from the live
 *    engine.** Their published delta is `_baseline − variant`, where `_baseline`
 *    is the pre-`e567cd4` AUBQC (which still carried the anneal). Mixing that sign
 *    convention with the live-engine rows would invert the one result the ledger
 *    exists to state.
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { AttributionPayload, BenchmarkPayload, CrossTargetPayload } from "../src/types/report";
import type { StatePayload } from "../src/types/state";
import {
  attributionRows,
  benchmarkMetricRows,
  coverageCells,
  deltaAgentSpread,
  finalRho,
  funnelStages,
  groupSummary,
  harvestFromRound,
  METHODS,
  noAnnealRows,
  partitionClasses,
  productionRow,
  revealedBindings,
  rhoPairedTest,
  rhoRatio,
  rowMatchesTarget,
  slotRoles,
  slotSplit,
  zeroShotRows,
} from "../src/lib/derive";
import { pvalue, signed } from "../src/lib/format";
import {
  countOf,
  flukeOdds,
  foldOverReference,
  foldText,
  gainText,
  gainWhole,
  isClear,
  librarySkipped,
  methodFull,
  methodName,
  noiseMultiple,
  noiseText,
  PLAIN,
  ratio,
  ratioText,
  relativeGain,
  shareWhole,
} from "../src/lib/plain";

const here = dirname(fileURLToPath(import.meta.url));
const contractPath = resolve(here, "../../api_contract.json");

interface Contract {
  state: StatePayload;
  benchmark: BenchmarkPayload;
  cross_target: CrossTargetPayload;
  attribution: AttributionPayload;
}

const contract = JSON.parse(readFileSync(contractPath, "utf8")) as Contract;

let checks = 0;
const failures: string[] = [];

function expect(label: string, condition: boolean, detail: string): void {
  checks += 1;
  if (!condition) failures.push(`${label}: ${detail}`);
}

function close(a: number, b: number, tolerance = 1e-9): boolean {
  return Math.abs(a - b) <= tolerance;
}

function expectClose(label: string, actual: number, expected: number, tolerance = 1e-9): void {
  checks += 1;
  // `NaN === NaN` is false, but "both undefined" is agreement for a ρ that the
  // method cannot produce — M1 has no scoring model, so its ρ is NaN in both.
  if (Number.isNaN(actual) && Number.isNaN(expected)) return;
  if (!close(actual, expected, tolerance)) {
    failures.push(`${label}: derived ${actual} vs contract ${expected}`);
  }
}

function mean(values: number[]): number {
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

// ═══════════════════════════════ View 1 · state ═══════════════════════════

function verifyState(): void {
  const state = contract.state;
  const revealed = revealedBindings(state);
  const best = state.kpis.best_qualified_binding;
  const roles = slotRoles(state, revealed, best);

  expect(
    "state: one role per slot",
    roles.length === state.batch_slots.length,
    `${roles.length} roles vs ${state.batch_slots.length} slots`,
  );

  const graduated = state.voi_diagnostics.slot_graduation === true;
  for (const role of roles) {
    const isVoi = role.slot.slot_type === "Max Joint Uncertainty";
    const isH0 = role.slot.slot_type === "Skeptic Falsification Probe";
    if (graduated) {
      expect(`state: slot ${role.slot.slot_id} graduated`, role.roleBucket === "harvest", role.roleBucket);
    } else {
      const expected = isVoi ? "voi" : isH0 ? "h0" : "exploit";
      expect(`state: slot ${role.slot.slot_id} bucket`, role.roleBucket === expected, `${role.roleBucket} vs ${expected}`);
    }
  }

  const crowned = roles.filter((role) => role.isNewBest);
  expect("state: at most one crowned slot", crowned.length <= 1, `${crowned.length} crowned`);

  for (const role of roles) {
    if (role.revealedBinding === null) continue;
    expect(
      `state: slot ${role.slot.slot_id} binding matches Pareto`,
      close(role.revealedBinding, revealed.get(role.slot.candidate_id) ?? Number.NaN),
      String(role.revealedBinding),
    );
  }

  // The KPI is the max over the revealed set, at the wire's 2-dp precision.
  const maxRevealed = Math.max(...state.pareto_data.map((point) => point.binding));
  expect("state: KPI equals max revealed binding", close(best, maxRevealed, 0.005), `${best} vs ${maxRevealed}`);

  const split = slotSplit(state);
  expect("state: split sums to the batch", split.exploit + split.voi + split.h0 === split.total, JSON.stringify(split));
  expect("state: five slots", split.total === 5, String(split.total));

  const deltas = state.batch_slots
    .map((slot) => slot.delta_agent)
    .filter((value): value is number => value !== null);
  const spread = deltaAgentSpread(state);
  if (deltas.length >= 2 && spread !== null) {
    expectClose("state: Δ_agent spread", spread, Math.max(...deltas) - Math.min(...deltas));
  }

  expect("state: objective law", state.voi_diagnostics.objective === "a_null := a_base", state.voi_diagnostics.objective);
  expect(
    "state: weights are diagnostics",
    state.voi_diagnostics.steers_null_objective === false,
    String(state.voi_diagnostics.steers_null_objective),
  );
  expectClose(
    "state: exploit + exploration = 1",
    state.voi_diagnostics.exploit_weight + state.voi_diagnostics.exploration_weight,
    1,
    1e-3,
  );

  for (const seat of state.agent_council) {
    expect(`state: seat ${seat.role} provider`, seat.provider.length > 0, "empty provider");
    expect(`state: seat ${seat.role} model`, seat.model.length > 0, "empty model");
    expect(
      `state: seat ${seat.role} tool reasons`,
      seat.tools_invoked.every((tool) => tool in seat.tool_activation_reasons || seat.tools_inactive.includes(tool)),
      "a cited tool has no activation reason and is not listed inactive",
    );
  }
}

function verifyRounds(): void {
  const state = contract.state;
  const totalRounds = 6;
  const harvest = harvestFromRound(totalRounds);
  expect("rounds: harvest boundary for R = 6", harvest === 5, String(harvest));
  expect("rounds: harvest boundary for R = 3", harvestFromRound(3) === 3, String(harvestFromRound(3)));
  expect("rounds: harvest boundary for R = 24", harvestFromRound(24) === 17, String(harvestFromRound(24)));
  expect("rounds: harvest boundary for R = 1", harvestFromRound(1) === 2, String(harvestFromRound(1)));

  const graduatedByHorizon = state.current_round >= harvest;
  expect(
    "rounds: graduation flag matches horizon",
    state.voi_diagnostics.slot_graduation === graduatedByHorizon,
    `flag ${state.voi_diagnostics.slot_graduation}, horizon ${graduatedByHorizon} at R${state.current_round}`,
  );
}

// ════════════════════════════ View 2 · benchmark ══════════════════════════

function verifyBenchmark(): void {
  const benchmark = contract.benchmark;

  const last = benchmark.spearman_rho_curves[benchmark.spearman_rho_curves.length - 1]!;
  const final = finalRho(benchmark.spearman_rho_curves as unknown as Array<Record<string, number | null>>);
  for (const method of METHODS) {
    const expected = last[method];
    expectClose(
      `benchmark: final ρ ${method}`,
      final[method] ?? Number.NaN,
      typeof expected === "number" ? expected : Number.NaN,
    );
  }
  expect("benchmark: M1 ρ undefined", final.m1_random === null, String(final.m1_random));

  const vsM3 = benchmark.paired_t_tests.m5_hybrid_minus_m3_ucb.metrics.spearman_rho_at_q35;
  expectClose("benchmark: Δρ vs M3", vsM3.mean_diff, (final.m5_hybrid ?? 0) - (final.m3_ucb ?? 0), 1e-3);
  expect("benchmark: ρ p-value is the published one", close(vsM3.p_value, 8.370558119805303e-32, 1e-40), String(vsM3.p_value));
  expect("benchmark: seeds won is 98/100", vsM3.seeds_won === 98 && vsM3.n === 100, `${vsM3.seeds_won}/${vsM3.n}`);
  expect("benchmark: ρ effect is far beyond its SE", Math.abs(vsM3.mean_diff) / vsM3.se > 5, String(vsM3.mean_diff / vsM3.se));

  const ratio = rhoRatio(final.m5_hybrid ?? null, final.m3_ucb ?? null);
  expectClose("benchmark: ρ ratio", ratio ?? Number.NaN, (final.m5_hybrid ?? 0) / (final.m3_ucb ?? 1));

  const aubqc = benchmark.paired_t_tests.m5_hybrid_minus_m3_ucb.metrics.aubqc;
  expect("benchmark: AUBQC gap is sub-SE", Math.abs(aubqc.mean_diff) < aubqc.se, `${aubqc.mean_diff} vs SE ${aubqc.se}`);
  expect("benchmark: AUBQC not significant", aubqc.p_value > 0.05, String(aubqc.p_value));

  const panel = benchmark.paired_t_tests.m5_hybrid_minus_m2_greedy.metrics.panel_aubqc;
  expect("benchmark: panel depth behind M2", panel.mean_diff < 0 && panel.p_value < 0.01, `${panel.mean_diff} p ${panel.p_value}`);
  const decile = benchmark.paired_t_tests.m5_hybrid_minus_m2_greedy.metrics.top_decile_hit_rate;
  expect("benchmark: decile behind M2", decile.mean_diff < 0 && decile.p_value < 0.01, `${decile.mean_diff} p ${decile.p_value}`);

  const zeroShot = zeroShotRows(benchmark);
  expect("benchmark: three zero-shot rows", zeroShot.length === 3, String(zeroShot.length));
  for (const row of zeroShot) {
    expect(`benchmark: ${row.key} negative`, row.rho < 0, String(row.rho));
    expect(`benchmark: ${row.key} below M5`, row.rho < (final.m5_hybrid ?? 0), `${row.rho} vs ${final.m5_hybrid}`);
  }

  for (const row of benchmarkMetricRows(benchmark)) {
    expect(`benchmark: ${row.key} has a best`, row.best !== null, "no best");
    expect(`benchmark: ${row.key} has an M5 rank`, row.m5Rank !== null, "no rank");
  }

  const sameSeed = benchmark.same_seed_comparison;
  // The published gap is rounded on the wire (4 dp), so it is compared at that
  // precision rather than rejected for a 2e-5 difference in a derived value.
  expectClose(
    "benchmark: same-seed gap",
    sameSeed.live.aubqc_gap_vs_best_baseline,
    sameSeed.live.aubqc - sameSeed.live.best_baseline_aubqc,
    1e-3,
  );
  expect("benchmark: same-seed initial ids", sameSeed.initial_ids.length === 5, String(sameSeed.initial_ids.length));
  expect(
    "benchmark: same-seed best baseline is a real method",
    (METHODS as readonly string[]).includes(sameSeed.live.best_baseline),
    sameSeed.live.best_baseline,
  );
  for (const method of METHODS) {
    expect(
      `benchmark: same-seed summary has ${method}`,
      Number.isFinite(sameSeed.summary[method]?.aubqc ?? Number.NaN),
      "missing aubqc",
    );
  }

  const checkpoints = benchmark.curves.map((row) => row.query_count);
  expect("benchmark: 7 checkpoints", checkpoints.length === 7, String(checkpoints.length));
  expect(
    "benchmark: ρ curves share the checkpoints",
    benchmark.spearman_rho_curves.map((row) => row.query_count).join(",") === checkpoints.join(","),
    "checkpoint mismatch",
  );

  expect(
    "benchmark: warm start not merged into aubqc",
    !Object.prototype.hasOwnProperty.call(benchmark.aubqc, "m5_warm_start"),
    "warm start leaked into aubqc",
  );
  expect(
    "benchmark: seed count agrees with source",
    benchmark.source.seeds === benchmark.seeds,
    `${benchmark.source.seeds} vs ${benchmark.seeds}`,
  );

  expect("format: p-value keeps magnitude", pvalue(vsM3.p_value).includes("10"), pvalue(vsM3.p_value));
  expect("format: signed keeps sign", signed(-0.0429, 4).startsWith("−"), signed(-0.0429, 4));
}

// ═══════════════════════════ View 3 · cross-target ════════════════════════

function verifyCrossTarget(): void {
  const cross = contract.cross_target;
  const benchmark = contract.benchmark;
  const attribution = contract.attribution;
  const partition = cross.learnability_gate.partition;

  const classes = partitionClasses(cross, attribution, benchmark);
  const total = classes.reduce((acc, entry) => acc + entry.targets.length, 0);
  expect("cross: partition is exhaustive", total === 17, `${total} targets across classes`);
  expect("cross: partition is disjoint", new Set(classes.flatMap((entry) => entry.targets)).size === total, "overlapping classes");
  expect(
    "cross: classes are exactly the metadata corpus",
    classes.flatMap((entry) => entry.targets).every((target) =>
      cross.lockbox_metadata.rows.some((row) => rowMatchesTarget(row, target)),
    ),
    "a class names a target outside the corpus",
  );

  // ── Class 1: M5 > M3 on ρ, and the paired ρ test exists somewhere. ──
  for (const target of partition.learnable_multiround) {
    if (target === "1n8z") {
      const reference = cross.reference_targets[0]!;
      const last = reference.spearman_rho_curves[reference.spearman_rho_curves.length - 1]!;
      expect("cross: 1n8z M5 > M3 on ρ", (last.m5_hybrid ?? 0) > (last.m3_ucb ?? 0), `${last.m5_hybrid} vs ${last.m3_ucb}`);
      const paired = benchmark.paired_t_tests.m5_hybrid_minus_m3_ucb.metrics.spearman_rho_at_q35;
      expect("cross: 1n8z paired ρ p ≤ 2e-4", paired.p_value <= 2e-4, String(paired.p_value));
      expectClose("cross: 1n8z Δρ agrees", paired.mean_diff, (last.m5_hybrid ?? 0) - (last.m3_ucb ?? 0), 1e-3);
      continue;
    }
    const row = cross.targets.find((entry) => entry.dataset === target && entry.coverage_label === "16.7pct");
    expect(`cross: ${target} measured at 16.7pct`, row !== undefined, "no row");
    if (!row) continue;
    const m5 = row.spearman_rho_at_last_checkpoint.m5_hybrid;
    const m3 = row.spearman_rho_at_last_checkpoint.m3_ucb;
    expect(`cross: ${target} M5 > M3 on ρ`, (m5 ?? 0) > (m3 ?? 0), `${m5} vs ${m3}`);
    const paired = rhoPairedTest(attribution, row.dataset, row.coverage_label);
    expect(`cross: ${target} has a paired ρ test`, paired !== null, "no attribution cell");
    if (!paired) continue;
    expectClose(`cross: ${target} paired Δρ = ρ gap`, paired.meanDiff, (m5 ?? 0) - (m3 ?? 0), 5e-4);
    expect(`cross: ${target} paired ρ p ≤ 2e-4`, paired.p_value <= 2e-4, String(paired.p_value));
  }

  // The cross-target `paired` block is an AUBQC test — asserted, so nobody
  // silently repurposes it as a ρ test.
  for (const row of cross.targets) {
    expect(
      `cross: ${row.dataset}@${row.coverage_label} paired metric is aubqc`,
      row.paired.m5_hybrid_minus_m3_ucb.metric === "aubqc",
      row.paired.m5_hybrid_minus_m3_ucb.metric,
    );
    expectClose(
      `cross: ${row.dataset}@${row.coverage_label} paired Δ = AUBQC gap`,
      row.paired.m5_hybrid_minus_m3_ucb.mean_diff,
      row.aubqc.m5_hybrid - row.aubqc.m3_ucb,
      1e-4,
    );
  }

  // ── Class 3: single round is a structural tie on AUBQC. ──
  for (const target of partition.single_round) {
    for (const cell of cross.targets.filter((entry) => entry.dataset === target)) {
      expect(
        `cross: ${target}@${cell.coverage_label} identical AUBQC`,
        close(cell.aubqc.m5_hybrid, cell.aubqc.m3_ucb, 1e-9),
        `${cell.aubqc.m5_hybrid} vs ${cell.aubqc.m3_ucb}`,
      );
      expect(
        `cross: ${target}@${cell.coverage_label} paired p = 1`,
        close(cell.paired.m5_hybrid_minus_m3_ucb.p_value, 1, 1e-9),
        String(cell.paired.m5_hybrid_minus_m3_ucb.p_value),
      );
    }
  }

  // ── Class 2: the learnability gate really failed. ──
  for (const target of partition.unlearnable) {
    const gateRows = cross.learnability_gate.rows.filter((row) => row.target_id === target && row.max_baseline_rho !== undefined);
    const measured = cross.targets.filter((row) => row.dataset === target);
    expect(`cross: ${target} has evidence`, gateRows.length > 0 || measured.length > 0, "no gate or measured rows");
    for (const row of gateRows) {
      expect(
        `cross: ${target} below gate (${row.coverage_label})`,
        (row.max_baseline_rho ?? 1) < cross.learnability_gate.rho_gate,
        `${row.max_baseline_rho} vs gate ${cross.learnability_gate.rho_gate}`,
      );
    }
    for (const row of measured) {
      const bestBaseline = Math.max(
        row.spearman_rho_at_last_checkpoint.m2_greedy ?? Number.NEGATIVE_INFINITY,
        row.spearman_rho_at_last_checkpoint.m3_ucb ?? Number.NEGATIVE_INFINITY,
        row.spearman_rho_at_last_checkpoint.m4_fixed_prior ?? Number.NEGATIVE_INFINITY,
      );
      expect(
        `cross: ${target}@${row.coverage_label} best baseline below gate`,
        bestBaseline < cross.learnability_gate.rho_gate,
        `${bestBaseline} vs gate ${cross.learnability_gate.rho_gate}`,
      );
    }
  }

  // ── Class 4: every member is a 1–8-sequence puddle, not a search problem. ──
  for (const target of partition.degenerate_1_or_2_cdrh3) {
    const row = cross.lockbox_metadata.rows.find((entry) => entry.target_id === target);
    expect(`cross: ${target} in metadata`, row !== undefined, "missing from lockbox rows");
    if (!row) continue;
    expect(`cross: ${target} not usable`, row.usable_as_search_problem === false, "flagged usable");
    expect(
      `cross: ${target} below the usability floor`,
      row.unique_cdrh3_pool_size < 35,
      `${row.unique_cdrh3_pool_size} distinct CDRH3`,
    );
  }
  const oneMhp = cross.lockbox_metadata.rows.find((row) => row.target_id === "1mhp_LC");
  expect("cross: 1mhp_LC is the 8-sequence outlier", oneMhp?.unique_cdrh3_pool_size === 8, String(oneMhp?.unique_cdrh3_pool_size));

  // ── The funnel: row-conserving, and it ends on the pre-registered target. ──
  const stages = funnelStages(cross.lockbox_metadata, cross.lockbox_metadata.min_nqual, cross.learnability_gate);
  expect("cross: funnel starts with all rows", stages[0]!.remaining.length === cross.lockbox_metadata.rows.length, "start mismatch");
  for (let index = 1; index < stages.length; index += 1) {
    const before = stages[index - 1]!;
    const stage = stages[index]!;
    expect(
      `cross: funnel stage '${stage.stage}' conserves rows`,
      before.remaining.length === stage.rejected.length + stage.remaining.length,
      `${before.remaining.length} -> ${stage.remaining.length} + ${stage.rejected.length}`,
    );
  }
  const survivors = stages[stages.length - 1]!.remaining;
  expect(
    "cross: funnel ends on aayl51 alone",
    survivors.length === 1 && survivors[0]!.target_id === "aayl51",
    survivors.map((row) => row.target_id).join(",") || "none",
  );
  // The stage labels are product copy (`lib/derive.ts` owns them), so the audit
  // keys on the rule the stage applies — the overlap stage is the only one whose
  // title mentions overlap — rather than on an exact string that a copy edit
  // would break.
  const overlapStage = stages.find((stage) => /overlap/i.test(stage.stage));
  expect("cross: overlap stage exists", overlapStage !== undefined, "no overlap stage");
  expect(
    "cross: aayl49 rejected on measured overlap",
    overlapStage?.rejected.some((row) => row.target_id === "aayl49") === true,
    overlapStage?.rejected.map((row) => row.target_id).join(",") ?? "none",
  );
  expect(
    "cross: HER2 control rejected",
    stages[stages.length - 1]!.rejected.some((row) => row.target_id.toLowerCase().includes("her2")),
    "4d5 row survived",
  );

  // ── Coverage matrix. ──
  const datasets = Array.from(new Set(cross.coverage_sensitivity.map((row) => row.dataset)));
  expect("cross: four coverage datasets", datasets.length === 4, datasets.join(","));
  for (const dataset of datasets) {
    expect(`cross: ${dataset} has three coverage cells`, coverageCells(cross, dataset).length === 3, String(coverageCells(cross, dataset).length));
  }
  for (const cell of cross.coverage_sensitivity) {
    if (cell.m5_rank === null) continue;
    expect(`cross: ${cell.dataset}@${cell.coverage_label} rank in 1..5`, cell.m5_rank >= 1 && cell.m5_rank <= 5, String(cell.m5_rank));
    if (cell.aubqc_m5 !== null && cell.aubqc_best_baseline !== null && cell.m5_minus_best_baseline !== null) {
      expectClose(
        `cross: ${cell.dataset}@${cell.coverage_label} Δ = m5 − best`,
        cell.m5_minus_best_baseline,
        cell.aubqc_m5 - cell.aubqc_best_baseline,
        1e-3,
      );
    }
  }

  // ── The reference row. ──
  const reference = cross.reference_targets[0];
  expect("cross: reference is 1n8z", reference?.dataset === "1n8z", reference?.dataset ?? "none");
  expect("cross: reference is flagged", reference?.reference_only === true, String(reference?.reference_only));
  expect("cross: reference checkpoints", (reference?.checkpoints.length ?? 0) === 7, String(reference?.checkpoints.length));
  expect(
    "cross: reference AUBQC matches the primary artifact",
    close(reference?.aubqc.m5_hybrid ?? Number.NaN, benchmark.aubqc.m5_hybrid, 1e-9),
    `${reference?.aubqc.m5_hybrid} vs ${benchmark.aubqc.m5_hybrid}`,
  );
  expect(
    "cross: 1n8z is absent from targets[]",
    cross.targets.every((row) => row.dataset !== "1n8z"),
    "1n8z leaked into the matrix",
  );

  // ── The one positive ablation on an unlearnable target. ──
  const aayl51 = cross.targets.find((row) => row.dataset === "aayl51" && row.coverage_label === "16.7pct");
  expect("cross: aayl51 measured", aayl51 !== undefined, "missing");
  if (aayl51) {
    const warm = aayl51.paired.m5_warm_start_minus_m5_hybrid;
    expect("cross: warm start wins on aayl51", warm.mean_diff > 0 && warm.p_value < 0.001, `${warm.mean_diff} p ${warm.p_value}`);
    expect(
      "cross: warm start summary agrees with the paired Δ",
      close(aayl51.m5_warm_start_summary.aubqc - aayl51.aubqc.m5_hybrid, warm.mean_diff, 1e-3),
      `${aayl51.m5_warm_start_summary.aubqc - aayl51.aubqc.m5_hybrid} vs ${warm.mean_diff}`,
    );
  }
}

// ═══════════════════════════ View 4 · attribution ═════════════════════════

function verifyAttribution(): void {
  const attribution = contract.attribution;
  const MECHANISM_ROWS = new Set([
    "no_anneal",
    "greedy_slots",
    "slot1_model",
    "harvest_model",
    "harvest_slot1",
    "all_model",
  ]);

  for (const cell of attribution.cells) {
    const production = productionRow(cell);
    expect(`attribution: ${cell.cell} production row`, production !== undefined, "no 'None' row");
    expect(`attribution: ${cell.cell} production method`, production?.method === "m5_hybrid", production?.method ?? "none");

    for (const [variant, row] of Object.entries(cell.rows)) {
      expect(
        `attribution: ${cell.cell}/${variant} seeds`,
        row.seeds === cell.seeds && row.aubqc_per_seed.length === cell.seeds,
        `${row.seeds} vs ${cell.seeds}`,
      );
    }

    for (const [variant, delta] of Object.entries(cell.production_minus)) {
      if (variant === "None" || !cell.rows[variant]) continue;
      const variantRow = cell.rows[variant]!;
      const preVoI = MECHANISM_ROWS.has(variant);
      const published = delta as typeof delta & { _baseline?: number };

      if (preVoI) {
        // These rows are `pre-e567cd4 production − variant`, where `_baseline` is
        // that pre-law engine's AUBQC. Assert both halves of that definition.
        expect(
          `attribution: ${cell.cell}/${variant} carries its baseline`,
          Number.isFinite(published._baseline ?? Number.NaN),
          "no _baseline field",
        );
        expectClose(
          `attribution: ${cell.cell}/${variant} ΔAUBQC = _baseline − variant`,
          delta.aubqc.mean_diff,
          (published._baseline ?? Number.NaN) - variantRow.aubqc,
          1e-6,
        );
        // The pre-law engine is a *different* engine from the live one, which is
        // exactly why this row cannot share the live production row's arrays.
        expect(
          `attribution: ${cell.cell}/${variant} baseline differs from live production`,
          !close(published._baseline ?? Number.NaN, production!.aubqc, 1e-9) ||
          variant === "no_anneal",
          `${published._baseline} vs ${production!.aubqc}`,
        );
      } else {
        const diffs = production!.aubqc_per_seed.map(
          (value, index) => value - (variantRow.aubqc_per_seed[index] ?? Number.NaN),
        );
        expectClose(`attribution: ${cell.cell}/${variant} ΔAUBQC = production − variant`, delta.aubqc.mean_diff, mean(diffs), 1e-9);
        expect(
          `attribution: ${cell.cell}/${variant} seeds won reproducible`,
          delta.aubqc.seeds_won === diffs.filter((value) => value > 0).length,
          `${delta.aubqc.seeds_won} vs ${diffs.filter((value) => value > 0).length}`,
        );
      }
    }

    // `no_anneal` is the shipped law, so its per-seed arrays must be *identical*
    // to the live production row. If they ever diverge, the causal story in the
    // panel ("this is the engine, minus the anneal") is no longer true.
    expect(
      `attribution: ${cell.cell} no_anneal ≡ live production`,
      cell.rows.no_anneal!.aubqc_per_seed.every((value, index) =>
        close(value, production!.aubqc_per_seed[index] ?? Number.NaN, 1e-12),
      ),
      "no_anneal and production disagree — the production law may have changed",
    );

    // The normalised view must be the exact negation of the published delta.
    for (const row of attributionRows(cell)) {
      expectClose(
        `attribution: ${cell.cell}/${row.variant} normalised sign`,
        row.variantVsProductionAubqc,
        -cell.production_minus[row.variant]!.aubqc.mean_diff,
        1e-12,
      );
      expect(
        `attribution: ${cell.cell}/${row.variant} basis labelled`,
        row.comparisonBasis === (MECHANISM_ROWS.has(row.variant) ? "pre-e567cd4 production" : "production"),
        row.comparisonBasis,
      );
    }
  }

  const slotRole = groupSummary(attribution, "slot_role");
  const sar = groupSummary(attribution, "sar_efficiency");
  const mechanism = groupSummary(attribution, "mechanism");
  expect("attribution: slot-role comparisons = 32", slotRole.comparisons === 32, String(slotRole.comparisons));
  expect("attribution: SAR comparisons = 24", sar.comparisons === 24, String(sar.comparisons));
  expect("attribution: mechanism comparisons = 16", mechanism.comparisons === 16, String(mechanism.comparisons));
  expect("attribution: smallest slot-role p ≥ 0.05", (slotRole.minP ?? 0) >= 0.05, String(slotRole.minP));
  expect("attribution: smallest SAR p ≥ 0.05", (sar.minP ?? 0) >= 0.05, String(sar.minP));
  expect("attribution: smallest mechanism p < 1e-10", (mechanism.minP ?? 1) < 1e-10, String(mechanism.minP));

  // The causal claim in the normalised convention: removing the anneal GAINS ρ
  // and never costs AUBQC. A significant AUBQC *loss* would falsify the shipped law.
  const rows = noAnnealRows(attribution);
  expect("attribution: no_anneal measured on 4 cells", rows.length === 4, String(rows.length));
  for (const row of rows) {
    expect(
      `attribution: removing the anneal gains ρ on ${row.cell}`,
      row.annealRemovalRho > 0,
      String(row.annealRemovalRho),
    );
    expectClose(
      `attribution: ${row.cell} normalised sign`,
      row.annealRemovalRho,
      -attribution.cells.find((cell) => cell.cell === row.cell)!.production_minus.no_anneal.rho.mean_diff,
      1e-12,
    );
    expect(
      `attribution: removing the anneal does not cost AUBQC on ${row.cell}`,
      row.annealRemovalAubqc >= -1e-9,
      `${row.annealRemovalAubqc} at p ${row.aubqcP}`,
    );
  }
  expect(
    "attribution: AUBQC is neutral (p ≥ 0.05) somewhere",
    rows.filter((row) => row.aubqcP >= 0.05).length >= 1,
    rows.map((row) => row.aubqcP.toFixed(3)).join(","),
  );
  expect(
    "attribution: at least two cells reach p ≤ 2e-4 on ρ",
    rows.filter((row) => row.rhoP <= 2e-4).length >= 2,
    rows.map((row) => row.rhoP.toExponential(1)).join(","),
  );
  expect(
    "attribution: the weakest ρ cell is still p < 0.05",
    rows.every((row) => row.rhoP < 0.05),
    rows.map((row) => row.rhoP.toExponential(1)).join(","),
  );
}

// ═════════════════════════════ consistency between views ═══════════════════

function verifyCrossViewConsistency(): void {
  const cross = contract.cross_target;
  for (const sensitivity of cross.coverage_sensitivity) {
    const target = cross.targets.find(
      (row) => row.dataset === sensitivity.dataset && row.coverage_label === sensitivity.coverage_label,
    );
    if (!target) continue;
    if (sensitivity.aubqc_m5 !== null) {
      expectClose(
        `cross: ${sensitivity.dataset}@${sensitivity.coverage_label} AUBQC agrees across payloads`,
        sensitivity.aubqc_m5,
        target.aubqc.m5_hybrid,
        5e-4,
      );
    }
    if (sensitivity.rho_m5 !== null) {
      expectClose(
        `cross: ${sensitivity.dataset}@${sensitivity.coverage_label} ρ agrees across payloads`,
        sensitivity.rho_m5,
        target.spearman_rho_at_last_checkpoint.m5_hybrid ?? Number.NaN,
        5e-5,
      );
    }
  }

  // The attribution production AUBQC must equal the cross-target M5 AUBQC on the
  // cells both cover; they are the same measurement published twice.
  for (const cell of contract.attribution.cells) {
    const [dataset, coverage] = cell.cell.split(" @ ");
    const target = contract.cross_target.targets.find(
      (row) => row.dataset === dataset && row.coverage_label === coverage,
    );
    if (!target) continue;
    expectClose(
      `consistency: ${cell.cell} production AUBQC = cross-target M5 AUBQC`,
      productionRow(cell)!.aubqc,
      target.aubqc.m5_hybrid,
      5e-4,
    );
  }
}

// ═══════════════════ the plain-language layer (`lib/plain.ts`) ═════════════

/**
 * Every headline figure a judge reads is produced by a function in
 * `src/lib/plain.ts`. This block asserts each one against the payload it
 * summarises, so a translation cannot drift from the number it translates.
 *
 * Three properties matter more than the arithmetic:
 *
 * 1. **The gain is relative to the baseline**, not an absolute difference — the
 *    two are easy to confuse in prose, and only one of them is what "+91%" means.
 * 2. **The odds ladder never overstates.** `flukeOdds` may return a weaker claim
 *    than the p-value supports, never a stronger one, so every tier is checked
 *    against the boundary it must not cross.
 * 3. **The fold-change is the inverse of the log scale.** `−log₁₀ K_D` means a
 *    difference of `d` is `10^d` times tighter, not `d` times.
 */
function verifyPlainLanguage(): void {
  const benchmark = contract.benchmark;
  const final = finalRho(benchmark.spearman_rho_curves as unknown as Array<Record<string, number | null>>);
  const m5 = final.m5_hybrid ?? Number.NaN;
  const m3 = final.m3_ucb ?? Number.NaN;
  const vsM3 = benchmark.paired_t_tests.m5_hybrid_minus_m3_ucb.metrics.spearman_rho_at_q35;

  // ── the headline gain ──
  const gain = relativeGain(m5, m3);
  expect("plain: gain is defined", gain !== null, "null for two finite scores");
  expectClose("plain: gain = (m5 − m3) / m3", gain ?? Number.NaN, (m5 - m3) / m3, 1e-12);
  // The gain must not be the absolute difference, which is the classic slip.
  expect(
    "plain: gain is relative, not absolute",
    Math.abs((gain ?? 0) - (m5 - m3)) > 1e-3,
    `gain ${gain} equals the absolute diff ${m5 - m3}`,
  );
  expect("plain: gain rounds to a positive percentage", gainWhole(gain).startsWith("+"), gainWhole(gain));
  expect("plain: gainWhole is a whole percentage", /^\+[0-9]+%$/.test(gainWhole(gain)), gainWhole(gain));
  expect("plain: gainText keeps a decimal", /^\+[0-9]+\.[0-9]%$/.test(gainText(gain)), gainText(gain));
  // "+91.1%" for this artifact — asserted against the payload, never typed in src/.
  expectClose("plain: gain percent", (gain ?? 0) * 100, 91.11, 0.02);

  const r = ratio(m5, m3);
  expectClose("plain: ratio = m5 / m3", r ?? Number.NaN, m5 / m3, 1e-12);
  expect("plain: ratioText has a times sign", ratioText(r).endsWith("×"), ratioText(r));
  expectClose("plain: ratio ≈ 1.91", r ?? Number.NaN, 1.91, 0.005);
  // The ratio and the gain are the same claim; they must not disagree.
  expectClose("plain: ratio = 1 + gain", r ?? Number.NaN, 1 + (gain ?? 0), 1e-12);

  // ── the noise multiple (replaces Δ / SE) ──
  const noise = noiseMultiple(vsM3.mean_diff, vsM3.se);
  expectClose("plain: noise multiple = |diff| / se", noise ?? Number.NaN, Math.abs(vsM3.mean_diff) / vsM3.se, 1e-12);
  expect("plain: noise multiple is well above 1", (noise ?? 0) > 5, String(noise));
  expect("plain: noiseText is an integer multiple", /^[0-9]+×$/.test(noiseText(noise)), noiseText(noise));
  expect("plain: noise rejects a zero SE", noiseMultiple(1, 0) === null, "returned a number for se = 0");

  // ── the odds ladder: conservative at every boundary ──
  // Each tier is checked at its own boundary and just inside it, because an
  // off-by-one boundary is exactly how a phrase ends up overstating the evidence.
  expect("plain: p ≥ 0.05 reads as chance", flukeOdds(0.05) === "could easily be chance", flukeOdds(0.05));
  expect("plain: p = 1 reads as chance", flukeOdds(1) === "could easily be chance", flukeOdds(1));
  expect("plain: p just under 0.05 is 'unlikely'", flukeOdds(0.049) === "unlikely to be chance", flukeOdds(0.049));
  // At exactly `p = 1e-3` the true odds are 1 in 1,000 — so "less than 1 in
  // 1,000" would be an overstatement, and the ladder must round *down* to the
  // weaker phrase. The tiers therefore start strictly below their boundary, and
  // these assertions pin that direction.
  expect("plain: p = 1e-3 does not claim 'less than 1 in 1,000'", flukeOdds(1e-3) === "unlikely to be chance", flukeOdds(1e-3));
  expect("plain: p just under 1e-3 earns the 1,000 tier", flukeOdds(9.9e-4) === "less than 1 in 1,000", flukeOdds(9.9e-4));
  expect("plain: p = 1e-6 does not claim 'less than 1 in a million'", flukeOdds(1e-6) === "less than 1 in 1,000", flukeOdds(1e-6));
  expect("plain: p just under 1e-6 earns the million tier", flukeOdds(9.9e-7) === "less than 1 in a million", flukeOdds(9.9e-7));
  expect("plain: p = 1e-12 does not claim the trillion tier", flukeOdds(1e-12) === "less than 1 in a million", flukeOdds(1e-12));
  expect("plain: p just under 1e-12 earns the trillion tier", flukeOdds(9.9e-13) === "less than 1 in a trillion", flukeOdds(9.9e-13));
  expect("plain: the published p reads as the strongest tier", flukeOdds(vsM3.p_value) === "less than 1 in a trillion", flukeOdds(vsM3.p_value));
  expect("plain: a missing p is not a phrase", flukeOdds(null) === "—", flukeOdds(null));
  // The ladder never claims more than the p-value proves: the trillion tier
  // requires p < 1e-12, and 1/p for this artifact is astronomically larger.
  expect("plain: trillion tier is honest", (vsM3.p_value ?? 1) < 1e-12, String(vsM3.p_value));
  expect("plain: clear() agrees with the ladder", isClear(vsM3.p_value) && !isClear(0.5), "isClear disagrees");

  // ── the fold-change (binding is a log scale) ──
  const delta = contract.state.kpis.delta_vs_wt;
  expectClose("plain: fold = 10^delta", foldOverReference(delta) ?? Number.NaN, Math.pow(10, delta), 1e-12);
  expect("plain: fold is not the raw delta", Math.abs((foldOverReference(delta) ?? 0) - delta) > 1, "fold equals the raw log difference");
  expect("plain: foldText has a times sign", foldText(delta).endsWith("×"), foldText(delta));
  expect("plain: fold of 0 is 1×", foldText(0) === "1.0×", foldText(0));

  // ── the library-skipped share ──
  const config = contract.cross_target.cross_target;
  const skipped = librarySkipped(
    contract.state.kpis.query_budget_spent,
    config.qualified_count,
  );
  expectClose(
    "plain: skipped = 1 − spent / qualified",
    skipped ?? Number.NaN,
    1 - contract.state.kpis.query_budget_spent / config.qualified_count,
    1e-12,
  );
  expect("plain: skipped is a plausible share", (skipped ?? 0) > 0 && (skipped ?? 0) < 1, String(skipped));
  expect("plain: skipped rejects a zero pool", librarySkipped(5, 0) === null, "divided by an empty pool");
  expect("plain: skipped rejects an over-budget run", librarySkipped(500, 100) === null, "returned a negative share");
  expect("plain: shareWhole is a whole percentage", /^[0-9]+%$/.test(shareWhole(skipped)), shareWhole(skipped));

  // ── counts ──
  expect("plain: countOf joins with 'of'", countOf(vsM3.seeds_won, vsM3.n) === `${vsM3.seeds_won} of ${vsM3.n}`, countOf(vsM3.seeds_won, vsM3.n));
  expect("plain: countOf handles a missing half", countOf(null, 5) === "—", countOf(null, 5));

  // ── the vocabulary itself ──
  // Every statistic the surface names must carry a scale line; a label without
  // one is the exact failure this redesign exists to prevent.
  for (const [key, entry] of Object.entries(PLAIN)) {
    expect(`plain: ${key} has a label`, entry.label.length > 0, "empty label");
    expect(`plain: ${key} has a scale`, entry.scale.length > 0, "empty scale");
    expect(`plain: ${key} has a token`, entry.token.length > 0, "empty token");
    expect(`plain: ${key} has a meaning`, entry.meaning.length > 20, "meaning too short");
    // Rule 2: the label is plain English, so it must not contain the token.
    expect(
      `plain: ${key} label is not its token`,
      entry.label.toLowerCase() !== entry.token.toLowerCase(),
      `label "${entry.label}" equals its token`,
    );
  }

  // Every method the tables print must have words, and no surface name may be
  // the bare `M1…M5` token.
  for (const method of METHODS) {
    const words = methodName(method);
    expect(`plain: ${method} has a name`, words.length > 0, "empty name");
    expect(`plain: ${method} is not shown as its token`, words !== method, `still "${method}"`);
    expect(`plain: ${method} has a full name`, methodFull(method).length >= words.length, "full name shorter than the short one");
  }
}

verifyState();
verifyRounds();
verifyBenchmark();
verifyCrossTarget();
verifyAttribution();
verifyCrossViewConsistency();
verifyPlainLanguage();

if (failures.length) {
  console.error(`\nnumber audit FAILED: ${failures.length} of ${checks} assertions`);
  for (const failure of failures.slice(0, 40)) console.error(`  ✗ ${failure}`);
  if (failures.length > 40) console.error(`  … and ${failures.length - 40} more`);
  process.exit(1);
}
console.log(`number audit ok: ${checks} assertions recomputed from api_contract.json`);
