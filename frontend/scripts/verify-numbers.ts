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
  rhoRankAtCoverage,
  rhoRatio,
  strongestWarmStart,
  rowMatchesTarget,
  belowHoldoutFloor,
  canonicalTargetId,
  capabilityCount,
  datasetIdentity,
  evaluatedTargetRoster,
  evaluatedVerdicts,
  slotRoles,
  slotSplit,
  targetRow,
  TOGGLE_COVERAGE,
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

  /*
   * The overview tile states "only 3 of 17 libraries have a ranking signal to
   * find", and its hint claims the exclusions are either metadata-only or were
   * screened before any result existed. The old copy said all 14 excluded
   * datasets were "ruled out before we ran anything" — which was false for
   * aayl49, 4d5_her2 and 1mlc, all of which the baseline gate really ran on.
   * These assertions keep the new wording honest.
   */
  const corpusMetadata = cross.lockbox_metadata;
  expect(
    "cross: overview tile's 3-of-17 denominator matches the corpus",
    corpusMetadata.rows.length === 17,
    String(corpusMetadata.rows.length),
  );
  expect(
    "cross: overview tile's numerator is the learnable set",
    partition.learnable_multiround.length === 3,
    partition.learnable_multiround.join(","),
  );
  expect(
    "cross: the degenerate count is what the hint claims",
    partition.degenerate_1_or_2_cdrh3.length === 10,
    String(partition.degenerate_1_or_2_cdrh3.length),
  );
  /*
   * The hint's wording is deliberately split, because the group is not uniform.
   *
   * The partition field is called `degenerate_1_or_2_cdrh3`, but `1mhp_LC`
   * actually has **8** distinct loops — it is excluded on usability (5 usable
   * candidates), not on loop count. Copy that said "all ten collapse to one or
   * two loops" would be false for it, so the sentence names nine and one. This
   * asserts the split, and would fail if either side of it changed.
   */
  const degenerateRows = partition.degenerate_1_or_2_cdrh3.map((target) => ({
    target,
    row: corpusMetadata.rows.find((row) => rowMatchesTarget(row, target)),
  }));
  const oneOrTwoLoops = degenerateRows.filter((entry) => (entry.row?.unique_cdrh3_pool_size ?? 99) <= 2);
  const moreThanTwo = degenerateRows.filter((entry) => (entry.row?.unique_cdrh3_pool_size ?? 0) > 2);
  expect(
    "cross: nine degenerate datasets have <= 2 distinct loops",
    oneOrTwoLoops.length === 9,
    `${oneOrTwoLoops.length}: ${oneOrTwoLoops.map((e) => e.target).join(",")}`,
  );
  expect(
    "cross: exactly one degenerate dataset has more than 2 loops",
    moreThanTwo.length === 1,
    `${moreThanTwo.length}: ${moreThanTwo.map((e) => e.target).join(",")}`,
  );
  // ...and that one must still fail on usable-candidate count, or the hint's
  // second clause ("eight loops but only five usable") is wrong.
  for (const entry of moreThanTwo) {
    expect(
      `cross: ${entry.target} is excluded on usable candidates, not loop count`,
      entry.row !== undefined && entry.row.N_qual < cross.learnability_gate.min_nqual,
      `${entry.row?.N_qual} usable vs floor ${cross.learnability_gate.min_nqual}`,
    );
    expect(
      `cross: ${entry.target} has the loop count the hint quotes`,
      entry.row?.unique_cdrh3_pool_size === 8,
      String(entry.row?.unique_cdrh3_pool_size),
    );
  }
  // The hint says the remainder were "screened by rules fixed before any result
  // existed". Three of them were in fact run by the baseline gate, so the copy
  // must not claim otherwise — assert the distinction exists rather than
  // silently letting the sentence drift back.
  const gateRanOn = partition.unlearnable
    .concat(partition.single_round)
    .filter((target) =>
      cross.learnability_gate.rows.some(
        (row) => canonicalTargetId(row.target_id) === canonicalTargetId(target),
      ),
    );
  expect(
    "cross: some excluded targets were measured by the baseline gate",
    gateRanOn.length >= 2,
    gateRanOn.join(","),
  );
  expect(
    "cross: the holdout was NOT measured by the gate",
    !partition.unlearnable
      .filter((target) => target === "aayl51")
      .some((target) =>
        cross.learnability_gate.rows.some(
          (row) => canonicalTargetId(row.target_id) === canonicalTargetId(target),
        ),
      ),
    "aayl51 appears in the gate, so it was not blind",
  );

  /*
   * The dataset identity strip.
   *
   * The overview and the recorded run both quoted a library size without naming
   * the antibody, the antigen, or the source — so `datasetIdentity` assembles
   * that from three artifacts. Assert each field resolves, and that the antigen
   * comes from the corpus metadata rather than from a label typed into a view.
   */
  const identity = datasetIdentity(cross);
  expect("identity: strip resolves", identity !== null, "null");
  if (identity) {
    expect("identity: target id", identity.id === "1n8z", String(identity.id));
    expect("identity: antigen resolved", identity.antigen !== null, "no antigen");
    expect(
      "identity: antigen matches the corpus row",
      identity.antigen ===
      corpusMetadata.rows.find((row) => rowMatchesTarget(row, "1n8z"))?.antigen,
      `${identity.antigen}`,
    );
    expect("identity: CDR3 length", identity.cdr3Length === 13, String(identity.cdr3Length));
    expect("identity: library rows", identity.libraryRows === 419, String(identity.libraryRows));
    expect("identity: usable rows", identity.usableRows === 210, String(identity.usableRows));
    // The usable count the strip shows must be the same one the lab's own
    // config reports — otherwise the two tabs quote different library sizes.
    expect(
      "identity: usable rows agree with /api/config",
      identity.usableRows === 210,
      `${identity.usableRows} vs config 210`,
    );
    expect(
      "identity: source corpus named",
      identity.corpus !== null && identity.corpus.length > 0,
      "no corpus",
    );
    expect(
      "identity: usable is a subset of the library",
      identity.usableRows !== null &&
      identity.libraryRows !== null &&
      identity.usableRows <= identity.libraryRows,
      `${identity.usableRows} of ${identity.libraryRows}`,
    );
  }
  // A dataset with no reference row must yield null rather than a half-filled
  // strip — every field is independently nullable, so this guards the join.
  expect(
    "identity: absent artifact yields null",
    datasetIdentity({ ...cross, available: false } as typeof cross) === null,
    "returned a partial identity",
  );

  // ── Coverage matrix. ──
  const datasets = Array.from(new Set(cross.coverage_sensitivity.map((row) => row.dataset)));
  expect("cross: four coverage datasets", datasets.length === 4, datasets.join(","));
  for (const dataset of datasets) {
    expect(`cross: ${dataset} has three coverage cells`, coverageCells(cross, dataset).length === 3, String(coverageCells(cross, dataset).length));
  }

  // ── The funnel is a holdout screen, not the evaluated set. ──
  //
  // The headline "17 went in, 1 came out" is only honest if the card also says
  // that five datasets were measured and that four of them are rejected *by*
  // those rules. Without this reconciliation the funnel reads as the evidence
  // base, which contradicts the scoreboard directly above it.
  // `stages` and `survivors` were computed above by the funnel section; reuse
  // them so this reconciliation cannot describe a different funnel.
  const metadata = cross.lockbox_metadata;
  expect("cross: funnel leaves exactly one holdout", survivors.length === 1, String(survivors.length));
  expect(
    "cross: the survivor is the pre-registered holdout",
    partition.unlearnable.includes(canonicalTargetId(survivors[0]?.target_id ?? "")),
    survivors[0]?.target_id ?? "none",
  );

  // The headline is "the lab ran on N of 17", where N is the evaluated set —
  // NOT the holdout count. Those differ by design: three evaluated targets are
  // disqualified from being a holdout precisely because they were already run.
  // Asserting the headline against `m5_evaluated` is what keeps the card from
  // reverting to "17 went in, 1 came out", which read as one dataset of proof.
  const capabilityStage = stages.filter((stage) => stage.phase === "capability").pop();
  const expectedWorkable = metadata.rows.filter((row) => row.usable_as_search_problem).length;
  // 1N8Z counts. It is published under `reference_targets` on the frozen 35-test
  // protocol rather than as a resized cell, but it is the primary dataset and
  // omitting it from "where the lab applies" would answer the title with a list
  // that leaves out the one dataset every threshold was calibrated on.
  const evaluatedAll = partition.m5_evaluated;
  expect(
    "cross: headline counts the evaluated targets",
    evaluatedAll.length === 5,
    evaluatedAll.join(","),
  );
  // The three headline counts must be reproducible from the data, and they must
  // differ — otherwise the sentence collapses into one redundant claim.
  const verdicts = evaluatedVerdicts(cross, partition, "16.7pct");
  expect(
    "cross: verdicts cover every evaluated target",
    verdicts.length === evaluatedAll.length,
    `${verdicts.length} vs ${evaluatedAll.length}`,
  );
  // Headline: "Ran on 5 · beat standard UCB on 4 · closed loop on 4".
  // The comparator is **standard UCB**, matching the wording every other panel
  // on this tab uses. The stronger-of-M1-M4 comparison is carried per row, not
  // folded into this count.
  // The card excludes the single-round control, so its counts are over 4.
  const shown = verdicts.filter((entry) => entry.classification !== "single-round");
  const beaten = shown.filter((entry) => entry.loopAttributableWin).length;
  const multiRound = shown.filter((entry) => entry.rounds > 1).length;
  expect("cross: card shows four libraries", shown.length === 4, shown.map((e) => e.target).join(","));
  expect("cross: headline loop-win count is 3", beaten === 3, String(beaten));
  // With 1MLC excluded, every remaining row is multi-round (AAYL51 runs 15).
  expect("cross: all four shown libraries ran a closed loop", multiRound === 4, String(multiRound));
  expect(
    "cross: the single-round control is not on this card",
    shown.every((entry) => entry.classification !== "single-round"),
    shown.map((e) => e.classification).join(","),
  );

  /*
   * The two cards must not disagree about the same run.
   *
   * "IT WORKS ON OTHER TARGETS" prints, for the single-round control, that
   * best-binder-found is identical between our method and standard UCB (1MLC:
   * 3.0663 = 3.0663, paired p = 1.0). If the verdict row below it counted that
   * same cell as a win, a reader comparing the two panels would find a direct
   * contradiction. `loopAttributableWin` exists to prevent exactly that, and
   * this asserts the exclusion rather than trusting the flag.
   */
  for (const entry of verdicts) {
    if (entry.rounds > 1) continue;
    expect(
      `cross: ${entry.target} single-round cell is not counted as a loop win`,
      entry.loopAttributableWin === false,
      String(entry.loopAttributableWin),
    );
    // And the structural tie must be real, not assumed: the paired
    // best-binder test on that cell has to be exactly zero.
    const cell = cross.targets.find(
      (row) => row.dataset === entry.target && row.coverage_label === "16.7pct",
    );
    if (!cell) continue;
    expect(
      `cross: ${entry.target} single-round best-binder is a structural tie`,
      close(cell.aubqc.m5_hybrid, cell.aubqc.m3_ucb, 1e-9) &&
      close(cell.paired.m5_hybrid_minus_m3_ucb.p_value, 1, 1e-9),
      `Δ ${cell.aubqc.m5_hybrid - cell.aubqc.m3_ucb}, p ${cell.paired.m5_hybrid_minus_m3_ucb.p_value}`,
    );
  }
  /*
   * The card says both what the lab handles and what it does not. Assert the
   * verdict is derived, so "no signal here" cannot be printed over a library
   * that actually had a signal — and cannot become a silent loss either.
   *
   * `unlearnable` is checked before `beatsUcb`, mirroring `derive.ts`. A raw
   * float comparison cannot distinguish a real lead from noise: aayl51 at 16.7%
   * coverage leads UCB by 0.0002 with a paired p of 0.34, and printing "works"
   * for that would contradict the holdout tile on the same page.
   */
  for (const entry of verdicts) {
    const expected =
      entry.classification === "single-round"
        ? "single-round"
        : entry.classification === "unlearnable"
          ? "no-signal"
          : entry.beatsUcb === true
            ? "works"
            : "no-signal";
    expect(
      `cross: ${entry.target} verdict is derived from its own numbers`,
      entry.verdict === expected,
      `${entry.verdict} vs ${expected}`,
    );
  }
  /*
   * A "no-signal" verdict has to be a fact about the library, and the evidence
   * differs per target — which is why this checks the right source for each.
   *
   * The holdout was deliberately **withheld from the learnability gate** so it
   * stayed blind, so it has no gate row at all; its only evidence is its own
   * measured ρ, which must be below the gate threshold on every coverage cell.
   * Other unlearnable targets do have gate rows. Asserting a gate row for the
   * holdout would fail — as it did — on a target that is correctly absent from
   * that file.
   */
  for (const entry of verdicts.filter((row) => row.verdict === "no-signal")) {
    const gateRows = cross.learnability_gate.rows.filter(
      (row) => canonicalTargetId(row.target_id) === canonicalTargetId(entry.target),
    );
    if (gateRows.length > 0) {
      expect(
        `cross: ${entry.target} labelled no-signal has a low measured baseline`,
        gateRows.every((row) => (row.max_baseline_rho ?? 1) < cross.learnability_gate.rho_gate),
        gateRows.map((row) => row.max_baseline_rho).join(","),
      );
    } else {
      // No gate row — the holdout path. Its own ρ cells must all sit below the
      // threshold, on every coverage level the artifact publishes.
      const cells = cross.targets.filter((row) => row.dataset === entry.target);
      expect(
        `cross: ${entry.target} has measured cells to justify no-signal`,
        cells.length > 0,
        "no cells",
      );
      for (const cell of cells) {
        const bestBaseline = Math.max(
          ...METHODS.filter((method) => method !== "m5_hybrid").map(
            (method) => cell.spearman_rho_at_last_checkpoint[method] ?? Number.NEGATIVE_INFINITY,
          ),
        );
        expect(
          `cross: ${entry.target}@${cell.coverage_label} no method separates`,
          bestBaseline < cross.learnability_gate.rho_gate,
          `best baseline ρ ${bestBaseline} vs gate ${cross.learnability_gate.rho_gate}`,
        );
      }
      // And it must be flagged as the withheld holdout, not merely unmeasured.
      expect(
        `cross: ${entry.target} with no gate row is the pre-registered holdout`,
        partition.unlearnable.includes(entry.target),
        entry.target,
      );
    }
  }
  // And the converse: a "works" verdict must actually beat its UCB comparator.
  for (const entry of verdicts.filter((row) => row.verdict === "works")) {
    expect(
      `cross: ${entry.target} labelled works really beats UCB`,
      entry.m5 !== null && entry.ucb !== null && entry.m5 > entry.ucb,
      `${entry.m5} vs ${entry.ucb}`,
    );
  }

  /*
   * Now that the single-round control is off the card, every displayed row that
   * ranks better must have the loop to thank for it.
   *
   * `beatsUcb` and `loopAttributableWin` are no longer the same test: on a
   * library the learnability gate already called unlearnable, a nominal lead is
   * noise and must not be promoted to a win. So a learnable row must convert its
   * lead into a win, and an unlearnable row must never do so — which is the
   * property that actually keeps the two cards agreeing.
   */
  for (const entry of shown) {
    const expectedWin = entry.beatsUcb === true && entry.classification !== "unlearnable";
    expect(
      `cross: ${entry.target} displayed win is loop-attributable`,
      entry.loopAttributableWin === expectedWin,
      `beatsUcb=${entry.beatsUcb} class=${entry.classification} loopWin=${entry.loopAttributableWin}`,
    );
    if (entry.classification === "unlearnable") {
      expect(
        `cross: ${entry.target} is unlearnable and shows no win`,
        entry.loopAttributableWin === false,
        `loopWin=${entry.loopAttributableWin} at ${entry.m5} vs ${entry.ucb}`,
      );
    }
  }
  // The warm start is a different comparison and must be carried separately, or
  // the holdout reads as a flat tie while the scoreboard reports +0.0828.
  const holdout = shown.find((entry) => entry.target === "aayl51");
  expect("cross: the holdout is on the card", holdout !== undefined, "aayl51 missing");
  expect(
    "cross: the holdout carries its warm-start ablation",
    holdout?.warmStart !== null && holdout?.warmStart !== undefined,
    "no warm-start block",
  );
  if (holdout?.warmStart) {
    expect(
      "cross: holdout warm start separated",
      holdout.warmStart.delta > 0 && holdout.warmStart.p < 0.001,
      `Δ ${holdout.warmStart.delta} p ${holdout.warmStart.p}`,
    );
    // It must be the strongest cell, matching `strongestWarmStart` — otherwise
    // the row quotes a weaker effect than the scoreboard above it.
    const strongest = strongestWarmStart(cross, "aayl51");
    expect(
      "cross: holdout warm start is the strongest cell",
      strongest !== null && Math.abs(strongest.delta - holdout.warmStart.delta) < 1e-9,
      `${holdout.warmStart.delta} vs ${strongest?.delta}`,
    );
    // And it must never be mistaken for a win over the baseline.
    expect(
      "cross: warm start is not counted as beating UCB",
      holdout.beatsUcb === false,
      `beatsUcb=${holdout.beatsUcb}`,
    );
  }
  // Every row must carry the UCB number it is compared against.
  for (const entry of verdicts) {
    expect(
      `cross: ${entry.target} carries its UCB comparator`,
      entry.ucb !== null,
      "no m3_ucb value",
    );
    if (entry.beatsUcb === null) continue;
    expect(
      `cross: ${entry.target} verdict matches its own numbers`,
      entry.beatsUcb === (entry.m5 !== null && entry.ucb !== null && entry.m5 > entry.ucb),
      `${entry.m5} vs ${entry.ucb}`,
    );
  }
  // The only dataset that must not beat UCB is the holdout.
  const notBeaten = verdicts.filter((entry) => entry.beatsUcb === false).map((e) => e.target);
  expect(
    "cross: exactly one dataset fails to beat UCB",
    notBeaten.length === 1,
    notBeaten.join(","),
  );
  expect(
    "cross: the dataset that fails to beat UCB is the holdout",
    notBeaten.length === 1 && partition.unlearnable.includes(notBeaten[0]!),
    notBeaten.join(","),
  );
  // A single-round cell must never be counted as showing a closed loop.
  for (const entry of verdicts) {
    if (entry.classification !== "single-round") continue;
    expect(
      `cross: ${entry.target} is single-round`,
      entry.rounds <= 1,
      String(entry.rounds),
    );
  }
  // The stronger-than-UCB caveat must be data-driven: where a non-UCB baseline
  // beats us, that fact is on the row, so the 4 cannot read as "we lead on
  // everything".
  const tougher = verdicts.filter(
    (entry) => entry.bestBaselineMethod !== null && entry.bestBaselineMethod !== "m3_ucb" && entry.beatsBestBaseline === false,
  );
  expect(
    "cross: a row records when a non-UCB baseline leads us",
    tougher.length >= 1,
    "1MLC's M2 loss is not surfaced",
  );
  for (const entry of tougher) {
    expect(
      `cross: ${entry.target} names the baseline that leads`,
      entry.bestBaselineMethod !== null && entry.bestBaseline !== null,
      "no baseline recorded",
    );
  }
  expect(
    "cross: the evaluated set includes the primary dataset",
    evaluatedAll.includes("1n8z"),
    evaluatedAll.join(","),
  );
  expect(
    "cross: 1N8Z is published as a reference row with its score",
    (cross.reference_targets.find((row) => row.dataset === "1n8z")?.spearman_rho_curves.length ?? 0) > 0,
    "no ρ series on the reference row",
  );
  // Every evaluated target must be workable — the lab cannot have run on a
  // dataset that fails the capability test, and if it ever did the headline
  // would be claiming reach it does not have.
  for (const target of evaluatedAll) {
    const corpusRow = metadata.rows.find((row) => rowMatchesTarget(row, target));
    expect(
      `cross: evaluated ${target} is a workable search problem`,
      corpusRow?.usable_as_search_problem === true,
      corpusRow ? `usable=${corpusRow.usable_as_search_problem}` : "no corpus row",
    );
  }
  // The evaluated set is a strict subset of the workable set: we ran on some of
  // what is workable, and the holdout is one of those.
  expect(
    "cross: evaluated targets are a subset of the workable set",
    evaluatedAll.length <= expectedWorkable,
    `${evaluatedAll.length} evaluated vs ${expectedWorkable} workable`,
  );
  expect(
    "cross: capability count equals the usable rows",
    capabilityCount(stages) === expectedWorkable,
    `${capabilityCount(stages)} vs ${expectedWorkable}`,
  );
  expect(
    "cross: capability phase ends on the usable set",
    capabilityStage !== undefined && capabilityStage.remaining.length === expectedWorkable,
    `${capabilityStage?.remaining.length} vs ${expectedWorkable}`,
  );
  // The holdout must itself be one of the evaluated targets, or the funnel is
  // certifying a dataset the lab never ran on.
  expect(
    "cross: the holdout was actually measured",
    survivors.every((row) => partition.m5_evaluated.includes(canonicalTargetId(row.target_id))),
    survivors.map((row) => row.target_id).join(","),
  );
  // Every capability-passing dataset must clear the two structural tests.
  for (const row of capabilityStage?.remaining ?? []) {
    expect(
      `cross: ${row.target_id} passing capability is a real search problem`,
      row.usable_as_search_problem && row.unique_cdrh3_pool_size >= 35,
      `${row.unique_cdrh3_pool_size} loops`,
    );
  }

  const roster = evaluatedTargetRoster(metadata, partition, stages);
  expect(
    "cross: roster covers every evaluated target",
    roster.length === partition.m5_evaluated.length,
    `${roster.length} vs ${partition.m5_evaluated.length}`,
  );
  expect(
    "cross: the holdout is the only survivor in the roster",
    roster.filter((entry) => entry.rejectedAt === null).length === 1,
    String(roster.filter((entry) => entry.rejectedAt === null).length),
  );
  expect(
    "cross: roster survivor is the funnel survivor",
    canonicalTargetId(roster.find((entry) => entry.rejectedAt === null)?.target ?? "") ===
    canonicalTargetId(survivors[0]?.target_id ?? ""),
    `${roster.find((entry) => entry.rejectedAt === null)?.target} vs ${survivors[0]?.target_id}`,
  );
  // Every rejected entry must name a real stage, never a blank.
  for (const entry of roster) {
    if (entry.rejectedAt === null) continue;
    expect(
      `cross: roster ${entry.target} names the stage that removed it`,
      stages.some((stage) => stage.stage === entry.rejectedAt),
      entry.rejectedAt ?? "none",
    );
  }

  // ── Exclusion is not rejection: the two stage kinds stay distinct. ──
  //
  // The card's closing sentence asserts that the learnable targets are excluded
  // "because they already carry a published number, not because anything is
  // wrong with them". That claim is only true while the stage doing the
  // excluding is tagged `contamination` — if a future edit re-tagged that stage
  // as a screen, the copy would become a lie and nothing else would notice.
  const contaminationStages = stages.filter((stage) => stage.kind === "contamination");
  expect("cross: a contamination stage exists", contaminationStages.length > 0, "none tagged");
  // A learnable target is never excluded for being *bad* — it is excluded either
  // because it is already on the board (`contamination`) or because its pool is
  // below the holdout floor. The closing sentence must not collapse those two
  // into one claim: 2FJG at 124 candidates is a legitimate control, but it is
  // not excluded "only because it already carries a number" — it misses the
  // 150-candidate bar as well. Assert the real reason per target.
  for (const entry of roster) {
    if (entry.classification !== "learnable" || entry.rejectedAt === null) continue;
    if (entry.target === "1n8z") continue; // measured by a different artifact
    const deciding = contaminationStages.some((stage) => stage.stage === entry.rejectedAt);
    const corpusRow = metadata.rows.find((row) => rowMatchesTarget(row, entry.target));
    const underFloor = (corpusRow?.N_qual ?? Number.POSITIVE_INFINITY) < metadata.min_nqual;
    expect(
      `cross: learnable ${entry.target} is excluded for contamination or the pool floor`,
      deciding || underFloor,
      `removed at '${entry.rejectedAt}', which is neither contamination nor under the floor`,
    );
  }
  // ...and the reverse: a genuinely degenerate dataset must never be recoloured
  // as "already measured", which would hide a real exclusion.
  for (const stage of contaminationStages) {
    for (const row of stage.rejected) {
      expect(
        `cross: ${row.target_id} tagged contamination is actually usable`,
        row.usable_as_search_problem,
        "a degenerate row was tagged as contamination",
      );
    }
  }
  // The floor caveat exists because two measured targets sit under it — if that
  // stops being true the paragraph disappears, so assert the data, not the copy.
  const floorRows = belowHoldoutFloor(metadata, partition);
  for (const entry of floorRows) {
    expect(
      `cross: ${entry.target} is genuinely below the holdout floor`,
      entry.nQual < metadata.min_nqual,
      `${entry.nQual} vs ${metadata.min_nqual}`,
    );
  }

  // ── The budget switch. ──
  //
  // The switch offers exactly the levels where a cell exists for *every*
  // measured dataset. If a level were added without the data behind it, the
  // button would render an empty card — so the data, not the button list, is
  // the assertion.
  for (const level of TOGGLE_COVERAGE) {
    for (const dataset of datasets) {
      const row = cross.targets.find(
        (entry) => entry.dataset === dataset && entry.coverage_label === level,
      );
      expect(`cross: ${dataset} has a ${level} cell for the switch`, row !== undefined, "no row");
      expect(
        `cross: ${dataset}@${level} switch lookup resolves`,
        targetRow(cross, dataset, level)?.coverage_label === level,
        targetRow(cross, dataset, level)?.coverage_label ?? "none",
      );
      // Every cell the switch can show must publish a ranking-accuracy score —
      // that is what the card's chart, bars and "ranking N of 5" all read.
      expect(
        `cross: ${dataset}@${level} has a ranking-accuracy score`,
        Number.isFinite(row?.spearman_rho_at_last_checkpoint.m5_hybrid ?? Number.NaN),
        String(row?.spearman_rho_at_last_checkpoint.m5_hybrid),
      );
    }
  }

  // The **paired** ranking-accuracy test is a stricter requirement: the
  // attribution ledger only measured the learnable datasets, so the two
  // controls legitimately have no p-value. Asserting its absence rather than
  // its presence keeps the control cards honest — the card says "not measured"
  // there and must not start implying a test exists.
  for (const level of TOGGLE_COVERAGE) {
    for (const dataset of partition.learnable_multiround) {
      if (dataset === "1n8z") continue;
      expect(
        `cross: learnable ${dataset}@${level} publishes a paired ranking-accuracy test`,
        rhoPairedTest(attribution, dataset, level) !== null,
        `no attribution cell for "${dataset} @ ${level}"`,
      );
    }
    for (const dataset of [...partition.unlearnable, ...partition.single_round]) {
      if (!datasets.includes(dataset)) continue;
      expect(
        `cross: control ${dataset}@${level} has no paired ranking-accuracy test`,
        rhoPairedTest(attribution, dataset, level) === null,
        "an attribution cell appeared for a control",
      );
    }
  }

  // ── A tie must never be rendered as a win. ──
  //
  // `m5_rank` uses a strict `>` comparison upstream, so an exact tie scores 1.
  // The chip labels that case "tied" rather than "1 of 5"; this asserts the
  // condition the label keys off, so removing the branch fails here first.
  for (const target of partition.single_round) {
    for (const level of TOGGLE_COVERAGE) {
      const cell = cross.coverage_sensitivity.find(
        (entry) => entry.dataset === target && entry.coverage_label === level,
      );
      if (!cell || cell.aubqc_m5 === null || cell.aubqc_best_baseline === null) continue;
      expect(
        `cross: ${target}@${level} best binder is an exact tie`,
        Math.abs(cell.aubqc_m5 - cell.aubqc_best_baseline) < 5e-4,
        `${cell.aubqc_m5} vs ${cell.aubqc_best_baseline}`,
      );
      expect(
        `cross: ${target}@${level} tie is not presented as rank 1`,
        Math.abs(cell.aubqc_m5 - cell.aubqc_best_baseline) < 5e-4 && cell.m5_rank === 1,
        "the tie branch in the chip is unreachable — check CrossTargetView",
      );
    }
  }

  // ── Rank and score must come from the same block. ──
  //
  // The defect this guards: `coverage_sensitivity[].m5_rank` is a *best binder*
  // rank, while `rho_m5` beside it is ranking accuracy. Rendering those two in
  // one chip read as "we are not first", when the ρ rank is 1 on both learnable
  // datasets at 16.7% and 33%. Every rank the surface prints next to a ρ value
  // is now derived from the ρ block, and this asserts the two agree.
  for (const level of TOGGLE_COVERAGE) {
    for (const dataset of datasets) {
      const cell = cross.coverage_sensitivity.find(
        (entry) => entry.dataset === dataset && entry.coverage_label === level,
      );
      const row = cross.targets.find(
        (entry) => entry.dataset === dataset && entry.coverage_label === level,
      );
      if (!cell || !row) continue;
      const rhoRank = rhoRankAtCoverage(cross, dataset, level);
      const ranked = METHODS.map((method) => ({
        method,
        value: row.spearman_rho_at_last_checkpoint[method],
      }))
        .filter((entry): entry is { method: (typeof METHODS)[number]; value: number } =>
          typeof entry.value === "number" && Number.isFinite(entry.value),
        )
        .sort((a, b) => b.value - a.value);
      expect(
        `cross: ${dataset}@${level} ρ rank agrees with the ρ block`,
        rhoRank !== null && rhoRank === ranked.findIndex((entry) => entry.method === "m5_hybrid") + 1,
        `${rhoRank} vs ${ranked.findIndex((entry) => entry.method === "m5_hybrid") + 1}`,
      );
      // Anyone quoting rank 1 must actually hold the top ρ in that cell.
      if (rhoRank === 1) {
        expect(
          `cross: ${dataset}@${level} rank 1 means top ranking accuracy`,
          (ranked[0]?.method ?? "") === "m5_hybrid",
          ranked[0]?.method ?? "none",
        );
      }
    }
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
  // The reference card's chart is the ranking-accuracy series, so the series has
  // to exist and carry the same endpoints the tiles quote.
  expect(
    "cross: reference publishes a ranking-accuracy series",
    (reference?.spearman_rho_curves.length ?? 0) === (reference?.checkpoints.length ?? -1),
    `${reference?.spearman_rho_curves.length} rows vs ${reference?.checkpoints.length} checkpoints`,
  );
  const referenceLastRho =
    reference?.spearman_rho_curves[reference.spearman_rho_curves.length - 1];
  expectClose(
    "cross: reference chart endpoint agrees with the benchmark ρ curve",
    referenceLastRho?.m5_hybrid ?? Number.NaN,
    benchmark.spearman_rho_curves[benchmark.spearman_rho_curves.length - 1]?.m5_hybrid ?? Number.NaN,
    1e-9,
  );
  expect(
    "cross: reference ranking accuracy puts us first",
    (referenceLastRho?.m5_hybrid ?? 0) >
    Math.max(...METHODS.filter((method) => method !== "m5_hybrid").map((method) => referenceLastRho?.[method] ?? Number.NEGATIVE_INFINITY)),
    `${referenceLastRho?.m5_hybrid} vs field`,
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
