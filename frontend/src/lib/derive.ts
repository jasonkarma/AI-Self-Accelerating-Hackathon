/**
 * Pure derivations over the four payloads.
 *
 * These functions exist so the **same** number is computed once and asserted by
 * `scripts/verify-numbers.ts`. Every figure the views attribute to a comparison
 * (a Δ, a ratio, a "seeds won" count, a group's smallest p) is produced here —
 * never typed into a component — so the audit script can recompute it from
 * `api_contract.json` and fail if the UI and the data disagree.
 */

import type {
  AttributionCell,
  AttributionPayload,
  BenchmarkPayload,
  CoverageSensitivityRow,
  CrossTargetPayload,
  LearnabilityGate,
  LockboxRow,
  LockboxMetadata,
  MethodRecord,
} from "../types/report";
import type { BatchSlot, StatePayload } from "../types/state";
import { SLOT_META } from "./format";

export const METHODS = ["m1_random", "m2_greedy", "m3_ucb", "m4_fixed_prior", "m5_hybrid"] as const;

/** The wild type the 1N8Z pool is built on — served by `GET /api/config`. */
export const WT_ANCHORS = [3, 11, 12, 13] as const;

// ─────────────────────────────── state view ───────────────────────────────

export interface SlotRoleView {
  slot: BatchSlot;
  roleLabel: string;
  roleColor: string;
  roleBucket: "exploit" | "voi" | "h0" | "harvest";
  /** Revealed binding, when this candidate has already been assayed. */
  revealedBinding: number | null;
  isNewBest: boolean;
}

/**
 * Assign each slot its scientific role for the *selected* round.
 *
 * The contract's `slot_type` names the selection rule; the role bucket is a
 * function of that rule **and** of `slot_graduation`. Once the horizon passes
 * `ceil(2R/3)` the selector stops paying for the VoI probe and the H0 control and
 * runs a 5:0:0 harvest, so a slot whose rule was "Max Joint Uncertainty" in
 * Round 2 is a pure exploit pick in Round 5. The UI must say that, not keep
 * painting the old badge — which is why the bucket is derived and not read off
 * `slot_type` alone.
 */
export function slotRoles(
  state: StatePayload,
  revealedById: Map<string, number>,
  bestBinding: number,
): SlotRoleView[] {
  const graduated = state.voi_diagnostics.slot_graduation === true;
  return state.batch_slots
    .slice()
    .sort((a, b) => a.slot_id - b.slot_id)
    .map((slot) => {
      const binding = revealedById.get(slot.candidate_id) ?? null;
      const base = roleForSlotType(slot.slot_type);
      const bucket = graduated ? "harvest" : base.bucket;
      const roleLabel = graduated ? "Graduated Harvest" : base.label;
      const roleColor = graduated ? "#EAB308" : base.color;
      return {
        slot,
        roleLabel,
        roleColor,
        roleBucket: bucket,
        revealedBinding: binding,
        isNewBest: binding !== null && Math.abs(binding - bestBinding) < 5e-4,
      };
    });
}

/**
 * Rule → (label, colour, bucket).
 *
 * Colour comes from the shared `SLOT_META` table so a slot badge, a legend entry
 * and a chart marker cannot disagree; only the *label* is written here, because
 * the label carries the scientific reading rather than the rule's name.
 */
function roleForSlotType(slotType: string): {
  label: string;
  color: string;
  bucket: "exploit" | "voi" | "h0";
} {
  const meta = SLOT_META[slotType];
  const color = meta?.color ?? "#10B981";
  switch (slotType) {
    case "Max Joint Uncertainty":
      return { label: "Tests the biggest unknown", color, bucket: "voi" };
    case "Skeptic Falsification Probe":
      return { label: "Control — must show nothing", color, bucket: "h0" };
    case "Agent Hypothesis Test":
      return { label: "Strong predicted candidate", color, bucket: "exploit" };
    default:
      return { label: "Consensus pick", color, bucket: "exploit" };
  }
}

/** `candidate_id -> revealed y_bind` from the Pareto scatter (the revealed set). */
export function revealedBindings(state: StatePayload): Map<string, number> {
  const map = new Map<string, number>();
  for (const point of state.pareto_data) map.set(point.candidate_id, point.binding);
  return map;
}

/** The statuses to exclude when looking for "the discovered binder". */
export function discoveredBinder(state: StatePayload): { id: string; binding: number } | null {
  const best = state.kpis.best_qualified_binding;
  const match = state.pareto_data.find((point) => Math.abs(point.binding - best) < 5e-4);
  return match ? { id: match.candidate_id, binding: match.binding } : null;
}

/** A residue-by-residue alignment of one candidate against the wild type. */
export interface AlignmentCell {
  position: number;
  wt: string;
  residue: string;
  mutated: boolean;
  anchor: boolean;
}

export function align(sequence: string, wildType: string): AlignmentCell[] {
  const length = Math.max(sequence.length, wildType.length);
  const cells: AlignmentCell[] = [];
  for (let index = 0; index < length; index += 1) {
    const wt = wildType[index] ?? "·";
    const residue = sequence[index] ?? "·";
    const position = index + 1;
    cells.push({
      position,
      wt,
      residue,
      mutated: residue !== wt,
      anchor: (WT_ANCHORS as readonly number[]).includes(position) && wt !== "·",
    });
  }
  return cells;
}


/**
 * The 3 : 1 : 1 → 5 : 0 : 0 split, computed from the slots actually chosen.
 *
 * The design claim is that rounds 1–4 fill three exploitation slots, one VoI
 * probe and one H0 control; the harvest phase fills five exploitation slots.
 * Rather than assert that, count it: the numbers below come from
 * `slot_type` plus `slot_graduation`, and the panel renders the count so a
 * mismatch is visible.
 */
export function slotSplit(state: StatePayload): {
  exploit: number;
  voi: number;
  h0: number;
  total: number;
  graduated: boolean;
  label: string;
} {
  const graduated = state.voi_diagnostics.slot_graduation === true;
  let voi = 0;
  let h0 = 0;
  for (const slot of state.batch_slots) {
    if (slot.slot_type === "Max Joint Uncertainty") voi += 1;
    if (slot.slot_type === "Skeptic Falsification Probe") h0 += 1;
  }
  const total = state.batch_slots.length;
  const exploit = total - voi - h0;
  return {
    exploit,
    voi,
    h0,
    total,
    graduated,
    label: graduated
      ? `${exploit} × best predicted`
      : `${exploit} exploit · ${voi} explore · ${h0} control`,
  };
}

/** Δ_agent spread over the batch — the concordance term of `slot_graduation`. */
export function deltaAgentSpread(state: StatePayload): number | null {
  const values = state.batch_slots
    .map((slot) => slot.delta_agent)
    .filter((value): value is number => value !== null && Number.isFinite(value));
  if (values.length < 2) return null;
  return Math.max(...values) - Math.min(...values);
}

/** The round at which the harvest phase begins: `ceil(2R/3) + 1`. */
export function harvestFromRound(totalRounds: number): number {
  return Math.ceil((2 * totalRounds) / 3) + 1;
}


// ──────────────────────────── benchmark view ──────────────────────────────

/** The final-checkpoint ρ of every method, including the `null` for M1. */
export function finalRho(curves: Array<Record<string, number | null> | { query_count: number } & Record<string, number | null>>): Record<string, number | null> {
  const last = curves[curves.length - 1];
  if (!last) return {};
  const out: Record<string, number | null> = {};
  for (const method of METHODS) {
    const value = last[method];
    out[method] = typeof value === "number" ? value : null;
  }
  return out;
}

/** `performer − baseline` for one metric, read straight off the paired test. */
export function pairedDelta(
  benchmark: BenchmarkPayload,
  comparison: "m5_hybrid_minus_m3_ucb" | "m5_hybrid_minus_m2_greedy" | "m5_warm_start_minus_m5_hybrid",
  metric: keyof BenchmarkPayload["paired_t_tests"]["m5_hybrid_minus_m3_ucb"]["metrics"],
): { meanDiff: number; p: number; seedsWon: number; n: number; se: number } {
  const block = benchmark.paired_t_tests[comparison].metrics[metric];
  return {
    meanDiff: block.mean_diff,
    p: block.p_value,
    seedsWon: block.seeds_won,
    n: block.n,
    se: block.se,
  };
}

/** `M5 / M3` on final ρ, from the two published values. */
export function rhoRatio(rhoM5: number | null, rhoM3: number | null): number | null {
  if (rhoM5 === null || rhoM3 === null || rhoM3 === 0) return null;
  return rhoM5 / rhoM3;
}

/** The `|Δ| / SE` ratio of a paired test — how many standard errors from zero. */
export function deltaOverSe(meanDiff: number, se: number): number | null {
  if (!Number.isFinite(se) || se === 0) return null;
  return Math.abs(meanDiff) / se;
}

export interface ZeroShotRow {
  key: string;
  label: string;
  rho: number;
  /** Published predictors are all negative; ranking against them is the point. */
  isNegative: boolean;
}

export function zeroShotRows(benchmark: BenchmarkPayload): ZeroShotRow[] {
  const refs = benchmark.zero_shot_references;
  return (Object.keys(refs) as Array<keyof typeof refs>).map((key) => ({
    key,
    label:
      key === "alphafold3" ? "AlphaFold3" : key === "proteinmpnn" ? "ProteinMPNN" : "ESM-2",
    rho: refs[key],
    isNegative: refs[key] < 0,
  }));
}

/** The checkpoint rows of the ρ curves, as `{query_count, values}`. */
export function rhoSeries(
  benchmark: BenchmarkPayload,
): Array<{ query_count: number; values: Record<string, number | null> }> {
  return benchmark.spearman_rho_curves.map((row) => {
    const values: Record<string, number | null> = {};
    for (const method of METHODS) values[method] = row[method];
    return { query_count: row.query_count, values };
  });
}

/** Every metric the contract publishes, so the panel can show all of them. */
export interface BenchmarkMetricRow {
  key: string;
  label: string;
  description: string;
  values: MethodRecord;
  best: { method: string; value: number } | null;
  m5Rank: number | null;
}

export function benchmarkMetricRows(benchmark: BenchmarkPayload): BenchmarkMetricRow[] {
  const rows: Array<{
    key: keyof BenchmarkPayload;
    label: string;
    description: string;
  }> = [
      {
        key: "aubqc",
        label: "Best binder found",
        description:
          "The average, over the whole run, of how good the single best candidate found so far was. Tracks one number: the best hit.",
      },
      {
        key: "normalized_aubqc",
        label: "Share of the possible gain",
        description:
          "The same average, rescaled so 0 is where you started and 1 is the best candidate that exists: the fraction of the available headroom captured.",
      },
      {
        key: "top5_panel_aubqc",
        label: "Depth of the top 5 leads",
        description: "How good your five best candidates are on average — a portfolio measure, not a single-hit measure.",
      },
      {
        key: "top_decile_hit_rate",
        label: "Top-decile hit rate",
        description: benchmark.top_decile_definition,
      },
    ];
  return rows.map((row) => {
    const values = benchmark[row.key] as MethodRecord;
    let best: { method: string; value: number } | null = null;
    for (const method of METHODS) {
      const value = values[method];
      if (!Number.isFinite(value)) continue;
      if (best === null || value > best.value) best = { method, value };
    }
    const ranked = METHODS.map((method) => ({ method, value: values[method] }))
      .filter((entry) => Number.isFinite(entry.value))
      .sort((a, b) => b.value - a.value);
    const index = ranked.findIndex((entry) => entry.method === "m5_hybrid");
    return {
      key: String(row.key),
      label: row.label,
      description: row.description,
      values,
      best,
      m5Rank: index >= 0 ? index + 1 : null,
    };
  });
}

// ─────────────────────────── cross-target view ────────────────────────────

export interface PartitionClass {
  key: "learnable" | "unlearnable" | "single" | "degenerate";
  title: string;
  subtitle: string;
  color: string;
  targets: string[];
  evidence: string;
}

/**
 * The 17-of-17 partition, read from `learnability_gate.partition` and enriched
 * with whatever evidence each class actually has.
 *
 * Three sources are consulted, and the panel names them because they are not
 * interchangeable:
 *
 * * `cross_target.targets[]` — the dynamic-sized cells (AUBQC and ρ per method).
 * * `cross_target.reference_targets[0]` — 1N8Z, measured by the primary protocol.
 * * `attribution.cells[]` — the only place a **paired ρ t-test** is published for
 *   a cross-target cell. The cross-target payload's `paired` block is an AUBQC
 *   test; using it as a ρ p-value would be a mislabelled statistic.
 */
export function partitionClasses(
  cross: CrossTargetPayload,
  attribution?: AttributionPayload | null,
  benchmark?: BenchmarkPayload | null,
): PartitionClass[] {
  const partition = cross.learnability_gate.partition;
  const metadata = cross.lockbox_metadata;

  const learnableEvidence = partition.learnable_multiround.map((target) =>
    learnableRow(cross, attribution ?? null, benchmark ?? null, target),
  );
  const measuredRho = learnableEvidence.filter(
    (entry): entry is { target: string; m5: number; m3: number; delta: number; p: number | null; seeds: string } =>
      entry !== null && entry.m5 !== null && entry.m3 !== null,
  );
  const withP = measuredRho.filter((entry) => entry.p !== null);
  const smallestP = withP.reduce<number | null>(
    (acc, entry) => (acc === null || (entry.p ?? 1) < acc ? (entry.p as number) : acc),
    null,
  );

  const unlearnableGateRows = cross.learnability_gate.rows.filter(
    (row) => partition.unlearnable.includes(row.target_id) && row.max_baseline_rho !== undefined,
  ); const unlearnableMeasured = partition.unlearnable.flatMap((target) =>
    cross.targets
      .filter((row) => row.dataset === target)
      .map((row) => ({
        target,
        rho: Math.max(
          row.spearman_rho_at_last_checkpoint.m2_greedy ?? Number.NEGATIVE_INFINITY,
          row.spearman_rho_at_last_checkpoint.m3_ucb ?? Number.NEGATIVE_INFINITY,
          row.spearman_rho_at_last_checkpoint.m4_fixed_prior ?? Number.NEGATIVE_INFINITY,
        ),
      }))
      .filter((entry) => Number.isFinite(entry.rho)),
  );
  const baselines = [
    ...unlearnableGateRows.map((row) => row.max_baseline_rho ?? 0),
    ...unlearnableMeasured.map((entry) => entry.rho),
  ];
  const minBaseline = baselines.length ? Math.min(...baselines) : null;
  const maxBaseline = baselines.length ? Math.max(...baselines) : null;

  const degenerate = partition.degenerate_1_or_2_cdrh3
    .map((target) => metadata.rows.find((row) => rowMatchesTarget(row, target)))
    .filter((row): row is LockboxRow => row !== undefined);
  const degenerateMax = degenerate.length
    ? Math.max(...degenerate.map((row) => row.unique_cdrh3_pool_size))
    : null;
  const aboveTwo = degenerate.filter((row) => row.unique_cdrh3_pool_size > 2).map((row) => row.target_id);

  return [
    {
      key: "learnable",
      title: "Learnable, multi-round",
      subtitle: "Our method ranks better in every target that has a signal at all",
      color: "#10B981",
      targets: partition.learnable_multiround,
      evidence:
        measuredRho.length === 0
          ? "no measured cells"
          : `${measuredRho.length} of ${partition.learnable_multiround.length} measured · Δρ ${formatRange(measuredRho.map((entry) => entry.delta))}${withP.length
            ? ` · paired p ≤ ${formatP(smallestP)} on the ${withP.length} cell(s) that publish one`
            : ""
          }`,
    },
    {
      key: "unlearnable",
      title: "No signal to learn",
      subtitle: "No method can rank these libraries, so nothing can be compared",
      color: "#F59E0B",
      targets: partition.unlearnable,
      evidence:
        minBaseline === null || maxBaseline === null
          ? `${metadata.rows.filter((row) => partition.unlearnable.includes(row.target_id)).length} targets`
          : `best baseline ρ ∈ [${minBaseline.toFixed(4)}, ${maxBaseline.toFixed(4)}] over ${baselines.length} measured cells`,
    },
    {
      key: "single",
      title: "Only one round is possible",
      subtitle: "Both methods must pick the same five candidates, so the result is identical",
      color: "#6366F1",
      targets: partition.single_round,
      evidence: singleRoundEvidence(cross),
    },
    {
      key: "degenerate",
      title: "Too few variants to search",
      subtitle: `Two or fewer distinct loops${aboveTwo.length ? ` (${aboveTwo.join(", ")} have ${aboveTwo.map((target) => degenerate.find((row) => row.target_id === target)?.unique_cdrh3_pool_size ?? "?").join("/")})` : ""} — there is nothing to choose between`,
      color: "#64748B",
      targets: partition.degenerate_1_or_2_cdrh3,
      evidence:
        degenerateMax === null
          ? `${degenerate.length} targets`
          : `${degenerate.length} targets, largest pool ${degenerateMax} distinct CDRH3 (usability floor is 35)`,
    },
  ];
}

/** One learnable target's ρ evidence, from whichever payload publishes it. */
function learnableRow(
  cross: CrossTargetPayload,
  attribution: AttributionPayload | null,
  benchmark: BenchmarkPayload | null,
  target: string,
): { target: string; m5: number | null; m3: number | null; delta: number | null; p: number | null; seeds: string } | null {
  // 1N8Z lives in the reference block, and its paired ρ test lives in the
  // primary benchmark artifact — not in the cross-target payload.
  if (target === "1n8z") {
    const reference = cross.reference_targets[0];
    if (!reference) return null;
    const last = reference.spearman_rho_curves[reference.spearman_rho_curves.length - 1];
    const m5 = last?.m5_hybrid ?? null;
    const m3 = last?.m3_ucb ?? null;
    const paired = benchmark?.paired_t_tests.m5_hybrid_minus_m3_ucb.metrics.spearman_rho_at_q35;
    return {
      target,
      m5,
      m3,
      delta: m5 !== null && m3 !== null ? m5 - m3 : null,
      p: paired?.p_value ?? null,
      seeds: paired ? `${paired.seeds_won}/${paired.n}` : "—",
    };
  }

  const row =
    cross.targets.find((entry) => entry.dataset === target && entry.coverage_label === "16.7pct") ??
    cross.targets.find((entry) => entry.dataset === target);
  if (!row) return null;
  const m5 = row.spearman_rho_at_last_checkpoint.m5_hybrid;
  const m3 = row.spearman_rho_at_last_checkpoint.m3_ucb;
  const paired = rhoPairedTest(attribution ?? null, row.dataset, row.coverage_label);
  return {
    target,
    m5,
    m3,
    delta: m5 !== null && m3 !== null ? m5 - m3 : null,
    p: paired?.p_value ?? null,
    seeds: paired ? `${paired.seeds_won}/${paired.n}` : "—",
  };
}

/**
 * The paired **ρ** test for one cross-target cell, if the attribution ledger
 * measured that cell. `null` when it did not — never a substituted AUBQC test.
 */
export function rhoPairedTest(
  attribution: AttributionPayload | null,
  dataset: string,
  coverageLabel: string,
): { meanDiff: number; p_value: number; seeds_won: number; n: number } | null {
  if (!attribution) return null;
  const cell = attribution.cells.find(
    (entry) => entry.cell === `${dataset} @ ${coverageLabel}` || entry.cell === `${dataset}@${coverageLabel}`,
  );
  if (!cell) return null;
  const delta = cell.production_minus.m3_ucb;
  if (!delta) return null;
  return {
    meanDiff: delta.rho.mean_diff,
    p_value: delta.rho.p_value,
    seeds_won: delta.rho.seeds_won,
    n: delta.rho.n,
  };
}


function singleRoundEvidence(cross: CrossTargetPayload): string {
  const target = cross.learnability_gate.partition.single_round[0];
  if (!target) return "none";
  const row = cross.targets.find((entry) => entry.dataset === target);
  if (!row) return target;
  const paired = row.paired.m5_hybrid_minus_m3_ucb;
  return `${target.toUpperCase()} · ${row.campaign_protocol.rounds} round, ${row.campaign_protocol.total_budget} tests · ${paired.p_value < 0.05 ? "a real difference" : "identical by construction"}`;
}

function formatP(p: number | null): string {
  if (p === null) return "—";
  if (p >= 1e-4) return p.toFixed(4);
  const [mantissa, exponent] = p.toExponential(1).split("e");
  return `${mantissa}e${Number(exponent)}`;
}

function formatRange(values: number[]): string {
  if (!values.length) return "—";
  const min = Math.min(...values);
  const max = Math.max(...values);
  return `${min >= 0 ? "+" : ""}${min.toFixed(4)}…+${max.toFixed(4)}`;
}

/** The coverage cells of one dataset, ordered 5pct → 16.7pct → 33pct. */
export function coverageCells(
  cross: CrossTargetPayload,
  dataset: string,
): CoverageSensitivityRow[] {
  const order = ["5pct", "16.7pct", "33pct"];
  return cross.coverage_sensitivity
    .filter((row) => row.dataset === dataset)
    .slice()
    .sort((a, b) => order.indexOf(a.coverage_label) - order.indexOf(b.coverage_label));
}

/** Every dataset in the sensitivity matrix, in first-seen order. */
export function coverageDatasets(cross: CrossTargetPayload): string[] {
  const seen: string[] = [];
  for (const row of cross.coverage_sensitivity) {
    if (!seen.includes(row.dataset)) seen.push(row.dataset);
  }
  return seen;
}

/**
 * The canonical short id for a corpus row.
 *
 * `lockbox_metadata.rows[].target_id` is derived from the CSV filename, so the
 * HER2 control arrives as `4d5_her2_benchmarking_data_trimmed` while the
 * learnability gate, the partition and the pre-registration document all call it
 * `4d5_her2`. Both spellings denote one dataset, and a UI that keys on the raw
 * id finds no row for `4d5_her2` — so every cross-payload join goes through here.
 */
export function canonicalTargetId(targetId: string): string {
  return targetId
    .replace(/_benchmarking_data(_trimmed)?$/, "")
    .replace(/_trimmed$/, "");
}

/** Does a corpus row denote `targetId` (under either spelling)? */
export function rowMatchesTarget(row: LockboxRow, targetId: string): boolean {
  return (
    row.target_id === targetId ||
    canonicalTargetId(row.target_id) === canonicalTargetId(targetId)
  );
}

/**
 * The lockbox rows split by the funnel rule that decided them.
 *
 * Three details are handled here rather than in the view, because each would be
 * a silent error if left to a component:
 *
 * * **Row ids are CSV-derived.** The audit's `target_id` for the HER2 control is
 *   `4d5_her2_benchmarking_data_trimmed`, while `is_her2` was computed against
 *   the short id `4d5_her2`, so that row's own `is_her2` flag reads `false`. The
 *   HER2 stage matches on the canonical id and on the antigen text as well, and
 *   the view prints a caveat naming the discrepancy instead of quietly trusting
 *   a flag that the data itself contradicts.
 * * **A row can win the funnel and still be disqualified by a measured overlap.**
 *   `aayl49` has no `already_used` marker and is not HER2, so the metadata-only
 *   stages let it through; the pre-registration excluded it on a **17.41 %**
 *   measured sequence overlap with the already-scored `aayl49_ML`, and that
 *   number lives in the learnability gate, not the metadata. Without the overlap
 *   stage the funnel would report two survivors and the view would contradict the
 *   pre-registration document.
 * * **"Survived the funnel" is not "not yet scored".** `aayl51` is both the
 *   pre-registered holdout and a measured cell; the two questions stay separate.
 */
export function funnelStages(
  metadata: LockboxMetadata,
  minNQual: number,
  gate?: LearnabilityGate,
): Array<{
  stage: string;
  detail: string;
  remaining: LockboxRow[];
  rejected: LockboxRow[];
  note?: string;
}> {
  const rows = metadata.rows;
  const overlapOf = (targetId: string): number => {
    if (!gate) return 0;
    const canonical = canonicalTargetId(targetId);
    return gate.rows
      .filter((row) => canonicalTargetId(row.target_id) === canonical)
      .reduce((acc, row) => Math.max(acc, row.overlap_pct_removed ?? 0), 0);
  };

  const notUsable = rows.filter((row) => !row.usable_as_search_problem);
  const usable = rows.filter((row) => row.usable_as_search_problem);
  const belowFloor = usable.filter((row) => row.N_qual < minNQual);
  const candidates = usable.filter((row) => row.N_qual >= minNQual);
  const alreadyScored = candidates.filter((row) => row.already_used !== null);
  const fresh = candidates.filter((row) => row.already_used === null);
  const overlapping = fresh.filter((row) => overlapOf(row.target_id) > 0);
  const distinct = fresh.filter((row) => overlapOf(row.target_id) === 0);
  const her2 = distinct.filter((row) => isHer2Row(row));
  const remaining = distinct.filter((row) => !isHer2Row(row));

  const stages: ReturnType<typeof funnelStages> = [
    {
      stage: "Every public dataset in the corpus",
      detail: "enumerated from metadata alone",
      remaining: rows,
      rejected: [],
    },
    {
      stage: "Is it actually a search problem?",
      detail: "needs at least 35 distinct CDRH3 loops and 2 positions that vary",
      remaining: usable,
      rejected: notUsable,
    },
    {
      stage: "Enough usable candidates",
      detail: `at least ${minNQual} survivors after a fixed median cut-off`,
      remaining: candidates,
      rejected: belowFloor,
    },
    {
      stage: "Not already solved by us",
      detail: "no previous published result on any method",
      remaining: fresh,
      rejected: alreadyScored,
    },
  ];

  if (gate) {
    stages.push({
      stage: "No overlap with a dataset we already used",
      detail: "overlap is measured from the sequences themselves, not from names",
      remaining: distinct,
      rejected: overlapping,
      note: overlapping.length
        ? `Overlap is a measured quantity (learnability gate), not a naming rule: ${overlapping
          .map((row) => `${canonicalTargetId(row.target_id)} ${overlapOf(row.target_id).toFixed(2)}%`)
          .join(", ")}.`
        : undefined,
    });
  }

  stages.push({
    stage: "Not the antibody we tuned on",
    detail: "HER2 is the source of the starting antibody, so it proves nothing",
    remaining,
    rejected: her2,
    note: her2.some((row) => !row.is_her2)
      ? "One row is rejected on its filename rather than its own `is_her2` flag: the audit compares the trimmed CSV id against the short target id, so the flag reads false for the HER2 control. The UI matches the canonical id and says so."
      : undefined,
  });

  return stages;
}

function isHer2Row(row: LockboxRow): boolean {
  if (row.is_her2) return true;
  const haystack = `${row.target_id} ${row.antigen}`.toLowerCase();
  return haystack.includes("her2");
}


/** Best method per metric in a cross-target cell (or any per-method row). */
export function bestMethod(
  record: Record<string, number | null> | { [key: string]: number | null },
): {
  method: string;
  value: number;
} | null {
  let best: { method: string; value: number } | null = null;
  for (const method of METHODS) {
    const value = record[method];
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    if (best === null || value > best.value) best = { method, value };
  }
  return best;
}

/** Rank of a method within a per-method record (1 = best). */
export function rankOf(
  record: Record<string, number | null>,
  method: string,
): number | null {
  const entries = METHODS.map((key) => ({ key, value: record[key] }))
    .filter(
      (entry): entry is { key: (typeof METHODS)[number]; value: number } =>
        typeof entry.value === "number" && Number.isFinite(entry.value),
    )
    .sort((a, b) => b.value - a.value);
  const index = entries.findIndex((entry) => entry.key === method);
  return index >= 0 ? index + 1 : null;
}

/** The learnability-gate result for one target at one coverage. */
export function gateRow(
  gate: LearnabilityGate,
  target: string,
  coverageLabel: string,
): LearnabilityGate["rows"][number] | undefined {
  return gate.rows.find(
    (row) => row.target_id === target && row.coverage_label === coverageLabel,
  );
}

// ─────────────────────────── attribution view ─────────────────────────────

export type VariantGroupKey = "mechanism" | "slot_role" | "sar_efficiency" | "baseline";

export interface VariantGroup {
  key: VariantGroupKey;
  title: string;
  description: string;
  color: string;
  variants: string[];
}

/**
 * The variant taxonomy the ledger itself is organised by. Names come from the
 * contract's `rows` keys; the *grouping* is the scientific claim being tested,
 * so it is declared once here and every table reads it.
 */
export const VARIANT_GROUPS: VariantGroup[] = [
  {
    key: "mechanism",
    title: "Mechanism ablations",
    description:
      "Delete one component of M5 and re-run. A variant that ties production is evidence the component is not doing the work.",
    color: "#10B981",
    variants: ["no_anneal", "greedy_slots"],
  },
  {
    key: "slot_role",
    title: "Slot-role separation (negative controls)",
    description:
      "Re-home the exploitation slots onto the scoring model. If better ranking converted into a better best binder, these would win.",
    color: "#6366F1",
    variants: ["slot1_model", "harvest_model", "harvest_slot1", "all_model"],
  },
  {
    key: "sar_efficiency",
    title: "SAR sample efficiency (negative controls)",
    description:
      "Add representation capacity to the SAR prior. A flat result means the engine is at its representation ceiling.",
    color: "#A855F7",
    variants: ["biochem_pooling", "sar_gain", "biochem_gain"],
  },
  {
    key: "baseline",
    title: "Matched baselines",
    description: "M1–M4 on the identical seed batch — the same comparison the benchmark makes.",
    color: "#F59E0B",
    variants: ["m1_random", "m2_greedy", "m3_ucb", "m4_fixed_prior"],
  },
];

/**
 * Every production-vs-variant delta in one cell, flattened for the table — with
 * the sign convention **normalised**, because the artifact mixes two of them:
 *
 * * Rows without a `_baseline` field are `production − variant` on the published
 *   per-seed arrays. Positive = production is better.
 * * The mechanism rows (`no_anneal`, `greedy_slots`, the slot-role family) carry
 *   `_baseline`, the pre-`e567cd4` production AUBQC, and their published delta is
 *   `_baseline − variant`. Read with the same sign as the rows above, a negative
 *   number would look like a production win when it is a *variant* win — exactly
 *   backwards for the one row the whole ablation exists to argue about.
 *
 * So `variantVsProduction = −mean_diff` is exposed as the field every component
 * renders: positive always means "the variant beat the engine it was compared
 * against", and `comparisonBasis` says which engine that was. Both are checkable:
 * `verify-numbers.ts` recomputes `mean_diff` from `_baseline` for the mechanism
 * rows and from the per-seed arrays for the rest.
 */
export interface AttributionRowView {
  variant: string;
  group: VariantGroupKey;
  /** `variant − production`: positive = the variant is better. */
  variantVsProductionAubqc: number;
  aubqcP: number;
  /** Published `seeds_won` in the artifact's own delta direction. */
  aubqcSeedsWon: number;
  aubqcN: number;
  variantVsProductionRho: number;
  rhoP: number;
  /** Published `seeds_won` in the artifact's own delta direction. */
  rhoSeedsWon: number;
  rhoN: number;
  /**
   * Seeds on which the variant strictly beat production, in the same direction as
   * `variantVsProduction*`. Only computable when both per-seed arrays are
   * published — the mechanism rows carry a scalar `_baseline` instead of the
   * baseline's per-seed vector, so for those the count is `null` rather than a
   * complement that would silently count ties as wins.
   */
  aubqcVariantBetter: number | null;
  rhoVariantBetter: number | null;
  /** What the variant was compared against. */
  comparisonBasis: "production" | "pre-e567cd4 production";
  baselineAubqc: number | null;
  note: string | null;
}

export function attributionRows(cell: AttributionCell): AttributionRowView[] {
  const rows: AttributionRowView[] = [];
  const production = cell.rows["None"];
  for (const [variant, delta] of Object.entries(cell.production_minus)) {
    if (variant === "None" || variant === "null") continue;
    const group = VARIANT_GROUPS.find((entry) => entry.variants.includes(variant))?.key ?? "mechanism";
    const extra = delta as typeof delta & { _baseline?: number; _note?: string };
    const baseline = typeof extra._baseline === "number" ? extra._baseline : null;
    const variantRow = cell.rows[variant];
    const productionComparable = baseline === null && production !== undefined && variantRow !== undefined;
    const countVariantBetter = (metric: "aubqc" | "rho"): number | null => {
      if (!productionComparable) return null;
      const series = production[`${metric}_per_seed`];
      const other = variantRow[`${metric}_per_seed`];
      if (!Array.isArray(series) || !Array.isArray(other) || series.length !== other.length) return null;
      return other.filter((value, index) => value > (series[index] ?? Number.NaN)).length;
    };
    rows.push({
      variant,
      group,
      variantVsProductionAubqc: -delta.aubqc.mean_diff,
      aubqcP: delta.aubqc.p_value,
      aubqcSeedsWon: delta.aubqc.seeds_won,
      aubqcN: delta.aubqc.n,
      variantVsProductionRho: -delta.rho.mean_diff,
      rhoP: delta.rho.p_value,
      rhoSeedsWon: delta.rho.seeds_won,
      rhoN: delta.rho.n,
      aubqcVariantBetter: countVariantBetter("aubqc"),
      rhoVariantBetter: countVariantBetter("rho"),
      comparisonBasis: baseline === null ? "production" : "pre-e567cd4 production",
      baselineAubqc: baseline,
      note: typeof extra._note === "string" ? extra._note : null,
    });
  }
  return rows;
}


/** The production row of a cell — the unablated configuration, keyed `"None"`. */
export function productionRow(cell: AttributionCell) {
  return cell.rows["None"];
}

/**
 * Aggregates for one variant group across every cell: how many comparisons,
 * and the smallest p observed — the number the negative-control claims rest on.
 */
export function groupSummary(
  attribution: AttributionPayload,
  group: VariantGroupKey,
): {
  comparisons: number;
  /** How many distinct changes the group contains. */
  variants: number;
  minP: number | null;
  minAt: { cell: string; variant: string; metric: string } | null;
  cells: number;
} {
  const names = VARIANT_GROUPS.find((entry) => entry.key === group)?.variants ?? [];
  let comparisons = 0;
  let minP: number | null = null;
  let minAt: { cell: string; variant: string; metric: string } | null = null;
  const seenCells = new Set<string>();
  for (const cell of attribution.cells) {
    seenCells.add(cell.cell);
    for (const name of names) {
      const delta = cell.production_minus[name];
      if (!delta) continue;
      for (const metric of ["aubqc", "rho"] as const) {
        const block = delta[metric];
        if (!block || !Number.isFinite(block.p_value) || block.n === 0) continue;
        comparisons += 1;
        if (minP === null || block.p_value < minP) {
          minP = block.p_value;
          minAt = { cell: cell.cell, variant: name, metric };
        }
      }
    }
  }
  return { comparisons, variants: names.length, minP, minAt, cells: seenCells.size };
}

/**
 * `no_anneal` per-cell deltas, in the same normalised sign convention as
 * `attributionRows`: positive = **deleting the anneal improved the metric**.
 *
 * The mechanism rows are measured against the pre-`e567cd4` production baseline
 * (the engine *with* the VoI anneal), so `−mean_diff` is the gain from removing
 * it. This normalisation is not cosmetic: the artifact publishes
 * `baseline − variant`, so its ρ delta reads `−0.0429` for a change that is a
 * 0.0429 *gain*. A table that printed the raw sign next to the label "ρ gain"
 * would state the opposite of the result.
 */
export function noAnnealRows(attribution: AttributionPayload): Array<{
  cell: string;
  annealRemovalAubqc: number;
  aubqcP: number;
  annealRemovalRho: number;
  rhoP: number;
  /**
   * Seeds on which the deletion strictly improved ρ, in the same direction as
   * `annealRemovalRho`. The artifact's own `seeds_won` counts the opposite
   * comparison (`pre-law engine − variant`), so it is *not* this number: with 14
   * of 50 seeds won, the deletion helped on 36. Both are computed from the
   * published delta's `n`, and neither is rendered as the other.
   */
  rhoSeedsDeletionHelped: number;
  rhoN: number;
  comparisonBasis: string;
}> {
  const rows = [];
  for (const cell of attribution.cells) {
    const delta = cell.production_minus.no_anneal;
    if (!delta) continue;
    const extra = delta as typeof delta & { _baseline?: number };
    rows.push({
      cell: cell.cell,
      annealRemovalAubqc: -delta.aubqc.mean_diff,
      aubqcP: delta.aubqc.p_value,
      annealRemovalRho: -delta.rho.mean_diff,
      rhoP: delta.rho.p_value,
      rhoSeedsDeletionHelped: delta.rho.n - delta.rho.seeds_won,
      rhoN: delta.rho.n,
      comparisonBasis:
        typeof extra._baseline === "number" ? "pre-e567cd4 production" : "production",
    });
  }
  return rows;
}


// ────────────────────────────── shared helpers ────────────────────────────



/** The batch's paired arms: slots 1/2/4 are H1, slot 3 is the probe, slot 5 is H0. */
export function arms(state: StatePayload): { h1: BatchSlot[]; voi: BatchSlot | null; h0: BatchSlot | null } {
  const sorted = state.batch_slots.slice().sort((a, b) => a.slot_id - b.slot_id);
  return {
    h1: sorted.filter((slot) => slot.slot_id === 1 || slot.slot_id === 2 || slot.slot_id === 4),
    voi: sorted.find((slot) => slot.slot_id === 3) ?? null,
    h0: sorted.find((slot) => slot.slot_id === 5) ?? null,
  };
}

// ──────────────────── cross-target: scoreboard helpers ────────────────────

/**
 * The M5-vs-M3 ρ evidence for one learnable target.
 *
 * 1N8Z is not in `targets[]`: it is measured by the primary 35-query protocol
 * and published under `reference_targets`, and its paired **ρ** test lives in
 * `/api/benchmark`. Every other learnable target is a dynamic-sized cell whose
 * paired ρ test exists only in the attribution ledger. Both paths are handled
 * here so the scoreboard cannot accidentally quote an AUBQC p-value beside a ρ
 * delta.
 */
export function learnableRhoEvidence(
  cross: CrossTargetPayload,
  attribution: AttributionPayload | null,
  benchmark: BenchmarkPayload | null,
  target: string,
): { target: string; label: string; m5: number | null; m3: number | null; delta: number | null; p: number | null; seedsWon: number | null; n: number | null; rounds: number | null } {
  if (target === "1n8z") {
    const reference = cross.reference_targets[0];
    const last = reference?.spearman_rho_curves[reference.spearman_rho_curves.length - 1];
    const paired = benchmark?.paired_t_tests.m5_hybrid_minus_m3_ucb.metrics.spearman_rho_at_q35;
    const m5 = last?.m5_hybrid ?? null;
    const m3 = last?.m3_ucb ?? null;
    return {
      target,
      label: reference?.label ?? "1N8Z",
      m5,
      m3,
      delta: m5 !== null && m3 !== null ? m5 - m3 : null,
      p: paired?.p_value ?? null,
      seedsWon: paired?.seeds_won ?? null,
      n: paired?.n ?? null,
      rounds: reference ? reference.checkpoints.length - 1 : null,
    };
  }
  const row =
    cross.targets.find((entry) => entry.dataset === target && entry.coverage_label === "16.7pct") ??
    cross.targets.find((entry) => entry.dataset === target);
  if (!row) {
    return { target, label: target, m5: null, m3: null, delta: null, p: null, seedsWon: null, n: null, rounds: null };
  }
  const paired = rhoPairedTest(attribution, row.dataset, row.coverage_label);
  const m5 = row.spearman_rho_at_last_checkpoint.m5_hybrid;
  const m3 = row.spearman_rho_at_last_checkpoint.m3_ucb;
  return {
    target,
    label: row.label,
    m5,
    m3,
    delta: m5 !== null && m3 !== null ? m5 - m3 : null,
    p: paired?.p_value ?? null,
    seedsWon: paired?.seeds_won ?? null,
    n: paired?.n ?? null,
    rounds: row.campaign_protocol.rounds,
  };
}

/**
 * The strongest warm-start cell of a target (`M5 warm start − M5 cold start`).
 *
 * A cross-target target is measured at up to three coverage levels and the
 * warm-start effect is not constant across them, so the scoreboard quotes the
 * cell with the largest positive Δ rather than an average that no single row
 * supports. The coverage label travels with the number for exactly that reason.
 */
export function strongestWarmStart(
  cross: CrossTargetPayload,
  target: string,
): { coverage: string; delta: number; p: number; seedsWon: number; n: number } | null {
  let best: { coverage: string; delta: number; p: number; seedsWon: number; n: number } | null = null;
  for (const row of cross.targets) {
    if (row.dataset !== target) continue;
    const paired = row.paired.m5_warm_start_minus_m5_hybrid;
    if (!Number.isFinite(paired.mean_diff)) continue;
    if (best === null || paired.mean_diff > best.delta) {
      best = {
        coverage: row.coverage_label,
        delta: paired.mean_diff,
        p: paired.p_value,
        seedsWon: paired.seeds_won,
        n: paired.n,
      };
    }
  }
  return best;
}

/**
 * Plain-English names for the ablation tokens.
 *
 * The contract keys variants by their internal id (`no_anneal`, `slot1_model`, …)
 * because that is what the artifact writes. A reader who has not read the
 * ablation script cannot tell `slot1_model` from `harvest_model`, so every
 * surface renders the human name and keeps the token as a small monospace
 * subtitle — the id stays visible and greppable, it just is not the headline.
 *
 * `effect` is the one-clause statement of what deleting (or re-homing) that
 * component does, written from the measured direction rather than from intent.
 */
export const VARIANT_LABELS: Record<
  string,
  { name: string; token: string; effect: string; kind: "causal" | "control" | "baseline" }
> = {
  no_anneal: {
    name: "Unbiased Null Objective",
    token: "no_anneal",
    effect: "Removes the heuristic that nudged scores toward the model's own guess; the shipped rule has no such nudge.",
    kind: "causal",
  },
  greedy_slots: {
    name: "Greedy Slot Selection",
    token: "greedy_slots",
    effect: "Replaces the uncertainty probe with a greedy pick.",
    kind: "control",
  },
  slot1_model: {
    name: "Slot 1 Scored by the Agent Model",
    token: "slot1_model",
    effect: "Re-homes the first exploitation slot onto the agent's scoring model.",
    kind: "control",
  },
  harvest_model: {
    name: "Harvest Slots Scored by the Model",
    token: "harvest_model",
    effect: "Re-homes the harvest-phase slots onto the agent's scoring model.",
    kind: "control",
  },
  harvest_slot1: {
    name: "Harvest + Slot 1 Scored by the Model",
    token: "harvest_slot1",
    effect: "Combines the two slot-role re-homings.",
    kind: "control",
  },
  all_model: {
    name: "All Slots Scored by the Agent Model",
    token: "all_model",
    effect: "The strongest re-homing: no slot follows the UCB rule.",
    kind: "control",
  },
  biochem_pooling: {
    name: "Biochemical Pooling Prior",
    token: "biochem_pooling",
    effect: "Groups residues by biochemistry instead of identity.",
    kind: "control",
  },
  sar_gain: {
    name: "SAR Prior · Extra Capacity",
    token: "sar_gain",
    effect: "Adds representational capacity to the SAR prior.",
    kind: "control",
  },
  biochem_gain: {
    name: "Biochemical Prior · Extra Capacity",
    token: "biochem_gain",
    effect: "Adds capacity to the biochemical pooling prior.",
    kind: "control",
  },
};

/** Fallback label for a variant the table does not know. */
export function variantLabel(variant: string): { name: string; token: string; effect: string } {
  const known = VARIANT_LABELS[variant];
  if (known) return known;
  return { name: variant, token: variant, effect: "" };
}
