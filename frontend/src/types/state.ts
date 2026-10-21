/**
 * Wire types for `GET /api/state` and the control routes on `app/server.py`.
 *
 * Transcribed from `api_contract.json` → `state`. Two rules govern this file:
 *
 * 1. **No number is typed as a literal.** The showcase never hard-codes `8.58`
 *    or `0.6578`; it renders whatever the backend returns. The type therefore
 *    says `number`, never `8.58`.
 * 2. **Optional means "the server may omit it".** `state_at(k)` adds
 *    `tool_outputs`, `declared_priors`, `hypotheses`, `quality_curve` and
 *    `aubqc`; the seed state omits them. Fields that the contract's golden
 *    payload always carries are required here, so a backend change surfaces as a
 *    compile error rather than a blank panel.
 */

/** `phase` is a two-value label form, per SPEC §9. */
export type Phase = "Cold-Start Tuition" | "Self-Accelerating Payoff";

/** `hcdr3_ribbon.status` — the contract's four evidentiary states. */
export type RibbonStatus = "neutral" | "anchor_conserved" | "low_risk_swap" | "epistasis_alert";

/** `batch_slots.slot_type` — the contract's four slot roles. */
export type SlotType =
  | "Dual-Surrogate UCB"
  | "Max Joint Uncertainty"
  | "Agent Hypothesis Test"
  | "Skeptic Falsification Probe";

/** The hypothesis verdict vocabulary used by the paired audit. */
export type Verdict = "SUPPORTED" | "FALSIFIED" | "INCONCLUSIVE";

export interface Kpis {
  best_qualified_binding: number;
  delta_vs_wt: number;
  query_budget_spent: number;
  total_budget: number;
  wet_lab_cost_saved_usd: number;
  hypotheses_supported: number;
  hypotheses_falsified: number;
}

/** A sparse, curated per-position ribbon entry (not all 13 positions). */
export interface RibbonEntry {
  position: number;
  wt_residue: string;
  soft_prior_weight: number;
  status: RibbonStatus;
}

/** One council seat's memo plus its Option-C activation annotation. */
export interface CouncilSeat {
  role: string;
  /** Provider id as the router reports it (`google`, `anthropic`, `openai`, …). */
  provider: string;
  /** The exact model id sent on the wire. */
  model: string;
  thinking_budget: number | null;
  thinking_trace: string;
  memo: string;
  /** What the seat actually cited, verbatim from the ledger. */
  tools_invoked: string[];
  /** Cited tools the round's drill schedule has switched on. */
  tools_invoked_active: string[];
  /** Cited tools the schedule has switched off (rendered greyed out, with a reason). */
  tools_inactive: string[];
  /** Trigger string per cited tool. */
  tool_activation_reasons: Record<string, string>;
}

/**
 * One of the five plate slots.
 *
 * The `a_*`/`voi`/`sar_maturity` fields are the selector's own diagnostics —
 * `a_null` is the H0 arm, `a_h1` the hypothesis arm, and `delta_agent` the
 * bounded prior that separates them. They are `number | null` because a slot
 * persisted by an older engine predates them.
 */
export interface BatchSlot {
  slot_id: number;
  slot_type: SlotType;
  candidate_id: string;
  sequence: string;
  mu_bind: number;
  sigma_bind: number;
  risk_score: number;
  risk_passed: boolean;
  a_null: number | null;
  a_h1: number | null;
  delta_agent: number | null;
  sigma_joint: number | null;
  optimistic: number | null;
  agent_promoted: number | null;
  voi: number | null;
  horizon: number | null;
  sar_maturity: number | null;
}

/** One revealed candidate in the risk/binding scatter. */
export interface ParetoPoint {
  candidate_id: string;
  risk: number;
  binding: number;
  /** `wild_type` | `pareto_frontier` | `queried_round_N`. */
  status: string;
}

/** One benchmark checkpoint row, carried inside `/api/state`. */
export interface BenchmarkCurveRow {
  query_count: number;
  m1_random: number | null;
  m2_greedy: number | null;
  m3_ucb: number | null;
  m4_fixed_prior: number | null;
  m5_hybrid: number | null;
}

/** One `position -> shrunk effect` row of the hierarchical SAR table. */
export interface PositionToleranceEffect {
  position: number;
  n: number;
  shrunk_effect: number;
}

/** The Option-C tool arsenal row. */
export interface ActiveTool {
  name: string;
  tier: string;
  active: boolean;
  trigger_reason: string;
  routed_to: string[];
}

/** Which drills are on for this round, and why. */
export interface ActiveToolsMetadata {
  round_number: number;
  queries_spent: number;
  voi: number;
  sar_maturity: number;
  active: string[];
  tools: ActiveTool[];
}

/**
 * The first-principles telemetry block.
 *
 * `exploit_weight` / `exploration_weight` are **diagnostics of the retired
 * anneal**, not coefficients of the shipped objective. `steers_null_objective`
 * is always `false` and the UI must say so; `objective` is `a_null := a_base`.
 */
export interface VoiDiagnostics {
  round_number: number;
  voi: number;
  horizon: number;
  sar_maturity: number;
  exploit_weight: number;
  exploration_weight: number;
  slot_split: string;
  sar_basis: string;
  sar_support: number;
  position_tolerance_effects: PositionToleranceEffect[];
  eta_sar: number | null;
  sar_maturity_target: number | null;
  slot_graduation: boolean | null;
  steers_null_objective: boolean;
  objective: string;
  objective_note: string;
  active_tools_metadata: ActiveToolsMetadata;
}

// ── Fields that `state_at(k)` adds (absent from the seed state) ────────────

/** One row of the deterministic SAR scan. */
export interface SarRow {
  position: number;
  residue: string;
  n: number;
  mean_y_bind: number;
  delta_vs_wt: number;
  std: number;
  candidate_ids: string[];
}

export interface SarScanOutput {
  tool: string;
  wt_binding?: number | null;
  rows: SarRow[];
}

export interface EpistasisClash {
  candidate_id: string;
  positions: number[];
  residual: number;
  direction: string;
}

export interface EpistasisOutput {
  tool: string;
  sigma?: number;
  threshold?: number;
  clashes: EpistasisClash[];
}

export interface DevelopabilityRow {
  candidate_id: string;
  sequence: string;
  max_5mer_hydropathy: number;
  aromatic_fraction: number;
  net_charge_abs_fraction: number;
  adjacent_charge_clusters: number;
  deamidation_motifs: string[];
  isomerization_motifs: string[];
  unpaired_cys: number;
}

export interface DevelopabilityOutput {
  tool: string;
  rows: DevelopabilityRow[];
}

export interface HotspotRow {
  candidate_id: string;
  sigma_bind: number;
  sigma_dev: number;
  sigma_joint: number;
}

export interface HotspotOutput {
  tool: string;
  rows: HotspotRow[];
}

export interface AuditVerdict {
  hypothesis_id: string | null;
  round_number: number;
  verdict: Verdict;
  reason: string;
  design: string;
  compared: string;
  statement: string;
  h1_arm_ids: string[];
  h1_target_id: string | null;
  h0_control_id: string | null;
  round_median: number | null;
  h1_binding: number | null;
  h1_arm_mean_binding: number | null;
  h1_mean_residual: number | null;
  h0_binding: number | null;
  h0_residual: number | null;
  residual_tolerance: number | null;
  decisive_lift: boolean;
  paired_lift: number | null;
  expected_binding: number | null;
  claim_about: string[];
  graded_on: string[];
  inverted_positions: number[];
  position_residuals: Record<string, number>;
}

export interface AuditOutput {
  tool: string;
  design: string;
  verdicts: AuditVerdict[];
  falsified_positions: number[];
}

export interface CostOutput {
  tool: string;
  queries: number;
  spent_usd: number;
  full_pool_cost_usd: number;
  saved_vs_exhaustive_usd: number;
  assay_days_parallel_batches: number;
  full_pool: number;
}

/** The six deterministic tools, keyed by registry name. */
export interface ToolOutputs {
  scan_positional_sar?: SarScanOutput;
  detect_epistasis_clashes?: EpistasisOutput;
  profile_developability_liabilities?: DevelopabilityOutput;
  inspect_joint_uncertainty_hotspots?: HotspotOutput;
  audit_past_hypotheses?: AuditOutput;
  compute_cost_and_effort_saved?: CostOutput;
}

/** A hypothesis as `/api/state` publishes it (already audited). */
export interface HypothesisRecord {
  hypothesis_id: string | null;
  round_number: number;
  statement: string;
  verdict: Verdict | null;
  evidence: AuditVerdict;
}

/** A declared positional prior (what the council argued, not what scored). */
export interface PositionalPrior {
  position: number;
  preferred_residues: string[];
  discouraged_residues: string[];
  weight_delta: number;
}

export interface QualityPoint {
  query_count: number;
  best_qualified_binding: number;
}

/**
 * The planner's decision for one round.
 *
 * The planner is the fourth agent: it reads the three scientist memos plus the
 * raw tool tables and emits a `RoundDecisionRecord`. Its output is structured
 * data rather than prose, so it never appears in `agent_council` — and until the
 * backend exposed this block there was no way for the console to show it.
 *
 * `{}` (not `null`) for the seed batch, which has no planning step.
 */
export interface PlannerRecord {
  round_number: number;
  hypothesis_id: string | null;
  /** The claim the planner wrote for this round, in its own words. */
  hypothesis_statement: string | null;
  /** Its verdict on the *previous* round's hypothesis. */
  previous_verdict: "N/A (Round 1)" | "SUPPORTED" | "FALSIFIED" | "INCONCLUSIVE" | null;
  confidence: "Low" | "Medium" | "High" | null;
  /** Where the planner judged the three seats to have disagreed. */
  disagreement_summary: string | null;
  supporting_candidate_ids: string[];
  targets: {
    explore: string | null;
    test: string | null;
    control: string | null;
  };
}

/** `GET /api/state` (and `GET /api/state?round=k`). */
export interface StatePayload {
  current_round: number;
  round_number?: number;
  phase: Phase;
  kpis: Kpis;
  hcdr3_ribbon: RibbonEntry[];
  /** The planner's decision; `{}` on the seed batch. */
  planner?: PlannerRecord | Record<string, never>;
  agent_council: CouncilSeat[];
  batch_slots: BatchSlot[];
  pareto_data: ParetoPoint[];
  benchmark_curves: BenchmarkCurveRow[];
  voi_diagnostics: VoiDiagnostics;

  // Added by `state_at(k)` / the live state after Round 1.
  declared_priors?: PositionalPrior[];
  hypotheses?: HypothesisRecord[];
  quality_curve?: QualityPoint[];
  aubqc?: number | null;
  tool_outputs?: ToolOutputs;
}

// ────────────────────────── Control-route payloads ────────────────────────

/** One compact timeline entry from `GET /api/rounds`. */
export interface TimelineEntry {
  round: number;
  label: string;
  phase: Phase;
  query_count: number;
  best_qualified_binding: number;
  delta_vs_wt: number;
  voi: number;
  sar_maturity: number;
  exploit_weight: number;
  hypotheses_supported: number;
  hypotheses_falsified: number;
  hypothesis_id: string | null;
  verdict: Verdict | null;
  statement: string | null;
  council_seats: number;
  slots: number;
  slot_candidate_ids: string[];
  active_tools: string[];
  inactive_tools: string[];
  tool_tiers: Record<string, string>;
}

/** `GET /api/rounds` — the stepper's cheap index. */
export interface RoundsPayload {
  seed: number;
  current_round: number;
  total_rounds: number;
  total_budget: number;
  batch_size: number;
  rounds: TimelineEntry[];
}

/** `GET /api/config` — the frozen protocol, echoed for provenance display. */
export interface ConfigPayload {
  dataset_sequence_sha256: string;
  is_synthetic: boolean;
  normalizer: { maxima: Record<string, number>; minima: Record<string, number> };
  pool_size: number;
  qualification_pass_rate: number;
  qualified_count: number;
  tau_dev: number;
  tau_risk: number;
  threshold_percentile: number;
  threshold_source: string;
  version: string;
  wild_type: string;
  wt_pass_fraction: number;
  wt_risk: number;
  total_budget: number;
  hitl: boolean;
  m5_policy_mode: string;
}

/** `GET /api/ledger` — enough to prove the hash chain, without the whole file. */
export interface LedgerPayload {
  verified: boolean;
  metadata: Record<string, unknown>;
  entries: Array<Record<string, unknown>>;
}

/**
 * `POST /api/rewind` and `POST /api/restore-golden`.
 * `restore-golden` additionally returns the re-hydrated state.
 */
export interface RewindResult {
  round?: number;
  current_round: number;
  ledger: string;
  entries?: number;
  verified?: boolean;
  state?: StatePayload;
  restored?: boolean;
  source?: string;
}

export interface HealthPayload {
  status: string;
  version: string;
}
