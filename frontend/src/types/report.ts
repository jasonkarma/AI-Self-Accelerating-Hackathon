/** Wire types for the virtual-lab console — transcribed from `api_contract.json`. */

/* ────────────── Frame / timestamp / provenance ────────────── */

export interface Frame {
  round: number;
  slot_quota: number;
  slot_used: number;
  seq_a: string;
  seq_b: string;
  y_bind: number | null;
  y_dev: number | null;
  is_control: boolean;
}

export interface RoundReport {
  round: number;
  frames: Frame[];
  is_final: boolean;
}

export interface CellReport {
  slot: number;
  a_h1: number | null;
  a_h2: number | null;
  a_base: number | null;
  delta_agent: number | null;
  n_qualified: number;
}

export interface CampaignKpis {
  aubqc: number | null;
  best_vs_wt: number | null;
  delta_vs_wt: number | null;
  library_skipped: number | null;
  query_budget_spent: number;
  qualified_library_size: number;
  round: number;
  n_rounds: number;
  slot_quota: number;
  potential_wt: string;
  cost_per_assay_usd: number;
  total_cost_usd: number;
}

export interface RoundRow {
  round: number;
  /** Each element is an array of (metric, value) pairs or null. */
  rows: Array<Array<[string, number | null]>>;
}

export interface SlotGraduation {
  slot: number;
  make_better: number;
  make_safer: number;
  graduation_target: string;
  graduated_at_round: number;
}

export interface HypothesisMemo {
  slot: number;
  target: string;
  hypo: string;
  memo: string;
}

/* ────────────── Payloads from `GET /api/…` ────────────── */

export interface StatePayload {
  campaign_rounds: RoundReport[];
  kpis: CampaignKpis;
  cells: CellReport[];
  hypothesis_memos: HypothesisMemo[];
  round_rows: RoundRow[];
  slot_graduations: SlotGraduation[];
  agent_council: AgentSeat[];
  agent_calls: AgentCall[];
  agent_errors: AgentError[];
  context: ContextEntry[];
  runs_to_date: number;
  confirmed_runs: number;
  is_restored: boolean;
  is_running: boolean;
  is_completed: boolean;
  restoration_message: string | null;
  restoration_sha256: string | null;
  lockbox_serialized: string | null;
  lockbox_pretty: string | null;
  ledgers_extant: number;
  seed: number;
}

export interface ConfigPayload {
  n_qualified: number;
  n_assays: number;
  query_budget_total: number;
  rounds: number;
  batch_size: number;
  objective: string;
  pool_label: string;
  pool_file: string;
  target_family: string;
}

export interface LedgerPayload {
  campaign_db: string;
  head_round: number;
  n_ledgers: number;
  sqlite_version: string;
  verified: boolean;
}

export interface HealthPayload {
  status: string;
  uptime: number;
  n_ledgers: number;
}

export interface BenchmarkPayload {
  rho: number | null;
  source: BenchmarkSource;
  spearman_rho_curves: RhoCurveRow[];
  aubqc: MethodRecord;
  aubqc_se: MethodRecord;
  normalized_aubqc: MethodRecord;
  aubqc_curves: CurveRow[];
  normalized_curves: CurveRow[];
  top5_panel_curves: CurveRow[];
  top5_panel_aubqc: MethodRecord;
  top_decile_hit_rate: MethodRecord;
  matched_initial_seeds: Record<string, string[]>;
  paired_t_tests: {
    m5_hybrid_minus_m3_ucb: PairedComparison;
    m5_hybrid_minus_m2_greedy: PairedComparison;
    m5_warm_start_minus_m5_hybrid: PairedComparison;
  };
  zero_shot_references: AlphafoldReferences;
  cross_validation: CrossValidation;
  prior_genesis_seeds: PriorGenesisSeed[];
  m5_warm_start: WarmStartBlock;
  learnability_gate: LearnabilityGate;
  tau_tests: Record<string, TauTest>;
  baseline_warm_start: number;
}

export interface BenchmarkSource {
  label: string;
  path: string;
  sha256: string;
}

export interface CurveRow {
  method: string;
  values: (number | null)[];
}

export interface RhoCurveRow {
  method: string;
  values: (number | null)[];
}

export interface MethodRecord {
  [method: string]: number | null;
}

export interface PairedComparison {
  n: number;
  mean_diff: number;
  se: number;
  t_stat: number;
  p_value: number;
  se_ratio: number;
  seeds_won: number;
}

export interface AlphafoldReferences {
  alphafold3: number | null;
  proteinmpnn: number | null;
  esm2: number | null;
}

export interface CrossValidation {
  fold: number;
  correlation: number;
  mae: number;
}

export interface PriorGenesisSeed {
  round_counts: number[];
  entropy: number[];
  coverage_curve: (number | null)[];
}

export interface WarmStartBlock {
  warm_start_aubqc: number | null;
  warm_start_rho: number | null;
  warm_start_wins: number;
  seed_id: string;
}

export interface LearnabilityGate {
  partition: PartitionEntry[];
  best_rho: number | null;
  worst_rho: number | null;
  reject: string | null;
  candidate_targets: number;
  holdout_targets: number;
  is_passing: boolean;
  rationale: string;
}

export interface PartitionEntry {
  target: string;
  dataset: string;
  median_rho: number | null;
  iqr: number | null;
  rank_accuracy: number | null;
  is_candidate: boolean;
  is_holdout: boolean;
}

export interface TauTest {
  kendall_tau: number | null;
  p_value: number | null;
  n: number;
  reject: string | null;
}

export interface AgentSeat {
  role: string;
  id: string;
  tier: string;
  memo: string;
  thinking_trace: string;
  confidence: number;
  tool_orders: Array<{
    name: string;
    args: Record<string, unknown>;
    ts: number;
  }>;
  tool_outputs: Array<{
    name: string;
    raw: string;
    parsed: {
      summary: string;
      diagnosis: string | null;
    } | null;
  }>;
}

export interface AgentCall {
  role: string;
  tool: string;
  elapsed_s: number;
  error: string | null;
}

export interface AgentError {
  role: string;
  tool: string;
  error: string;
}

export interface ContextEntry {
  role: string;
  note: string;
  detail: string;
}

/* ────────────── Cross-target sweep ────────────── */

export interface CrossTargetRow {
  dataset: string;
  label: string;
  n_qualified: number;
  n_rounds: number;
  aubqc: MethodRecord;
  rho_rank: number | null;
  rho_pool: number | null;
  rho_coverage: number | null;
  top_decile_hit_rate: MethodRecord;
  spearman_rho: MethodRecord;
  spearman_rho_se: MethodRecord;
  coverage: number;
  coverage_label: string;
}

export interface CrossTargetPayload {
  available: boolean;
  detail?: string;
  path?: string;
  m5_policy_mode: string;
  seeds: number;
  curves: CurveRow[];
  aubqc: MethodRecord;
  aubqc_se: MethodRecord;
  normalized_curves: CurveRow[];
  normalized_aubqc: MethodRecord;
  top5_panel_curves: CurveRow[];
  top5_panel_aubqc: MethodRecord;
  spearman_rho_curves: RhoCurveRow[];
  top_decile_hit_rate: MethodRecord;
  top_decile_definition: string;
  matched_initial_seeds: Record<string, string[]>;
  paired_t_tests: {
    m5_hybrid_minus_m3_ucb: PairedComparison;
    m5_hybrid_minus_m2_greedy: PairedComparison;
    m5_warm_start_minus_m5_hybrid: PairedComparison;
  };
  zero_shot_references: {
    alphafold3: number;
    proteinmpnn: number;
    esm2: number;
  };
  zero_shot_definition: string;
  m5_warm_start: WarmStartBlock;
  cross_target: {
    dataset: string;
    label: string;
    source: string;
    repo: string;
    qualified_count: number;
    target_count: number;
    all_targets: string[];
    references: string[];
  };
  targets: CrossTargetRow[];
  reference_targets: CrossTargetRow[];
  coverage_sensitivity: CoverageSensitivity[];
  learnability_gate: LearnabilityGate;
}

export interface CoverageSensitivity {
  label: string;
  coverage_pct: number;
  n_rounds: number;
  spearman_rho: MethodRecord;
  aubqc: MethodRecord;
}

/* ────────────── Attribution ────────────── */

export interface AttributionPayload {
  available: boolean;
  detail?: string;
  path?: string;
  cells: AttributionCell[];
  targets: AttributionTarget[];
}

export interface AttributionCell {
  cell: string;
  rows: Record<string, {
    aubqc: number | null;
    aubqc_se: number | null;
    spearman_rho: number | null;
    top_decile_hit_rate: number | null;
  }>;
}

export interface AttributionTarget {
  dataset: string;
  label: string;
  coverage: number;
}

/* ────────────── Ptarmigan-1 cross-modality extension ────────────── */

export interface PtarmiganHit {
  rank: number;
  compound_id: string;
  name: string;
  smiles: string;
  engagement: number;
  score: number;
  predicted_residue: number;
  predicted_residue_letter: string;
}

export interface PtarmiganMethodSummary {
  aubqc: number | null;
  rho: number | null;
  top_decile_hit_rate: number | null;
}

export interface PtarmiganPairedEntry {
  metric: string;
  n: number;
  mean_diff: number;
  se: number;
  t_stat: number;
  p_value: number;
  seeds_won: number;
}

export interface PtarmiganCoverage {
  known: boolean;
  sequence_length: number;
  residue_start: number;
  residue_end: number;
  training_max_residues: number;
  ood_reasons: string[];
  is_truncated: boolean;
}

export interface PtarmiganTargetConfig {
  gene: string;
  chembl_id: string;
  uniprot: string;
  n_residues: number;
}

export interface PtarmiganTargetPayload {
  screen_id?: string;
  target_gene?: string;
  target_uniprot?: string;
  target_config?: PtarmiganTargetConfig;
  library_size?: number;
  n_hits?: number;
  screen_cost_usd?: string;
  ptarmigan_model_version?: string;
  hotspot_residue_indices?: number[];
  hotspot_n_residues?: number;
  coverage?: PtarmiganCoverage;
  hits?: PtarmiganHit[];
  methods?: Record<string, PtarmiganMethodSummary>;
  paired_comparisons?: Record<string, Record<string, PtarmiganPairedEntry>>;
  metadata?: {
    seeds: number;
    rounds: number;
    batch_size: number;
    initial_queries: number;
    total_queries_per_seed: number;
    methods: string[];
  };
  summary?: string;
  stage_a_payload?: Record<string, unknown>;
}

/** `GET /api/ptarmigan` — the Ptarmigan-1 cross-modality extension combined artifact. */
export interface PtarmiganPayload {
  available: boolean;
  detail?: string;
  path?: string;
  screen_id?: string;
  target_gene?: string;
  target_uniprot?: string;
  target_config?: PtarmiganTargetConfig;
  library_size?: number;
  n_hits?: number;
  screen_cost_usd?: string;
  ptarmigan_model_version?: string;
  hotspot_residue_indices?: number[];
  hotspot_n_residues?: number;
  coverage?: PtarmiganCoverage;
  hits?: PtarmiganHit[];
  methods?: Record<string, PtarmiganMethodSummary>;
  paired_comparisons?: Record<string, Record<string, PtarmiganPairedEntry>>;
  metadata?: {
    seeds: number;
    rounds: number;
    batch_size: number;
    initial_queries: number;
    total_queries_per_seed: number;
    methods: string[];
  };
  summary?: string;
  /** Root-level CDK2 stays compatible; targets map carries per-target payloads. */
  targets?: Record<string, PtarmiganTargetPayload>;
  n_targets?: number;
  target_genes?: string[];
  cross_summary?: string;
  stage_a_payload?: Record<string, unknown>;
}

/* ────────────── Lockbox ────────────── */

export interface LockboxRow {
  label: string;
  dataset: string;
  cell: string;
  n_qualified: number;
  n_rounds: number;
  coverage: number;
  aubqc: number | null;
  rho: number | null;
}

/* ────────────── Cross-target view helpers ────────────── */

/** One row in the response matrix, grouped by coverage level. */
export interface CrossTargetCell {
  dataset: string;
  coverage_label: string;
  rho: number | null;
  rho_se: number | null;
}
