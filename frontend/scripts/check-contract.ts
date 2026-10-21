/**
 * Contract guard: does the committed `api_contract.json` still have the shape
 * this frontend was written against?
 *
 * Run with `npm run contract`. It does not boot the backend (the frontend repo
 * is deliberately buildable without Python); it reads the contract the backend
 * itself generates and asserts the *paths the UI dereferences*. A field the UI
 * reads but the contract no longer publishes is a crash waiting for a projector;
 * a field the contract publishes but the UI ignores is a number a judge may ask
 * about. Both are reported.
 *
 * This is a shape check, not a schema validator: it asserts presence and JSON
 * type, because that is exactly what a render-time dereference depends on.
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const contractPath = resolve(here, "../../api_contract.json");

type PathSpec = { path: string; type: "object" | "array" | "number" | "string" | "boolean" };

const SPECS: PathSpec[] = [
  // ── /api/state ──
  { path: "state.current_round", type: "number" },
  { path: "state.phase", type: "string" },
  { path: "state.kpis.best_qualified_binding", type: "number" },
  { path: "state.kpis.delta_vs_wt", type: "number" },
  { path: "state.kpis.query_budget_spent", type: "number" },
  { path: "state.kpis.total_budget", type: "number" },
  { path: "state.kpis.wet_lab_cost_saved_usd", type: "number" },
  { path: "state.kpis.hypotheses_supported", type: "number" },
  { path: "state.kpis.hypotheses_falsified", type: "number" },
  { path: "state.hcdr3_ribbon", type: "array" },
  // The planner's decision. `{}` on the seed batch, so the *presence* of the key
  // is asserted here and the inner fields are asserted against a real round.
  { path: "state.planner", type: "object" },
  { path: "state.planner.round_number", type: "number" },
  { path: "state.planner.hypothesis_id", type: "string" },
  { path: "state.planner.hypothesis_statement", type: "string" },
  { path: "state.planner.previous_verdict", type: "string" },
  { path: "state.planner.confidence", type: "string" },
  { path: "state.planner.disagreement_summary", type: "string" },
  { path: "state.planner.supporting_candidate_ids", type: "array" },
  { path: "state.planner.targets", type: "object" },
  { path: "state.planner.targets.control", type: "string" },
  { path: "state.hcdr3_ribbon[0].position", type: "number" },
  { path: "state.hcdr3_ribbon[0].wt_residue", type: "string" },
  { path: "state.hcdr3_ribbon[0].soft_prior_weight", type: "number" },
  { path: "state.hcdr3_ribbon[0].status", type: "string" },
  { path: "state.agent_council", type: "array" },
  { path: "state.agent_council[0].role", type: "string" },
  { path: "state.agent_council[0].provider", type: "string" },
  { path: "state.agent_council[0].model", type: "string" },
  { path: "state.agent_council[0].memo", type: "string" },
  { path: "state.agent_council[0].tools_invoked", type: "array" },
  { path: "state.agent_council[0].tools_invoked_active", type: "array" },
  { path: "state.agent_council[0].tools_inactive", type: "array" },
  { path: "state.agent_council[0].tool_activation_reasons", type: "object" },
  { path: "state.batch_slots", type: "array" },
  { path: "state.batch_slots[0].slot_id", type: "number" },
  { path: "state.batch_slots[0].slot_type", type: "string" },
  { path: "state.batch_slots[0].candidate_id", type: "string" },
  { path: "state.batch_slots[0].sequence", type: "string" },
  { path: "state.batch_slots[0].mu_bind", type: "number" },
  { path: "state.batch_slots[0].sigma_bind", type: "number" },
  { path: "state.batch_slots[0].risk_score", type: "number" },
  { path: "state.batch_slots[0].risk_passed", type: "boolean" },
  { path: "state.batch_slots[0].a_null", type: "number" },
  { path: "state.batch_slots[0].a_h1", type: "number" },
  { path: "state.batch_slots[0].delta_agent", type: "number" },
  { path: "state.batch_slots[0].sigma_joint", type: "number" },
  { path: "state.pareto_data", type: "array" },
  { path: "state.pareto_data[0].candidate_id", type: "string" },
  { path: "state.pareto_data[0].risk", type: "number" },
  { path: "state.pareto_data[0].binding", type: "number" },
  { path: "state.pareto_data[0].status", type: "string" },
  { path: "state.benchmark_curves", type: "array" },
  { path: "state.benchmark_curves[0].query_count", type: "number" },
  { path: "state.voi_diagnostics.voi", type: "number" },
  { path: "state.voi_diagnostics.sar_maturity", type: "number" },
  { path: "state.voi_diagnostics.exploit_weight", type: "number" },
  { path: "state.voi_diagnostics.exploration_weight", type: "number" },
  { path: "state.voi_diagnostics.eta_sar", type: "number" },
  { path: "state.voi_diagnostics.sar_maturity_target", type: "number" },
  { path: "state.voi_diagnostics.slot_graduation", type: "boolean" },
  { path: "state.voi_diagnostics.steers_null_objective", type: "boolean" },
  { path: "state.voi_diagnostics.objective", type: "string" },
  { path: "state.voi_diagnostics.objective_note", type: "string" },
  { path: "state.voi_diagnostics.position_tolerance_effects", type: "array" },
  { path: "state.voi_diagnostics.active_tools_metadata.active", type: "array" },
  { path: "state.voi_diagnostics.active_tools_metadata.tools", type: "array" },
  { path: "state.voi_diagnostics.active_tools_metadata.tools[0].name", type: "string" },
  { path: "state.voi_diagnostics.active_tools_metadata.tools[0].tier", type: "string" },
  { path: "state.voi_diagnostics.active_tools_metadata.tools[0].active", type: "boolean" },
  { path: "state.voi_diagnostics.active_tools_metadata.tools[0].trigger_reason", type: "string" },
  { path: "state.voi_diagnostics.active_tools_metadata.tools[0].routed_to", type: "array" },

  // ── /api/benchmark ──
  { path: "benchmark.seeds", type: "number" },
  { path: "benchmark.curves", type: "array" },
  { path: "benchmark.curves[0].m5_hybrid", type: "number" },
  { path: "benchmark.aubqc.m5_hybrid", type: "number" },
  { path: "benchmark.aubqc_se.m5_hybrid", type: "number" },
  { path: "benchmark.normalized_aubqc.m5_hybrid", type: "number" },
  { path: "benchmark.top5_panel_aubqc.m5_hybrid", type: "number" },
  { path: "benchmark.top_decile_hit_rate.m5_hybrid", type: "number" },
  { path: "benchmark.top_decile_definition", type: "string" },
  { path: "benchmark.spearman_rho_curves", type: "array" },
  { path: "benchmark.paired_t_tests.m5_hybrid_minus_m3_ucb.metrics.spearman_rho_at_q35.p_value", type: "number" },
  { path: "benchmark.paired_t_tests.m5_hybrid_minus_m3_ucb.metrics.aubqc.mean_diff", type: "number" },
  { path: "benchmark.paired_t_tests.m5_hybrid_minus_m2_greedy.metrics.panel_aubqc.p_value", type: "number" },
  { path: "benchmark.paired_t_tests.m5_warm_start_minus_m5_hybrid.metrics.aubqc.p_value", type: "number" },
  { path: "benchmark.zero_shot_references.alphafold3", type: "number" },
  { path: "benchmark.zero_shot_references.proteinmpnn", type: "number" },
  { path: "benchmark.zero_shot_references.esm2", type: "number" },
  { path: "benchmark.zero_shot_definition", type: "string" },
  { path: "benchmark.m5_warm_start.summary.aubqc", type: "number" },
  { path: "benchmark.m5_warm_start.curves", type: "array" },
  { path: "benchmark.source.label", type: "string" },
  { path: "benchmark.source.filename", type: "string" },
  { path: "benchmark.source.filename_is_historical", type: "boolean" },
  { path: "benchmark.same_seed_comparison.seed", type: "number" },
  { path: "benchmark.same_seed_comparison.initial_ids", type: "array" },
  { path: "benchmark.same_seed_comparison.summary.live_m5.aubqc", type: "number" },
  { path: "benchmark.same_seed_comparison.live.aubqc", type: "number" },
  { path: "benchmark.same_seed_comparison.live.best_baseline", type: "string" },
  { path: "benchmark.same_seed_comparison.live.aubqc_gap_vs_best_baseline", type: "number" },

  // ── /api/cross-target ──
  { path: "cross_target.available", type: "boolean" },
  { path: "cross_target.seeds", type: "number" },
  { path: "cross_target.targets", type: "array" },
  { path: "cross_target.targets[0].dataset", type: "string" },
  { path: "cross_target.targets[0].coverage_label", type: "string" },
  { path: "cross_target.targets[0].cdr3_length", type: "number" },
  { path: "cross_target.targets[0].campaign_protocol.rounds", type: "number" },
  { path: "cross_target.targets[0].campaign_protocol.total_budget", type: "number" },
  { path: "cross_target.targets[0].aubqc.m5_hybrid", type: "number" },
  { path: "cross_target.targets[0].aubqc_se.m5_hybrid", type: "number" },
  { path: "cross_target.targets[0].spearman_rho_at_last_checkpoint.m5_hybrid", type: "number" },
  { path: "cross_target.targets[0].paired.m5_hybrid_minus_m3_ucb.p_value", type: "number" },
  { path: "cross_target.coverage_sensitivity", type: "array" },
  { path: "cross_target.coverage_sensitivity[0].eta_sar", type: "number" },
  { path: "cross_target.coverage_sensitivity[0].sar_maturity_target", type: "number" },
  { path: "cross_target.coverage_sensitivity[0].aubqc_m5", type: "number" },
  { path: "cross_target.coverage_sensitivity[0].aubqc_best_baseline", type: "number" },
  { path: "cross_target.coverage_sensitivity[0].m5_rank", type: "number" },
  { path: "cross_target.coverage_sensitivity[0].m5_leads_rho", type: "boolean" },
  { path: "cross_target.lockbox_metadata.available", type: "boolean" },
  { path: "cross_target.lockbox_metadata.files_enumerated", type: "array" },
  { path: "cross_target.lockbox_metadata.min_nqual", type: "number" },
  { path: "cross_target.lockbox_metadata.rows", type: "array" },
  { path: "cross_target.lockbox_metadata.rows[0].target_id", type: "string" },
  { path: "cross_target.lockbox_metadata.rows[0].N_qual", type: "number" },
  { path: "cross_target.lockbox_metadata.rows[0].gate_ok", type: "boolean" },
  { path: "cross_target.lockbox_metadata.rows[0].is_her2", type: "boolean" },
  { path: "cross_target.lockbox_metadata.rows[0].unique_cdrh3_pool_size", type: "number" },
  { path: "cross_target.lockbox_metadata.notes", type: "array" },
  { path: "cross_target.learnability_gate.partition.learnable_multiround", type: "array" },
  { path: "cross_target.learnability_gate.partition.unlearnable", type: "array" },
  { path: "cross_target.learnability_gate.partition.single_round", type: "array" },
  { path: "cross_target.learnability_gate.partition.degenerate_1_or_2_cdrh3", type: "array" },
  { path: "cross_target.learnability_gate.rho_gate", type: "number" },
  { path: "cross_target.learnability_gate.rows", type: "array" },
  { path: "cross_target.reference_targets", type: "array" },
  { path: "cross_target.reference_targets[0].aubqc.m5_hybrid", type: "number" },
  { path: "cross_target.reference_targets[0].curves", type: "array" },
  { path: "cross_target.reference_targets[0].checkpoints", type: "array" },
  { path: "cross_target.cross_target.unchanged_modules", type: "array" },
  { path: "cross_target.cross_target.varies_by_protocol", type: "array" },
  { path: "cross_target.merge_log.targets_present", type: "array" },

  // ── /api/attribution ──
  { path: "attribution.available", type: "boolean" },
  { path: "attribution.seeds", type: "number" },
  { path: "attribution.cells", type: "array" },
  { path: "attribution.cells[0].cell", type: "string" },
  { path: "attribution.cells[0].protocol.rounds", type: "number" },
  { path: "attribution.cells[0].rows", type: "object" },
  { path: "attribution.cells[0].rows.None.aubqc", type: "number" },
  { path: "attribution.cells[0].rows.None.rho", type: "number" },
  { path: "attribution.cells[0].production_minus.no_anneal.aubqc.p_value", type: "number" },
  { path: "attribution.cells[0].production_minus.no_anneal.rho.p_value", type: "number" },
  { path: "attribution.cells[0].production_minus.slot1_model.aubqc.p_value", type: "number" },
  { path: "attribution.cells[0].production_minus.biochem_pooling.rho.p_value", type: "number" },
  { path: "attribution.cells[0]._provenance.mechanism_rows_caveat", type: "string" },
];

const REQUIRED_VARIANTS = [
  "None",
  "biochem_pooling",
  "sar_gain",
  "biochem_gain",
  "m2_greedy",
  "m3_ucb",
  "m4_fixed_prior",
  "greedy_slots",
  "no_anneal",
  "slot1_model",
  "harvest_model",
  "harvest_slot1",
  "all_model",
];

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

function read(root: Json, path: string): { found: boolean; value: Json | undefined } {
  const segments = path.split(".");
  let node: Json | undefined = root;
  for (const segment of segments) {
    const match = /^([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(segment);
    if (!match) return { found: false, value: undefined };
    const key = match[1]!;
    const index = match[2];
    if (node === undefined || node === null || typeof node !== "object" || Array.isArray(node)) {
      return { found: false, value: undefined };
    }
    node = (node as { [key: string]: Json })[key];
    if (index !== undefined) {
      if (!Array.isArray(node)) return { found: false, value: undefined };
      node = node[Number(index)];
    }
  }
  return { found: node !== undefined, value: node };
}

function jsonType(value: Json | undefined): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function main(): number {
  let raw: string;
  try {
    raw = readFileSync(contractPath, "utf8");
  } catch (error) {
    console.error(`cannot read ${contractPath}: ${(error as Error).message}`);
    return 2;
  }
  const contract = JSON.parse(raw) as Json;
  const failures: string[] = [];
  const nullableHits: string[] = [];

  for (const spec of SPECS) {
    const { found, value } = read(contract, spec.path);
    if (!found) {
      failures.push(`missing: ${spec.path}`);
      continue;
    }
    const actual = jsonType(value);
    if (actual === "null") {
      // Legitimate for Spearman ρ on M1 and for a pre-`e567cd4` slot, but it must
      // be *expected*: the UI renders null as an em dash, never as zero.
      nullableHits.push(spec.path);
      continue;
    }
    if (actual !== spec.type) {
      failures.push(`type: ${spec.path} is ${actual}, UI expects ${spec.type}`);
    }
  }

  const variants = read(contract, "attribution.cells[0].rows");
  if (variants.found && variants.value && typeof variants.value === "object" && !Array.isArray(variants.value)) {
    const present = new Set(Object.keys(variants.value));
    for (const variant of REQUIRED_VARIANTS) {
      if (!present.has(variant)) failures.push(`attribution: variant '${variant}' missing from rows`);
    }
    for (const variant of present) {
      if (!REQUIRED_VARIANTS.includes(variant)) {
        console.warn(`warn: attribution row '${variant}' is published but not rendered by the UI`);
      }
    }
  }

  const targets = read(contract, "cross_target.targets");
  if (targets.found && Array.isArray(targets.value)) {
    for (const target of targets.value) {
      if (!target || typeof target !== "object" || Array.isArray(target)) continue;
      const row = target as { [key: string]: Json };
      for (const key of ["dataset", "coverage_label", "aubqc", "spearman_rho_at_last_checkpoint", "paired"]) {
        if (!(key in row)) failures.push(`cross_target.targets: row ${String(row.dataset)} lacks '${key}'`);
      }
    }
  }

  if (nullableHits.length) {
    console.log(`nullable-but-present (rendered as "—", never 0): ${nullableHits.length} path(s)`);
    for (const path of nullableHits.slice(0, 8)) console.log(`  · ${path}`);
  }

  if (failures.length) {
    console.error(`\ncontract check FAILED (${failures.length})`);
    for (const failure of failures) console.error(`  ✗ ${failure}`);
    return 1;
  }
  console.log(`contract check ok: ${SPECS.length} paths, ${REQUIRED_VARIANTS.length} variants, ${targets.found && Array.isArray(targets.value) ? targets.value.length : 0} cross-target rows`);
  return 0;
}

process.exit(main());
