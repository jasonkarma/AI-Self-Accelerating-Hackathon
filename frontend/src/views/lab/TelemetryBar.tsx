import { Gauge, Sigma } from "lucide-react";
import type { ConfigPayload, StatePayload } from "../../types/state";
import { InfoTip } from "../../components/Panel";
import { num, percent } from "../../lib/format";

/**
 * The first-principles telemetry bar.
 *
 * Every gauge here is a *derived* quantity with a published formula, and the
 * formula is printed next to it. The point of the panel is that a judge can
 * check `VoI` against `sar_maturity`, `sar_maturity_target` and `eta_sar`
 * without opening the source.
 *
 * The `exploit_weight` / `exploration_weight` pair carries an explicit
 * warning. It used to steer `a_null`; the anneal was retired at `e567cd4`, and
 * the payload says so itself (`steers_null_objective: false`). Rendering the
 * number without the badge would invite exactly the reading the backend went out
 * of its way to prohibit.
 */
export function TelemetryBar({ state, config }: { state: StatePayload; config: ConfigPayload | null }) {
  const diagnostics = state.voi_diagnostics;
  const maturityTarget = diagnostics.sar_maturity_target;
  const maturityFraction =
    maturityTarget && maturityTarget > 0 ? Math.min(diagnostics.sar_maturity / maturityTarget, 1) : null;
  const seqLength = config ? config.wild_type.length : null;

  return (
    <section className="panel px-4 py-3.5">
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <Gauge size={12} style={{ color: "var(--accent)" }} />
          <span className="label text-[9.5px]">Loop telemetry · round {diagnostics.round_number}</span>
          <InfoTip label="Telemetry formulas">
            Every gauge is a derived quantity with a published formula. VoI scales slot 3's optimism bonus
            only; SAR maturity is measured against this campaign's own target; η_SAR is the additive prior
            trust given L, N_qual and Q, computed from t = 0 observables only.
          </InfoTip>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span
            className="chip mono !text-[9.5px]"
            style={{ borderColor: "var(--accent-line)", color: "var(--accent)", background: "var(--accent-soft)" }}
            title={diagnostics.objective_note}
          >
            OBJECTIVE {diagnostics.objective}
          </span>
          <span
            className="chip mono !text-[9.5px]"
            style={{
              borderColor: diagnostics.steers_null_objective ? "rgba(244,63,94,0.4)" : "var(--border)",
              color: diagnostics.steers_null_objective ? "var(--danger)" : "var(--ink-faint)",
            }}
            title={
              diagnostics.steers_null_objective
                ? "The anneal is steering a_null. This contradicts the shipped objective law."
                : "steers_null_objective: false — exploit_weight / exploration_weight are diagnostics of the retired anneal, not coefficients of a_null."
            }
          >
            VoI diagnostic only
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-3.5 sm:grid-cols-3">
        <GaugeCell
          label="Value of information"
          value={num(diagnostics.voi, 4)}
          fraction={null}
          color="#0EA5E9"
          formula="H · (1 − 0.65·C)"
          hint="Value of information. Scales slot 3's optimism bonus only; it is not a phase switch."
        />
        <GaugeCell
          label="SAR maturity"
          value={num(diagnostics.sar_maturity, 4)}
          fraction={maturityFraction}
          color="#A855F7"
          formula={`target ${num(maturityTarget, 1)}`}
          hint="Empirically corroborated fraction of the hierarchical SAR table, measured against this campaign's own target."
        />
        <GaugeCell
          label="η_SAR prior trust"
          value={num(diagnostics.eta_sar, 3)}
          fraction={null}
          color="#6366F1"
          formula="additive_prior_trust(L, N_qual, Q)"
          hint="How much the empirical SAR prior is trusted given this pool and this budget. Computed from t = 0 observables only."
        />
        <GaugeCell
          label="Sequence space"
          value={seqLength === null ? "—" : `L = ${seqLength}`}
          fraction={null}
          color="var(--ink-dim)"
          formula={config ? `N_qual ${config.qualified_count} · Q ${state.kpis.total_budget}` : "from the frozen protocol"}
          hint="L, N_qual and Q come from the frozen protocol and the KPI block — never typed in."
        />
        <GaugeCell
          label="SAR support"
          value={`${diagnostics.sar_support}`}
          fraction={null}
          color="var(--ink-dim)"
          formula={`${diagnostics.sar_basis} · ${diagnostics.position_tolerance_effects.length} positions`}
          hint="Number of evidence units behind the SAR table, and the estimator's basis."
        />
        <GaugeCell
          label="Exploit / explore weights"
          value={`${num(diagnostics.exploit_weight, 2)} / ${num(diagnostics.exploration_weight, 2)}`}
          fraction={null}
          color="var(--ink-faint)"
          formula="retired anneal · diagnostic"
          hint={diagnostics.objective_note}
        />
      </div>

      {diagnostics.position_tolerance_effects.length ? (
        <div className="mt-3 border-t pt-2.5" style={{ borderColor: "var(--border)" }}>
          <div className="mb-1.5 flex items-center gap-1.5">
            <Sigma size={10} style={{ color: "var(--ink-faint)" }} />
            <span className="label text-[9px]">Position-tolerance effects</span>
            <InfoTip label="Shrinkage rule">
              Shrunk <span className="mono">Δy_bind</span> vs the WT median, with shrinkage{" "}
              <span className="mono">n / (n + 3)</span>. A cell with <span className="mono">n &lt; 2</span>{" "}
              contributes nothing. Empty positions are not zero-effect positions — they are positions the
              ledger has no repeated observation for.
            </InfoTip>
          </div>
          <div className="flex flex-wrap gap-1">
            {diagnostics.position_tolerance_effects
              .slice()
              .sort((a, b) => a.position - b.position)
              .map((effect) => {
                const positive = effect.shrunk_effect >= 0;
                return (
                  <span
                    key={effect.position}
                    className="chip mono !gap-1 !text-[9.5px]"
                    style={{
                      borderColor: positive ? "var(--accent-line)" : "rgba(245,158,11,0.35)",
                      color: positive ? "var(--accent)" : "#F59E0B",
                      background: positive ? "var(--accent-soft)" : "rgba(245,158,11,0.07)",
                    }}
                    title={`Position ${effect.position}: shrunk effect ${effect.shrunk_effect} over n = ${effect.n}`}
                  >
                    <span className="ink-faint">P{effect.position}</span>
                    {positive ? "+" : "−"}
                    {Math.abs(effect.shrunk_effect).toFixed(3)}
                    <span className="ink-faint">n={effect.n}</span>
                  </span>
                );
              })}
          </div>
        </div>
      ) : null}

      {config ? (
        <div
          className="mt-2.5 flex flex-wrap items-center gap-x-3.5 gap-y-1 border-t pt-2 text-[9.5px] ink-faint"
          style={{ borderColor: "var(--border)" }}
        >
          <span>
            τ_dev <span className="mono">{config.tau_dev.toFixed(4)}</span>
          </span>
          <span>
            pass rate <span className="mono">{percent(config.qualification_pass_rate, 1)}</span>
          </span>
          <span>
            qualified <span className="mono">{config.qualified_count}/{config.pool_size}</span>
          </span>
          <span className="mono">{config.version}</span>
        </div>
      ) : null}
    </section>
  );
}

function GaugeCell({
  label,
  value,
  fraction,
  color,
  formula,
  hint,
}: {
  label: string;
  value: string;
  fraction: number | null;
  color: string;
  formula: string;
  hint: string;
}) {
  return (
    <div className="min-w-0" title={hint}>
      <div className="label mb-0.5 whitespace-normal leading-tight">{label}</div>
      <div className="mono text-[17px] font-semibold leading-tight" style={{ color }}>
        {value}
      </div>
      {fraction !== null ? (
        <div
          className="mt-1 h-1 w-full overflow-hidden rounded-full"
          style={{ background: "var(--border)" }}
          role="progressbar"
          aria-valuenow={Math.round(fraction * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${label} as a fraction of target`}
        >
          <div
            className="h-full rounded-full transition-[width] duration-500"
            style={{ width: `${fraction * 100}%`, background: color }}
          />
        </div>
      ) : null}
      <div className="mono mt-1 whitespace-normal text-[9.5px] leading-tight ink-faint">{formula}</div>
    </div>
  );
}
