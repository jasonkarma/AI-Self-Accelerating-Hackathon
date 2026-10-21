import { Gauge, Sigma } from "lucide-react";
import type { ConfigPayload, StatePayload } from "../../types/state";
import { InfoTip } from "../../components/Panel";
import { num } from "../../lib/format";

/**
 * The engine's internal state.
 *
 * This panel used to sit in the demo path under the three numbers that matter,
 * which invited the reader to treat `VoI` and `η_SAR` as results. They are not
 * results — they are the loop's own decision variables — so the panel now lives
 * inside the "under the hood" disclosure at the bottom of the view.
 *
 * Each gauge keeps its published formula next to it, because the whole point is
 * that a sceptical judge can check it. What changed is the *label*: the surface
 * says what the quantity is for, and the internal name sits in the tooltip.
 */
export function TelemetryBar({ state, config }: { state: StatePayload; config: ConfigPayload | null }) {
  const diagnostics = state.voi_diagnostics;
  const maturityTarget = diagnostics.sar_maturity_target;
  const maturityFraction =
    maturityTarget && maturityTarget > 0 ? Math.min(diagnostics.sar_maturity / maturityTarget, 1) : null;
  const seqLength = config ? config.wild_type.length : null;

  return (
    <section className="rounded-xl border px-4 py-3.5" style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <Gauge size={12} style={{ color: "var(--accent)" }} />
          <span className="label text-[9.5px]">loop internals · round {diagnostics.round_number}</span>
          <InfoTip label="What these gauges are">
            None of these is a result. They are the quantities the loop uses to decide what to test next, and
            each carries the formula the backend publishes. The value of information, for instance, only
            scales the exploration slot's optimism bonus.
          </InfoTip>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span
            className="chip mono !text-[9.5px]"
            style={{ borderColor: "var(--accent-line)", color: "var(--accent)", background: "var(--accent-soft)" }}
            title={diagnostics.objective_note}
          >
            objective {diagnostics.objective}
          </span>
          <span
            className="chip mono !text-[9.5px]"
            style={{
              borderColor: diagnostics.steers_null_objective ? "rgba(244,63,94,0.4)" : "var(--border)",
              color: diagnostics.steers_null_objective ? "var(--danger)" : "var(--ink-faint)",
            }}
            title={
              diagnostics.steers_null_objective
                ? "The retired anneal is steering the objective. This would contradict the shipped rule."
                : "The exploit/explore weights are leftovers from a retired heuristic. They do not affect the score."
            }
          >
            not steering the score
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-3.5 sm:grid-cols-3">
        <GaugeCell
          label="How much a test is worth"
          value={num(diagnostics.voi, 4)}
          fraction={null}
          color="#0EA5E9"
          formula="H · (1 − 0.65·C)"
          token="VoI"
          hint="Value of information. Only scales the exploration slot's optimism bonus; it is not a phase switch."
        />
        <GaugeCell
          label="How well the position table is filled"
          value={num(diagnostics.sar_maturity, 4)}
          fraction={maturityFraction}
          color="#A855F7"
          formula={`target ${num(maturityTarget, 1)}`}
          token="sar_maturity"
          hint="The fraction of the position/residue table that has been corroborated by measurements, against this campaign's own target."
        />
        <GaugeCell
          label="How much the prior is trusted"
          value={num(diagnostics.eta_sar, 3)}
          fraction={null}
          color="#6366F1"
          formula="additive_prior_trust(L, N_qual, Q)"
          token="η_SAR"
          hint="How far the hand-written sequence prior is allowed to move a score, given the library size and the budget. Computed from the starting conditions only."
        />
        <GaugeCell
          label="Sequence length"
          value={seqLength === null ? "—" : `L = ${seqLength}`}
          fraction={null}
          color="var(--ink-dim)"
          formula={config ? `${config.qualified_count} of ${config.pool_size} candidates qualify` : "from the frozen protocol"}
          token="L · N_qual · Q"
          hint="The three numbers every other quantity is scaled against, read from the frozen protocol and the campaign totals."
        />
        <GaugeCell
          label="Evidence behind the position table"
          value={`${diagnostics.sar_support}`}
          fraction={null}
          color="var(--ink-dim)"
          formula={`${diagnostics.sar_basis} · ${diagnostics.position_tolerance_effects.length} positions`}
          token="sar_support"
          hint="How many independent observations the position/residue table is built from, and the estimator's basis."
        />
        <GaugeCell
          label="Retired heuristic weights"
          value={`${num(diagnostics.exploit_weight, 2)} / ${num(diagnostics.exploration_weight, 2)}`}
          fraction={null}
          color="var(--ink-faint)"
          formula="kept for audit · does not affect the score"
          token="exploit_weight · exploration_weight"
          hint={diagnostics.objective_note}
        />
      </div>

      {diagnostics.position_tolerance_effects.length ? (
        <div className="mt-3 border-t pt-2.5" style={{ borderColor: "var(--border)" }}>
          <div className="mb-1.5 flex items-center gap-1.5">
            <Sigma size={10} style={{ color: "var(--ink-faint)" }} />
            <span className="label text-[9px]">What each position does on average</span>
            <InfoTip label="How these are estimated">
              Each position's typical effect on binding, shrunk toward zero when there are few observations
              (<span className="mono">n / (n + 3)</span>); a position with <span className="mono">n &lt; 2</span>{" "}
              contributes nothing. An absent position is one the record has no repeated observation for — not a
              position with zero effect.
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
                    title={`Position ${effect.position}: average effect ${effect.shrunk_effect} over n = ${effect.n} observations`}
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
          <span title="The safety cut-off on developability risk, set at the pool median.">
            safety cut-off <span className="mono">{config.tau_dev.toFixed(4)}</span>
          </span>
          <span>
            pass rate{" "}
            <span className="mono">{(config.qualification_pass_rate * 100).toFixed(1)}%</span>
          </span>
          <span>
            qualified{" "}
            <span className="mono">
              {config.qualified_count}/{config.pool_size}
            </span>
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
  token,
  hint,
}: {
  label: string;
  value: string;
  fraction: number | null;
  color: string;
  formula: string;
  token: string;
  hint: string;
}) {
  return (
    <div className="min-w-0">
      <div className="label mb-0.5 flex items-center gap-1 whitespace-normal leading-tight">
        <span>{label}</span>
        <InfoTip label={`${token} — what it means`}>{hint}</InfoTip>
      </div>
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
          <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${fraction * 100}%`, background: color }} />
        </div>
      ) : null}
      <div className="mono mt-1 whitespace-normal text-[9.5px] leading-tight ink-faint">{formula}</div>
      <div className="mono mt-0.5 text-[8.5px] leading-tight ink-faint" style={{ opacity: 0.75 }}>
        {token}
      </div>
    </div>
  );
}
