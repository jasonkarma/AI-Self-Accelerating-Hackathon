import { AlertTriangle, RefreshCw, Star } from "lucide-react";
import { useState } from "react";
import type { ConfigPayload, RoundsPayload, StatePayload } from "../../types/state";
import { Callout, InfoTip, KpiTile, Note, Panel } from "../../components/Panel";
import { RoundTimeline } from "../../components/RoundTimeline";
import { TelemetryBar } from "./TelemetryBar";
import { CouncilPanel, FalsificationTracker } from "./CouncilPanel";
import { PlatePanel } from "./PlatePanel";
import { DiscoveryPanel } from "./DiscoveryPanel";
import { LiveRoundRunner } from "./LiveRoundRunner";
import { discoveredBinder, slotSplit } from "../../lib/derive";
import { int, num, percent, signed } from "../../lib/format";
import type { Resource } from "../../hooks/useResource";

interface Props {
  state: Resource<StatePayload>;
  rounds: RoundsPayload | null;
  config: ConfigPayload | null;
  selectedRound: number;
  onSelectRound: (round: number) => void;
  onRestoreGolden: () => Promise<void>;
  restoreError: string | null;
  onReloadAll: () => void;
  playing: boolean;
  onPlayToggle: () => void;
}

/**
 * View 1 — the golden Seed-21 campaign.
 *
 * The layout is a workspace, not a report scroll:
 *
 * * **Header band** — the claim in one line plus five KPIs.
 * * **Round stepper** — a compact two-line timeline, with the live-round control
 *   embedded in its toolbar so no separate drawer is needed.
 * * **Left 60 %** — the five-slot plate, then the two discovery charts.
 * * **Right 40 %** — the scientist council (tabbed), the falsification tracker
 *   and the loop telemetry.
 *
 * Everything a judge needs to see the *result* sits above the fold; the
 * methodology that used to occupy three paragraphs above the plate is now one
 * `Note` disclosure at the bottom of the page.
 */
export function LabView({
  state,
  rounds,
  config,
  selectedRound,
  onSelectRound,
  onRestoreGolden,
  restoreError,
  onReloadAll,
  playing,
  onPlayToggle,
}: Props) {
  const [restoring, setRestoring] = useState(false);
  const totalRounds = rounds?.total_rounds ?? 6;

  async function handleRestore() {
    setRestoring(true);
    try {
      await onRestoreGolden();
    } finally {
      setRestoring(false);
    }
  }

  if (state.status === "error") {
    return (
      <div className="panel px-4 py-6">
        <div className="flex items-center gap-2 text-[13px] font-semibold" style={{ color: "var(--danger)" }}>
          <AlertTriangle size={14} /> Could not load the campaign state
        </div>
        <p className="mt-1.5 text-[12px] leading-relaxed ink-dim">{state.error.message}</p>
        <button
          type="button"
          onClick={onReloadAll}
          className="chip mono mt-3"
          style={{ borderColor: "var(--border)", color: "var(--accent)" }}
        >
          <RefreshCw size={11} /> retry
        </button>
      </div>
    );
  }

  if (!state.data) {
    return (
      <div className="panel flex items-center gap-2 px-4 py-6 text-[12px] ink-faint">
        <RefreshCw size={13} className="animate-spin" /> Loading campaign state…
      </div>
    );
  }

  const payload = state.data;
  const binder = discoveredBinder(payload);
  const split = slotSplit(payload);
  const isLive = payload.current_round === (rounds?.current_round ?? payload.current_round);
  const championSlot = payload.batch_slots.find((slot) => slot.candidate_id === binder?.id);

  // Screening efficiency: assays run vs the assays a full pool sweep would need.
  const poolSize = config?.pool_size ?? null;
  const qualified = config?.qualified_count ?? null;
  const spent = payload.kpis.query_budget_spent;
  const screened = qualified ?? poolSize;
  const efficiency = screened !== null && screened > 0 ? 1 - spent / screened : null;

  return (
    <div className="space-y-3.5">
      {/* ── Hero band ─────────────────────────────────────────────────────── */}
      <section className="panel hero-wash overflow-hidden">
        <div className="grid gap-5 px-5 py-4 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:items-center">
          <div className="min-w-0">
            <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
              <span
                className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold"
                style={{ background: "var(--gold-soft)", color: "var(--gold)", border: "1px solid var(--gold-line)" }}
              >
                <Star size={10} fill="var(--gold)" />
                {isLive ? "GLOBAL #1 BINDER FOUND" : `BEST AS OF ROUND ${payload.current_round}`}
              </span>
              <span className="text-[11px] ink-faint">
                {isLive ? "Live campaign" : `Reconstructed · as of round ${payload.current_round}`} ·{" "}
                {payload.phase}
              </span>
            </div>
            <h2 className="text-[24px] font-semibold leading-tight tracking-[-0.015em] ink sm:text-[27px]">
              {binder ? (
                <>
                  <span className="mono" style={{ color: "var(--gold)" }}>
                    {num(binder.binding, 3)}
                  </span>{" "}
                  −log₁₀ K<sub>D</sub> {isLive ? "in" : "after"}{" "}
                  <span className="mono">{int(spent)}</span> assays
                </>
              ) : (
                <>
                  {num(payload.kpis.best_qualified_binding, 3)} best qualified after{" "}
                  <span className="mono">{int(spent)}</span> assays
                </>
              )}
            </h2>
            <p className="mt-1.5 max-w-[62ch] text-[12.5px] leading-relaxed ink-dim">
              {binder ? (
                <>
                  <span className="mono" style={{ color: "var(--ink)" }}>
                    {binder.id}
                  </span>{" "}
                  is the global #1 in the {config ? int(config.pool_size) : "candidate"} pool, found in slot{" "}
                  {championSlot?.slot_id ?? "—"} of round {payload.current_round}. Δ vs the frozen reference is{" "}
                  <span className="mono" style={{ color: "var(--accent)" }}>
                    {signed(payload.kpis.delta_vs_wt, 3)}
                  </span>
                  .
                </>
              ) : (
                <>
                  Best qualified candidate among the {payload.pareto_data.length} revealed so far. Δ vs the
                  frozen reference is <span className="mono">{signed(payload.kpis.delta_vs_wt, 3)}</span>.
                </>
              )}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            <KpiTile
              label="Best qualified"
              value={num(payload.kpis.best_qualified_binding, 3)}
              caption={`Δ ${signed(payload.kpis.delta_vs_wt, 3)} vs reference`}
              emphasis="gold"
              hint="Max measured −log₁₀ K_D among candidates that passed the developability gate."
            />
            <KpiTile
              label="Screening efficiency"
              value={screened === null ? "—" : `${int(spent)} / ${int(screened)}`}
              caption={efficiency === null ? "qualified assays" : `saved ${percent(efficiency, 1)} of a full sweep`}
              emphasis="accent"
              hint="Assays actually run, against the number of qualified candidates a full pool sweep would require."
            />
            <KpiTile
              label="Hypotheses graded"
              value={`${payload.kpis.hypotheses_supported}✓ ${payload.kpis.hypotheses_falsified}✗`}
              caption="paired H1 vs H0, wet-lab graded"
              hint="Each hypothesis is graded against its own paired negative control in the same batch."
            />
            <KpiTile
              label="Candidates revealed"
              value={int(payload.pareto_data.length)}
              caption={`of ${config ? int(config.pool_size) : "—"} in the pool`}
              hint="Assayed candidates visible in the discovery scatter."
            />
            <KpiTile
              label="Batch split"
              value={split.graduated ? "5 : 0 : 0" : `${split.exploit} : ${split.voi} : ${split.h0}`}
              caption={split.graduated ? "harvest phase" : "exploit : VoI : control"}
              hint="Counted from the five selected slots' own rules, not asserted."
            />
            <KpiTile
              label="Round horizon"
              value={`${payload.current_round}/${totalRounds}`}
              caption={`${payload.phase} · VoI ${num(payload.voi_diagnostics.voi, 3)}`}
              hint="Rounds spent against the campaign's budgeted horizon, with the current value of information."
            />
          </div>
        </div>
      </section>

      <RoundTimeline
        rounds={rounds}
        selected={selectedRound}
        onSelect={onSelectRound}
        playing={playing}
        onPlayToggle={onPlayToggle}
        onRestoreGolden={handleRestore}
        busy={restoring}
        liveControl={isLive ? <LiveRoundRunner currentRound={payload.current_round} totalRounds={totalRounds} onRoundComplete={onReloadAll} /> : null}
      />

      {restoreError ? (
        <Callout tone="danger" title="Restoring the golden ledger failed">
          {restoreError}
        </Callout>
      ) : null}

      {/* ── Split workspace: plate + live record on the left, reasoning right ── */}
      <div className="grid gap-3.5 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] xl:items-start">
        <div className="min-w-0">
          <PlatePanel state={payload} config={config} totalRounds={totalRounds} />
        </div>

        <div className="min-w-0 space-y-3.5">
          <CouncilPanel state={payload} />
          <FalsificationTracker state={payload} />
          <TelemetryBar state={payload} config={config} />
        </div>
      </div>

      {/* ── The two discovery charts, side by side, beneath the split ─────── */}
      <DiscoveryPanel state={payload} config={config} />

      {/* ── Methodology, one disclosure ───────────────────────────────────── */}
      <Panel eyebrow="Methodology" title="How to read this view" flush={false}>
        <div className="grid gap-2 sm:grid-cols-3">
          <MiniNote title="μ_bind is a prediction, not a measurement" tone="#0EA5E9">
            The surrogate's <span className="mono">μ_bind ± σ_bind</span> is fitted on revealed rows. Only{" "}
            <span className="mono">−log₁₀ K_D</span> came back from the wet lab; the audit grades the
            hypothesis on the difference between the two.
          </MiniNote>
          <MiniNote title="exploit / explore weights do not steer a_null" tone="#F59E0B">
            They report what the retired VoI anneal <em>would</em> have done.{" "}
            <span className="mono">steers_null_objective: false</span> is published by the backend, and the{" "}
            <span className="mono">no_anneal</span> ablation in View 4 is the evidence.
          </MiniNote>
          <MiniNote title="One seed is a demo, not a result" tone="#10B981">
            This view is a single recorded campaign. The 100-seed panel in View 2 is where the method
            comparison lives.
          </MiniNote>
        </div>
        <Note summary="Data access and write scope">
          Selecting an earlier round issues <span className="mono">GET /api/state?round=k</span>, which
          reconstructs that round from the ledger prefix and never touches the database. The only write
          exposed here is <span className="mono">POST /api/restore-golden</span>, behind its own button, plus
          the live-round stream.
        </Note>
        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] ink-faint">
          <span className="mono">payload round {payload.current_round}</span>
          <span className="mono">{payload.benchmark_curves.length} checkpoints</span>
          <InfoTip label="Objective law">
            The shipped objective is <span className="mono">a_null := a_base</span>. The agent prior can
            re-order candidates but can never alter the metric.
          </InfoTip>
        </div>
      </Panel>
    </div>
  );
}

function MiniNote({ title, tone, children }: { title: string; tone: string; children: React.ReactNode }) {
  return (
    <div
      className="rounded-xl border px-3 py-2.5"
      style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}
    >
      <div className="text-[11.5px] font-semibold leading-snug" style={{ color: tone }}>
        {title}
      </div>
      <p className="mt-1 text-[11px] leading-relaxed ink-faint">{children}</p>
    </div>
  );
}
