import { AlertTriangle, ChevronDown, RefreshCw, Star } from "lucide-react";
import { useState } from "react";
import type { BenchmarkPayload, CrossTargetPayload } from "../../types/report";
import type { ConfigPayload, RoundsPayload, StatePayload } from "../../types/state";
import { Callout, DatasetStrip, InfoTip, KpiTile, Note } from "../../components/Panel";
import { Headline, TokenChip } from "../../components/Plain";
import { RoundTimeline } from "../../components/RoundTimeline";
import { TelemetryBar } from "./TelemetryBar";
import { CouncilPanel, FalsificationTracker } from "./CouncilPanel";
import { PlannerCard } from "./PlannerCard";
import { PlatePanel } from "./PlatePanel";
import { Hcdr3Ribbon } from "./Hcdr3Ribbon";
import { DiscoveryPanel } from "./DiscoveryPanel";
import { LiveRoundRunner } from "./LiveRoundRunner";
import { datasetIdentity, discoveredBinder, slotSplit } from "../../lib/derive";
import { int, num } from "../../lib/format";
import { foldText, librarySkipped, shareWhole } from "../../lib/plain";
import type { Resource } from "../../hooks/useResource";

interface Props {
  state: Resource<StatePayload>;
  rounds: RoundsPayload | null;
  config: ConfigPayload | null;
  /** Loaded alongside the lab so the ranking-accuracy chart has a source. */
  benchmark?: BenchmarkPayload | null;
  /**
   * Only for the dataset identity strip — the lab's own endpoints never say
   * which antibody these candidates belong to.
   */
  cross?: CrossTargetPayload | null;
  selectedRound: number;
  onSelectRound: (round: number) => void;
  onRestoreGolden: () => Promise<void>;
  restoreError: string | null;
  onReloadAll: () => void;
  playing: boolean;
  onPlayToggle: () => void;
}

/**
 * View 1 — the recorded campaign.
 *
 * The page answers, in order: *what did it find* (one fold-change number), *how
 * much did that cost* (35 tests out of the library), *did it tell the truth*
 * (hypotheses it disproved on itself), and *how did it decide* (the plate and
 * the scientist council).
 *
 * The engine's internal gauges (`VoI`, `η_SAR`, SAR maturity, the retired anneal
 * weights) are real and published, but they are not a demo surface: they live in
 * the "under the hood" disclosure at the bottom, where a judge who asks "how
 * does the loop actually pick?" can reach them in one click.
 */
export function LabView({
  state,
  rounds,
  config,
  benchmark,
  selectedRound,
  onSelectRound,
  onRestoreGolden,
  restoreError,
  onReloadAll,
  playing,
  onPlayToggle,
  cross = null,
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
          <AlertTriangle size={14} /> The campaign isn't loading
        </div>
        <p className="mt-1.5 text-[12px] leading-relaxed ink-dim">{state.error.message}</p>
        <button
          type="button"
          onClick={onReloadAll}
          className="chip mono mt-3"
          style={{ borderColor: "var(--border)", color: "var(--accent)" }}
        >
          <RefreshCw size={11} /> try again
        </button>
      </div>
    );
  }

  if (!state.data) {
    return (
      <div className="panel flex items-center gap-2 px-4 py-6 text-[12px] ink-faint">
        <RefreshCw size={13} className="animate-spin" /> Loading the campaign…
      </div>
    );
  }

  const payload = state.data;
  const identity = datasetIdentity(cross ?? null);
  const binder = discoveredBinder(payload);
  const split = slotSplit(payload);
  const isLive = payload.current_round === (rounds?.current_round ?? payload.current_round);
  const championSlot = payload.batch_slots.find((slot) => slot.candidate_id === binder?.id);

  // How much of the qualified library the lab never had to test.
  const poolSize = config?.pool_size ?? null;
  const qualified = config?.qualified_count ?? null;
  const spent = payload.kpis.query_budget_spent;
  const screened = qualified ?? poolSize;
  const skipped = librarySkipped(spent, screened);
  const disproved = payload.kpis.hypotheses_falsified;
  const graded = payload.kpis.hypotheses_supported + payload.kpis.hypotheses_falsified;

  return (
    <div className="space-y-3.5">
      {/* ── Hero band ─────────────────────────────────────────────────────── */}
      <section className="panel hero-wash">
        <div className="grid gap-5 px-5 py-4 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:items-center">
          <div className="min-w-0">
            <div className="mb-2 flex flex-wrap items-center gap-1.5">
              <span
                className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold"
                style={{ background: "var(--gold-soft)", color: "var(--gold)", border: "1px solid var(--gold-line)" }}
              >
                <Star size={10} fill="var(--gold)" />
                {isLive ? "BEST CANDIDATE IN THE WHOLE LIBRARY" : `BEST AS OF ROUND ${payload.current_round}`}
              </span>
              <span className="text-[11px] ink-faint">
                {isLive ? "the recorded run" : `replay · as of round ${payload.current_round}`} · {payload.phase}
              </span>
            </div>

            {/*
             * What the run is *about*. The panel below says "out of 210
             * candidates" without ever naming the antibody or the antigen, so a
             * reader could quote the fold-change without knowing what it measures.
             */}
            <DatasetStrip identity={identity} className="mb-3" />

            <Headline
              value={foldText(payload.kpis.delta_vs_wt)}
              label="tighter binding than the antibody we started from"
              tone="gold"
              hint={
                <>
                  Binding is measured on a log scale, so the reported improvement of{" "}
                  <span className="mono">{num(payload.kpis.delta_vs_wt, 3)}</span> means{" "}
                  <span className="mono">10^{num(payload.kpis.delta_vs_wt, 3)}</span> — that many times tighter
                  binding than the wild-type reference.
                </>
              }
              support={
                binder ? (
                  <>
                    <span className="mono" style={{ color: "var(--gold)" }}>{binder.id}</span> is the best binder
                    out of {identity?.usableRows ? int(identity.usableRows) : config ? int(config.qualified_count) : "the"}{" "}
                    usable variants
                    {identity?.antigen ? ` of ${identity.antigen.split(" (")[0]}` : ""}, found in slot{" "}
                    {championSlot?.slot_id ?? "—"} of round {payload.current_round} — after only{" "}
                    <span className="mono">{int(spent)}</span> lab tests.
                  </>
                ) : (
                  <>
                    Best qualifying candidate among the {payload.pareto_data.length} tested so far, after{" "}
                    <span className="mono">{int(spent)}</span> lab tests.
                  </>
                )
              }
            />

            <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
              <span
                className="chip mono !text-[10px]"
                style={{ borderColor: "var(--gold-line)", color: "var(--gold)", background: "var(--gold-soft)" }}
              >
                score {num(payload.kpis.best_qualified_binding, 2)}
              </span>
              <span className="text-[10.5px] ink-faint">
                every binding value is a measured assay from a published dataset — the model never grades
                itself
              </span>
              <TokenChip
                token="10 ^ Δ vs reference"
                meaning="Binding values are −log₁₀ K_D: one unit is a ten-fold change in binding strength."
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-2">
            <KpiTile
              label="Library skipped"
              value={skipped === null ? "—" : shareWhole(skipped)}
              caption={`${int(spent)} tests out of ${screened === null ? "—" : int(screened)} candidates`}
              emphasis="accent"
              hint="The loop names the five candidates to test each round, so most of the qualified library never needs an assay."
            />
            <KpiTile
              label="Rank in library"
              value="#1"
              caption={config ? `of ${int(config.pool_size)} candidates` : "best found"}
              emphasis="gold"
              hint="The champion is the highest measured binding anywhere in the pool — not merely the best of what was tested."
            />
            <KpiTile
              label="Hypotheses disproved"
              value={`${int(disproved)} of ${int(graded)}`}
              caption="by its own control wells"
              hint="Every hypothesis is graded against a matched control assayed in the same batch, so a wrong guess is caught in the round it was made. Publishing the failures is the point."
            />
            <KpiTile
              label="Rounds so far"
              value={`${payload.current_round} of ${totalRounds}`}
              caption={`${payload.phase}`}
              hint="Rounds completed against the campaign's planned horizon."
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
        <Callout tone="danger" title="Resetting the recorded run failed">
          {restoreError}
        </Callout>
      ) : null}

      {/* ── The batch + the scientists who chose it ────────────────────────── */}
      <div className="grid gap-3.5 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] xl:items-start">
        <div className="min-w-0 space-y-3.5">
          <Hcdr3Ribbon state={payload} wildType={config?.wild_type ?? "SRWGGDGCYAMDY"} />
          <PlatePanel state={payload} config={config} totalRounds={totalRounds} />
        </div>

        <div className="min-w-0 space-y-3.5">
          <CouncilPanel state={payload} />
          {/* The decision the three memos feed into — placed directly under the
              debate so the reading order is argument → conclusion. */}
          <PlannerCard state={payload} />
          <FalsificationTracker state={payload} />
        </div>
      </div>

      {/* ── What was found, and at what cost ───────────────────────────────── */}
      <DiscoveryPanel state={payload} config={config} rounds={rounds} benchmark={benchmark} />

      {/* ── Under the hood: everything a judge has to ask for ──────────────── */}
      <div className="space-y-2">
        <details className="disclosure panel px-4 py-2.5">
          <summary className="flex cursor-pointer items-center gap-1.5 text-[12px] font-medium ink-dim">
            <ChevronDown size={12} />
            Under the hood: live engine telemetry
          </summary>
          <div className="mt-3">
            <TelemetryBar state={payload} config={config} />
          </div>
        </details>

        <details className="disclosure panel px-4 py-2.5">
          <summary className="flex cursor-pointer items-center gap-1.5 text-[12px] font-medium ink-dim">
            <ChevronDown size={12} />
            How to read this view
          </summary>
          <div className="mt-3 space-y-2">
            <div className="grid gap-2 sm:grid-cols-3">
              <MiniNote title="Predictions are not measurements" tone="#0EA5E9">
                The model's predicted binding is fitted on already-tested candidates. Only the assay result
                comes back from the lab — and the audit grades the hypothesis on the gap between the two.
              </MiniNote>
              <MiniNote title="The scoring rule cannot be bent by the agents" tone="#F59E0B">
                The scientist council can re-order candidates but can never change the objective itself. The
                "unbiased objective" ablation in the last tab is the evidence.
              </MiniNote>
              <MiniNote title="One run is a demo, not a result" tone="#10B981">
                This view is a single recorded campaign. The 100-run comparison in the second tab is where the
                method claim lives.
              </MiniNote>
            </div>
            <Note summary="What is read-only, and what writes">
              Selecting an earlier round asks the server to rebuild that round from the recorded history — it
              never touches the database. The only write exposed here is the "reset to recorded run" button,
              behind its own click, plus the live-round stream.
            </Note>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] ink-faint">
              <span className="mono">round {payload.current_round}</span>
              <span className="mono">{payload.benchmark_curves.length} checkpoints</span>
              <span className="mono">{split.label}</span>
              <InfoTip label="Objective">
                The shipped objective is <span className="mono">a_null := a_base</span>. The agents' prior can
                re-order candidates but can never alter the score being optimised.
              </InfoTip>
            </div>
          </div>
        </details>
      </div>
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
