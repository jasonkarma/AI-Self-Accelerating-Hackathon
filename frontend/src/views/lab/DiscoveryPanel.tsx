import { useMemo } from "react";
import type { BenchmarkPayload } from "../../types/report";
import type { ConfigPayload, RoundsPayload, StatePayload } from "../../types/state";
import { ScatterPlot, LineChart, type ScatterPoint } from "../../components/charts";
import { InfoTip, Legend, Panel } from "../../components/Panel";
import { finalRho, rhoSeries } from "../../lib/derive";
import { num } from "../../lib/format";

/**
 * The discovery record: two charts, side by side, above the fold.
 *
 * Left is the only place in the console where the *developability* constraint is
 * visible at the same time as binding: every revealed candidate, with the
 * champion ringed. Right is the campaign's own staircase against the 100-seed
 * medians carried in the same payload — shown together so a reader cannot mistake
 * one seed for a method result.
 */
export function DiscoveryPanel({
  state,
  config,
  rounds,
  benchmark: benchmarkPayload,
}: {
  state: StatePayload;
  config: ConfigPayload | null;
  rounds?: RoundsPayload | null;
  /** The 100-run artifact — the only source of a ranking-accuracy series. */
  benchmark?: BenchmarkPayload | null;
}) {
  const threshold = config ? 1 - config.tau_dev : null;
  const best = state.kpis.best_qualified_binding;

  const points = useMemo<ScatterPoint[]>(
    () =>
      state.pareto_data.map((point) => {
        const isChampion = Math.abs(point.binding - best) < 5e-4;
        const isWildType = point.status === "wild_type";
        const isFrontier = point.status === "pareto_frontier";
        return {
          id: point.candidate_id,
          x: point.risk,
          y: point.binding,
          color: isChampion ? "#F0B429" : isWildType ? "#22D3EE" : isFrontier ? "#10B981" : "#64748B",
          radius: isChampion ? 6 : isFrontier ? 4.2 : 3,
          ring: !isFrontier && !isWildType && !isChampion,
          label: `${point.candidate_id} · ${point.status.replace(/_/g, " ")}`,
        };
      }),
    [state.pareto_data, best],
  );

  /*
   * The live staircase comes from `/api/rounds`, not from `quality_curve`.
   *
   * `quality_curve` is only published for reconstructed rounds, so on the live
   * state it is absent and the chart fell through to its empty branch — which is
   * why the "best candidate" panel could claim the opening batch hadn't finished
   * while the hero above it showed a finished campaign. The timeline carries the
   * same per-round `best_qualified_binding` for every round, including the last,
   * so it is the honest source for both the staircase and its final point.
   */
  const curve: Array<{ query_count: number; best: number }> = (() => {
    const live = (rounds?.rounds ?? [])
      .filter((entry) => Number.isFinite(entry.best_qualified_binding))
      .map((entry) => ({ query_count: entry.query_count, best: entry.best_qualified_binding }));
    if (live.length) return live;
    return (state.quality_curve ?? []).map((point) => ({
      query_count: point.query_count,
      best: point.best_qualified_binding,
    }));
  })();
  // Ranking accuracy over the 100 runs, per checkpoint. `null` for a live state
  // with no benchmark artifact loaded, which the panel renders as an explicit
  // "not published for one run" rather than an empty axis.
  const accuracySeries = useMemo(
    () => (benchmarkPayload ? rhoSeries(benchmarkPayload) : []),
    [benchmarkPayload],
  );
  const finalAccuracy = useMemo(
    () =>
      benchmarkPayload
        ? finalRho(
          benchmarkPayload.spearman_rho_curves as unknown as Array<Record<string, number | null>>,
        )
        : null,
    [benchmarkPayload],
  );


  // The champion's own draw order, so the scatter can be read as a discovery.
  const championId = state.pareto_data.find((point) => Math.abs(point.binding - best) < 5e-4)?.candidate_id;

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Panel
        eyebrow="What was found"
        title="Every candidate tested: how well it binds vs. how drug-like it is"
        lede={
          championId
            ? `${state.pareto_data.length} candidates tested. ${championId} is the best binder anywhere in the library — ringed in gold.`
            : `${state.pareto_data.length} candidates tested so far.`
        }
        actions={
          <Legend
            entries={[
              { label: "best", color: "#F0B429" },
              { label: "top 10", color: "#10B981" },
              { label: "tested", color: "#64748B" },
              { label: "start", color: "#22D3EE" },
            ]}
          />
        }
      >
        {points.length ? (
          <ScatterPlot
            points={points}
            height={264}
            xLabel="safety risk → (further right is worse)"
            yLabel="binding strength →"
            ariaLabel="Every tested candidate: safety risk on one axis, measured binding on the other"
            thresholdY={{ value: best, label: `best found ${num(best, 2)}` }}
          />
        ) : (
          <p className="text-[12px] ink-faint">Nothing measured at this round yet.</p>
        )}
        <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] ink-faint">
          <span>
            {state.pareto_data.filter((point) => point.status === "pareto_frontier").length} in the best-known
            top 10
          </span>
          {threshold !== null && config ? (
            <span className="mono">
              candidates pass the safety filter while risk stays under {threshold.toFixed(2)}
            </span>
          ) : null}
          <InfoTip label="Why the safety axis matters">
            A candidate that fails the safety filter cannot enter the shortlist or the headline measure,
            however tightly it binds. That is why the lab spends slots on drug-likeness rather than chasing
            raw affinity alone. The exact threshold is{" "}
            <span className="mono">τ_dev = {config?.tau_dev.toFixed(4) ?? "—"}</span>, the pool median.
          </InfoTip>
        </div>
      </Panel>

      <Panel
        eyebrow="The climb, and what it means"
        title="Best binder found, and the measure that predicts it"
        lede="The top chart is this one recorded run. The bottom chart is the axis the method comparison rests on — ranking accuracy, averaged over 100 runs."
        actions={
          <Legend
            entries={[
              { label: "this run", color: "#22D3EE" },
              { label: "ranking accuracy · ours", color: "#10B981" },
              { label: "ranking accuracy · standard", color: "#6366F1" },
            ]}
          />
        }
      >
        {curve.length ? (
          <LineChart
            x={curve.map((point) => point.query_count)}
            series={[
              {
                key: "live",
                label: "This run",
                color: "#22D3EE",
                values: curve.map((point) => point.best),
                width: 2.4,
              },
            ]}
            height={196}
            xLabel="lab tests spent"
            yLabel="best binder so far"
            formatY={(value) => value.toFixed(2)}
            ariaLabel="Best binder found over the recorded campaign"
            referenceLines={[{ value: best, label: `final ${num(best, 2)}`, color: "#10B981" }]}
          />
        ) : (
          <p className="text-[12px] ink-faint">The opening batch hasn't completed yet.</p>
        )}

        {/*
         * This used to be a second copy of the same binding curve, averaged over
         * 100 runs. Two line charts of almost the same number, stacked, invited
         * exactly the question the panel could not answer ("why is 8.52 not
         * 8.58?"). The average is still available in the hero's tooltip and in
         * the metric table; what the lab view was missing is the axis the whole
         * method claim rests on, which only the benchmark artifact publishes.
         */}
        <div className="mt-3 border-t pt-3" style={{ borderColor: "var(--border)" }}>
          <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2">
            <span className="label text-[9.5px]">Ranking accuracy by tests spent · 100-run average</span>
            <span className="mono text-[9.5px] ink-faint">0 = guessing · 1 = perfect</span>
          </div>
          {accuracySeries.length > 1 ? (
            <LineChart
              x={accuracySeries.map((row) => row.query_count)}
              series={[
                {
                  key: "m5_hybrid",
                  label: "Virtual Lab (ours)",
                  color: "#10B981",
                  values: accuracySeries.map((row) => row.values.m5_hybrid),
                  width: 2.4,
                },
                {
                  key: "m3_ucb",
                  label: "Standard UCB",
                  color: "#6366F1",
                  values: accuracySeries.map((row) => row.values.m3_ucb),
                  dashed: true,
                  width: 1.5,
                },
                {
                  key: "m2_greedy",
                  label: "Greedy",
                  color: "#F59E0B",
                  values: accuracySeries.map((row) => row.values.m2_greedy),
                  dashed: true,
                  width: 1.3,
                },
              ]}
              height={150}
              xLabel="lab tests spent"
              yLabel="ranking accuracy"
              formatY={(value) => value.toFixed(2)}
              referenceLines={[{ value: 0, label: "0 · guessing", color: "var(--border-strong)" }]}
              ariaLabel="Ranking accuracy by tests spent, averaged over 100 runs"
            />
          ) : (
            <p className="text-[11px] ink-faint">
              Ranking accuracy is only published for the 100-run benchmark, not for a single recorded campaign.
            </p>
          )}
          <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 text-[10px] ink-faint">
            <span>
              at the final checkpoint{" "}
              <span className="mono" style={{ color: "#10B981" }}>
                ours {num(finalAccuracy?.m5_hybrid ?? null, 2)}
              </span>{" "}
              vs{" "}
              <span className="mono" style={{ color: "#6366F1" }}>
                standard {num(finalAccuracy?.m3_ucb ?? null, 2)}
              </span>
            </span>
            <InfoTip label="Why this chart is not about one run">
              Ranking accuracy is a property of a scoring model across the whole library, not of a single
              draw, so it is only meaningful averaged over independent runs. This run cannot have its own
              curve without re-deriving it from the ledger — the honest statement is that the recorded run{" "}
              <em>used</em> the loop, and the 100-run benchmark measures whether the loop ranks better.
            </InfoTip>
          </div>
        </div>
      </Panel>
    </div>
  );
}
