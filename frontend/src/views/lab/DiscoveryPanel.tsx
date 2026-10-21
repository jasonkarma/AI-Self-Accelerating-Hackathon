import { useMemo } from "react";
import type { ConfigPayload, StatePayload } from "../../types/state";
import { ScatterPlot, LineChart, type ScatterPoint } from "../../components/charts";
import { InfoTip, Legend, Panel } from "../../components/Panel";
import { bestMethod, METHODS } from "../../lib/derive";
import { methodMeta, num } from "../../lib/format";

/**
 * The discovery record: two charts, side by side, above the fold.
 *
 * Left is the only place in the console where the *developability* constraint is
 * visible at the same time as binding: every revealed candidate, with the
 * champion ringed. Right is the campaign's own staircase against the 100-seed
 * medians carried in the same payload — shown together so a reader cannot mistake
 * one seed for a method result.
 */
export function DiscoveryPanel({ state, config }: { state: StatePayload; config: ConfigPayload | null }) {
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

  const curve = state.quality_curve ?? [];
  const benchmark = state.benchmark_curves;
  const x = benchmark.map((row) => row.query_count);

  const finalBest = useMemo(() => {
    const lastRow = benchmark[benchmark.length - 1];
    if (!lastRow) return null;
    const record: Record<string, number | null> = {};
    for (const method of METHODS) record[method] = lastRow[method];
    return bestMethod(record);
  }, [benchmark]);

  // The champion's own draw order, so the scatter can be read as a discovery.
  const championId = state.pareto_data.find((point) => Math.abs(point.binding - best) < 5e-4)?.candidate_id;

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Panel
        eyebrow="Discovery record"
        title="Revealed pool: binding vs developability risk"
        lede={
          championId
            ? `${state.pareto_data.length} candidates assayed. ${championId} is the global #1 — ringed in gold.`
            : `${state.pareto_data.length} candidates assayed so far.`
        }
        actions={
          <Legend
            entries={[
              { label: "champion", color: "#F0B429" },
              { label: "frontier", color: "#10B981" },
              { label: "revealed", color: "#64748B" },
              { label: "WT", color: "#22D3EE" },
            ]}
          />
        }
      >
        {points.length ? (
          <ScatterPlot
            points={points}
            height={264}
            xLabel="risk (1 − y_dev)"
            yLabel="−log₁₀ K_D"
            ariaLabel="Scatter of revealed candidates: developability risk against measured binding"
            thresholdY={{ value: best, label: `best qualified ${num(best, 3)}` }}
          />
        ) : (
          <p className="text-[12px] ink-faint">No reveals yet at this round.</p>
        )}
        <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] ink-faint">
          <span>
            {state.pareto_data.filter((point) => point.status === "pareto_frontier").length} on the top-10
            frontier
          </span>
          {threshold !== null && config ? (
            <span className="mono">
              τ_dev {config.tau_dev.toFixed(4)} → risk ≤ {threshold.toFixed(2)} qualifies
            </span>
          ) : null}
          <InfoTip label="Why the risk axis matters">
            A candidate right of the τ_dev threshold cannot enter the top-5 panel or AUBQC, however well it
            binds. That is why the lab spends slots on qualification rather than chasing raw affinity.
          </InfoTip>
        </div>
      </Panel>

      <Panel
        eyebrow="Campaign trajectory"
        title="Best qualified binding, live vs 100-seed medians"
        lede="The campaign's own staircase against the benchmark medians carried in the same payload."
        actions={
          <Legend
            entries={[
              { label: "live", color: "#22D3EE" },
              { label: "M5", color: "#10B981" },
              { label: "M3", color: "#6366F1" },
              { label: "M2", color: "#F59E0B" },
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
                label: "Live campaign",
                color: "#22D3EE",
                values: curve.map((point) => point.best_qualified_binding),
                width: 2.4,
              },
            ]}
            height={182}
            xLabel="assays spent (q)"
            yLabel="best qualified −log₁₀ K_D"
            formatY={(value) => value.toFixed(2)}
            ariaLabel="Live campaign best qualified binding by query count"
            referenceLines={[{ value: best, label: `final ${num(best, 3)}`, color: "#10B981" }]}
          />
        ) : (
          <p className="text-[12px] ink-faint">The seed round has no completed batch yet.</p>
        )}
        <div className="mt-2 border-t pt-2" style={{ borderColor: "var(--border)" }}>
          <LineChart
            x={x}
            series={METHODS.map((method) => ({
              key: method,
              label: methodMeta(method).short,
              color: methodMeta(method).color,
              values: benchmark.map((row) => row[method]),
              dashed: method !== "m5_hybrid",
              width: method === "m5_hybrid" ? 2.2 : 1.3,
            }))}
            height={116}
            xLabel="q"
            yLabel="100-seed median"
            formatY={(value) => value.toFixed(2)}
            ariaLabel="Matched 100-seed median benchmark curves"
          />
          <div className="mt-1 flex flex-wrap items-center justify-between gap-2 text-[10px] ink-faint">
            <span>
              best final checkpoint{" "}
              <span className="mono" style={{ color: finalBest ? methodMeta(finalBest.method).color : undefined }}>
                {finalBest ? `${methodMeta(finalBest.method).short} ${num(finalBest.value, 3)}` : "—"}
              </span>
            </span>
            <InfoTip label="One seed is a demo, not a result">
              A single seed beating a 100-seed median proves the loop runs, not that the method wins. The
              method comparison lives in the 100-seed benchmark panel.
            </InfoTip>
          </div>
        </div>
      </Panel>
    </div>
  );
}
