import { useMemo } from "react";
import type { ConfigPayload, StatePayload } from "../../types/state";
import { ScatterPlot, LineChart, type ScatterPoint } from "../../components/charts";
import { InfoTip, Legend, Panel } from "../../components/Panel";
import { bestMethod, METHODS } from "../../lib/derive";
import { methodMeta, num } from "../../lib/format";
import { methodName } from "../../lib/plain";

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
        eyebrow="The climb"
        title="How good the best candidate got, round by round"
        lede="The recorded campaign's own staircase, next to what the same loop and its competitors average over 100 runs."
        actions={
          <Legend
            entries={[
              { label: "this run", color: "#22D3EE" },
              { label: "Virtual Lab", color: "#10B981" },
              { label: "Standard", color: "#6366F1" },
              { label: "Greedy", color: "#F59E0B" },
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
                values: curve.map((point) => point.best_qualified_binding),
                width: 2.4,
              },
            ]}
            height={182}
            xLabel="lab tests spent"
            yLabel="best binder so far"
            formatY={(value) => value.toFixed(2)}
            ariaLabel="Best binder found over the recorded campaign"
            referenceLines={[{ value: best, label: `final ${num(best, 2)}`, color: "#10B981" }]}
          />
        ) : (
          <p className="text-[12px] ink-faint">The opening batch hasn't completed yet.</p>
        )}
        <div className="mt-2 border-t pt-2" style={{ borderColor: "var(--border)" }}>
          <LineChart
            x={x}
            series={METHODS.map((method) => ({
              key: method,
              label: methodName(method),
              color: methodMeta(method).color,
              values: benchmark.map((row) => row[method]),
              dashed: method !== "m5_hybrid",
              width: method === "m5_hybrid" ? 2.2 : 1.3,
            }))}
            height={116}
            xLabel="lab tests spent"
            yLabel="average over 100 runs"
            formatY={(value) => value.toFixed(2)}
            ariaLabel="Average best binder found over 100 runs, per method"
          />
          <div className="mt-1 flex flex-wrap items-center justify-between gap-2 text-[10px] ink-faint">
            <span>
              strongest final average{" "}
              <span className="mono" style={{ color: finalBest ? methodMeta(finalBest.method).color : undefined }}>
                {finalBest ? `${methodName(finalBest.method)} ${num(finalBest.value, 2)}` : "—"}
              </span>
            </span>
            <InfoTip label="One run is a demo, not a result">
              A single run beating a hundred-run average proves the loop works, not that the method wins. The
              method comparison lives in the second tab.
            </InfoTip>
          </div>
        </div>
      </Panel>
    </div>
  );
}
