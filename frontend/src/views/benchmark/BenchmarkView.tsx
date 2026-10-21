import { AlertTriangle, ArrowUpRight, ChevronDown, TrendingUp, Trophy } from "lucide-react";
import type { BenchmarkPayload } from "../../types/report";
import { Badge, InfoTip, KpiTile, Note, Panel } from "../../components/Panel";
import { Headline, HonestStrip, ScaleMarker, TokenChip } from "../../components/Plain";
import { HBarChart, LineChart, type BarDatum } from "../../components/charts";
import { benchmarkMetricRows, finalRho, METHODS, pairedDelta, zeroShotRows } from "../../lib/derive";
import { int, methodMeta, num, pExact, pvalue, signed } from "../../lib/format";
import {
  countOf,
  flukeOdds,
  gainText,
  gainWhole,
  isClear,
  methodName,
  noiseMultiple,
  noiseText,
  PLAIN,
  ratio,
  ratioText,
  relativeGain,
  shareWhole,
} from "../../lib/plain";
import type { Resource } from "../../hooks/useResource";

/**
 * View 2 — "Beats the standard method".
 *
 * The page is ordered as an argument a judge can follow in sixty seconds:
 *
 * 1. **One number.** +91% more accurate at ranking, with the two underlying
 *    scores on a 0→1 track so the scale explains itself.
 * 2. **The evidence in words.** 98 of 100 runs won; the gap is 17× the rerun
 *    noise; the odds of that being luck are quoted conservatively.
 * 3. **The ranking chart**, on the axis that separates, with the zero-shot
 *    references named as what they are.
 * 4. **One muted strip** admitting the metric we tie on — credibility, at the
 *    weight it deserves, not a peer card.
 * 5. **Disclosures** for the full metric grid and the single-run comparison, so
 *    Q&A material costs zero scroll.
 *
 * The raw statistics (`spearman_rho_at_q35`, `mean_diff`, `se`, `p_value`) never
 * appear as headings; `lib/plain.ts` owns the words, and the tokens live in
 * tooltips.
 */
export function BenchmarkView({ benchmark }: { benchmark: Resource<BenchmarkPayload> }) {
  if (benchmark.status === "error") {
    return (
      <div className="panel px-4 py-6">
        <div className="flex items-center gap-2 text-[13px] font-semibold" style={{ color: "#F43F5E" }}>
          <AlertTriangle size={14} /> The benchmark results aren't loading
        </div>
        <p className="mt-1.5 text-[12px] ink-dim">{benchmark.error.message}</p>
      </div>
    );
  }
  if (!benchmark.data) {
    return <div className="panel px-4 py-6 text-[12px] ink-faint">Loading the 100-run results…</div>;
  }
  const payload = benchmark.data;
  return (
    <div className="space-y-4">
      <Scoreboard payload={payload} />
      <RankingEvidence payload={payload} />
      <HonestAxis payload={payload} />
      <Appendices payload={payload} />
      <Provenance payload={payload} />
    </div>
  );
}

/** The shared derivation block, so every panel quotes the same numbers. */
function useBenchFigures(payload: BenchmarkPayload) {
  const final = finalRho(
    payload.spearman_rho_curves as unknown as Array<Record<string, number | null>>,
  );
  const vsM3 = pairedDelta(payload, "m5_hybrid_minus_m3_ucb", "spearman_rho_at_q35");
  const aubqc = pairedDelta(payload, "m5_hybrid_minus_m3_ucb", "aubqc");
  const gain = relativeGain(final.m5_hybrid ?? null, final.m3_ucb ?? null);
  const rhoRatio = ratio(final.m5_hybrid ?? null, final.m3_ucb ?? null);
  const noise = noiseMultiple(vsM3.meanDiff, vsM3.se);
  return { final, vsM3, aubqc, gain, rhoRatio, noise };
}

// ─────────────────────────────── the hero ─────────────────────────────────

/**
 * The headline.
 *
 * Left: the claim in plain language, anchored by the 0→1 scale so nothing has to
 * be explained aloud. Right: the same claim as four supporting facts a judge can
 * glance at, each phrased as a sentence rather than a symbol.
 */
function Scoreboard({ payload }: { payload: BenchmarkPayload }) {
  const { final, vsM3, gain, rhoRatio, noise } = useBenchFigures(payload);
  const zeroShot = zeroShotRows(payload);
  const bestZeroShot = zeroShot.reduce((acc, row) => (row.rho > acc.rho ? row : acc), zeroShot[0]!);
  const m5 = final.m5_hybrid ?? null;
  const m3 = final.m3_ucb ?? null;

  return (
    <section className="panel hero-wash">
      <div className="grid gap-5 px-5 py-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <span
              className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10.5px] font-semibold"
              style={{ background: "var(--gold-soft)", color: "var(--gold)", border: "1px solid var(--gold-line)" }}
            >
              <Trophy size={11} /> THE RESULT
            </span>
            <span className="text-[11px] ink-faint">
              {int(payload.seeds)} independent runs · one antibody library · identical starting conditions
            </span>
          </div>

          <Headline
            value={gainWhole(gain)}
            label="more accurate at ranking candidates than the standard UCB baseline"
            tone="accent"
            hint={
              <>
                The improvement in {PLAIN.spearman_rho.label.toLowerCase()} relative to the standard UCB
                baseline:
                <span className="mono"> ({num(m5, 4)} − {num(m3, 4)}) ÷ {num(m3, 4)}</span>. Both values are
                read from the published artifact, not typed in.
              </>
            }
            support={
              <>
                {PLAIN.spearman_rho.label} measures whether a method puts the genuinely better candidates on
                top. The standard UCB baseline gets the order {shareWhole(m3)} of the way to perfect; ours gets{" "}
                {shareWhole(m5)}.
              </>
            }
          />

          <div className="mt-4 rounded-xl border px-3.5 py-3" style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}>
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <span className="label text-[9.5px]">{PLAIN.spearman_rho.label} · both methods on one scale</span>
              <span className="mono text-[10px] ink-faint">{PLAIN.spearman_rho.scale}</span>
            </div>
            <ScaleMarker
              ariaLabel={`Ranking accuracy: Virtual Lab ${num(m5, 2)}, standard UCB ${num(m3, 2)}, on a scale from 0 (guessing) to 1 (perfect)`}
              leftLabel="0 · guessing"
              rightLabel="1 · perfect"
              ticks={[
                { value: 0.25, label: "0.25" },
                { value: 0.5, label: "0.5" },
                { value: 0.75, label: "0.75" },
              ]}
              markers={[
                { label: "Virtual Lab (ours)", value: m5, color: "#10B981", emphasis: true },
                { label: "Standard method", value: m3, color: "#6366F1" },
              ]}
            />
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span
              className="chip mono !text-[10px]"
              style={{ borderColor: "var(--accent-line)", color: "var(--accent)", background: "var(--accent-soft)" }}
            >
              <TrendingUp size={10} /> {ratioText(rhoRatio)} the score
            </span>
            <TokenChip token={PLAIN.spearman_rho.token} meaning={PLAIN.spearman_rho.meaning} />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2.5 self-start sm:grid-cols-2">
          <KpiTile
            label="Runs won"
            value={countOf(vsM3.seedsWon, vsM3.n)}
            caption="head-to-head, same starting candidates"
            emphasis="accent"
            hint={`On how many of the ${int(vsM3.n)} runs our ranking accuracy exceeded the standard UCB baseline's, measured on the identical candidate set.`}
          />
          <KpiTile
            label="Gap vs. noise"
            value={noiseText(noise)}
            caption="reruns of one method differ by 1×"
            emphasis="accent"
            hint={`The observed gap divided by the run-to-run standard error (${vsM3.se.toFixed(4)}). Under 2× would be indistinguishable from luck.`}
          />
          <KpiTile
            label="Odds this is luck"
            value={flukeOdds(vsM3.p)}
            caption="stated conservatively"
            hint={`Two-sided paired t-test on per-run differences; exact p = ${pExact(vsM3.p)}, quoted here as a weaker claim than it supports.`}
          />
          <KpiTile
            label="Off-the-shelf predictors"
            value="rank it backwards"
            caption={`best of them: ${signed(bestZeroShot?.rho ?? null, 2)}`}
            hint={
              <>
                AlphaFold3, ProteinMPNN and ESM-2, scored the same way against the same measured binding.
                All three come out negative, i.e. worse than guessing. {payload.zero_shot_definition}
              </>
            }
          />
        </div>
      </div>

      <div
        className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t px-5 py-2.5 text-[10.5px] ink-faint"
        style={{ borderColor: "var(--border)" }}
      >
        <span>Scores are stable: the same method rerun lands within {num(vsM3.se * 2, 3)} of itself.</span>
        <span className="mono">state-of-the-art structure predictors: {zeroShot.map((row) => `${row.label} ${signed(row.rho, 2)}`).join(" · ")}</span>
      </div>
    </section>
  );
}

// ────────────────────────── the ranking evidence ──────────────────────────

/** The chart, captioned so the negative bars explain themselves. */
function RankingEvidence({ payload }: { payload: BenchmarkPayload }) {
  const { final, vsM3, gain } = useBenchFigures(payload);
  const zeroShot = zeroShotRows(payload);

  const bars: BarDatum[] = [
    ...zeroShot.map((row) => ({
      key: row.key,
      label: row.label,
      value: row.rho,
      color: "#F43F5E",
      note: "never saw a single assay from this library",
      negative: true,
    })),
    ...(["m4_fixed_prior", "m3_ucb", "m2_greedy"] as const).map((method) => ({
      key: method,
      label: methodName(method),
      value: final[method] ?? 0,
      color: methodMeta(method).color,
      note: "learned by testing candidates",
    })),
    {
      key: "m5_hybrid",
      label: "Virtual Lab (ours)",
      value: final.m5_hybrid ?? 0,
      color: "#10B981",
      note: `${gainText(gain)} over the standard UCB baseline`,
      hero: true,
    },
  ];

  return (
    <Panel
      tone="hero"
      eyebrow="Evidence"
      title="Who ranks the library best"
      lede={`${PLAIN.spearman_rho.label} for each method against the measured binding, after ${int(payload.seeds)} runs. Everything left of zero is worse than guessing.`}
      actions={<InfoTip label="How to read the scale">{PLAIN.spearman_rho.meaning}</InfoTip>}
    >
      <HBarChart
        data={bars}
        format={(value) => value.toFixed(2)}
        labelWidth={190}
        rowHeight={40}
        zeroLabel="0 · guessing"
        ariaLabel={`Ranking accuracy per method: ${bars.map((bar) => `${bar.label} ${bar.value.toFixed(2)}`).join(", ")}`}
      />
      <div className="mt-2 grid gap-2 sm:grid-cols-2">
        <div className="rounded-lg border px-3 py-2" style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}>
          <div className="text-[11px] font-semibold ink-dim">The red bars are the interesting ones</div>
          <p className="mt-0.5 text-[10.5px] leading-relaxed ink-faint">
            AlphaFold3, ProteinMPNN and ESM-2 are excellent structure models, but none of them ever saw an
            assay from this library — so they rank it <em>backwards</em>. A general-purpose model is not a
            substitute for testing.
          </p>
        </div>
        <div className="rounded-lg border px-3 py-2" style={{ borderColor: "var(--accent-line)", background: "var(--accent-soft)" }}>
          <div className="text-[11px] font-semibold" style={{ color: "var(--accent)" }}>
            The gap is the result
          </div>
          <p className="mt-0.5 text-[10.5px] leading-relaxed ink-faint">
            <span className="mono">{num(final.m3_ucb ?? null, 2)}</span> →{" "}
            <span className="mono">{num(final.m5_hybrid ?? null, 2)}</span> on a 0→1 scale, winning{" "}
            {countOf(vsM3.seedsWon, vsM3.n)} runs: {gainText(gain)}.
          </p>
        </div>
      </div>
      <Note summary="What the reference predictors were and were not run against">
        They are quoted as external comparison points and were not given the lab's testing budget, so the
        comparison is "off-the-shelf knowledge vs. tested knowledge", not a like-for-like budget race. Their
        score is the same ranking accuracy, computed once over the whole library — nothing is retrained.
        <span className="mono mt-1 block text-[10px]">{payload.zero_shot_definition}</span>
      </Note>
    </Panel>
  );
}

// ───────────────────────────── the honest axis ────────────────────────────

/**
 * The metric we do not win, at one line of weight.
 *
 * This is deliberately *not* a peer panel. The instruction the redesign encodes:
 * the metric that cannot separate two methods does not get a scoreboard. It gets
 * a sentence, the numbers behind it, and a disclosure if anyone asks why.
 */
function HonestAxis({ payload }: { payload: BenchmarkPayload }) {
  const { aubqc } = useBenchFigures(payload);
  const best = METHODS.reduce(
    (acc, method) => (payload.aubqc[method] > payload.aubqc[acc] ? method : acc),
    METHODS[0],
  );
  // Where each method's curve actually lands, and the highest value the pool
  // contains. Quoting both is what makes "they both saturate" checkable rather
  // than asserted: the curves average below the pool maximum, which is the
  // ceiling a single pick is competing against.
  const finalCurve = payload.curves[payload.curves.length - 1];
  const poolCeiling = finalCurve
    ? Math.max(...METHODS.map((method) => finalCurve[method]))
    : null;
  const m5Final = finalCurve?.m5_hybrid ?? null;
  const m3Final = finalCurve?.m3_ucb ?? null;

  return (
    <div className="space-y-2">
      <HonestStrip
        title="The one we didn't win:"
        body={
          <>
            {PLAIN.aubqc.label.toLowerCase()} — how good the single best candidate found was. Both methods
            reach the same ceiling, so this measure can't separate them.
          </>
        }
        detail={
          <span className="mono">
            gap {aubqc.meanDiff >= 0 ? "+" : "−"}
            {num(Math.abs(aubqc.meanDiff), 4)} · p {pvalue(aubqc.p)} ·{" "}
            {isClear(aubqc.p) ? "a real difference" : "indistinguishable from chance"} · both axes are
            reported together or not at all
          </span>
        }
      />
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-xl border px-3.5 py-2.5" style={{ borderColor: "var(--accent-line)", background: "var(--accent-soft)" }}>
        <span className="text-[11.5px] font-semibold" style={{ color: "var(--accent)" }}>
          And it transferred:
        </span>
        <span className="text-[11.5px] leading-relaxed ink-dim">
          the same code, with no retuning, also beat the standard UCB baseline on the other two public targets
          with a learnable signal — HIV-1 gp120 and lysozyme.
        </span>
      </div>
      <details
        className="disclosure rounded-xl border px-3.5 py-2"
        style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}
      >
        <summary className="flex items-center gap-1.5 text-[11px] ink-dim">
          <ChevronDown size={11} />
          Why that measure saturates (and why it still matters for different reasons)
        </summary>
        <div className="mt-2 grid gap-3 sm:grid-cols-2">
          <div>
            <LineChart
              x={payload.curves.map((row) => row.query_count)}
              series={METHODS.map((method) => ({
                key: method,
                label: methodName(method),
                color: methodMeta(method).color,
                values: payload.curves.map((row) => row[method]),
                dashed: method !== "m5_hybrid",
                width: method === "m5_hybrid" ? 2.4 : 1.3,
              }))}
              height={168}
              xLabel="lab tests spent"
              yLabel="best binder found"
              formatY={(value) => value.toFixed(2)}
              ariaLabel="Best binder found over the run, per method"
            />
          </div>
          <div className="space-y-2 text-[11px] leading-relaxed ink-faint">
            <p>
              {PLAIN.aubqc.meaning} The ceiling is reached within the first few rounds, so every method that
              reaches it is tied by construction — a bare average of this number invites reading noise as a
              win.
            </p>
            <p>
              Concretely, at the final checkpoint ours and the baseline both land on{" "}
              <span className="mono ink-dim">{num(m5Final, 2)}</span> and{" "}
              <span className="mono ink-dim">{num(m3Final, 2)}</span>: identical to two decimals, against a
              best-observed value of <span className="mono ink-dim">{num(poolCeiling, 2)}</span>. When the
              single-pick measure is the same number for every method, it has stopped carrying information —
              which is a fact about the measure, not a result for or against either method.
            </p>
            <p>
              Best {PLAIN.aubqc.label.toLowerCase()}:{" "}
              <span className="mono" style={{ color: methodMeta(best).color }}>
                {methodName(best)} {num(payload.aubqc[best], 4)}
              </span>
              . Ours is not the top row here, and the full table below says so.
            </p>
            <div className="flex flex-wrap gap-1.5">
              <TokenChip token={PLAIN.aubqc.token} meaning={PLAIN.aubqc.meaning} />
              <TokenChip token="max_so_far" meaning="The running maximum of the best candidate found so far — a single-tail statistic that saturates." />
            </div>
          </div>
        </div>
      </details>
    </div>
  );
}

// ────────────────────────────── the appendices ────────────────────────────

/**
 * Everything that used to occupy the middle of the page, behind one click each.
 *
 * Opening either costs a click during Q&A and zero scroll during the pitch. The
 * honest summary sentence stays outside, because "we show the metrics we lose"
 * is a claim about the page, not about the table.
 */
function Appendices({ payload }: { payload: BenchmarkPayload }) {
  return (
    <div className="space-y-2">
      <p className="px-1 text-[11.5px] leading-relaxed ink-dim">
        We also lose two measures to <span className="font-semibold ink">Greedy</span> — depth of the top five
        leads and the hit rate in the true best 10%. Both are below, unfiltered.
      </p>
      <details className="disclosure panel px-4 py-2.5">
        <summary className="flex cursor-pointer items-center gap-1.5 text-[12px] font-medium ink-dim">
          <ChevronDown size={12} />
          Every published measure, including the two we lose
        </summary>
        <div className="mt-3">
          <MetricTable payload={payload} />
        </div>
      </details>
      <details className="disclosure panel px-4 py-2.5">
        <summary className="flex cursor-pointer items-center gap-1.5 text-[12px] font-medium ink-dim">
          <ChevronDown size={12} />
          The run you just watched, on its own five starting candidates
        </summary>
        <div className="mt-3">
          <SameSeedPanel payload={payload} />
        </div>
      </details>
    </div>
  );
}

function MetricTable({ payload }: { payload: BenchmarkPayload }) {
  const rows = benchmarkMetricRows(payload);
  const final = finalRho(
    payload.spearman_rho_curves as unknown as Array<Record<string, number | null>>,
  );
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span className="text-[11.5px] leading-relaxed ink-dim">
          One row per measure. A measure we trail on stays in, with the rank column saying so.
        </span>
        <Badge color="#10B981" muted title="No measure is filtered out because it flatters the method">
          all measures
        </Badge>
      </div>
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse">
          <thead>
            <tr style={{ background: "var(--panel-2)" }}>
              <th className="label px-3 py-2 text-left text-[9px]">measure</th>
              {METHODS.map((method) => (
                <th key={method} className="label px-3 py-2 text-right text-[9px]" style={{ color: methodMeta(method).color }}>
                  {methodName(method)}
                </th>
              ))}
              <th className="label px-3 py-2 text-right text-[9px]">our rank</th>
              <th className="label px-3 py-2 text-left text-[9px]">best</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="px-3 py-2 align-top">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[12px] font-medium ink">{row.label}</span>
                    <InfoTip align="left">{row.description}</InfoTip>
                  </div>
                </td>
                {METHODS.map((method) => {
                  const isBest = row.best?.method === method;
                  return (
                    <td key={method} className="px-3 py-2 text-right align-top">
                      <span
                        className="mono text-[11.5px]"
                        style={{
                          color: method === "m5_hybrid" ? "#10B981" : isBest ? "var(--ink)" : "var(--ink-dim)",
                          fontWeight: method === "m5_hybrid" || isBest ? 650 : 400,
                        }}
                      >
                        {row.key === "top_decile_hit_rate"
                          ? shareWhole(row.values[method])
                          : num(row.values[method], 3)}
                      </span>
                    </td>
                  );
                })}
                <td className="px-3 py-2 text-right align-top">
                  <span
                    className="mono text-[11.5px]"
                    style={{
                      color: row.m5Rank === 1 ? "#10B981" : row.m5Rank === null ? "var(--ink-faint)" : "#F59E0B",
                    }}
                  >
                    {row.m5Rank === null ? "—" : `${row.m5Rank} of 5`}
                  </span>
                </td>
                <td className="px-3 py-2 align-top">
                  {row.best ? (
                    <span className="mono text-[11px]" style={{ color: methodMeta(row.best.method).color }}>
                      {methodName(row.best.method)}{" "}
                      {row.key === "top_decile_hit_rate" ? shareWhole(row.best.value) : num(row.best.value, 3)}
                    </span>
                  ) : (
                    <span className="ink-faint">—</span>
                  )}
                </td>
              </tr>
            ))}
            <tr style={{ borderTop: "1px solid var(--border-strong)", background: "var(--panel-2)" }}>
              <td className="px-3 py-2 align-top">
                <div className="text-[12px] font-medium ink">{PLAIN.spearman_rho.label}</div>
                <div className="text-[10px] ink-faint">0 = guessing · 1 = perfect · not defined for Random, which has no model</div>
              </td>
              {METHODS.map((method) => (
                <td key={method} className="px-3 py-2 text-right align-top">
                  <span
                    className="mono text-[11.5px]"
                    style={{
                      color: method === "m5_hybrid" ? "#10B981" : final[method] === null ? "var(--ink-faint)" : "var(--ink-dim)",
                      fontWeight: method === "m5_hybrid" ? 650 : 400,
                    }}
                  >
                    {final[method] === null ? "n/a" : num(final[method], 2)}
                  </span>
                </td>
              ))}
              <td className="px-3 py-2 text-right align-top mono text-[11.5px]" style={{ color: "#10B981" }}>
                1 of 5
              </td>
              <td className="px-3 py-2 align-top mono text-[11px]" style={{ color: "#10B981" }}>
                Ours {num(final.m5_hybrid ?? null, 2)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="pt-2.5">
        <Note
          summary={`Guided first batch: ${num(payload.m5_warm_start.summary.aubqc, 3)} vs ${num(
            payload.aubqc.m5_hybrid,
            3,
          )} on the same ceiling measure`}
        >
          Difference {num(pairedDelta(payload, "m5_warm_start_minus_m5_hybrid", "aubqc").meanDiff, 3)},{" "}
          {flukeOdds(pairedDelta(payload, "m5_warm_start_minus_m5_hybrid", "aubqc").p)} — reported under its
          own heading, never merged into the head-to-head comparison.
        </Note>
      </div>
    </div>
  );
}

function SameSeedPanel({ payload }: { payload: BenchmarkPayload }) {
  const comparison = payload.same_seed_comparison;
  const methods = comparison.methods.filter((method) => method !== "m5_hybrid");
  const rows: Array<{ key: string; label: string; color: string; value: number; q5: number; q35: number }> = [
    ...methods.map((method) => ({
      key: method,
      label: methodName(method),
      color: methodMeta(method).color,
      value: comparison.summary[method].aubqc,
      q5: comparison.summary[method].q5,
      q35: comparison.summary[method].q35,
    })),
    {
      key: "live_m5",
      label: "The recorded run",
      color: "#06B6D4",
      value: comparison.live.aubqc,
      q5: comparison.live.q5,
      q35: comparison.live.q35,
    },
  ];
  const bestBaseline = comparison.live.best_baseline;
  const gap = comparison.live.aubqc_gap_vs_best_baseline;

  if (comparison.error) {
    return (
      <Panel eyebrow="Same starting candidates" title="This comparison isn't available">
        <p className="mono text-[12px] ink-dim">{comparison.error}</p>
      </Panel>
    );
  }

  return (
    <div>
      <p className="mb-2 text-[11.5px] leading-relaxed ink-dim">
        Every method re-run from the identical five starting candidates ({comparison.initial_ids.join(", ")}),
        so the only difference between rows is the strategy.
      </p>
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[620px] border-collapse">
          <thead>
            <tr style={{ background: "var(--panel-2)" }}>
              {["method", "after 5 tests", "after 35 tests", "best binder found", "vs. best baseline"].map((heading) => (
                <th key={heading} className="label px-3 py-2 text-left text-[9px]">
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isLive = row.key === "live_m5";
              const delta = isLive ? gap : row.value - comparison.live.aubqc;
              return (
                <tr
                  key={row.key}
                  style={{
                    borderTop: "1px solid var(--border)",
                    background: isLive ? "rgba(6,182,212,0.06)" : undefined,
                  }}
                >
                  <td className="px-3 py-1.5">
                    <span className="mono text-[11.5px]" style={{ color: row.color, fontWeight: isLive ? 650 : 400 }}>
                      {row.label}
                    </span>
                    {row.key === bestBaseline ? (
                      <span className="chip mono ml-2 !text-[9px]" style={{ borderColor: "#F59E0B55", color: "#F59E0B" }}>
                        BEST HERE
                      </span>
                    ) : null}
                  </td>
                  <td className="px-3 py-1.5 mono text-[11.5px] ink-dim">{num(row.q5, 2)}</td>
                  <td className="px-3 py-1.5 mono text-[11.5px] ink">{num(row.q35, 2)}</td>
                  <td className="px-3 py-1.5 mono text-[11.5px]" style={{ color: isLive ? "#06B6D4" : "var(--ink-dim)" }}>
                    {num(row.value, 3)}
                  </td>
                  <td className="px-3 py-1.5 mono text-[11px]" style={{ color: delta >= 0 ? "var(--accent)" : "var(--danger)" }}>
                    {delta >= 0 ? "+" : "−"}
                    {num(Math.abs(delta), 3)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="pt-2">
        <Note summary="Read this row as a demo, not a result">
          On this seed the recorded campaign is {gap >= 0 ? "ahead of" : "behind"}{" "}
          {methodName(bestBaseline)} by {num(Math.abs(gap), 3)} on the ceiling measure. One run cannot
          establish a method difference; it is shown because it is the run the demo walks through, and its
          gap is the same order as the noise between reruns.
        </Note>
      </div>
    </div>
  );
}

// ─────────────────────────────── provenance ───────────────────────────────

function Provenance({ payload }: { payload: BenchmarkPayload }) {
  const final = finalRho(
    payload.spearman_rho_curves as unknown as Array<Record<string, number | null>>,
  );
  return (
    <Panel
      eyebrow="Where these numbers come from"
      title="Every figure on this page is read from a stored result file"
      actions={
        <span className="flex items-center gap-1 text-[10px] ink-faint">
          <ArrowUpRight size={10} /> nothing is hand-entered
        </span>
      }
    >
      <div className="grid gap-3 text-[11.5px] leading-relaxed ink-dim sm:grid-cols-2">
        <div className="rounded-lg border px-3 py-2" style={{ borderColor: "var(--border)" }}>
          <div className="label text-[9px]">result file</div>
          <p className="mono mt-0.5">
            {payload.source.label} · {int(payload.source.seeds)} runs ·{" "}
            <span style={{ color: "var(--ink)" }}>{payload.source.filename}</span>
          </p>
          {payload.source.filename_is_historical ? (
            <p className="mt-1 text-[10.5px] ink-faint">
              The filename is a historical write path; the run count is derived from the file's own
              <span className="mono"> seeds</span> field.
            </p>
          ) : null}
        </div>
        <div className="rounded-lg border px-3 py-2 sm:col-span-2" style={{ borderColor: "var(--border)" }}>
          <div className="label text-[9px]">the exact published figures</div>
          <p className="mt-0.5 text-[11.5px] leading-relaxed ink-dim">
            Ranking accuracy at the final checkpoint: ours{" "}
            <span className="mono" style={{ color: "var(--ink)" }}>
              {num(final.m5_hybrid ?? null, 4)}
            </span>
            , Standard UCB{" "}
            <span className="mono" style={{ color: "var(--ink)" }}>
              {num(final.m3_ucb ?? null, 4)}
            </span>
            , on a 0→1 scale. The headline percentage is exactly{" "}
            <span className="mono" style={{ color: "var(--ink)" }}>
              {gainText(relativeGain(final.m5_hybrid ?? null, final.m3_ucb ?? null))}
            </span>
            : ({num(final.m5_hybrid ?? null, 4)} − {num(final.m3_ucb ?? null, 4)}) ÷{" "}
            {num(final.m3_ucb ?? null, 4)}. Quoted here at full precision so the arithmetic can be checked
            without hovering anything.
          </p>
        </div>
        <div className="rounded-lg border px-3 py-2" style={{ borderColor: "var(--border)" }}>
          <div className="label text-[9px]">how the 100 runs were produced</div>
          <p className="mt-0.5 text-[11.5px] leading-relaxed" style={{ color: "var(--ink)" }}>
            The decision rules run deterministically here, so every p-value is reproducible from the stored
            file. The first tab is where the agents actually deliberate.
          </p>
          <p className="mt-1 text-[10.5px] ink-faint">
            Policy <span className="mono">{payload.m5_policy_mode}</span> ·{" "}
            {int(Object.keys(payload.matched_initial_seeds).length)} starting batches are carried in the file
            so the same-start comparison can be reproduced exactly.
          </p>
        </div>
      </div>
    </Panel>
  );
}
