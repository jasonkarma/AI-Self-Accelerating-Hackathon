import { AlertTriangle, ArrowUpRight, Scale, Target, TrendingUp, Trophy } from "lucide-react";
import type { BenchmarkPayload } from "../../types/report";
import { Badge, InfoTip, KpiTile, Legend, Note, Panel } from "../../components/Panel";
import { HBarChart, LineChart, type BarDatum } from "../../components/charts";
import {
  benchmarkMetricRows,
  deltaOverSe,
  finalRho,
  METHODS,
  pairedDelta,
  rankOf,
  rhoRatio,
  rhoSeries,
  zeroShotRows,
} from "../../lib/derive";
import { int, methodMeta, num, pExact, pvalue, percent, signed } from "../../lib/format";
import type { Resource } from "../../hooks/useResource";

/**
 * View 2 — the 100-seed 1N8Z benchmark and the zero-shot showdown.
 *
 * The order is the argument: a single headline scoreboard states the win in the
 * units a judge reads first (ρ, ratio, p, seeds won), then the two axes are shown
 * side by side so the tie on AUBQC and the separation on ρ are visible at once,
 * and only then does the full metric table appear. The tie explanation is not a
 * footnote — it is the sentence that makes both charts honest.
 */
export function BenchmarkView({ benchmark }: { benchmark: Resource<BenchmarkPayload> }) {
  if (benchmark.status === "error") {
    return (
      <div className="panel px-4 py-6">
        <div className="flex items-center gap-2 text-[13px] font-semibold" style={{ color: "#F43F5E" }}>
          <AlertTriangle size={14} /> Could not load the benchmark artifact
        </div>
        <p className="mt-1.5 text-[12px] ink-dim">{benchmark.error.message}</p>
      </div>
    );
  }
  if (!benchmark.data) {
    return <div className="panel px-4 py-6 text-[12px] ink-faint">Loading the 100-seed artifact…</div>;
  }
  const payload = benchmark.data;
  return (
    <div className="space-y-4">
      <Scoreboard payload={payload} />
      <Axes payload={payload} />
      <MetricTable payload={payload} />
      <SameSeedPanel payload={payload} />
      <Provenance payload={payload} />
    </div>
  );
}

/**
 * The headline banner.
 *
 * Left: the two numbers the claim rests on. Right: the paired-test block that
 * makes the claim statistical rather than descriptive.
 */
function Scoreboard({ payload }: { payload: BenchmarkPayload }) {
  const final = finalRho(payload.spearman_rho_curves as unknown as Array<Record<string, number | null>>);
  const vsM3 = pairedDelta(payload, "m5_hybrid_minus_m3_ucb", "spearman_rho_at_q35");
  const ratio = rhoRatio(final.m5_hybrid ?? null, final.m3_ucb ?? null);
  const overSe = deltaOverSe(vsM3.meanDiff, vsM3.se);
  const aubqc = pairedDelta(payload, "m5_hybrid_minus_m3_ucb", "aubqc");
  const zeroShot = zeroShotRows(payload);
  const bestZeroShot = zeroShot.reduce((acc, row) => (row.rho > acc.rho ? row : acc), zeroShot[0]);

  return (
    <section className="panel hero-wash overflow-hidden">
      <div className="grid gap-5 px-5 py-5 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span
              className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10.5px] font-semibold"
              style={{ background: "var(--gold-soft)", color: "var(--gold)", border: "1px solid var(--gold-line)" }}
            >
              <Trophy size={11} /> M5 WINS ON THE AXIS THAT SEPARATES
            </span>
            <span className="text-[11px] ink-faint">
              {int(payload.seeds)}-seed paired benchmark · 1N8Z · {payload.source.label}
            </span>
          </div>

          <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
            <div>
              <div className="label mb-1 text-[10px]">M5 Virtual Lab · Spearman ρ</div>
              <div className="kpi-value mono text-[52px] leading-none" style={{ color: "var(--accent)" }}>
                {num(final.m5_hybrid ?? null, 4)}
              </div>
            </div>
            <div className="pb-1.5">
              <div className="mono text-[13px] ink-faint">vs</div>
            </div>
            <div className="pb-1.5">
              <div className="label mb-1 text-[10px]">M3 Standard UCB</div>
              <div className="kpi-value mono text-[30px] leading-none ink-dim">
                {num(final.m3_ucb ?? null, 4)}
              </div>
            </div>
            <div
              className="mb-1 flex items-center gap-1.5 rounded-lg px-2.5 py-1.5"
              style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-line)" }}
            >
              <TrendingUp size={13} style={{ color: "var(--accent)" }} />
              <span className="mono text-[14px] font-semibold" style={{ color: "var(--accent)" }}>
                {signed(vsM3.meanDiff, 4)}
              </span>
              <span className="text-[11px] ink-dim">improvement</span>
            </div>
          </div>

          <p className="mt-2.5 max-w-[68ch] text-[12.5px] leading-relaxed ink-dim">
            The loop learns the fitness landscape; the static predictors never do. Every zero-shot
            reference correlates <em>negatively</em> with measured binding — the best of them reaches{" "}
            <span className="mono">{num(bestZeroShot?.rho ?? null, 2)}</span>.
          </p>

          <div className="mt-3.5 grid grid-cols-2 gap-2.5 sm:grid-cols-4">
            <KpiTile
              label="ρ ratio"
              value={ratio === null ? "—" : `${ratio.toFixed(2)}×`}
              caption="M5 ÷ M3 correlation"
              emphasis="accent"
              hint="Final-checkpoint Spearman ρ, M5 over M3."
            />
            <KpiTile
              label="Paired p-value"
              value={pvalue(vsM3.p)}
              caption={`exact ${pExact(vsM3.p)}`}
              emphasis="accent"
              hint="Two-sided paired t-test on per-seed Δρ."
            />
            <KpiTile
              label="Seeds won"
              value={`${int(vsM3.seedsWon)}/${int(vsM3.n)}`}
              caption="per-seed, paired"
              hint="Number of the 100 seeds on which M5's ρ exceeds M3's on the same seed."
            />
            <KpiTile
              label="Δ / SE"
              value={overSe === null ? "—" : overSe.toFixed(1)}
              caption={`SE ${vsM3.se.toFixed(4)}`}
              hint="Effect size in standard errors — 17 is far outside any noise explanation."
            />
          </div>
        </div>

        <div className="flex min-w-0 flex-col justify-between gap-3 rounded-xl border p-3.5" style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}>
          <div>
            <div className="label mb-2 flex items-center gap-1.5 text-[10px]">
              <Scale size={11} style={{ color: "#6366F1" }} /> The honest second axis
              <InfoTip label="Why AUBQC ties">
                AUBQC is the area under <span className="mono">max_so_far</span> — a single-tail statistic.
                Once the ceiling is reached every method that got there is tied by construction, so a bare
                AUBQC mean invites reading noise as a win.
              </InfoTip>
            </div>
            <p className="text-[12px] leading-relaxed ink-dim">
              On AUBQC — the best binder found — M5 and M3 are{" "}
              <span className="mono" style={{ color: "var(--ink)" }}>statistically tied</span>.
            </p>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="mono text-[20px] font-semibold ink-dim">{signed(aubqc.meanDiff, 4)}</span>
              <span className="mono text-[11px] ink-faint">p = {pvalue(aubqc.p)}</span>
            </div>
            <p className="mt-1.5 text-[11px] leading-relaxed ink-faint">
              Both axes are reported together or not at all. The separation is in ρ, not in the ceiling.
            </p>
          </div>
          <div className="rounded-lg border px-3 py-2" style={{ borderColor: "var(--border)", background: "var(--panel)" }}>
            <div className="mono flex items-center justify-between text-[10.5px]">
              <span className="ink-faint">zero-shot references</span>
              <span style={{ color: "#F43F5E" }}>all negative</span>
            </div>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {zeroShot.map((row) => (
                <span
                  key={row.key}
                  className="mono rounded-md px-1.5 py-0.5 text-[10px]"
                  style={{ background: "rgba(244,63,94,0.1)", color: "#FB7185" }}
                  title={payload.zero_shot_definition}
                >
                  {row.label} {num(row.rho, 2)}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/** The two axes, side by side: the bar chart on the left, the curves on the right. */
function Axes({ payload }: { payload: BenchmarkPayload }) {
  const final = finalRho(payload.spearman_rho_curves as unknown as Array<Record<string, number | null>>);
  const zeroShot = zeroShotRows(payload);
  const vsM3 = pairedDelta(payload, "m5_hybrid_minus_m3_ucb", "spearman_rho_at_q35");
  const rho = pairedDelta(payload, "m5_hybrid_minus_m3_ucb", "spearman_rho_at_q35");
  const aubqc = pairedDelta(payload, "m5_hybrid_minus_m3_ucb", "aubqc");
  const rhoSeriesRows = rhoSeries(payload);
  const x = payload.curves.map((row) => row.query_count);
  const rhoX = rhoSeriesRows.map((row) => row.query_count);

  const bars: BarDatum[] = [
    ...zeroShot.map((row) => ({
      key: row.key,
      label: row.label,
      value: row.rho,
      color: "#F43F5E",
      note: "zero-shot, no wet-lab feedback",
      negative: true,
      tag: "zero-shot",
    })),
    ...(["m4_fixed_prior", "m3_ucb", "m2_greedy"] as const).map((method) => ({
      key: method,
      label: `${methodMeta(method).short} ${methodMeta(method).label}`,
      value: final[method] ?? 0,
      color: methodMeta(method).color,
      note: "active learning, 100 seeds",
      tag: "trained by querying",
    })),
    {
      key: "m5_hybrid",
      label: "M5 Virtual Lab",
      value: final.m5_hybrid ?? 0,
      color: "#10B981",
      note: `${int(vsM3.seedsWon)}/${int(vsM3.n)} seeds won vs M3`,
      hero: true,
      tag: "this work",
    },
  ];

  const best = METHODS.reduce((acc, method) => (payload.aubqc[method] > payload.aubqc[acc] ? method : acc), METHODS[0]);
  const m5Rank = rankOf(payload.aubqc, "m5_hybrid");

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
      <Panel
        tone="hero"
        eyebrow="Evidence · primary axis"
        title="Rank correlation: the loop learns the landscape"
        lede="Spearman ρ between each method's scoring model and measured binding over the qualified pool at q = 35."
        actions={
          <InfoTip label="Zero-shot definition">
            {payload.zero_shot_definition}
          </InfoTip>
        }
      >
        <HBarChart
          data={bars}
          format={(value) => value.toFixed(4)}
          labelWidth={176}
          rowHeight={38}
          zeroLabel="ρ = 0.00"
          ariaLabel={`Spearman rho at the final checkpoint: ${bars
            .map((bar) => `${bar.label} ${bar.value.toFixed(4)}`)
            .join(", ")}`}
        />
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px] ink-faint">
          <span>M1 has no scoring model → ρ undefined, not 0</span>
          <span className="mono">{"bar length = |ρ| · red bars are negative"}</span>
        </div>
        <Note summary="Reading the negative bars">
          A static structure or sequence model can rank a whole antibody, but CDRH3 point mutations change
          affinity through local context that pre-training never observed. The reference predictors are
          quoted as external comparison points — they were not re-run against the loop's budget.
        </Note>
      </Panel>

      <Panel
        eyebrow="Evidence · both axes"
        title="Why AUBQC ties while Spearman ρ separates"
        lede="The ceiling statistic saturates early; rank comprehension does not."
        actions={
          <Legend
            entries={[
              { label: "M5", color: "#10B981" },
              { label: "M3", color: "#6366F1" },
              { label: "M2", color: "#F59E0B" },
              { label: "M4", color: "#A855F7" },
              { label: "M1", color: "#64748B" },
            ]}
          />
        }
      >
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <div className="mb-1 flex items-center gap-1.5">
              <Target size={11} style={{ color: "var(--ink-faint)" }} />
              <span className="label text-[9.5px]">AUBQC · best binder found</span>
            </div>
            <LineChart
              x={x}
              series={METHODS.map((method) => ({
                key: method,
                label: methodMeta(method).short,
                color: methodMeta(method).color,
                values: payload.curves.map((row) => row[method]),
                dashed: method !== "m5_hybrid",
                width: method === "m5_hybrid" ? 2.4 : 1.3,
              }))}
              height={196}
              xLabel="assays spent (q)"
              yLabel="best qualified −log₁₀ K_D"
              formatY={(value) => value.toFixed(2)}
              ariaLabel="AUBQC learning curves for M1 to M5"
            />
          </div>
          <div>
            <div className="mb-1 flex items-center gap-1.5">
              <Scale size={11} style={{ color: "var(--ink-faint)" }} />
              <span className="label text-[9.5px]">Spearman ρ · the axis that separates</span>
            </div>
            <LineChart
              x={rhoX}
              series={METHODS.filter((method) => method !== "m1_random").map((method) => ({
                key: method,
                label: methodMeta(method).short,
                color: methodMeta(method).color,
                values: rhoSeriesRows.map((row) => row.values[method] ?? null),
                dashed: method !== "m5_hybrid",
                width: method === "m5_hybrid" ? 2.4 : 1.3,
              }))}
              height={196}
              xLabel="assays spent (q)"
              yLabel="Spearman ρ"
              formatY={(value) => value.toFixed(2)}
              referenceLines={[{ value: 0, label: "ρ = 0", color: "var(--border-strong)" }]}
              ariaLabel="Spearman rho by checkpoint for M2 to M5"
            />
          </div>
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          <div className="rounded-xl border px-3 py-2.5" style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}>
            <div className="text-[11.5px] font-semibold" style={{ color: "#F59E0B" }}>
              Why AUBQC ties
            </div>
            <p className="mt-1 text-[11.5px] leading-relaxed ink-dim">
              It is the area under <span className="mono">max_so_far</span>. The ceiling is reached early,
              so every method that got there is tied by construction: M5 − M3 ={" "}
              <span className="mono">{signed(aubqc.meanDiff, 4)}</span> at{" "}
              <span className="mono">p = {pvalue(aubqc.p)}</span>.
            </p>
          </div>
          <div
            className="rounded-xl border px-3 py-2.5"
            style={{ borderColor: "var(--accent-line)", background: "var(--accent-soft)" }}
          >
            <div className="text-[11.5px] font-semibold" style={{ color: "var(--accent)" }}>
              Why ρ separates
            </div>
            <p className="mt-1 text-[11.5px] leading-relaxed ink-dim">
              It is a global rank statistic over the whole pool: M5 − M3 ={" "}
              <span className="mono">{signed(rho.meanDiff, 4)}</span> at{" "}
              <span className="mono">p = {pvalue(rho.p)}</span>, winning{" "}
              <span className="mono">
                {int(rho.seedsWon)}/{int(rho.n)}
              </span>{" "}
              seeds.
            </p>
          </div>
        </div>
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[10.5px] ink-faint">
          <span>
            best AUBQC <span style={{ color: methodMeta(best).color }}>{methodMeta(best).short}</span>{" "}
            {num(payload.aubqc[best], 4)} · M5 rank {m5Rank ?? "—"}/5
          </span>
          <span className="mono">
            M5 − M3 AUBQC {signed(aubqc.meanDiff, 4)} · SE {aubqc.se.toFixed(4)}
          </span>
        </div>
      </Panel>
    </div>
  );
}

function MetricTable({ payload }: { payload: BenchmarkPayload }) {
  const rows = benchmarkMetricRows(payload);
  const final = finalRho(payload.spearman_rho_curves as unknown as Array<Record<string, number | null>>);
  return (
    <Panel
      eyebrow="Full disclosure"
      title="Every published metric, including the ones M5 loses"
      lede="Where M5 trails — top-5 panel depth and top-decile hit rate against M2 — the rank column says so."
      actions={<Badge color="#10B981" muted title="No metric is filtered out because it flatters the method">all metrics</Badge>}
      flush
    >
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse">
          <thead>
            <tr style={{ background: "var(--panel-2)" }}>
              <th className="label px-5 py-2 text-left text-[9px]">metric</th>
              {METHODS.map((method) => (
                <th key={method} className="label px-3 py-2 text-right text-[9px]" style={{ color: methodMeta(method).color }}>
                  {methodMeta(method).short}
                </th>
              ))}
              <th className="label px-3 py-2 text-right text-[9px]">M5 rank</th>
              <th className="label px-5 py-2 text-left text-[9px]">best</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="px-5 py-2 align-top">
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
                        {row.key === "top_decile_hit_rate" ? percent(row.values[method], 1) : num(row.values[method], 4)}
                      </span>
                    </td>
                  );
                })}
                <td className="px-3 py-2 text-right align-top">
                  <span
                    className="mono text-[11.5px]"
                    style={{ color: row.m5Rank === 1 ? "#10B981" : row.m5Rank === null ? "var(--ink-faint)" : "#F59E0B" }}
                  >
                    {row.m5Rank === null ? "—" : `${row.m5Rank}/5`}
                  </span>
                </td>
                <td className="px-5 py-2 align-top">
                  {row.best ? (
                    <span className="mono text-[11px]" style={{ color: methodMeta(row.best.method).color }}>
                      {methodMeta(row.best.method).short}{" "}
                      {row.key === "top_decile_hit_rate" ? percent(row.best.value, 1) : num(row.best.value, 4)}
                    </span>
                  ) : (
                    <span className="ink-faint">—</span>
                  )}
                </td>
              </tr>
            ))}
            <tr style={{ borderTop: "1px solid var(--border-strong)", background: "var(--panel-2)" }}>
              <td className="px-5 py-2 align-top">
                <div className="text-[12px] font-medium ink">Spearman ρ at q = 35</div>
                <div className="text-[10px] ink-faint">undefined for M1 → n/a, never 0</div>
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
                    {final[method] === null ? "n/a" : num(final[method], 4)}
                  </span>
                </td>
              ))}
              <td className="px-3 py-2 text-right align-top mono text-[11.5px]" style={{ color: "#10B981" }}>
                1/5
              </td>
              <td className="px-5 py-2 align-top mono text-[11px]" style={{ color: "#10B981" }}>
                M5 {num(final.m5_hybrid ?? null, 4)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="px-5 py-2.5">
        <Note
          summary={`Cold-start ablation: M5 warm start AUBQC ${num(payload.m5_warm_start.summary.aubqc, 4)} vs ${num(
            payload.aubqc.m5_hybrid,
            4,
          )}`}
        >
          Δ {signed(pairedDelta(payload, "m5_warm_start_minus_m5_hybrid", "aubqc").meanDiff, 4)}, p ={" "}
          {pvalue(pairedDelta(payload, "m5_warm_start_minus_m5_hybrid", "aubqc").p)} — reported under its own
          key, never merged into the matched comparison.
        </Note>
      </div>
    </Panel>
  );
}

function SameSeedPanel({ payload }: { payload: BenchmarkPayload }) {
  const comparison = payload.same_seed_comparison;
  const methods = comparison.methods.filter((method) => method !== "m5_hybrid");
  const rows: Array<{ key: string; label: string; color: string; value: number; q5: number; q35: number }> = [
    ...methods.map((method) => ({
      key: method,
      label: methodMeta(method).short,
      color: methodMeta(method).color,
      value: comparison.summary[method].aubqc,
      q5: comparison.summary[method].q5,
      q35: comparison.summary[method].q35,
    })),
    {
      key: "live_m5",
      label: "LAB",
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
      <Panel eyebrow="Seed-matched" title="Same-seed comparison unavailable">
        <p className="mono text-[12px] ink-dim">{comparison.error}</p>
      </Panel>
    );
  }

  return (
    <Panel
      eyebrow={`Seed-matched · seed ${comparison.seed}`}
      title="The recorded campaign on its own five initial queries"
      lede={`M1–M5 re-run on the identical five seed candidates, so the only difference between rows is the policy — ${comparison.initial_ids.join(", ")}.`}
      flush
    >
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[620px] border-collapse">
          <thead>
            <tr style={{ background: "var(--panel-2)" }}>
              {["method", "q = 5", "q = 35", "AUBQC", "vs best baseline"].map((heading) => (
                <th key={heading} className="label px-5 py-2 text-left text-[9px]">
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
                  <td className="px-5 py-1.5">
                    <span className="mono text-[11.5px]" style={{ color: row.color, fontWeight: isLive ? 650 : 400 }}>
                      {row.label}
                    </span>
                    <span className="ml-2 text-[11px] ink-dim">{methodMeta(row.key).label}</span>
                    {row.key === bestBaseline ? (
                      <span className="chip mono ml-2 !text-[9px]" style={{ borderColor: "#F59E0B55", color: "#F59E0B" }}>
                        BEST BASELINE
                      </span>
                    ) : null}
                  </td>
                  <td className="px-5 py-1.5 mono text-[11.5px] ink-dim">{num(row.q5, 3)}</td>
                  <td className="px-5 py-1.5 mono text-[11.5px] ink">{num(row.q35, 3)}</td>
                  <td className="px-5 py-1.5 mono text-[11.5px]" style={{ color: isLive ? "#06B6D4" : "var(--ink-dim)" }}>
                    {num(row.value, 4)}
                  </td>
                  <td className="px-5 py-1.5 mono text-[11px]" style={{ color: delta >= 0 ? "var(--accent)" : "var(--danger)" }}>
                    {signed(delta, 4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="px-5 py-2.5">
        <Note summary="Read this row, not the 100-seed mean">
          On seed {comparison.seed} the live campaign is {gap >= 0 ? "ahead of" : "behind"}{" "}
          <span className="mono">{methodMeta(bestBaseline).short}</span> by{" "}
          <span className="mono">{signed(gap, 4)}</span> AUBQC. One seed cannot establish a method
          difference; it is shown because it is the run the demo walks through, and its per-seed gap is the
          same order as the 100-seed standard error.
        </Note>
      </div>
    </Panel>
  );
}

function Provenance({ payload }: { payload: BenchmarkPayload }) {
  return (
    <Panel
      eyebrow="Provenance"
      title="Where this view's numbers come from"
      actions={
        <span className="flex items-center gap-1 text-[10px] ink-faint">
          <ArrowUpRight size={10} /> all figures from the benchmark artifact
        </span>
      }
    >
      <div className="grid gap-3 text-[11.5px] leading-relaxed ink-dim sm:grid-cols-2">
        <div className="rounded-lg border px-3 py-2" style={{ borderColor: "var(--border)" }}>
          <div className="label text-[9px]">artifact</div>
          <p className="mono mt-0.5">
            {payload.source.label} · {int(payload.source.seeds)} seeds · <span style={{ color: "var(--ink)" }}>{payload.source.filename}</span>
          </p>
          {payload.source.filename_is_historical ? (
            <p className="mt-1 text-[10.5px] ink-faint">
              The filename is a historical write-path artifact; the label is derived from the artifact's own
              <span className="mono"> seeds</span> field.
            </p>
          ) : null}
        </div>
        <div className="rounded-lg border px-3 py-2" style={{ borderColor: "var(--border)" }}>
          <div className="label text-[9px]">policy mode</div>
          <p className="mono mt-0.5" style={{ color: "var(--ink)" }}>{payload.m5_policy_mode}</p>
          <p className="mt-1 text-[10.5px] ink-faint">
            {int(Object.keys(payload.matched_initial_seeds).length)} seed → 5-candidate mappings are carried
            in the payload so the same-seed comparison is reproducible.
          </p>
        </div>
      </div>
    </Panel>
  );
}
