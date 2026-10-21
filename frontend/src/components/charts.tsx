import { useId, useState } from "react";
import type { ReactNode } from "react";
import { useMeasure } from "../hooks/useMeasure";

// ─────────────────────────────── scales ───────────────────────────────────

export function linearScale(
  domain: [number, number],
  range: [number, number],
): (value: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  if (span === 0) return () => (r0 + r1) / 2;
  return (value: number) => r0 + ((value - d0) / span) * (r1 - r0);
}

/** "Nice" ticks, so axes read 0.0 / 0.2 / 0.4 rather than 0.013 / 0.287. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const span = max - min;
  const rawStep = span / Math.max(1, count);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalised = rawStep / magnitude;
  const step =
    (normalised <= 1 ? 1 : normalised <= 2 ? 2 : normalised <= 5 ? 5 : 10) * magnitude;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = start; value <= max + step * 0.001; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

function path(points: Array<[number, number]>): string {
  return points.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x},${y}`).join(" ");
}

// ─────────────────────────────── tooltip ──────────────────────────────────

interface TooltipState {
  x: number;
  y: number;
  content: ReactNode;
}

function Tooltip({ state }: { state: TooltipState | null }) {
  if (!state) return null;
  return (
    <div
      className="pointer-events-none absolute z-20 max-w-[260px] rounded-lg border px-2.5 py-2 text-[11px] leading-snug shadow-lg"
      style={{
        left: state.x,
        top: state.y,
        transform: "translate(-50%, -110%)",
        background: "var(--panel)",
        borderColor: "var(--border-strong)",
        color: "var(--ink)",
      }}
    >
      {state.content}
    </div>
  );
}

// ─────────────────────────────── line chart ───────────────────────────────

export interface LineSeries {
  key: string;
  label: string;
  color: string;
  /** `null` entries are gaps, not zeros — the line breaks and resumes. */
  values: Array<number | null>;
  dashed?: boolean;
  width?: number;
}

interface LineChartProps {
  x: number[];
  series: LineSeries[];
  height?: number;
  yLabel: string;
  xLabel: string;
  yDomain?: [number, number];
  /** Horizontal reference lines (e.g. the zero line on a ρ plot). */
  referenceLines?: Array<{ value: number; label: string; color?: string }>;
  /** Shaded bands drawn under the series (e.g. a ± SE ribbon). */
  bands?: Array<{ lower: number[]; upper: number[]; color: string; label: string }>;
  formatY?: (value: number) => string;
  formatX?: (value: number) => string;
  /** Extra tooltip lines per series index. */
  annotate?: (index: number) => ReactNode;
  ariaLabel: string;
}

export function LineChart({
  x,
  series,
  height = 240,
  yLabel,
  xLabel,
  yDomain,
  referenceLines = [],
  bands = [],
  formatY = (value) => value.toFixed(2),
  formatX = (value) => String(value),
  annotate,
  ariaLabel,
}: LineChartProps) {
  const [ref, size] = useMeasure<HTMLDivElement>();
  const [hover, setHover] = useState<TooltipState | null>(null);
  const gradientId = useId();
  const margin = { top: 12, right: 16, bottom: 30, left: 46 };
  const width = Math.max(size.width, 240);
  const innerW = Math.max(width - margin.left - margin.right, 10);
  const innerH = Math.max(height - margin.top - margin.bottom, 10);

  const flat = series.flatMap((entry) => entry.values.filter((value): value is number => value !== null));
  const bandValues = bands.flatMap((band) => [...band.lower, ...band.upper]);
  const all = [...flat, ...bandValues, ...referenceLines.map((line) => line.value)];
  const lo = yDomain ? yDomain[0] : all.length ? Math.min(...all) : 0;
  const hi = yDomain ? yDomain[1] : all.length ? Math.max(...all) : 1;
  const padded: [number, number] =
    yDomain ?? [lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.12];
  const yScale = linearScale(padded, [margin.top + innerH, margin.top]);
  const xScale = linearScale([x[0] ?? 0, x[x.length - 1] ?? 1], [margin.left, margin.left + innerW]);
  const ticks = niceTicks(padded[0], padded[1], 4);

  function handleMove(event: React.MouseEvent<SVGSVGElement>) {
    if (!x.length) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - rect.left;
    let nearest = 0;
    let bestDistance = Number.POSITIVE_INFINITY;
    x.forEach((value, index) => {
      const distance = Math.abs(xScale(value) - px);
      if (distance < bestDistance) {
        bestDistance = distance;
        nearest = index;
      }
    });
    setHover({
      x: xScale(x[nearest] ?? 0),
      y: margin.top + 8,
      content: (
        <div className="space-y-1">
          <div className="mono font-semibold" style={{ color: "var(--ink)" }}>
            q = {formatX(x[nearest] ?? 0)}
          </div>
          {series.map((entry) => {
            const value = entry.values[nearest];
            return (
              <div key={entry.key} className="flex items-center justify-between gap-3">
                <span className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-1.5 w-1.5 rounded-full"
                    style={{ background: entry.color }}
                  />
                  <span style={{ color: "var(--ink-dim)" }}>{entry.label}</span>
                </span>
                <span className="mono" style={{ color: entry.color }}>
                  {value === null || value === undefined ? "n/a" : formatY(value)}
                </span>
              </div>
            );
          })}
          {annotate ? annotate(nearest) : null}
        </div>
      ),
    });
  }

  return (
    <div ref={ref} className="relative w-full" style={{ height }} role="img" aria-label={ariaLabel}>
      <svg
        width={width}
        height={height}
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
        className="overflow-visible"
      >
        <defs>
          {series.map((entry) => (
            <linearGradient key={entry.key} id={`${gradientId}-${entry.key}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={entry.color} stopOpacity={0.22} />
              <stop offset="100%" stopColor={entry.color} stopOpacity={0} />
            </linearGradient>
          ))}
        </defs>

        {ticks.map((tick) => (
          <g key={tick}>
            <line
              x1={margin.left}
              x2={margin.left + innerW}
              y1={yScale(tick)}
              y2={yScale(tick)}
              stroke="var(--border)"
              strokeDasharray="3 5"
            />
            <text
              x={margin.left - 8}
              y={yScale(tick)}
              textAnchor="end"
              dominantBaseline="middle"
              className="mono"
              fontSize="10"
              fill="var(--ink-faint)"
            >
              {formatY(tick)}
            </text>
          </g>
        ))}

        {referenceLines.map((line) => (
          <g key={line.label}>
            <line
              x1={margin.left}
              x2={margin.left + innerW}
              y1={yScale(line.value)}
              y2={yScale(line.value)}
              stroke={line.color ?? "var(--border-strong)"}
              strokeDasharray="6 4"
            />
            <text
              x={margin.left + innerW}
              y={yScale(line.value) - 4}
              textAnchor="end"
              fontSize="9.5"
              fill={line.color ?? "var(--ink-faint)"}
              className="mono"
            >
              {line.label}
            </text>
          </g>
        ))}

        {bands.map((band) => {
          const upper = band.upper.map((value, index) => [xScale(x[index] ?? 0), yScale(value)] as [number, number]);
          const lower = band.lower
            .map((value, index) => [xScale(x[index] ?? 0), yScale(value)] as [number, number])
            .reverse();
          return (
            <path
              key={band.label}
              d={`${path(upper)} L${lower.map(([px, py]) => `${px},${py}`).join(" L")} Z`}
              fill={band.color}
              fillOpacity={0.12}
              stroke="none"
            />
          );
        })}

        {series.map((entry) => {
          const points = entry.values
            .map((value, index) =>
              value === null ? null : ([xScale(x[index] ?? 0), yScale(value)] as [number, number]),
            )
            .filter((point): point is [number, number] => point !== null);
          if (!points.length) return null;
          const areaPath = `${path(points)} L${points[points.length - 1]![0]},${margin.top + innerH} L${points[0]![0]},${margin.top + innerH} Z`;
          return (
            <g key={entry.key}>
              {!entry.dashed && series.length <= 2 ? (
                <path d={areaPath} fill={`url(#${gradientId}-${entry.key})`} />
              ) : null}
              <path
                d={path(points)}
                fill="none"
                stroke={entry.color}
                strokeWidth={entry.width ?? 2}
                strokeDasharray={entry.dashed ? "6 4" : undefined}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {points.map(([px, py], index) => (
                <circle
                  key={`${entry.key}-${index}`}
                  cx={px}
                  cy={py}
                  r={2.6}
                  fill="var(--panel)"
                  stroke={entry.color}
                  strokeWidth={1.6}
                />
              ))}
            </g>
          );
        })}

        <line
          x1={margin.left}
          x2={margin.left + innerW}
          y1={margin.top + innerH}
          y2={margin.top + innerH}
          stroke="var(--border-strong)"
        />

        {x.map((value) => (
          <text
            key={value}
            x={xScale(value)}
            y={margin.top + innerH + 14}
            textAnchor="middle"
            className="mono"
            fontSize="10"
            fill="var(--ink-faint)"
          >
            {formatX(value)}
          </text>
        ))}

        <text
          x={margin.left}
          y={height - 2}
          fontSize="9.5"
          fill="var(--ink-faint)"
          className="mono"
        >
          {xLabel}
        </text>
        <text
          x={0}
          y={margin.top - 2}
          fontSize="9.5"
          fill="var(--ink-faint)"
          className="mono"
        >
          {yLabel}
        </text>
      </svg>
      <Tooltip state={hover} />
    </div>
  );
}

// ─────────────────────────────── bar chart ────────────────────────────────

export interface BarDatum {
  key: string;
  label: string;
  value: number;
  color: string;
  /** A second line under the bar (e.g. `± SE`). */
  note?: string;
  hero?: boolean;
  /** Render to the left of zero (used for the zero-shot negative ρ panel). */
  negative?: boolean;
  /** A short tag drawn right-aligned in the label gutter, e.g. "zero-shot". */
  tag?: string;
}

interface BarChartProps {
  data: BarDatum[];
  height?: number;
  format?: (value: number) => string;
  /** Fixed domain; defaults to `[min(0, lo), max(0, hi)]` with padding. */
  domain?: [number, number];
  ariaLabel: string;
  showZeroLine?: boolean;
  /**
   * Width of the label gutter in px. Defaults to a generous 168 so long method
   * names (AlphaFold3, ProteinMPNN, M4 Fixed Structural Prior) are never clipped
   * and never collide with a negative bar, which is drawn to the *right* of the
   * gutter's zero line.
   */
  labelWidth?: number;
  /** Row height; 34 gives each label two comfortable lines. */
  rowHeight?: number;
  /** Caption printed under the zero line, e.g. "ρ = 0.00". */
  zeroLabel?: string;
  /** Right-hand gutter for the value labels. */
  valueWidth?: number;
}

/**
 * A horizontal bar chart with a **reserved label gutter**.
 *
 * The layout is the fix for the collision this chart used to have: labels are
 * right-aligned inside a fixed-width gutter, the plotting area starts after it,
 * and a negative bar grows leftwards only into the plot (never under the text).
 * The zero line is drawn on top of the bars with a small caption, and each value
 * label sits on the far side of its bar end with a fixed margin, so a negative
 * bar's value can never intersect the zero line's label.
 */
export function HBarChart({
  data,
  height,
  format = (value) => value.toFixed(3),
  domain,
  ariaLabel,
  showZeroLine = true,
  labelWidth = 168,
  rowHeight = 34,
  zeroLabel,
  valueWidth = 74,
}: BarChartProps) {
  const [ref, size] = useMeasure<HTMLDivElement>();
  const [hover, setHover] = useState<TooltipState | null>(null);
  const padding = { top: 10, right: valueWidth, bottom: showZeroLine ? 22 : 8, left: labelWidth };
  const computedHeight = height ?? data.length * rowHeight + padding.top + padding.bottom;
  const width = Math.max(size.width, labelWidth + valueWidth + 120);
  const innerW = Math.max(width - padding.left - padding.right, 40);
  const values = data.map((datum) => datum.value);
  const lo = domain ? domain[0] : Math.min(0, ...values);
  const hiRaw = domain ? domain[1] : Math.max(0, ...values);
  // 12 % headroom so a bar that reaches the maximum still has room for its label.
  const hi = domain ? domain[1] : hiRaw + Math.abs(hiRaw) * 0.12;
  const span = hi - lo || 1;
  const xScale = (value: number) => padding.left + ((value - lo) / span) * innerW;
  const zero = xScale(0);
  const baseY = padding.top + data.length * rowHeight + 4;

  return (
    <div ref={ref} className="relative w-full" style={{ height: computedHeight }} role="img" aria-label={ariaLabel}>
      <svg width={width} height={computedHeight} onMouseLeave={() => setHover(null)}>
        {data.map((datum, index) => {
          const y = padding.top + index * rowHeight;
          const barHeight = 18;
          const start = Math.min(xScale(datum.value), zero);
          const barWidth = Math.max(Math.abs(xScale(datum.value) - zero), 2);
          const hero = datum.hero === true;
          const labelX = datum.value >= 0 ? start + barWidth + 10 : start - 10;
          return (
            <g
              key={datum.key}
              onMouseEnter={() =>
                setHover({
                  x: Math.min(Math.max(labelX, padding.left + 40), width - 60),
                  y,
                  content: (
                    <div>
                      <div className="mono font-semibold" style={{ color: datum.color }}>
                        {datum.label}
                      </div>
                      <div className="mono">{format(datum.value)}</div>
                      {datum.note ? <div className="mono ink-faint">{datum.note}</div> : null}
                    </div>
                  ),
                })
              }
            >
              {hero ? (
                <rect
                  x={0}
                  y={y - 3}
                  width={width}
                  height={barHeight + 6}
                  rx={7}
                  fill="var(--accent-soft)"
                />
              ) : null}
              <text
                x={padding.left - 14}
                y={y + barHeight / 2 - (datum.tag ? 5 : 0)}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize="11.5"
                fill={hero ? datum.color : "var(--ink-dim)"}
                fontWeight={hero ? 650 : 450}
              >
                {datum.label}
              </text>
              {datum.tag ? (
                <text
                  x={padding.left - 14}
                  y={y + barHeight / 2 + 8}
                  textAnchor="end"
                  dominantBaseline="middle"
                  fontSize="9"
                  fill="var(--ink-faint)"
                  className="mono"
                >
                  {datum.tag}
                </text>
              ) : null}
              <rect
                x={start}
                y={y}
                width={barWidth}
                height={barHeight}
                rx={4}
                fill={datum.color}
                opacity={hero ? 0.95 : 0.7}
                stroke={hero ? datum.color : "none"}
                strokeWidth={hero ? 1 : 0}
              />
              {/* A negative bar grows left of zero and its value is printed just
                  right of the zero line, in the empty space the bar leaves behind —
                  so a value can never land on the axis caption or the label gutter. */}
              <text
                x={datum.value >= 0 ? start + barWidth + 8 : zero + 8}
                y={y + barHeight / 2}
                textAnchor="start"
                dominantBaseline="middle"
                fontSize="11"
                fill={datum.value < 0 ? datum.color : hero ? datum.color : "var(--ink)"}
                fontWeight={hero ? 650 : 450}
                className="mono"
              >
                {format(datum.value)}
              </text>
            </g>
          );
        })}

        {showZeroLine ? (
          <g>
            <line
              x1={zero}
              x2={zero}
              y1={padding.top - 4}
              y2={baseY - 2}
              stroke="var(--border-strong)"
              strokeWidth={1.2}
            />
            <text
              x={zero}
              y={baseY + 12}
              textAnchor={zero < padding.left + 40 ? "start" : zero > width - 60 ? "end" : "middle"}
              fontSize="10"
              fill="var(--ink-faint)"
              className="mono"
            >
              {zeroLabel ?? "0"}
            </text>
          </g>
        ) : null}
      </svg>
      <Tooltip state={hover} />
    </div>
  );
}

// ────────────────────────────── scatter plot ──────────────────────────────

export interface ScatterPoint {
  id: string;
  x: number;
  y: number;
  color: string;
  radius?: number;
  label?: string;
  ring?: boolean;
}

interface ScatterProps {
  points: ScatterPoint[];
  height?: number;
  xLabel: string;
  yLabel: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  formatX?: (value: number) => string;
  formatY?: (value: number) => string;
  ariaLabel: string;
  /** A horizontal line at the qualified threshold, if the axes are probabilities. */
  thresholdY?: { value: number; label: string };
}

export function ScatterPlot({
  points,
  height = 300,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  formatX = (value) => value.toFixed(2),
  formatY = (value) => value.toFixed(1),
  ariaLabel,
  thresholdY,
}: ScatterProps) {
  const [ref, size] = useMeasure<HTMLDivElement>();
  const [hover, setHover] = useState<TooltipState | null>(null);
  const margin = { top: 12, right: 18, bottom: 34, left: 48 };
  const width = Math.max(size.width, 260);
  const innerW = Math.max(width - margin.left - margin.right, 10);
  const innerH = Math.max(height - margin.top - margin.bottom, 10);
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const xPad = ((Math.max(...xs, 0.2) - Math.min(...xs, 0)) || 0.2) * 0.08;
  const yPad = ((Math.max(...ys, 1) - Math.min(...ys, 0)) || 1) * 0.08;
  const xDom: [number, number] = xDomain ?? [Math.min(...xs, 0) - xPad, Math.max(...xs, 0.5) + xPad];
  const yDom: [number, number] = yDomain ?? [Math.min(...ys) - yPad, Math.max(...ys) + yPad];
  const xScale = linearScale(xDom, [margin.left, margin.left + innerW]);
  const yScale = linearScale(yDom, [margin.top + innerH, margin.top]);
  const xTicks = niceTicks(xDom[0], xDom[1], 4);
  const yTicks = niceTicks(yDom[0], yDom[1], 4);

  return (
    <div ref={ref} className="relative w-full" style={{ height }} role="img" aria-label={ariaLabel}>
      <svg width={width} height={height} onMouseLeave={() => setHover(null)}>
        {xTicks.map((tick) => (
          <g key={`x-${tick}`}>
            <line x1={xScale(tick)} x2={xScale(tick)} y1={margin.top} y2={margin.top + innerH} stroke="var(--border)" strokeDasharray="3 5" />
            <text x={xScale(tick)} y={margin.top + innerH + 14} textAnchor="middle" fontSize="10" fill="var(--ink-faint)" className="mono">
              {formatX(tick)}
            </text>
          </g>
        ))}
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line x1={margin.left} x2={margin.left + innerW} y1={yScale(tick)} y2={yScale(tick)} stroke="var(--border)" strokeDasharray="3 5" />
            <text x={margin.left - 8} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" fontSize="10" fill="var(--ink-faint)" className="mono">
              {formatY(tick)}
            </text>
          </g>
        ))}
        {thresholdY ? (
          <g>
            <line x1={margin.left} x2={margin.left + innerW} y1={yScale(thresholdY.value)} y2={yScale(thresholdY.value)} stroke="#F59E0B" strokeDasharray="6 4" strokeOpacity={0.7} />
            <text x={margin.left + 4} y={yScale(thresholdY.value) - 4} fontSize="9.5" fill="#F59E0B" className="mono">
              {thresholdY.label}
            </text>
          </g>
        ) : null}
        {points.map((point) => (
          <circle
            key={point.id}
            cx={xScale(point.x)}
            cy={yScale(point.y)}
            r={point.radius ?? 4}
            fill={point.color}
            fillOpacity={point.ring ? 0.22 : 0.82}
            stroke={point.color}
            strokeWidth={point.ring ? 1.8 : 0.8}
            onMouseEnter={() =>
              setHover({
                x: xScale(point.x),
                y: yScale(point.y),
                content: (
                  <div className="space-y-0.5">
                    <div className="mono font-semibold">{point.label ?? point.id}</div>
                    <div className="mono">
                      {yLabel} {formatY(point.y)}
                    </div>
                    <div className="mono">
                      {xLabel} {formatX(point.x)}
                    </div>
                  </div>
                ),
              })
            }
          />
        ))}
        <text x={margin.left + innerW} y={height - 2} textAnchor="end" fontSize="9.5" fill="var(--ink-faint)" className="mono">
          {xLabel}
        </text>
        <text x={0} y={margin.top - 2} fontSize="9.5" fill="var(--ink-faint)" className="mono">
          {yLabel}
        </text>
      </svg>
      <Tooltip state={hover} />
    </div>
  );
}

// ──────────────────────────── error-bar plot ──────────────────────────────

export interface ErrorBarDatum {
  key: string;
  label: string;
  value: number;
  error: number;
  color: string;
  hero?: boolean;
  note?: string;
}

/**
 * Point estimate ± SE with a shared y-axis.
 *
 * Drawn as points-with-whiskers rather than bars because the interesting
 * quantity on a near-tied panel is *the gap relative to the error bar*, and a
 * bar chart's filled area makes a 0.0009 difference look like a win.
 */
export function ErrorBarChart({
  data,
  height = 260,
  format = (value) => value.toFixed(4),
  ariaLabel,
  zoom = true,
}: {
  data: ErrorBarDatum[];
  height?: number;
  format?: (value: number) => string;
  ariaLabel: string;
  zoom?: boolean;
}) {
  const [ref, size] = useMeasure<HTMLDivElement>();
  const [hover, setHover] = useState<TooltipState | null>(null);
  const margin = { top: 14, right: 18, bottom: 34, left: 58 };
  const width = Math.max(size.width, 280);
  const innerW = Math.max(width - margin.left - margin.right, 20);
  const innerH = Math.max(height - margin.top - margin.bottom, 20);

  const lows = data.map((datum) => datum.value - datum.error);
  const highs = data.map((datum) => datum.value + datum.error);
  const lo = zoom ? Math.min(...lows) : Math.min(0, ...lows);
  const hi = zoom ? Math.max(...highs) : Math.max(...highs);
  const pad = (hi - lo) * 0.25 || 0.001;
  const yDom: [number, number] = [lo - pad, hi + pad];
  const yScale = linearScale(yDom, [margin.top + innerH, margin.top]);
  const step = innerW / Math.max(data.length, 1);
  const ticks = niceTicks(yDom[0], yDom[1], 4);

  return (
    <div ref={ref} className="relative w-full" style={{ height }} role="img" aria-label={ariaLabel}>
      <svg width={width} height={height} onMouseLeave={() => setHover(null)}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={margin.left} x2={margin.left + innerW} y1={yScale(tick)} y2={yScale(tick)} stroke="var(--border)" strokeDasharray="3 5" />
            <text x={margin.left - 8} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" fontSize="10" fill="var(--ink-faint)" className="mono">
              {format(tick)}
            </text>
          </g>
        ))}
        {data.map((datum, index) => {
          const cx = margin.left + step * (index + 0.5);
          const y = yScale(datum.value);
          const yLow = yScale(datum.value - datum.error);
          const yHigh = yScale(datum.value + datum.error);
          return (
            <g
              key={datum.key}
              onMouseEnter={() =>
                setHover({
                  x: cx,
                  y,
                  content: (
                    <div className="space-y-0.5">
                      <div className="mono font-semibold" style={{ color: datum.color }}>
                        {datum.label}
                      </div>
                      <div className="mono">
                        {format(datum.value)} ± {format(datum.error)}
                      </div>
                      {datum.note ? <div className="mono ink-faint">{datum.note}</div> : null}
                    </div>
                  ),
                })
              }
            >
              <line x1={cx} x2={cx} y1={yHigh} y2={yLow} stroke={datum.color} strokeWidth={datum.hero ? 2 : 1.2} strokeOpacity={0.85} />
              <line x1={cx - 5} x2={cx + 5} y1={yHigh} y2={yHigh} stroke={datum.color} strokeWidth={datum.hero ? 2 : 1.2} />
              <line x1={cx - 5} x2={cx + 5} y1={yLow} y2={yLow} stroke={datum.color} strokeWidth={datum.hero ? 2 : 1.2} />
              <circle cx={cx} cy={y} r={datum.hero ? 5 : 3.6} fill={datum.color} stroke="var(--panel)" strokeWidth={1.5} />
              <text x={cx} y={margin.top + innerH + 16} textAnchor="middle" fontSize="10" fill={datum.hero ? datum.color : "var(--ink-faint)"} className="mono" fontWeight={datum.hero ? 600 : 400}>
                {datum.label}
              </text>
            </g>
          );
        })}
        <line x1={margin.left} x2={margin.left + innerW} y1={margin.top + innerH} y2={margin.top + innerH} stroke="var(--border-strong)" />
      </svg>
      <Tooltip state={hover} />
    </div>
  );
}

