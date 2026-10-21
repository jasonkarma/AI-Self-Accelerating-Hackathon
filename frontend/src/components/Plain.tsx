import type { ReactNode } from "react";
import { InfoTip } from "./Panel";

/**
 * The three primitives the plain-language surface is built from.
 *
 * They exist so the rewrite cannot drift back into raw statistics: a headline
 * number is always rendered by `Headline`, a bounded statistic always carries a
 * `ScaleMarker`, and an internal token may only reach the page inside
 * `TokenChip` (where its tooltip holds the definition).
 */

// ───────────────────────────── the hero number ────────────────────────────

/**
 * One number, in the largest type on the view.
 *
 * The rule this component encodes: the big number is a *plain-language* figure
 * (`+91%`, `83%`, `7.4×`) and the raw statistic it was derived from travels in
 * `support`, one line below, at body size. There is exactly one of these per
 * view; anything else that wants to be large is a `KpiTile`.
 */
export function Headline({
    value,
    label,
    support,
    hint,
    tone = "accent",
    size = "xl",
}: {
    value: ReactNode;
    label: ReactNode;
    support?: ReactNode;
    hint?: ReactNode;
    tone?: "accent" | "gold" | "ink";
    size?: "lg" | "xl";
}) {
    const color = tone === "accent" ? "var(--accent)" : tone === "gold" ? "var(--gold)" : "var(--ink)";
    const scale = size === "xl" ? "text-[52px] sm:text-[60px]" : "text-[38px] sm:text-[44px]";
    return (
        <div className="min-w-0">
            <div
                className={`kpi-value mono ${scale} leading-[0.95] tracking-[-0.03em]`}
                style={{ color }}
            >
                {value}
            </div>
            <div className="mt-1.5 flex flex-wrap items-baseline gap-x-1.5 text-[14px] font-semibold leading-snug ink">
                <span>{label}</span>
                {hint ? <InfoTip label="What this number is">{hint}</InfoTip> : null}
            </div>
            {support ? (
                <div className="mt-1 max-w-[62ch] text-[12px] leading-relaxed ink-dim">{support}</div>
            ) : null}
        </div>
    );
}

// ──────────────────────────── the scale track ─────────────────────────────

/**
 * A 0 → 1 track that makes a bounded statistic legible without a sentence.
 *
 * `0.66` against `0.34` means nothing on its own; the same two markers on a bar
 * labelled *guessing* … *perfect* is immediate. This is the single element that
 * removes the need to explain what a rank correlation is during a demo.
 */
export function ScaleMarker({
    markers,
    min = 0,
    max = 1,
    leftLabel,
    rightLabel,
    ticks,
    ariaLabel,
}: {
    markers: Array<{ label: string; value: number | null; color: string; emphasis?: boolean }>;
    min?: number;
    max?: number;
    leftLabel: string;
    rightLabel: string;
    ticks?: Array<{ value: number; label: string }>;
    ariaLabel: string;
}) {
    const span = max - min || 1;
    const clamp = (value: number) => Math.min(1, Math.max(0, (value - min) / span));
    const drawn = markers.filter((marker): marker is typeof marker & { value: number } =>
        typeof marker.value === "number" && Number.isFinite(marker.value),
    );

    return (
        <div className="w-full" role="img" aria-label={ariaLabel}>
            <div className="relative h-[26px]">
                {/* the track */}
                <div
                    className="absolute inset-x-0 top-[11px] h-[3px] rounded-full"
                    style={{ background: "var(--border-strong)" }}
                />
                {/* optional guide ticks */}
                {(ticks ?? []).map((tick) => (
                    <div
                        key={tick.label}
                        className="absolute top-[7px] h-[11px] w-px"
                        style={{ left: `${clamp(tick.value) * 100}%`, background: "var(--border)" }}
                        title={tick.label}
                    />
                ))}
                {/* markers */}
                {drawn.map((marker) => (
                    <div
                        key={marker.label}
                        className="absolute top-0 flex -translate-x-1/2 flex-col items-center"
                        style={{ left: `${clamp(marker.value) * 100}%` }}
                    >
                        <span
                            aria-hidden
                            className="rounded-full"
                            style={{
                                width: marker.emphasis ? 13 : 9,
                                height: marker.emphasis ? 13 : 9,
                                marginTop: marker.emphasis ? 4 : 6,
                                background: marker.color,
                                boxShadow: marker.emphasis ? `0 0 0 3px ${marker.color}33` : undefined,
                            }}
                        />
                    </div>
                ))}
            </div>
            <div className="flex items-center justify-between text-[10px] ink-faint">
                <span>{leftLabel}</span>
                <span>{rightLabel}</span>
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
                {markers.map((marker) => (
                    <span key={marker.label} className="flex items-center gap-1.5 text-[11px]">
                        <span
                            aria-hidden
                            className="inline-block h-2 w-2 rounded-full"
                            style={{ background: marker.color }}
                        />
                        <span className="ink-dim">{marker.label}</span>
                        <span className="mono font-semibold" style={{ color: marker.color }}>
                            {typeof marker.value === "number" && Number.isFinite(marker.value)
                                ? marker.value.toFixed(2)
                                : "—"}
                        </span>
                    </span>
                ))}
            </div>
        </div>
    );
}

// ─────────────────────────────── the chip ─────────────────────────────────

/**
 * An internal token, rendered as a chip whose tooltip carries its definition.
 *
 * Rule 4 of the redesign: a technical token may appear on the surface only
 * inside one of these. `no_anneal` as a chip next to a plain-English headline is
 * honest and greppable; `no_anneal` as the value of a hero KPI tile is not.
 */
export function TokenChip({
    token,
    meaning,
    color = "var(--ink-faint)",
}: {
    token: string;
    meaning: ReactNode;
    color?: string;
}) {
    return (
        <span
            className="chip mono !text-[9.5px]"
            style={{ borderColor: "var(--border)", color }}
            title={typeof meaning === "string" ? meaning : undefined}
        >
            {token}
            <InfoTip label={`What ${token} means`}>{meaning}</InfoTip>
        </span>
    );
}

// ───────────────────────── a demoted, muted strip ─────────────────────────

/**
 * The "we did not win here" strip.
 *
 * A metric we tie or lose on still belongs on the page — it is what makes the
 * rest believable — but at one line of weight, below the result, never as a
 * peer card. This component is that one line.
 */
export function HonestStrip({
    title,
    body,
    detail,
}: {
    title: ReactNode;
    body: ReactNode;
    detail?: ReactNode;
}) {
    return (
        <section
            className="rounded-xl border px-3.5 py-2.5"
            style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}
        >
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                <span className="text-[11.5px] font-semibold ink-dim">{title}</span>
                <span className="text-[11.5px] leading-relaxed ink-faint">{body}</span>
            </div>
            {detail ? <div className="mt-1 text-[10.5px] leading-relaxed ink-faint">{detail}</div> : null}
        </section>
    );
}
