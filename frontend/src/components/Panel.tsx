import type { ReactNode } from "react";

interface Props {
  title: ReactNode;
  /** Small uppercase class label, e.g. "EVIDENCE · PRIMARY AXIS". */
  eyebrow?: ReactNode;
  /** One short line. Anything longer belongs in a `Note` or an `InfoTip`. */
  lede?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  /** Keeps the panel body flush with the border (for tables and charts). */
  flush?: boolean;
  tone?: "default" | "hero" | "warning" | "gold";
}

/**
 * The one container every panel uses.
 *
 * `eyebrow` is the section class, `title` the claim itself, `lede` the single
 * sentence a reader needs before the numbers mean anything. Longer methodology
 * prose is deliberately *not* accepted here — it lives in `Note` disclosures and
 * `InfoTip` hover cards, so the scan path down the page stays numeric.
 */
export function Panel({
  title,
  eyebrow,
  lede,
  actions,
  children,
  className = "",
  flush = false,
  tone = "default",
}: Props) {
  const ring =
    tone === "hero"
      ? "ring-1 ring-[color:var(--accent-line)]"
      : tone === "gold"
        ? "ring-1 ring-[color:var(--gold-line)]"
        : tone === "warning"
          ? "ring-1 ring-[#F59E0B]/30"
          : "";
  return (
    <section className={`panel ${ring} ${className}`}>
      <header className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-5 pb-3 pt-4">
        <div className="min-w-0">
          {eyebrow ? <div className="label mb-1.5 text-[10px]">{eyebrow}</div> : null}
          <h2 className="text-[15.5px] font-semibold leading-snug tracking-[-0.01em] ink">{title}</h2>
          {lede ? <p className="mt-1 max-w-[76ch] text-[12.5px] leading-relaxed ink-dim">{lede}</p> : null}
        </div>
        {actions ? <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div> : null}
      </header>
      <div className={flush ? "" : "px-5 pb-5"}>{children}</div>
    </section>
  );
}

/** A metric cell: label above, tabular value, optional footnote. */
export function Stat({
  label,
  value,
  sub,
  accent,
  title,
  size = "md",
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  accent?: string;
  title?: string;
  size?: "sm" | "md" | "lg" | "xl";
}) {
  const scale =
    size === "xl"
      ? "text-[30px]"
      : size === "lg"
        ? "text-[22px]"
        : size === "sm"
          ? "text-[13px]"
          : "text-[18px]";
  return (
    <div className="min-w-0" title={title}>
      <div className="label mb-1 text-[10px]">{label}</div>
      <div className={`kpi-value mono ${scale}`} style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      {sub ? <div className="mt-1 text-[11px] leading-tight ink-faint">{sub}</div> : null}
    </div>
  );
}

/**
 * A KPI tile: the hero row on every view.
 *
 * `emphasis` is the one number the viewer should be able to read from the back
 * of the room; everything else in the tile is support. Gold is reserved for a
 * discovered champion, emerald for a method win, neutral otherwise.
 */
export function KpiTile({
  label,
  value,
  unit,
  caption,
  emphasis = "none",
  hint,
}: {
  label: ReactNode;
  value: ReactNode;
  unit?: ReactNode;
  caption?: ReactNode;
  emphasis?: "none" | "accent" | "gold";
  /**
   * The definition behind the tile. A plain string becomes the native `title`
   * tooltip; anything richer (a card with monospace and inline code) is rendered
   * as an `InfoTip` beside the label, because a native title can only hold text.
   */
  hint?: ReactNode;
}) {
  const color = emphasis === "accent" ? "var(--accent)" : emphasis === "gold" ? "var(--gold)" : "var(--ink)";
  const border =
    emphasis === "accent" ? "var(--accent-line)" : emphasis === "gold" ? "var(--gold-line)" : "var(--border)";
  const richHint = hint !== undefined && typeof hint !== "string";
  return (
    <div
      className="relative min-w-0 overflow-hidden rounded-xl border px-3.5 py-3"
      style={{
        borderColor: border,
        background:
          emphasis === "none"
            ? "var(--panel-2)"
            : emphasis === "accent"
              ? "linear-gradient(150deg, var(--accent-soft), transparent 70%)"
              : "linear-gradient(150deg, var(--gold-soft), transparent 70%)",
      }}
      title={typeof hint === "string" ? hint : undefined}
    >
      <div className="label mb-1 flex items-center gap-1 text-[9.5px]">
        <span className="truncate">{label}</span>
        {richHint ? <InfoTip label={`What ${typeof label === "string" ? label : "this number"} means`}>{hint}</InfoTip> : null}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="kpi-value mono text-[24px]" style={{ color }}>
          {value}
        </span>
        {unit ? <span className="mono text-[11px] ink-faint">{unit}</span> : null}
      </div>
      {caption ? <div className="mt-0.5 truncate text-[10.5px] leading-tight ink-faint">{caption}</div> : null}
    </div>
  );
}

/** A small coloured chip with a semantic dot. */
export function Badge({
  children,
  color,
  title,
  muted = false,
}: {
  children: ReactNode;
  color: string;
  title?: string;
  muted?: boolean;
}) {
  return (
    <span
      className="chip mono"
      title={title}
      style={{
        borderColor: muted ? "var(--border)" : `${color}55`,
        background: muted ? "transparent" : `${color}14`,
        color: muted ? "var(--ink-faint)" : color,
        opacity: muted ? 0.75 : 1,
      }}
    >
      <span
        aria-hidden
        className="inline-block h-1.5 w-1.5 rounded-full"
        style={{ background: muted ? "var(--ink-faint)" : color }}
      />
      {children}
    </span>
  );
}

/** Legend row shared by every chart, so a colour means one thing everywhere. */
export function Legend({
  entries,
}: {
  entries: Array<{ label: string; color: string; note?: string; dashed?: boolean }>;
}) {
  return (
    <ul className="flex flex-wrap items-center gap-x-3.5 gap-y-1.5">
      {entries.map((entry) => (
        <li key={entry.label} className="flex items-center gap-1.5 text-[11px] ink-dim">
          <span
            aria-hidden
            className="inline-block h-[3px] w-3.5 rounded-full"
            style={{
              background: entry.dashed
                ? `repeating-linear-gradient(90deg, ${entry.color} 0 3px, transparent 3px 6px)`
                : entry.color,
            }}
          />
          <span className="mono" style={{ color: entry.color }}>
            {entry.label}
          </span>
          {entry.note ? <span className="ink-faint">{entry.note}</span> : null}
        </li>
      ))}
    </ul>
  );
}

/** An honest "we could not render this" state — never an empty axis. */
export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="panel-flat flex flex-col items-start gap-2 px-4 py-5">
      <div className="text-[13px] font-semibold ink">{title}</div>
      <div className="max-w-2xl text-[12px] leading-relaxed ink-dim">{detail}</div>
      {action}
    </div>
  );
}

/** A caveat that must travel with a claim. */
export function Callout({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warn" | "danger" | "success";
  title: ReactNode;
  children: ReactNode;
}) {
  const palette: Record<string, { border: string; bg: string; fg: string }> = {
    info: { border: "#0EA5E9", bg: "rgba(14,165,233,0.07)", fg: "#38BDF8" },
    warn: { border: "#F59E0B", bg: "rgba(245,158,11,0.07)", fg: "#FBBF24" },
    danger: { border: "#F43F5E", bg: "rgba(244,63,94,0.07)", fg: "#FB7185" },
    success: { border: "#10B981", bg: "rgba(16,185,129,0.07)", fg: "#34D399" },
  };
  const tone_ = palette[tone] ?? palette.info!;
  return (
    <div
      className="rounded-xl border px-3.5 py-2.5"
      style={{ borderColor: `${tone_.border}3d`, background: tone_.bg }}
    >
      <div className="text-[11.5px] font-semibold leading-snug" style={{ color: tone_.fg }}>
        {title}
      </div>
      <div className="mt-1 text-[11.5px] leading-relaxed ink-dim">{children}</div>
    </div>
  );
}

/**
 * An `ⓘ` with a hover/focus card.
 *
 * Pure CSS (`group-hover` + `focus-within`), so it works in the static render
 * harness and never needs a portal. The card is capped in width and aligned to
 * the right edge by default because these labels sit near panel headers.
 */
export function InfoTip({
  children,
  label = "More detail",
  align = "right",
}: {
  children: ReactNode;
  label?: string;
  align?: "left" | "right";
}) {
  return (
    <span className="group relative inline-flex align-middle">
      <button
        type="button"
        aria-label={label}
        className="focus-ring inline-flex h-4 w-4 items-center justify-center rounded-full border text-[9px] font-semibold ink-faint transition-colors hover:border-[color:var(--accent-line)] hover:text-[color:var(--accent)]"
        style={{ borderColor: "var(--border-strong)" }}
      >
        i
      </button>
      <span
        role="tooltip"
        className={`pointer-events-none invisible absolute top-6 z-40 w-[280px] rounded-xl border px-3 py-2 text-[11px] font-normal normal-case leading-relaxed opacity-0 shadow-xl transition-opacity duration-150 group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100 ${align === "right" ? "right-0" : "left-0"
          }`}
        style={{
          background: "var(--panel-raised)",
          borderColor: "var(--border-strong)",
          color: "var(--ink-dim)",
          letterSpacing: "normal",
        }}
      >
        {children}
      </span>
    </span>
  );
}

/**
 * A single-line methodology note that opens into the full prose.
 *
 * This is the text-reduction primitive: the page shows one short caption, and
 * the paragraph a reviewer wants is one click away rather than occupying four
 * lines above the chart.
 */
export function Note({ summary, children }: { summary: ReactNode; children: ReactNode }) {
  return (
    <details className="disclosure group mt-2">
      <summary className="flex items-center gap-1.5 text-[10.5px] leading-snug ink-faint transition-colors hover:text-[color:var(--ink-dim)]">
        <span
          aria-hidden
          className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border text-[8px] font-semibold"
          style={{ borderColor: "var(--border-strong)" }}
        >
          i
        </span>
        <span>{summary}</span>
        <span aria-hidden className="transition-transform group-open:rotate-90">
          ›
        </span>
      </summary>
      <div
        className="mt-1.5 rounded-lg border-l-2 px-3 py-2 text-[11px] leading-relaxed ink-faint"
        style={{ borderColor: "var(--border-strong)", background: "var(--panel-2)" }}
      >
        {children}
      </div>
    </details>
  );
}
