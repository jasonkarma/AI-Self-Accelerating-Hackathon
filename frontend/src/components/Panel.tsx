import { useCallback, useRef, useState, type ReactNode } from "react";

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

/**
 * A one-line identification of the dataset a page's numbers are about.
 *
 * The overview and the recorded run both quoted a library size without ever
 * saying whose library it was — which antibody, against what, from where. This
 * strip answers that in one row, and every field is read from an artifact: a
 * missing value renders as an em dash rather than a plausible-looking name.
 */
export function DatasetStrip({
  identity,
  className = "",
}: {
  identity: {
    id: string | null;
    antigen: string | null;
    cdr3Length: number | null;
    libraryRows: number | null;
    usableRows: number | null;
    corpus: string | null;
  } | null;
  className?: string;
}) {
  if (!identity) return null;
  const dash = "—";
  const fmt = (value: number | null) =>
    value === null ? dash : Math.round(value).toLocaleString("en-US");
  const cells: Array<{ label: string; value: string; mono?: boolean }> = [
    { label: "antibody target", value: identity.id ? identity.id.toUpperCase() : dash, mono: true },
    { label: "binds", value: identity.antigen ?? dash },
    {
      label: "region optimised",
      // CDRH3, not "CDR3-H3": the latter fuses the Kabat loop label with the
      // IMGT label and names a region that does not exist. This is the one
      // place in the repo that had it wrong — every other reference (README,
      // LOCKBOX_PREREGISTRATION.md, derive.ts, SequenceAlignment.tsx) already
      // uses `CDRH3`, and the field itself is `cdr3Length`.
      value: identity.cdr3Length === null ? dash : `CDRH3 · ${identity.cdr3Length} residues`,
      mono: true,
    },
    {
      label: "library",
      value:
        identity.libraryRows === null
          ? dash
          : `${fmt(identity.usableRows)} usable of ${fmt(identity.libraryRows)}`,
      mono: true,
    },
    { label: "measured by", value: identity.corpus ?? dash },
  ];
  return (
    <div
      className={`flex flex-wrap items-stretch gap-x-5 gap-y-2 rounded-xl border px-3.5 py-2.5 ${className}`}
      style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}
    >
      {cells.map((cell) => (
        <div key={cell.label} className="min-w-0">
          <div className="label text-[8.5px]">{cell.label}</div>
          <div
            className={`${cell.mono ? "mono" : ""} truncate text-[11px] leading-snug ink`}
            title={cell.value}
          >
            {cell.value}
          </div>
        </div>
      ))}
    </div>
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
    /*
     * The gradient lives on an absolutely-positioned layer rather than on the
     * tile itself, so the tile no longer needs `overflow-hidden`.
     *
     * That matters for the `InfoTip` beside the label: an absolutely-positioned
     * tooltip is clipped by the first ancestor with `overflow: hidden`, which is
     * why the hint cards were being cut off by their own tiles. Clipping the
     * background layer instead keeps the rounded gradient and lets the tooltip
     * escape to the panel above.
     */
    <div
      className="relative min-w-0 rounded-xl border px-3.5 py-3"
      style={{ borderColor: border }}
      title={typeof hint === "string" ? hint : undefined}
    >
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0 overflow-hidden rounded-xl"
        style={{
          background:
            emphasis === "none"
              ? "var(--panel-2)"
              : emphasis === "accent"
                ? "linear-gradient(150deg, var(--accent-soft), transparent 70%)"
                : "linear-gradient(150deg, var(--gold-soft), transparent 70%)",
        }}
      />
      <div className="label relative mb-1 flex items-center gap-1 text-[9.5px]">
        <span className="truncate">{label}</span>
        {richHint ? <InfoTip label={`What ${typeof label === "string" ? label : "this number"} means`}>{hint}</InfoTip> : null}
      </div>
      <div className="relative flex items-baseline gap-1">
        <span className="kpi-value mono text-[24px]" style={{ color }}>
          {value}
        </span>
        {unit ? <span className="mono text-[11px] ink-faint">{unit}</span> : null}
      </div>
      {caption ? (
        <div className="relative mt-0.5 truncate text-[10.5px] leading-tight ink-faint">{caption}</div>
      ) : null}
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
 * The card is **fixed-positioned at the trigger's measured coordinates**, not
 * absolutely positioned inside it. That is the only way it can survive the
 * places these triggers actually live:
 *
 * * a horizontally scrolling table (`overflow-x: auto` clips both axes, so a
 *   card inside the ablation ledger was cut in half);
 * * a panel with `overflow: hidden` for its rounded corner;
 * * a trigger near the viewport's left edge, where right-aligning a 280 px card
 *   pushed it off-screen entirely.
 *
 * Measuring at open time also lets the card flip above the trigger when there
 * is no room below, and clamp itself inside the viewport horizontally. It is
 * still not a portal — the element stays in this subtree — so the static
 * render harness keeps working: with no pointer, no card is mounted at all.
 */
export function InfoTip({
  children,
  label = "More detail",
  align = "left",
}: {
  children: ReactNode;
  label?: string;
  align?: "left" | "right";
}) {
  const [spot, setSpot] = useState<{ left: number; top: number; width: number } | null>(null);
  const trigger = useRef<HTMLButtonElement>(null);

  const place = useCallback(() => {
    const el = trigger.current;
    if (!el || typeof window === "undefined") return;
    const rect = el.getBoundingClientRect();
    const width = Math.min(300, window.innerWidth - 16);
    // Anchor to the trigger's leading edge by default; both edges are clamped
    // into the viewport afterwards, so neither can push the card off-screen.
    const raw = align === "right" ? rect.right - width : rect.left;
    const left = Math.min(Math.max(8, raw), Math.max(8, window.innerWidth - width - 8));
    // 180 px is a conservative height estimate: enough to decide whether the
    // card fits below the trigger or has to open upwards.
    const below = rect.bottom + 8;
    const top = below + 180 > window.innerHeight ? Math.max(8, rect.top - 188) : below;
    setSpot({ left, top, width });
  }, [align]);

  const clear = useCallback(() => setSpot(null), []);

  return (
    <span
      className="inline-flex align-middle"
      onMouseEnter={place}
      onMouseLeave={clear}
      onFocus={place}
      onBlur={clear}
    >
      <button
        ref={trigger}
        type="button"
        aria-label={label}
        className="focus-ring inline-flex h-4 w-4 items-center justify-center rounded-full border text-[9px] font-semibold ink-faint transition-colors hover:border-[color:var(--accent-line)] hover:text-[color:var(--accent)]"
        style={{ borderColor: "var(--border-strong)" }}
      >
        i
      </button>
      {spot ? (
        <span
          role="tooltip"
          className="pointer-events-none fixed z-[80] rounded-xl border px-3 py-2 text-[11px] font-normal normal-case leading-relaxed shadow-xl"
          style={{
            left: spot.left,
            top: spot.top,
            width: spot.width,
            background: "var(--panel-raised)",
            borderColor: "var(--border-strong)",
            color: "var(--ink-dim)",
            letterSpacing: "normal",
            // Long unbroken tokens (field names, `a_null := a_base`) must wrap
            // rather than push the card wider than its measured box.
            overflowWrap: "anywhere",
          }}
        >
          {children}
        </span>
      ) : null}
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
