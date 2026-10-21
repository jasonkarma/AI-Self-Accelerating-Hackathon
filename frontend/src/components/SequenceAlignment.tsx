import { useMemo } from "react";
import { align } from "../lib/derive";
import { ANCHOR_COLOR, FALSIFIED_COLOR, MUTATION_COLOR } from "../lib/format";

/**
 * An anchor that has been *mutated* is the one case a reader must not miss, so it
 * gets its own rose token rather than reusing the amber "refuted" colour: the two
 * mean different things (a conserved position was changed vs a declared rule was
 * inverted by the audit).
 */
const ALERT_COLOR = "#FB7185";

interface Props {
  sequence: string;
  wildType: string;
  /** Positions the round's audit has refuted — rendered with a strike marker. */
  falsifiedPositions?: number[];
  /** Compact single-line form for table cells. */
  dense?: boolean;
  label?: string;
}

/**
 * Residue-by-residue alignment of a 13-mer CDRH3 against the wild type.
 *
 * Three things are encoded, and only three:
 * * **mutation** — the residue differs from `SRWGGDGFYAMDY` (filled emerald);
 * * **anchor** — positions 3, 11, 12, 13, which the priors must conserve
 *   (cyan underscore); an anchor that *is* mutated is highlighted rose, because
 *   that is the one case a reader must not miss;
 * * **refuted** — a position whose declared rule the paired audit inverted
 *   (amber overline).
 *
 * Everything is monospace with tabular numerals so two candidates line up
 * column-for-column even when their mutation counts differ.
 */
export function SequenceAlignment({ sequence, wildType, falsifiedPositions = [], dense, label }: Props) {
  const cells = useMemo(() => align(sequence, wildType), [sequence, wildType]);
  const refuted = useMemo(() => new Set(falsifiedPositions), [falsifiedPositions]);

  if (dense) {
    return (
      <span className="mono text-[12px] tracking-[0.14em]" title={`${sequence} · WT ${wildType}`}>
        {cells.map((cell) => {
          const color = cell.mutated
            ? cell.anchor
              ? ALERT_COLOR
              : MUTATION_COLOR
            : cell.anchor
              ? ANCHOR_COLOR
              : "var(--ink-dim)";
          return (
            <span
              key={cell.position}
              style={{ color, fontWeight: cell.mutated ? 600 : 400 }}
            >
              {cell.residue}
            </span>
          );
        })}
      </span>
    );
  }

  return (
    <div className="min-w-0">
      {label ? <div className="label mb-1">{label}</div> : null}
      <div
        className="inline-grid gap-[3px]"
        style={{ gridTemplateColumns: `repeat(${cells.length}, minmax(0, 1fr))` }}
        role="table"
        aria-label="CDRH3 alignment"
      >
        {cells.map((cell) => {
          const mutationColor = cell.anchor ? ALERT_COLOR : MUTATION_COLOR;
          const baseColor = cell.anchor ? ANCHOR_COLOR : "var(--ink)";
          return (
            <div key={`wt-${cell.position}`} className="text-center">
              <div
                className="mono text-[9px] leading-none"
                style={{ color: cell.anchor ? ANCHOR_COLOR : "var(--ink-faint)" }}
              >
                {cell.wt}
              </div>
              <div
                className="mono flex h-7 w-[22px] items-center justify-center rounded-[5px] text-[13px] font-semibold"
                style={{
                  color: cell.mutated ? mutationColor : baseColor,
                  background: cell.mutated
                    ? cell.anchor
                      ? "rgba(251,113,133,0.16)"
                      : "rgba(52,211,153,0.14)"
                    : "transparent",
                  borderBottom: cell.anchor
                    ? `2px solid ${cell.mutated ? ALERT_COLOR : ANCHOR_COLOR}`
                    : "2px solid transparent",
                  textDecorationLine: refuted.has(cell.position) ? "overline" : undefined,
                  textDecorationColor: FALSIFIED_COLOR,
                  textDecorationThickness: "2px",
                }}
                title={`${cell.position}${cell.wt}>${cell.residue}${cell.anchor ? " · anchor" : ""}${
                  cell.mutated ? " · mutated" : ""
                }${refuted.has(cell.position) ? " · prior refuted by paired audit" : ""}`}
              >
                {cell.residue}
              </div>
              <div className="mono text-[8.5px] leading-none ink-faint">{cell.position}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The legend for the alignment, so the encoding is never implicit. */
export function AlignmentLegend() {
  const items = [
    { label: "WT residue", color: "var(--ink-faint)", style: "plain" },
    { label: "mutation", color: MUTATION_COLOR, style: "fill" },
    { label: "anchor 3/11/12/13", color: ANCHOR_COLOR, style: "underline" },
    { label: "anchor mutated", color: ALERT_COLOR, style: "fill" },
    { label: "prior refuted", color: FALSIFIED_COLOR, style: "overline" },
  ];
  return (
    <ul className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] ink-faint">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1">
          <span
            aria-hidden
            className="mono inline-flex h-3.5 w-3.5 items-center justify-center rounded-[3px] text-[9px] font-semibold"
            style={{
              color: item.color,
              background: item.style === "fill" ? `${item.color}22` : "transparent",
              borderBottom: item.style === "underline" ? `2px solid ${item.color}` : undefined,
              textDecorationLine: item.style === "overline" ? "overline" : undefined,
              textDecorationColor: item.color,
            }}
          >
            A
          </span>
          {item.label}
        </li>
      ))}
    </ul>
  );
}
