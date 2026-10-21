import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw, SkipForward, Star } from "lucide-react";
import type { RoundsPayload, TimelineEntry } from "../types/state";
import { harvestFromRound } from "../lib/derive";
import { num } from "../lib/format";

interface Props {
  rounds: RoundsPayload | null;
  selected: number;
  onSelect: (round: number) => void;
  playing: boolean;
  onPlayToggle: () => void;
  onRestoreGolden: () => void;
  busy: boolean;
  /** Live-round control, rendered inline so the timeline is the only toolbar. */
  liveControl?: React.ReactNode;
}

/**
 * The round scrubber — a two-line stepper, not a card wall.
 *
 * Line 1 is the round label (`Seed`, `R1`…`R6`) with a state dot; line 2 is that
 * round's best qualified `−log₁₀ K_D`, so the staircase is legible without
 * opening anything. The terminal round of the harvest phase is badged `★ R6
 * DISCOVERY` in gold once the record actually holds the champion.
 *
 * Two different things live here and are deliberately kept apart:
 *
 * * **Selection** (`onSelect`) is read-only — it asks the server for
 *   `GET /api/state?round=k`, which reconstructs that round from the ledger
 *   prefix and never touches the database. Stepping back is free.
 * * **Rewind** (`onRestoreGolden`) is a *write* — it copies the frozen golden
 *   ledger over the live one. It is behind an explicit button, so a walkthrough
 *   cannot destroy its own state by clicking left too many times.
 */
export function RoundTimeline({
  rounds,
  selected,
  onSelect,
  playing,
  onPlayToggle,
  onRestoreGolden,
  busy,
  liveControl,
}: Props) {
  if (!rounds) {
    return (
      <div className="panel-flat flex items-center gap-2 px-3 py-2.5 text-[12px] ink-faint">
        <RotateCcw size={13} className="animate-spin" />
        Loading round index…
      </div>
    );
  }

  const total = rounds.total_rounds;
  const entries = rounds.rounds;
  const harvest = harvestFromRound(total);
  const current = rounds.current_round;

  return (
    <div className="panel px-3.5 py-3">
      <div className="flex items-stretch gap-3">
        <div className="flex shrink-0 flex-col justify-between gap-1.5 pr-3" style={{ borderRight: "1px solid var(--border)" }}>
          <div className="flex items-center gap-1">
            <IconButton onClick={() => onSelect(Math.max(0, selected - 1))} disabled={selected <= 0} label="Previous round">
              <ChevronLeft size={14} />
            </IconButton>
            <IconButton
              onClick={onPlayToggle}
              label={playing ? "Pause playback" : "Replay the campaign"}
              active={playing}
              title={playing ? "Pause" : "Replay the recorded campaign from the seed batch"}
            >
              {playing ? <Pause size={13} /> : <Play size={13} />}
            </IconButton>
            <IconButton
              onClick={() => onSelect(Math.min(current, selected + 1))}
              disabled={selected >= current}
              label="Next round"
            >
              <ChevronRight size={14} />
            </IconButton>
          </div>
          <button
            type="button"
            onClick={() => onSelect(current)}
            className="focus-ring flex items-center justify-center gap-1 rounded-md border px-1.5 py-1 text-[10px] ink-faint"
            style={{ borderColor: "var(--border)" }}
            title="Jump to the newest round in the ledger"
          >
            <SkipForward size={10} /> R{current}
          </button>
        </div>

        <ol className="scroll-thin flex min-w-0 flex-1 items-stretch gap-1.5 overflow-x-auto pb-0.5">
          {entries.map((entry) => (
            <li key={entry.round} className="shrink-0">
              <RoundChip
                entry={entry}
                active={entry.round === selected}
                onSelect={() => onSelect(entry.round)}
                isHarvest={entry.round >= harvest}
                // The star marks the round that *currently holds* the champion:
                // the newest round, once the harvest phase has begun.
                isChampion={entry.round === current && entry.round >= harvest}
              />
            </li>
          ))}
        </ol>
      </div>

      <div
        className="mt-2.5 flex flex-wrap items-center justify-between gap-2 border-t pt-2.5"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10.5px] ink-faint">
            {entries[selected]?.phase ?? ""} · round {selected} of {current}
          </span>
          {entries[selected]?.hypothesis_id ? (
            <span className="chip mono !text-[9.5px]" style={{ borderColor: "var(--border)", color: "var(--ink-dim)" }}>
              {entries[selected]?.hypothesis_id}
            </span>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {liveControl}
          <button
            type="button"
            onClick={onRestoreGolden}
            disabled={busy}
            className="focus-ring flex h-7 items-center gap-1.5 rounded-md border px-2 text-[10.5px] font-medium disabled:opacity-40"
            style={{ borderColor: "var(--border)", color: "var(--ink-dim)" }}
            title="Copies the frozen Seed-21 ledger over the live one and re-hydrates the campaign. A write; the stepper is read-only."
          >
            <RotateCcw size={11} className={busy ? "animate-spin" : undefined} />
            Restore golden
          </button>
        </div>
      </div>
    </div>
  );
}

function IconButton({
  children,
  onClick,
  disabled,
  label,
  title,
  active,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  label: string;
  title?: string;
  active?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={title ?? label}
      className="focus-ring flex h-6 w-6 items-center justify-center rounded-md border transition-colors disabled:opacity-30"
      style={{
        borderColor: active ? "var(--accent-line)" : "var(--border)",
        color: active ? "var(--accent)" : "var(--ink-dim)",
        background: active ? "var(--accent-soft)" : "transparent",
      }}
    >
      {children}
    </button>
  );
}

function RoundChip({
  entry,
  active,
  onSelect,
  isHarvest,
  isChampion,
}: {
  entry: TimelineEntry;
  active: boolean;
  onSelect: () => void;
  isHarvest: boolean;
  isChampion: boolean;
}) {
  const isSeed = entry.round === 0;
  const champion = isChampion && isHarvest;
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={active ? "step" : undefined}
      className="focus-ring group relative flex h-full min-w-[74px] flex-col items-center justify-center rounded-xl border px-2 py-1.5 transition-all"
      style={{
        borderColor: active ? (champion ? "var(--gold)" : "var(--accent)") : "var(--border)",
        background: active
          ? champion
            ? "var(--gold-soft)"
            : "var(--accent-soft)"
          : "var(--panel-2)",
        boxShadow: active ? "0 0 0 1px inset transparent" : undefined,
      }}
      title={
        isSeed
          ? "Seed batch — five random draws, no planning step"
          : `${entry.label} · ${entry.phase}${entry.verdict ? ` · ${entry.verdict}` : ""}`
      }
    >
      <span className="flex items-center gap-1">
        {champion ? (
          <Star size={9} style={{ color: "var(--gold)" }} fill="var(--gold)" />
        ) : null}
        <span
          className="mono text-[10.5px] font-semibold leading-none"
          style={{
            color: active ? (champion ? "var(--gold)" : "var(--accent)") : "var(--ink-dim)",
          }}
        >
          {isSeed ? "SEED" : entry.label}
        </span>
      </span>
      <span
        className="mono mt-1 text-[11.5px] leading-none"
        style={{ color: active ? "var(--ink)" : "var(--ink-dim)" }}
      >
        {num(entry.best_qualified_binding, 2)}
      </span>
      <span className="mono mt-0.5 text-[8.5px] leading-none ink-faint">
        {isSeed ? "random" : isHarvest ? "harvest" : "3:1:1"}
      </span>
    </button>
  );
}
