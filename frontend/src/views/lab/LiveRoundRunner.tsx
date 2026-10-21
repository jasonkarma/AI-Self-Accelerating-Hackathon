import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, PlayCircle, Square } from "lucide-react";
import { streamRoundUrl, type StreamEvent } from "../../lib/api";

/**
 * The live round runner — `GET /api/stream-round` over `EventSource`.
 *
 * This is the only control in the console that **spends wet-lab budget**. The
 * backend's stream endpoint triggers a round as a side effect of connecting, so
 * the button is an explicit, labelled action rather than something a page load
 * can do by accident, and it is disabled once the campaign has reached its
 * horizon.
 *
 * The component renders **only the control** into the timeline toolbar; the
 * event feed is mounted beneath it and stays empty (and invisible) until a stream
 * actually emits something, so no empty drawer occupies the workspace on load.
 * The vocabulary is the one in SPEC §9.1 (`round_started` → `tool_call` →
 * `agent_thinking` → `agent_memo` → `planner_synthesis` → `slot_selected` →
 * `oracle_reveal` → `round_complete` → `stream_end`), and it never fabricates an
 * event: if the server emits nothing, the feed stays empty and says so.
 */
export function LiveRoundRunner({
  currentRound,
  totalRounds,
  onRoundComplete,
}: {
  currentRound: number;
  totalRounds: number;
  onRoundComplete: () => void;
}) {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);
  const completedRef = useRef(false);

  const stop = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
    setRunning(false);
  }, []);

  useEffect(() => stop, [stop]);

  const start = useCallback(() => {
    if (sourceRef.current) return;
    setEvents([]);
    setError(null);
    setRunning(true);
    completedRef.current = false;

    const source = new EventSource(streamRoundUrl());
    sourceRef.current = source;

    const record = (name: string) => (event: MessageEvent<string>) => {
      let data: Record<string, unknown> = {};
      try {
        const parsed: unknown = JSON.parse(event.data);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          data = parsed as Record<string, unknown>;
        }
      } catch {
        data = { raw: event.data };
      }
      setEvents((previous) => [...previous, { event: name, data }]);
      if (name === "round_complete" && !completedRef.current) {
        completedRef.current = true;
        onRoundComplete();
      }
      if (name === "stream_end") {
        source.close();
        sourceRef.current = null;
        setRunning(false);
      }
      if (name === "error" && typeof data.message === "string") {
        setError(data.message);
      }
    };

    for (const name of [
      "round_started",
      "tool_call",
      "agent_thinking",
      "agent_memo",
      "planner_synthesis",
      "batch_pending_approval",
      "slot_selected",
      "oracle_reveal",
      "round_complete",
      "error",
      "stream_end",
    ]) {
      source.addEventListener(name, record(name) as EventListener);
    }
    source.onerror = () => {
      // The server closes the stream after `stream_end`; a browser-level error
      // after that is expected and must not be reported as a failure.
      if (completedRef.current) return;
      setError("stream disconnected before round_complete — the campaign state is unchanged for this round");
      stop();
    };
  }, [onRoundComplete, stop]);

  const exhausted = currentRound >= totalRounds;

  return (
    <>
      <button
        type="button"
        onClick={running ? stop : start}
        disabled={!running && exhausted}
        title={
          exhausted
            ? `The campaign has spent its ${totalRounds}-round horizon. Restore the golden ledger to replay it.`
            : "Connecting triggers one round on the server and streams every event. Spends 5 wet-lab queries."
        }
        className="focus-ring flex h-7 items-center gap-1.5 rounded-md border px-2 text-[10.5px] font-medium disabled:opacity-40"
        style={{
          borderColor: running ? "rgba(244,63,94,0.4)" : "var(--accent-line)",
          color: running ? "var(--danger)" : "var(--accent)",
          background: running ? "rgba(244,63,94,0.07)" : "var(--accent-soft)",
        }}
      >
        {running ? <Square size={10} /> : <PlayCircle size={11} />}
        {running ? "Stop stream" : exhausted ? "Horizon spent" : `Run round ${currentRound + 1}`}
      </button>

      {(events.length > 0 || error) && (
        <div className="mt-2 max-h-40 overflow-y-auto rounded-lg border px-2 py-1" style={{ borderColor: "var(--border)" }}>
          {error ? (
            <div className="flex items-start gap-1.5 px-1 py-1 text-[10.5px]" style={{ color: "#F59E0B" }}>
              <AlertTriangle size={10} className="mt-[2px] shrink-0" />
              {error}
            </div>
          ) : null}
          <ol>
            {events.map((event, index) => (
              <EventRow key={`${event.event}-${index}`} event={event} />
            ))}
          </ol>
          {running && events.length === 0 ? (
            <div className="px-1 py-1 text-[10.5px] ink-faint">Waiting for the first event…</div>
          ) : null}
        </div>
      )}
    </>
  );
}

function EventRow({ event }: { event: StreamEvent }) {
  const color = eventColor(event.event);
  const detail = summarise(event);
  return (
    <li className="flex items-start gap-2 py-0.5">
      <span className="mono w-[104px] shrink-0 text-[9.5px]" style={{ color }}>
        {event.event}
      </span>
      <span className="mono min-w-0 flex-1 break-words text-[10px] ink-dim">{detail}</span>
    </li>
  );
}

function eventColor(name: string): string {
  switch (name) {
    case "round_started":
      return "#0EA5E9";
    case "tool_call":
      return "#A855F7";
    case "agent_thinking":
    case "agent_memo":
      return "#6366F1";
    case "planner_synthesis":
      return "#F59E0B";
    case "slot_selected":
      return "#10B981";
    case "oracle_reveal":
      return "#22D3EE";
    case "round_complete":
    case "stream_end":
      return "#10B981";
    case "error":
      return "#F43F5E";
    default:
      return "var(--ink-faint)";
  }
}

/** A one-line rendering of each event's payload, using only fields it carries. */
function summarise(event: StreamEvent): string {
  const data = event.data;
  const text = (key: string): string | null =>
    typeof data[key] === "string" || typeof data[key] === "number" ? String(data[key]) : null;
  switch (event.event) {
    case "round_started": {
      const round = text("round");
      const budget = text("budget_spent");
      const total = text("total_budget");
      return `round ${round ?? "?"} · ${text("phase") ?? ""} · budget ${budget ?? "?"}/${total ?? "?"}`;
    }
    case "tool_call": {
      const duration = typeof data.duration_ms === "number" ? `${data.duration_ms.toFixed(0)} ms` : "—";
      return `${text("tool") ?? "?"} → ${text("role") ?? "?"} · ${duration}`;
    }
    case "agent_thinking":
      return `${text("role") ?? "?"} · ${String(data.chunk ?? data.token ?? "").slice(0, 96)}`;
    case "agent_memo":
      return `${text("role") ?? "?"} · ${String(data.memo ?? "").slice(0, 120)}`;
    case "planner_synthesis": {
      const hypothesis = text("hypothesis_id");
      return `${hypothesis ?? "decision"} · ${String(data.hypothesis_statement ?? "").slice(0, 100)}`;
    }
    case "slot_selected":
      return `slot ${text("slot_id") ?? "?"} · ${text("candidate_id") ?? "?"} · ${text("slot_type") ?? ""}`;
    case "oracle_reveal": {
      const binding = typeof data.y_bind === "number" ? data.y_bind.toFixed(3) : "—";
      const best = data.new_best === true ? " · NEW BEST" : "";
      return `${text("candidate_id") ?? "?"} · y_bind ${binding}${best}`;
    }
    case "round_complete": {
      const summary = data.round_summary;
      const queried =
        summary && typeof summary === "object" && "queried" in summary
          ? (summary as { queried: unknown }).queried
          : null;
      return Array.isArray(queried) ? `batch measured: ${queried.join(", ")}` : "round complete";
    }
    case "error":
      return text("message") ?? "unknown error";
    case "stream_end":
      return data.ok === true ? "ok" : "closed";
    case "batch_pending_approval":
      return "awaiting human approval (HITL mode) — POST /api/round/approve or /api/round/reject";
    default:
      return JSON.stringify(data).slice(0, 140);
  }
}

