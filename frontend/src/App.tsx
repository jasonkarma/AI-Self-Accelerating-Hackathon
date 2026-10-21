import { useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { endpoints } from "./lib/api";
import { useResource, useTheme } from "./hooks/useResource";
import { Header, type TabKey } from "./components/Header";
import { OverviewView } from "./views/overview/OverviewView";
import { LabView } from "./views/lab/LabView";
import { BenchmarkView } from "./views/benchmark/BenchmarkView";
import { CrossTargetView } from "./views/cross/CrossTargetView";
import { AttributionView } from "./views/attribution/AttributionView";
import type { StatePayload } from "./types/state";

/**
 * The console shell.
 *
 * Data-loading policy, and why:
 *
 * * **View 1 polls the round index, not the state.** `/api/rounds` is the cheap
 *   endpoint (`timeline()` refits no surrogate); it is polled at a slow cadence
 *   while the tab is visible so a live round triggered by `uvicorn` shows up
 *   without the reader reloading. The state itself is fetched only when the
 *   selected round changes.
 * * **Views 2–4 are fetched once.** They are frozen artifacts on disk; a timer
 *   would only add noise. The header's RELOAD button re-fetches everything.
 * * **Nothing is memoized across reloads.** Every panel is a pure function of the
 *   payload it is handed, so a re-fetch cannot leave a stale number behind.
 */
export default function App() {
  const { theme, toggle } = useTheme();
  // The demo opens on the cold open; `overview` is the only tab that is not a
  // report, and it exists so the first thirty seconds need no narration.
  const [tab, setTab] = useState<TabKey>("overview");
  const [selectedRound, setSelectedRound] = useState<number | null>(null);
  const [restoreError, setRestoreError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const rounds = useResource((signal) => endpoints.rounds(signal), [nonce]);
  const config = useResource((signal) => endpoints.config(signal), [nonce]);
  const ledger = useResource((signal) => endpoints.ledger(signal), [nonce]);
  const health = useResource((signal) => endpoints.health(signal), [nonce]);

  const benchmark = useResource((signal) => endpoints.benchmark(signal), [nonce]);
  const crossTarget = useResource((signal) => endpoints.crossTarget(signal), [nonce]);
  const attribution = useResource((signal) => endpoints.attribution(signal), [nonce]);

  const currentRound = rounds.data?.current_round ?? null;
  const effectiveRound = selectedRound ?? currentRound;

  const state = useResource<StatePayload>(
    (signal) =>
      endpoints.state(
        effectiveRound !== null && effectiveRound !== currentRound ? effectiveRound : undefined,
        signal,
      ),
    [effectiveRound, nonce, currentRound],
  );

  // Clamp the selection if the ledger got shorter under us (a rewind).
  useEffect(() => {
    if (selectedRound !== null && currentRound !== null && selectedRound > currentRound) {
      setSelectedRound(currentRound);
    }
  }, [currentRound, selectedRound]);

  // Playback: walk the recorded rounds at a readable pace, then stop.
  //
  // The timer lives here rather than in the view so a tab switch cannot leave an
  // orphaned interval advancing the selection in the background. It steps
  // *selection* only — never a POST — so replaying the campaign is free.
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!currentRound) return;
    if (!playing) return;
    const timer = window.setInterval(() => {
      setSelectedRound((value) => {
        const next = (value ?? 0) + 1;
        if (next >= currentRound) {
          setPlaying(false);
          return currentRound;
        }
        return next;
      });
    }, 1400);
    return () => window.clearInterval(timer);
  }, [playing, currentRound]);

  const onPlayToggle = useCallback(() => {
    setPlaying((value) => {
      const next = !value;
      // Starting playback always rewinds to the seed batch, so a viewer who
      // presses play twice sees the campaign from the beginning rather than a
      // jump from wherever the scrubber happened to be.
      if (next) setSelectedRound(0);
      return next;
    });
  }, []);

  const onRestoreGolden = useCallback(async () => {
    setRestoreError(null);
    try {
      await endpoints.restoreGolden();
      setSelectedRound(null);
      setNonce((value) => value + 1);
    } catch (error) {
      setRestoreError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const reload = useCallback(() => {
    setNonce((value) => value + 1);
  }, []);

  const loading = useMemo(
    () =>
      [rounds, config, ledger, health, benchmark, crossTarget, attribution, state].some(
        (resource) => resource.status === "loading",
      ),
    [rounds, config, ledger, health, benchmark, crossTarget, attribution, state],
  );

  return (
    <div className="min-h-full canvas">
      <Header
        tab={tab}
        onTab={setTab}
        theme={theme}
        onTheme={toggle}
        onReload={reload}
        loading={loading}
        ledger={ledger.data}
        ledgerError={ledger.status === "error"}
        health={health.data}
        config={config.data}
      />

      <main className="mx-auto max-w-[1520px] px-4 py-5 sm:px-6">
        {/*
         * View transitions are a fade + 6 px lift, not a slide. A slide implies
         * the tabs are a sequence; they are four independent reports, and a reader
         * who flips between View 2 and View 4 is comparing, not advancing.
         */}
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
          >
            {tab === "overview" ? (
              <OverviewView
                benchmark={benchmark}
                cross={crossTarget}
                state={state}
                config={config.data}
                onTab={setTab}
              />
            ) : null}
            {tab === "lab" ? (
              <LabView
                state={state}
                rounds={rounds.data}
                config={config.data}
                benchmark={benchmark.data}
                selectedRound={effectiveRound ?? 0}
                onSelectRound={(round) => setSelectedRound(round)}
                onRestoreGolden={onRestoreGolden}
                restoreError={restoreError}
                onReloadAll={reload}
                playing={playing}
                onPlayToggle={onPlayToggle}
              />
            ) : null}
            {tab === "benchmark" ? <BenchmarkView benchmark={benchmark} /> : null}
            {tab === "cross" ? (
              <CrossTargetView cross={crossTarget} attribution={attribution} benchmark={benchmark} />
            ) : null}
            {tab === "attribution" ? <AttributionView attribution={attribution} /> : null}
          </motion.div>
        </AnimatePresence>

        <footer
          className="mt-8 flex flex-wrap items-center justify-between gap-2 border-t pb-8 pt-3 text-[10px] ink-faint"
          style={{ borderColor: "var(--border)" }}
        >
          <span>Every figure is read from a stored result file or the live API — nothing is hand-entered.</span>
          <span className="mono">
            {currentRound !== null ? `ledger round ${currentRound}` : "ledger…"} ·{" "}
            {benchmark.data ? benchmark.data.source.label : "benchmark…"} ·{" "}
            {crossTarget.data ? `${crossTarget.data.seeds} runs per cross-target setting` : "cross-target…"}
          </span>
        </footer>
      </main>

      <RoundPolling enabled={tab === "lab"} onTick={() => rounds.reload()} />
    </div>
  );
}

/**
 * Slow, tab-scoped polling of the cheap round index.
 *
 * It exists so a round triggered from the terminal (`curl -N /api/stream-round`)
 * appears in the UI without a manual reload. 8 s is chosen to be much slower
 * than a round takes to *render* and much faster than a demo pause; it is
 * disabled while another tab is open because none of the other views depend on
 * the live ledger.
 */
function RoundPolling({ enabled, onTick }: { enabled: boolean; onTick: () => void }) {
  useEffect(() => {
    if (!enabled) return;
    // `onTick` is recreated each render (it closes over `rounds.reload`), so it is
    // deliberately not a dependency: depending on it would tear down and rebuild
    // the interval on every render and the poll would never fire. The effect is
    // keyed on `enabled` alone — start when the tab opens, stop when it closes.
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") onTick();
    }, 8000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);
  return null;
}
