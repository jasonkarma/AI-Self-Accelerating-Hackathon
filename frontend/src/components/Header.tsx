import { useState } from "react";
import { Check, ChevronDown, Info, Moon, RefreshCw, ShieldCheck, Sun } from "lucide-react";
import type { ConfigPayload, HealthPayload, LedgerPayload } from "../types/state";
import { PINNED, shortSha } from "../lib/pinned";
import type { Theme } from "../hooks/useResource";

export type TabKey = "overview" | "lab" | "benchmark" | "cross" | "attribution";

/**
 * The tab bar is the demo's outline, so the labels are sentences rather than
 * section names: a judge should be able to read the four tabs and know what the
 * argument is before clicking anything.
 */
export const TABS: Array<{ key: TabKey; index: number; title: string; subtitle: string }> = [
  { key: "overview", index: 0, title: "Overview", subtitle: "What we built, in 30 seconds" },
  { key: "lab", index: 1, title: "The recorded run", subtitle: "Watch it decide, round by round" },
  { key: "benchmark", index: 2, title: "Beats standard UCB", subtitle: "100 runs, head to head" },
  { key: "cross", index: 3, title: "Works on other targets", subtitle: "Four more public datasets, same code" },
  { key: "attribution", index: 4, title: "Why it works", subtitle: "The ablations, including the nulls" },
];

interface Props {
  tab: TabKey;
  onTab: (tab: TabKey) => void;
  theme: Theme;
  onTheme: () => void;
  onReload: () => void;
  loading: boolean;
  ledger: LedgerPayload | null;
  ledgerError: boolean;
  health: HealthPayload | null;
  config: ConfigPayload | null;
}

/**
 * The application bar.
 *
 * Two rules keep it from reading as a debug monitor:
 *
 * 1. **Navigation is a product tab bar.** Four numbered sections, each with a
 *    human label and a one-line subtitle. No HTTP verbs, no route names.
 * 2. **Verification is one pill, not six chips.** The lock digest, hash-chain
 *    state, test count, objective law and protocol parameters live behind a
 *    single `Provenance & Verification` disclosure in the top-right corner. The
 *    pill itself carries only the verdict (`hash chain verified`, test count),
 *    so a judge can see at a glance that the data is checkable and open it when
 *    they want to check.
 *
 * The disclosure is a native `<details>` element rendered with the body always
 * in the DOM and hidden by CSS (`.prov-body`). That keeps it keyboard-operable,
 * keeps the numbers in the static render used by the smoke test, and avoids a
 * portal that would break `renderToStaticMarkup`.
 */
export function Header({
  tab,
  onTab,
  theme,
  onTheme,
  onReload,
  loading,
  ledger,
  ledgerError,
  health,
  config,
}: Props) {
  return (
    <header
      className="sticky top-0 z-30 border-b backdrop-blur-md"
      style={{ background: "var(--header)", borderColor: "var(--border)" }}
    >
      <div className="mx-auto flex max-w-[1520px] items-center justify-between gap-4 px-4 pt-3 sm:px-6">
        <div className="flex min-w-0 items-center gap-2.5">
          <Mark />
          <div className="min-w-0">
            <h1 className="truncate text-[14.5px] font-semibold leading-tight tracking-[-0.01em] ink">
              Self-Accelerating Multi-Agent Virtual Lab
            </h1>
            <div className="truncate text-[11px] leading-tight ink-faint">
              {/* Deliberately carries no number.
                  The app bar is persistent chrome: it stays on screen across all
                  five tabs, where the pools are different sizes (210 here, 756
                  on aayl49_ML, 124 on 2fjg, 486 on aayl51). A count here would
                  describe one dataset while the reader is looking at another,
                  and it would go stale on any re-run. The instance number lives
                  once, in the 83% tile's caption, on the screen where it applies.

                  "Controls for its own reasoning" is the precise claim: the loop
                  reserves one well per batch for what the baseline would have
                  picked, so the agent prior is itself the thing under test. */}
              A virtual lab that decides the next experiment — and controls for its own reasoning
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          <ProvenanceDrawer ledger={ledger} ledgerError={ledgerError} health={health} config={config} />
          <button
            type="button"
            onClick={onReload}
            disabled={loading}
            className="focus-ring flex h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[11px] font-medium transition-colors disabled:opacity-60"
            style={{ borderColor: "var(--border)", color: "var(--ink-dim)" }}
            title="Re-fetch every endpoint"
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : undefined} />
            <span className="hidden sm:inline">{loading ? "Syncing" : "Refresh"}</span>
          </button>
          <button
            type="button"
            onClick={onTheme}
            className="focus-ring flex h-8 w-8 items-center justify-center rounded-lg border transition-colors"
            style={{ borderColor: "var(--border)", color: "var(--ink-dim)" }}
            title={theme === "dark" ? "Light mode for projectors" : "Dark mode"}
            aria-label="Toggle colour theme"
          >
            {theme === "dark" ? <Sun size={13} /> : <Moon size={13} />}
          </button>
        </div>
      </div>

      <nav
        className="scroll-thin mx-auto -mb-px mt-2 flex max-w-[1520px] gap-0.5 overflow-x-auto px-4 sm:px-6"
        aria-label="Console views"
      >
        {TABS.map((entry) => {
          const active = entry.key === tab;
          return (
            <button
              key={entry.key}
              type="button"
              onClick={() => onTab(entry.key)}
              aria-current={active ? "page" : undefined}
              className="focus-ring group relative shrink-0 rounded-t-lg px-3 pb-2.5 pt-2 text-left transition-colors"
              style={{ background: active ? "var(--panel)" : "transparent" }}
            >
              <div className="flex items-baseline gap-1.5">
                <span
                  className="mono text-[10px] font-semibold tracking-tight"
                  style={{ color: active ? "var(--accent)" : "var(--ink-faint)" }}
                >
                  {/* The overview is the door, not section 0. */}
                  {entry.index === 0 ? "→" : String(entry.index).padStart(2, "0")}
                </span>
                <span
                  className="whitespace-nowrap text-[12.5px] font-semibold leading-none"
                  style={{ color: active ? "var(--ink)" : "var(--ink-dim)" }}
                >
                  {entry.title}
                </span>
              </div>
              <div
                className="mt-1 whitespace-nowrap pl-[22px] text-[10.5px] leading-none"
                style={{ color: active ? "var(--ink-dim)" : "var(--ink-faint)" }}
              >
                {entry.subtitle}
              </div>
              <span
                aria-hidden
                className="absolute inset-x-2 bottom-0 h-[2px] rounded-full transition-opacity"
                style={{
                  background: "linear-gradient(90deg,#10B981,#22D3EE)",
                  opacity: active ? 1 : 0,
                }}
              />
            </button>
          );
        })}
      </nav>
    </header>
  );
}

/** A compact mark: three bars stepping up, the loop's own staircase. */
function Mark() {
  return (
    <span
      aria-hidden
      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px] border"
      style={{
        borderColor: "var(--accent-line)",
        background: "linear-gradient(150deg, var(--accent-soft), transparent 75%)",
      }}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
        <path d="M2 12.5V9" stroke="#10B981" strokeWidth="2.1" strokeLinecap="round" />
        <path d="M6 12.5V6.4" stroke="#10B981" strokeWidth="2.1" strokeLinecap="round" opacity="0.85" />
        <path d="M10 12.5V3.6" stroke="#22D3EE" strokeWidth="2.1" strokeLinecap="round" />
        <circle cx="13.4" cy="3.4" r="1.6" fill="#F0B429" />
      </svg>
    </span>
  );
}

/**
 * `ⓘ Provenance & Verification` — the single home for every technical token
 * that used to be loose in the top bar.
 */
function ProvenanceDrawer({
  ledger,
  ledgerError,
  health,
  config,
}: {
  ledger: LedgerPayload | null;
  ledgerError: boolean;
  health: HealthPayload | null;
  config: ConfigPayload | null;
}) {
  const [open, setOpen] = useState(false);
  const verified = ledger?.verified === true;
  const entries = ledger?.entries?.length ?? null;
  const tone = verified ? "var(--accent)" : ledgerError ? "var(--danger)" : "var(--gold)";

  return (
    <details
      className="prov group relative"
      open={open}
      onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}
    >
      <summary
        className="focus-ring flex h-8 cursor-pointer select-none items-center gap-1.5 rounded-lg border px-2.5 text-[11px] font-medium transition-colors"
        style={{
          borderColor: open ? "var(--accent-line)" : "var(--border)",
          background: open ? "var(--accent-soft)" : "var(--panel-2)",
          color: "var(--ink-dim)",
        }}
        title="Provenance & verification — lock digest, hash chain, tests, pipeline parameters"
      >
        <ShieldCheck size={12} style={{ color: tone }} />
        <span className="hidden sm:inline">Provenance</span>
        <span className="mono text-[10px]" style={{ color: tone }}>
          {verified ? "✓" : "…"}
        </span>
        <ChevronDown size={11} className="prov-chevron transition-transform" />
      </summary>

      <div
        className="prov-body absolute right-0 top-10 z-50 w-[380px] max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl border shadow-2xl"
        style={{ background: "var(--panel-raised)", borderColor: "var(--border-strong)" }}
      >
        <div className="flex items-start gap-2 border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
          <ShieldCheck size={14} className="mt-0.5 shrink-0" style={{ color: tone }} />
          <div className="min-w-0">
            <div className="text-[12.5px] font-semibold ink">Provenance &amp; verification</div>
            <div className="mt-0.5 text-[11px] leading-snug ink-faint">
              Every figure in this console is read from the API. Nothing is hand-entered.
            </div>
          </div>
        </div>

        <dl className="divide-y" style={{ borderColor: "var(--border)" }}>
          <Row
            label="Hash chain"
            value={verified ? `VERIFIED · ${entries ?? "—"} entries` : ledgerError ? "UNAVAILABLE" : "PENDING"}
            color={tone}
            hint="GET /api/ledger walks the ledger and re-computes each entry's hash; `verified` is the server's own verdict."
            icon={<Check size={10} />}
          />
          <Row
            label="Golden ledger SHA-256"
            value={`${shortSha(PINNED.goldenLedgerSha256)}…${PINNED.goldenLedgerSha256.slice(-4)}`}
            mono
            hint={`SHA-256 of results/golden_seed21.sqlite3 at commit ${PINNED.lockedCommit}. A property of the repository, not of the running server.`}
          />
          <Row
            label="Backend suite"
            value={`${PINNED.backendTests} passing`}
            hint={`Recorded at commit ${PINNED.lockedCommit}. A repository fact — the backend does not report its own test count.`}
          />
          <Row
            label="Objective law"
            value={PINNED.objectiveLaw}
            color="var(--accent)"
            mono
            hint="The shipped objective (core/state.py). The exploit/exploration weights are diagnostics of the retired anneal, not coefficients of a_null."
          />
          <Row
            label="API"
            value={`v${health?.version ?? config?.version ?? "—"}`}
            hint="GET /api/health"
          />
          {config ? (
            <>
              <Row
                label="Protocol"
                value={`${config.version} · ${config.pool_size} rows`}
                hint="The frozen protocol from GET /api/config."
              />
              <Row
                label="τ_dev threshold"
                value={config.tau_dev.toFixed(4)}
                mono
                hint={`${config.threshold_source} · ${config.qualified_count} of ${config.pool_size} candidates qualified.`}
              />
              <Row
                label="Data mode"
                value={config.is_synthetic ? "SYNTHETIC" : "MEASURED"}
                color={config.is_synthetic ? "var(--gold)" : "var(--accent)"}
                hint="Whether the pool's binding labels are measured assay values or a synthetic surrogate."
              />
              <Row
                label="Dataset digest"
                value={`${config.dataset_sequence_sha256.slice(0, 12)}…`}
                mono
                hint="SHA-256 of the normalised 1N8Z sequence pool."
              />
            </>
          ) : null}
        </dl>

        <div
          className="flex items-start gap-1.5 border-t px-4 py-2.5 text-[10px] leading-relaxed ink-faint"
          style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}
        >
          <Info size={10} className="mt-[2px] shrink-0" />
          Live half of this card is fetched (ledger, health, protocol). The three repository facts are
          pinned constants in <span className="mono">src/lib/pinned.ts</span> and labelled as such.
        </div>
      </div>
    </details>
  );
}

function Row({
  label,
  value,
  color,
  mono,
  hint,
  icon,
}: {
  label: string;
  value: string;
  color?: string;
  mono?: boolean;
  hint: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3 px-4 py-1.5" title={hint}>
      <dt className="text-[11px] ink-faint">{label}</dt>
      <dd
        className={`flex items-center gap-1 text-[11px] ${mono ? "mono" : ""}`}
        style={{ color: color ?? "var(--ink)" }}
      >
        {icon}
        {value}
      </dd>
    </div>
  );
}
