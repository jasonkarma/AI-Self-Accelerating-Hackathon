import { CheckCircle2, ChevronDown, Gavel, HelpCircle, XCircle } from "lucide-react";
import { useState } from "react";
import type { PlannerRecord, StatePayload } from "../../types/state";
import { InfoTip } from "../../components/Panel";
import { renderInline } from "../../components/Markdown";

/**
 * The fourth agent.
 *
 * Three specialists argue in parallel and write memos. A fourth — the planner —
 * reads all three memos plus the raw tool tables and emits a structured
 * `RoundDecisionRecord`: the hypothesis worth testing next, the verdict on the
 * *previous* hypothesis, and its own account of where the three seats disagreed.
 *
 * That output is why the council is not just a debate club: the memos are
 * input, and this card is the decision. It had no UI at all until the backend
 * exposed it, because `agent_council` stores memos and the planner produces a
 * record rather than prose.
 *
 * The verdict line is the quiet star. "2 of 6 hypotheses disproved" on the
 * overview is this card, counted: a planner that marks its own earlier claim
 * FALSIFIED is the falsification loop in one row.
 */
export function PlannerCard({ state }: { state: StatePayload }) {
    const [open, setOpen] = useState(false);
    const planner = state.planner as PlannerRecord | undefined;
    const hasRecord = planner !== undefined && typeof planner.hypothesis_id === "string";

    if (!hasRecord) {
        return (
            <section className="panel px-4 py-3.5">
                <div className="label mb-1 flex items-center gap-1.5 text-[9.5px]">
                    <Gavel size={10} style={{ color: "var(--ink-faint)" }} />
                    Decision Specialist
                </div>
                <p className="text-[11.5px] leading-relaxed ink-faint">
                    The opening batch has no planning step — five candidates are drawn at random before any evidence
                    exists. From Round 1 onward this card carries the hypothesis the planner chose to test.
                </p>
            </section>
        );
    }

    const verdict = verdictMeta(planner.previous_verdict);
    const VerdictIcon = verdict.icon;

    return (
        <section className="panel">
            <header className="flex flex-wrap items-start justify-between gap-2 px-4 pb-2 pt-3.5">
                <div className="min-w-0">
                    <div className="label mb-1 flex items-center gap-1.5 text-[9.5px]">
                        <Gavel size={10} style={{ color: "var(--gold)" }} />
                        Decision Specialist · round {planner.round_number}
                    </div>
                    <h2 className="text-[13px] font-semibold leading-snug ink">
                        A fourth agent reads all three arguments and commits to one claim
                    </h2>
                </div>
                <span
                    className="mono rounded-full px-2 py-0.5 text-[9px] font-semibold"
                    style={{ background: verdict.bg, color: verdict.color }}
                    title={verdict.hint}
                >
                    {verdict.tag}
                </span>
            </header>

            <div className="px-4 pb-3">
                <div
                    className="rounded-lg border px-2.5 py-2"
                    style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}
                >
                    <div className="mb-1 flex items-center gap-1.5">
                        <span className="label text-[8.5px]">the claim it chose to test</span>
                        {planner.hypothesis_id ? (
                            <span className="mono text-[8.5px] ink-faint">{planner.hypothesis_id}</span>
                        ) : null}
                    </div>
                    <p className="line-clamp-3 text-[11px] leading-relaxed ink">
                        {planner.hypothesis_statement ? renderInline(planner.hypothesis_statement) : "—"}
                    </p>
                </div>

                <div className="mt-2 flex items-start gap-2 rounded-lg border px-2.5 py-2" style={{ borderColor: "var(--border)" }}>
                    <VerdictIcon size={11} style={{ color: verdict.color, marginTop: 2, flexShrink: 0 }} />
                    <div className="min-w-0">
                        <div className="text-[10px] leading-snug" style={{ color: verdict.color }}>
                            {verdict.headline}
                        </div>
                        <div className="text-[9.5px] leading-snug ink-faint">
                            Confidence {planner.confidence ?? "—"}
                            {planner.targets?.control ? (
                                <>
                                    {" · "}control well <span className="mono">{planner.targets.control}</span>
                                </>
                            ) : null}
                        </div>
                    </div>
                    <InfoTip label="Why the planner grades its own last claim">
                        The planner writes the round's hypothesis, then—next round—grades it against the matched control
                        that was assayed alongside it. A planner that marks its own earlier claim FALSIFIED is the
                        falsification loop working; the count on the overview is these verdicts tallied.
                    </InfoTip>
                </div>

                {planner.disagreement_summary ? (
                    <div className="mt-2">
                        <button
                            type="button"
                            onClick={() => setOpen((value) => !value)}
                            aria-expanded={open}
                            className="focus-ring flex w-full items-start gap-1.5 text-left text-[10px] leading-snug ink-dim transition-colors hover:text-[color:var(--ink)]"
                        >
                            <ChevronDown
                                size={10}
                                className="mt-[2px] shrink-0 transition-transform"
                                style={{ transform: open ? "rotate(180deg)" : undefined }}
                            />
                            <span>
                                <span className="font-semibold">Where the three seats disagreed</span> — the planner's own
                                account
                            </span>
                        </button>
                        {open ? (
                            <p className="mt-1 rounded-lg border-l-2 px-2.5 py-1.5 text-[10.5px] leading-relaxed ink-dim" style={{ borderColor: "var(--gold-line)", background: "var(--panel-2)" }}>
                                {planner.disagreement_summary}
                            </p>
                        ) : null}
                    </div>
                ) : null}
            </div>
        </section>
    );
}

function verdictMeta(verdict: PlannerRecord["previous_verdict"]): {
    tag: string;
    headline: string;
    color: string;
    bg: string;
    hint: string;
    icon: typeof CheckCircle2;
} {
    switch (verdict) {
        case "SUPPORTED":
            return {
                tag: "LAST CLAIM HELD",
                headline: "The previous round's hypothesis survived its control.",
                color: "var(--accent)",
                bg: "var(--accent-soft)",
                hint: "The planner's own earlier claim was confirmed by the matched control assayed beside it.",
                icon: CheckCircle2,
            };
        case "FALSIFIED":
            return {
                tag: "LAST CLAIM DISPROVED",
                headline: "The previous round's hypothesis failed its control.",
                color: "#F59E0B",
                bg: "rgba(245,158,11,0.12)",
                hint: "The planner's own earlier claim was contradicted by the assay — and it says so here rather than quietly moving on.",
                icon: XCircle,
            };
        case "INCONCLUSIVE":
            return {
                tag: "LAST CLAIM UNRESOLVED",
                headline: "The previous round's hypothesis was not separated from its control.",
                color: "#94A3B8",
                bg: "rgba(148,163,184,0.12)",
                hint: "The assay did not separate the hypothesis from its control, so the claim stays open.",
                icon: HelpCircle,
            };
        default:
            return {
                tag: "FIRST CLAIM",
                headline: "Nothing to grade yet — this is the first hypothesis.",
                color: "var(--ink-dim)",
                bg: "var(--panel-2)",
                hint: "Round 1 has no previous claim to grade.",
                icon: HelpCircle,
            };
    }
}
