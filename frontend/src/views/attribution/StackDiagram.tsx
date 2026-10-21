import { ArrowDown, X } from "lucide-react";
import { InfoTip } from "../../components/Panel";

/**
 * What the loop is made of, and which layer the `no_anneal` ablation removed.
 *
 * Written for the narrow column beside the causal-winner card, so it stacks
 * vertically: evidence in at the top, a plate out at the bottom. Each stage
 * states three things — **what comes in**, **what it does**, and **what comes
 * out** — because the previous horizontal version labelled only the arrows,
 * which left the reader to infer the mechanics from four noun phrases.
 *
 * The stage that matters is 03. The deleted heuristic lived there and nowhere
 * else: between the agents' *opinions* and the selector's *number*. Saying
 * "no agent, tool or slot rule changed" is the whole reason the ablation can be
 * interpreted as a single-cause result.
 */
export function StackDiagram() {
    return (
        <section className="panel px-3.5 py-3.5">
            <div className="mb-2.5 flex items-start justify-between gap-2">
                <div className="min-w-0">
                    <div className="label mb-1 flex items-center gap-1.5 text-[9px]">
                        How one round is decided
                        <InfoTip label="Why this diagram is here">
                            The ablation below says <em>what</em> changed. This says <em>where</em> in the engine the change
                            landed — and, more usefully, what it could <em>not</em> touch. Agents can re-order candidates;
                            they cannot rescale the score.
                        </InfoTip>
                    </div>
                    <div className="text-[12px] leading-snug ink-dim">
                        Evidence in at the top, five candidates out at the bottom.
                    </div>
                </div>
            </div>

            <div className="space-y-0">
                <Stage
                    index="01"
                    title="Tools measure"
                    color="#22D3EE"
                    input="the assay results so far"
                    action="builds tables: per-position effects, epistasis clashes, liabilities, past mistakes"
                    output="numbers, not opinions"
                    note="Every figure an agent cites originates here — the models cannot invent one."
                />

                <Connector label="the same tables, to all three" />

                <Stage
                    index="02"
                    title="Three agents argue"
                    color="#6366F1"
                    input="those tables only"
                    action="binding proposes · manufacturability vetoes · skeptic attacks"
                    output="three memos, written in parallel"
                    note="No agent reads another's memo. Disagreement is structural, not sequential."
                />

                <Connector label="opinions become one number, with a cap" highlight />

                <Stage
                    index="03"
                    title="Acquisition scores"
                    color="#10B981"
                    input="the memos' bounded prior + the surrogate"
                    action="merges prediction, uncertainty and the agent prior into a single ranking"
                    output="every candidate, ranked"
                    note="The agent prior is capped here — agents can re-order the list, never rescale it. The surrogate is an ensemble of randomized trees over fixed-length residue features, chosen because 5–35 measurements is far too little data for a deep model and too little to fit a meaningful kernel by hand."
                    highlight
                    removed="the heuristic we deleted sat exactly here, nudging every score toward the model's own guess"
                />

                <Connector label="top five, by rule" />

                <Stage
                    index="04"
                    title="Slots get filled"
                    color="#F0B429"
                    input="that ranking"
                    action="applies the 3-exploit : 1-explore : 1-control mix (5:0:0 once the horizon passes two thirds)"
                    output="the five wells you assay"
                    note="The mix is a portfolio rule, not a model output."
                />
            </div>

            <div
                className="mt-3 rounded-lg border px-2.5 py-2"
                style={{ borderColor: "rgba(240,180,41,0.4)", background: "var(--gold-soft)" }}
            >
                <div className="text-[10.5px] font-semibold leading-snug" style={{ color: "var(--gold)" }}>
                    Only stage 03 changed
                </div>
                <p className="mt-0.5 text-[10px] leading-relaxed ink-dim">
                    The removed heuristic added a term between the agents' prior and the final score. No tool changed,
                    no agent changed, no slot rule changed — so when ranking accuracy improved in every setting, there is
                    only one thing it can be attributed to. The control well is the visible consequence: a control that
                    inherits an exploration bonus is no longer a control.
                </p>
            </div>
        </section>
    );
}

function Connector({ label, highlight = false }: { label: string; highlight?: boolean }) {
    const color = highlight ? "var(--gold)" : "var(--ink-faint)";
    return (
        <div className="flex items-center gap-1.5 py-1 pl-[18px]">
            <ArrowDown size={10} style={{ color }} />
            <span className="mono text-[8.5px] leading-tight" style={{ color }}>
                {label}
            </span>
        </div>
    );
}

function Stage({
    index,
    title,
    color,
    input,
    action,
    output,
    note,
    highlight = false,
    removed,
}: {
    index: string;
    title: string;
    color: string;
    input: string;
    action: string;
    output: string;
    note: string;
    highlight?: boolean;
    removed?: string;
}) {
    return (
        <div
            className="rounded-lg border px-2.5 py-2"
            style={{
                borderColor: highlight ? `${color}88` : "var(--border)",
                background: highlight ? `${color}12` : "var(--panel-2)",
            }}
        >
            <div className="flex items-baseline gap-1.5">
                <span className="mono text-[8.5px] font-semibold" style={{ color }}>
                    {index}
                </span>
                <span className="text-[11.5px] font-semibold leading-tight ink">{title}</span>
            </div>

            <dl className="mt-1 space-y-0.5">
                <Row term="in" value={input} />
                <Row term="does" value={action} />
                <Row term="out" value={output} strong />
            </dl>

            {removed ? (
                <div className="mt-1.5 flex items-start gap-1 rounded border px-1.5 py-1" style={{ borderColor: "rgba(244,63,94,0.45)", background: "rgba(244,63,94,0.08)" }}>
                    <X size={9} className="mt-[2px] shrink-0" style={{ color: "#FB7185" }} />
                    <span className="text-[9px] leading-snug" style={{ color: "#FB7185" }}>
                        {removed}
                    </span>
                </div>
            ) : null}

            <p className="mt-1 text-[9px] leading-snug ink-faint">{note}</p>
        </div>
    );
}

function Row({ term, value, strong = false }: { term: string; value: string; strong?: boolean }) {
    return (
        <div className="flex gap-1 text-[9.5px] leading-snug">
            <dt className="mono w-[26px] shrink-0 text-[8px] uppercase tracking-wide ink-faint">{term}</dt>
            <dd className={strong ? "ink" : "ink-dim"}>{value}</dd>
        </div>
    );
}
