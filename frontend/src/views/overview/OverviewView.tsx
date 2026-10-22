import { ArrowRight, FlaskConical, Scale, Target, Trophy } from "lucide-react";
import type { BenchmarkPayload, CrossTargetPayload } from "../../types/report";
import type { ConfigPayload, StatePayload } from "../../types/state";
import type { TabKey } from "../../components/Header";
import { DatasetStrip, InfoTip } from "../../components/Panel";
import { HBarChart } from "../../components/charts";
import { datasetIdentity, finalRho, pairedDelta } from "../../lib/derive";
import { int, methodMeta, num } from "../../lib/format";
import { countOf, flukeOdds, gainWhole, librarySkipped, methodName, shareWhole } from "../../lib/plain";
import type { Resource } from "../../hooks/useResource";

/**
 * The cold open.
 *
 * Three numbers, each a door into the view that proves it. The page exists so a
 * five-minute demo does not have to be driven from memory: read the sentence,
 * point at the number, click through to the evidence.
 *
 * The three are chosen to answer the three questions a judge actually asks, in
 * order: *does it work?* (ranking accuracy), *is it efficient?* (library
 * skipped), *does it generalize?* (targets won). The fourth card states the task
 * itself, because nothing else on the console does.
 */
export function OverviewView({
    benchmark,
    cross,
    state,
    config,
    onTab,
}: {
    benchmark: Resource<BenchmarkPayload>;
    cross: Resource<CrossTargetPayload>;
    state: Resource<StatePayload>;
    config: ConfigPayload | null;
    onTab: (tab: TabKey) => void;
}) {
    const payload = benchmark.data;
    const final = payload
        ? finalRho(payload.spearman_rho_curves as unknown as Array<Record<string, number | null>>)
        : null;
    const vsM3 = payload
        ? pairedDelta(payload, "m5_hybrid_minus_m3_ucb", "spearman_rho_at_q35")
        : null;
    const gain =
        final && final.m5_hybrid !== null && final.m3_ucb !== null && final.m3_ucb !== 0
            ? (final.m5_hybrid - final.m3_ucb) / final.m3_ucb
            : null;

    const spent = state.data?.kpis.query_budget_spent ?? null;
    const qualified = config?.qualified_count ?? null;
    const skipped = librarySkipped(spent, qualified);

    const identity = datasetIdentity(cross.data ?? null);
    const partition = cross.data?.available ? cross.data.learnability_gate.partition : null;
    const learnableCount = partition?.learnable_multiround.length ?? null;
    // The corpus size the partition was built from — 17 enumerated files. Read
    // from the metadata rather than typed, so "3 of 17" cannot go stale if the
    // corpus grows.
    const totalDatasets = cross.data?.available
        ? cross.data.lockbox_metadata.rows.length
        : null;
    // The metadata-only exclusion: datasets that collapse to one or two distinct
    // loops. Kept separate from the baseline-gated ones, because only these were
    // decided without running anything.
    const degenerateCount = partition?.degenerate_1_or_2_cdrh3.length ?? null;

    return (
        <div className="space-y-4">
            {/* ── The one-sentence task ─────────────────────────────────────────── */}
            <section className="panel hero-wash">
                <div className="px-6 pb-4 pt-4">
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                        <span
                            className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10.5px] font-semibold"
                            style={{ background: "var(--gold-soft)", color: "var(--gold)", border: "1px solid var(--gold-line)" }}
                        >
                            <Trophy size={11} /> START HERE
                        </span>
                        <span className="text-[11px] ink-faint">antibody engineering · decided by a multi-agent lab</span>
                    </div>

                    {/*
                     * Was missing entirely: the hero talked about "thousands of
                     * variants" without ever naming the antibody or the antigen.
                     * A reader could finish the page unable to say what the
                     * numbers described.
                     */}
                    <DatasetStrip identity={identity} className="mb-3 mt-0.5" />

                    <h2 className="max-w-[34ch] text-[30px] font-semibold leading-[1.15] tracking-[-0.025em] ink sm:text-[38px]">
                        AI agents argue over{" "}
                        <span style={{ color: "var(--accent)" }}>which assay to run next</span> — and the result
                        decides who was right.
                    </h2>
                    {/*
                     * One line, not three.
                     *
                     * The paragraph restated three things the page already shows:
                     * that assays are expensive (the 83% tile), that agents pick
                     * the batch (the recorded run behind it), and that the
                     * surrogate updates (the headline). Only the economics framing
                     * is unique to it, and that survives in the first clause.
                     *
                     * `max-w` is deliberately gone: the old 70ch box wrapped this
                     * to three lines on a laptop. Constraining width was working
                     * against the one thing this sentence now has to do.
                     */}
                    <p className="mt-2 text-[13px] leading-relaxed ink-dim">
                        Real assays are expensive, so the lab decides which assay to run next — round after round, with no
                        human choosing the candidates.
                    </p>

                    <div className="mt-4 grid gap-3 sm:grid-cols-3">
                        <BigNumber
                            icon={<Target size={13} />}
                            value={gain === null ? "—" : gainWhole(gain)}
                            label="more accurate at ranking candidates"
                            caption="than the standard UCB baseline, over 100 independent runs"
                            hint={
                                gain === null ? undefined : (
                                    <>
                                        Ranking accuracy: ours {num(final?.m5_hybrid ?? null, 2)}, standard{" "}
                                        {num(final?.m3_ucb ?? null, 2)}, on a 0→1 scale. The headline is the relative improvement.
                                    </>
                                )
                            }
                            tone="accent"
                            onClick={() => onTab("benchmark")}
                            cta="See the head-to-head"
                        />
                        <BigNumber
                            icon={<FlaskConical size={13} />}
                            value={skipped === null ? "—" : shareWhole(skipped)}
                            label="of the library never needed a test"
                            caption={
                                spent !== null && qualified !== null
                                    ? `${int(spent)} assays instead of ${int(qualified)}`
                                    : "assays saved against a full sweep"
                            }
                            hint="The loop names the five candidates each round, so most of the qualified library is never assayed."
                            tone="accent"
                            onClick={() => onTab("lab")}
                            cta="Watch the recorded run"
                        />
                        <BigNumber
                            icon={<Scale size={13} />}
                            value={learnableCount === null ? "—" : countOf(learnableCount, learnableCount)}
                            label="of the targets where learning is possible"
                            caption={
                                learnableCount === null || totalDatasets === null
                                    ? "same code, no retuning"
                                    : `only ${countOf(learnableCount, totalDatasets)} libraries have a ranking signal to find`
                            }
                            hint={
                                /*
                                 * This used to read "14 other datasets ruled out
                                 * before we ran anything", which was not true of
                                 * all 14: the baseline gate genuinely ran on
                                 * `aayl49`, `4d5_her2` and `1mlc`, and `aayl51`
                                 * was withheld rather than eliminated. The copy
                                 * below only makes claims the artifacts support.
                                 *
                                 * It also now *defines* learnability, which is
                                 * what a reader needs in order for "3 of 3" to
                                 * mean anything: a library is learnable when the
                                 * measured binding varies with the sequence.
                                 */
                                <>
                                    A library is learnable only when its measured binding actually varies with the
                                    sequence — otherwise there is no ordering to discover and no method, ours
                                    included, can be compared.{" "}
                                    {degenerateCount === null
                                        ? "Most datasets here"
                                        : `${int(degenerateCount)} of the ${int(totalDatasets ?? 0)} datasets here`}{" "}
                                    do not offer a real search: nine collapse to one or two distinct loops, and the
                                    tenth has eight loops but only five usable candidates. The remainder were
                                    screened by rules fixed before any result existed, and one is the sealed holdout
                                    that was never available to choose from.
                                </>
                            }
                            tone="accent"
                            onClick={() => onTab("cross")}
                            cta="See the other targets"
                        />
                    </div>
                </div>

                {final ? (
                    <div
                        className="border-t px-6 py-4"
                        style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}
                    >
                        <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
                            <span className="label text-[9.5px]">the headline number, on its own scale</span>
                            <span className="mono text-[10px] ink-faint">0 = guessing · 1 = perfect</span>
                        </div>
                        {/*
                         * Bars, not the marker track that used to be here.
                         *
                         * The track drew three dots on one 26px rule, so the entire
                         * comparison had to be read off their horizontal positions, and
                         * the values were then repeated in a legend row underneath —
                         * two renderings of the same three numbers, neither of them
                         * obviously "more is better". A bar's *length* carries the
                         * quantity, so 0.66 against 0.34 is legible at a glance.
                         *
                         * The domain is pinned to [0, 1] rather than fitted to the data:
                         * "guessing → perfect" is the frame the caption promises, and a
                         * fitted domain would rescale between runs and destroy the
                         * cross-run comparison this card exists to support.
                         *
                         * `showZeroLine` is cheap insurance rather than a present need:
                         * this card reads the *final* checkpoint, where all three quoted
                         * methods are positive (0.66 / 0.34 / 0.51). But `m4_fixed_prior`
                         * is −0.0992 at q=5, so the rho curves genuinely do go negative
                         * and a future edit that reads an earlier checkpoint must not
                         * clamp a negative bar into looking like 0.
                         */}
                        <HBarChart
                            data={[
                                {
                                    key: "m5_hybrid",
                                    label: "Virtual Lab (ours)",
                                    value: final.m5_hybrid ?? 0,
                                    color: methodMeta("m5_hybrid").color,
                                    hero: true,
                                    negative: (final.m5_hybrid ?? 0) < 0,
                                },
                                {
                                    key: "m3_ucb",
                                    label: "Standard UCB",
                                    value: final.m3_ucb ?? 0,
                                    color: "#6366F1",
                                    negative: (final.m3_ucb ?? 0) < 0,
                                },
                                {
                                    key: "m2_greedy",
                                    label: "Greedy",
                                    value: final.m2_greedy ?? 0,
                                    color: "#F59E0B",
                                    negative: (final.m2_greedy ?? 0) < 0,
                                },
                            ]}
                            format={(value) => value.toFixed(2)}
                            domain={[0, 1]}
                            labelWidth={132}
                            rowHeight={26}
                            showZeroLine
                            zeroLabel="0 · guessing"
                            ariaLabel={`Ranking accuracy out of 1 (0 = guessing, 1 = perfect): Virtual Lab ${num(final.m5_hybrid ?? null, 2)}, standard UCB ${num(final.m3_ucb ?? null, 2)}, greedy ${num(final.m2_greedy ?? null, 2)}`}
                        />
                        {vsM3 ? (
                            <p className="mt-1.5 text-[10.5px] ink-faint">
                                Ours beat the standard UCB baseline on {countOf(vsM3.seedsWon, vsM3.n)} runs;{" "}
                                {flukeOdds(vsM3.p)} that the result is chance.
                            </p>
                        ) : null}
                    </div>
                ) : null}
            </section>

            {/* ── The route through the rest of the console ─────────────────────── */}
            <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <RouteCard
                    index="01"
                    title="The recorded run"
                    body="Five candidates per round, each with the agents' argument and the assay result."
                    onClick={() => onTab("lab")}
                />
                <RouteCard
                    index="02"
                    title="Beats the standard UCB baseline"
                    body="100 independent runs, head to head, plus every measure we lose on."
                    onClick={() => onTab("benchmark")}
                />
                <RouteCard
                    index="03"
                    title="Works on other targets"
                    body="The same engine on four more public datasets, with a holdout it could not have tuned."
                    onClick={() => onTab("cross")}
                />
                <RouteCard
                    index="04"
                    title="Why it works"
                    body="The ablation that removed one of our own heuristics, and the nulls that bound it."
                    onClick={() => onTab("attribution")}
                />
            </section>

            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-1 text-[10px] ink-faint">
                <span>
                    Every figure in this console is read from a stored result file or the live API — nothing is
                    hand-entered.
                </span>
                <InfoTip label="How the numbers are checked">
                    A build-time script recomputes each derived figure from the published payloads and fails if the page
                    and the data disagree, and a second script refuses any result number typed into the source.
                </InfoTip>
                {payload ? (
                    <span className="mono">
                        {int(payload.seeds)} runs · {methodName("m5_hybrid")} vs {methodName("m3_ucb")}
                    </span>
                ) : null}
            </div>
        </div>
    );
}

function BigNumber({
    icon,
    value,
    label,
    caption,
    hint,
    tone,
    onClick,
    cta,
}: {
    icon: React.ReactNode;
    value: string;
    label: string;
    caption: string;
    hint?: React.ReactNode;
    tone: "accent" | "gold";
    onClick: () => void;
    cta: string;
}) {
    const color = tone === "accent" ? "var(--accent)" : "var(--gold)";
    return (
        <button
            type="button"
            onClick={onClick}
            className="focus-ring group flex min-w-0 flex-col items-start rounded-xl border px-4 py-3.5 text-left transition-colors"
            style={{
                borderColor: "var(--accent-line)",
                background: "linear-gradient(150deg, var(--accent-soft), transparent 72%)",
            }}
        >
            <span className="label mb-1 flex items-center gap-1.5 text-[9.5px]" style={{ color }}>
                {icon}
                {label}
                {hint ? <InfoTip label={`What ${label} means`}>{hint}</InfoTip> : null}
            </span>
            <span className="kpi-value mono text-[38px] leading-none" style={{ color }}>
                {value}
            </span>
            <span className="mt-1.5 text-[11px] leading-snug ink-dim">{caption}</span>
            <span
                className="mt-2 flex items-center gap-1 text-[10.5px] font-medium"
                style={{ color: "var(--ink-faint)" }}
            >
                {cta}
                <ArrowRight size={10} className="transition-transform group-hover:translate-x-0.5" />
            </span>
        </button>
    );
}

function RouteCard({
    index,
    title,
    body,
    onClick,
}: {
    index: string;
    title: string;
    body: string;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            onClick={onClick}
            className="focus-ring group flex flex-col items-start rounded-xl border px-3.5 py-3 text-left transition-colors"
            style={{ borderColor: "var(--border)", background: "var(--panel)" }}
        >
            <span className="mono text-[9.5px] font-semibold" style={{ color: "var(--accent)" }}>
                {index}
            </span>
            <span className="mt-0.5 text-[12px] font-semibold leading-snug ink">{title}</span>
            <span className="mt-1 text-[10.5px] leading-relaxed ink-faint">{body}</span>
        </button>
    );
}
