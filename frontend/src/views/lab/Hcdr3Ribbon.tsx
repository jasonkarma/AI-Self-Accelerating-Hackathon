import type { RibbonEntry, RibbonStatus, StatePayload } from "../../types/state";
import { InfoTip } from "../../components/Panel";
import { num } from "../../lib/format";

/**
 * The HCDR3 loop, position by position.
 *
 * Every assay in this campaign is a mutation of a 13-residue loop, but the
 * interface showed the loop only as aligned text inside each plate row. That
 * made the *structure* of the problem invisible: which positions are load-bearing
 * anchors, which tolerate swaps, and which one behaves non-additively.
 *
 * The payload already carries that judgement in `hcdr3_ribbon` — four evidentiary
 * states and a signed prior weight per position — and nothing rendered it. This
 * component is the whole of that data, drawn:
 *
 * * **anchor** — mutating it tends to break binding; the prior protects it.
 * * **low-risk swap** — a substitution here is usually tolerated.
 * * **epistasis alert** — the position does not behave additively, so its effect
 *   depends on what else is mutated. This is the one a judge asks about.
 * * **neutral** — no evidence either way yet.
 *
 * The ribbon is sparse by design: the backend curates entries, and a position it
 * does not list is one the ledger has no opinion on. Rendering those as "no
 * evidence" rather than hiding them keeps the loop's full length visible, so the
 * reader can see this is a 13-position problem and not a 7-position one.
 */

const STATUS_META: Record<
    RibbonStatus,
    { label: string; short: string; color: string; meaning: string }
> = {
    anchor_conserved: {
        label: "Anchor · keep",
        short: "anchor",
        color: "#F0B429",
        meaning:
            "Structurally load-bearing. Mutating this position tends to break binding, so the search is biased against it — the anchor shows up in almost every strong binder.",
    },
    low_risk_swap: {
        label: "Safe to vary",
        short: "low-risk",
        color: "#22D3EE",
        meaning:
            "A substitution here is usually tolerated, which makes it a cheap place to explore. The signed weight is how much the position shifted the score in the campaign so far.",
    },
    epistasis_alert: {
        label: "Non-additive",
        short: "epistasis",
        color: "#F43F5E",
        meaning:
            "The effect of mutating this position depends on what else is mutated — it is not additive. A purely position-by-position model misprices it, which is why the epistasis-clash tool exists and why this position is flagged rather than trusted.",
    },
    neutral: {
        label: "No evidence",
        short: "neutral",
        color: "#64748B",
        meaning:
            "The record has no repeated observation here, so the loop has no opinion. Absence of evidence is not evidence of no effect.",
    },
};

const STATUS_ORDER: RibbonStatus[] = [
    "anchor_conserved",
    "low_risk_swap",
    "epistasis_alert",
    "neutral",
];

export function Hcdr3Ribbon({ state, wildType }: { state: StatePayload; wildType: string }) {
    const entries = state.hcdr3_ribbon ?? [];
    const byPosition = new Map(entries.map((entry) => [entry.position, entry]));
    const length = Math.max(wildType.length, ...entries.map((entry) => entry.position), 0);
    if (!length) return null;

    const counts = STATUS_ORDER.map((status) => ({
        status,
        count: entries.filter((entry) => entry.status === status).length,
    })).filter((entry) => entry.count > 0);

    const alerts = entries.filter((entry) => entry.status === "epistasis_alert");

    return (
        <section className="panel px-4 py-3">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-1.5">
                    <span className="label text-[9.5px]">The loop being optimised · {length} positions</span>
                    <InfoTip label="What this ribbon is">
                        One cell per residue of the starting antibody's HCDR3 loop. The colour is the evidence the
                        campaign has accumulated about that position: load-bearing anchors are protected, tolerant
                        positions are explored, and non-additive positions are flagged. This is why the lab does not treat
                        all {length} positions as interchangeable.
                    </InfoTip>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                    {counts.map(({ status, count }) => (
                        <span key={status} className="flex items-center gap-1 text-[9.5px]">
                            <span
                                aria-hidden
                                className="inline-block h-2 w-2 rounded-sm"
                                style={{ background: STATUS_META[status].color }}
                            />
                            <span className="ink-faint">{STATUS_META[status].short}</span>
                            <span className="mono ink-dim">{count}</span>
                        </span>
                    ))}
                </div>
            </div>

            <div className="flex flex-wrap gap-1">
                {Array.from({ length }, (_, index) => {
                    const position = index + 1;
                    const entry: RibbonEntry | undefined = byPosition.get(position);
                    const status: RibbonStatus = entry?.status ?? "neutral";
                    const meta = STATUS_META[status];
                    const residue = entry?.wt_residue ?? wildType[index] ?? "·";
                    const weight = entry?.soft_prior_weight ?? null;
                    const known = entry !== undefined;

                    return (
                        <div
                            key={position}
                            className="flex min-w-[42px] flex-col items-center rounded-lg border px-1.5 py-1"
                            style={{
                                borderColor: known ? `${meta.color}66` : "var(--border)",
                                background: known ? `${meta.color}14` : "transparent",
                                opacity: known ? 1 : 0.55,
                            }}
                            title={`${position}${residue} — ${known ? meta.label : "no recorded evidence"}${weight !== null && weight !== 0 ? ` · prior weight ${weight > 0 ? "+" : ""}${weight.toFixed(2)}` : ""
                                }`}
                        >
                            <span className="mono text-[8.5px] leading-none ink-faint">{position}</span>
                            <span
                                className="mono text-[13px] font-semibold leading-tight"
                                style={{ color: known ? meta.color : "var(--ink-dim)" }}
                            >
                                {residue}
                            </span>
                            <span className="mono text-[8px] leading-none ink-faint">
                                {weight !== null && Math.abs(weight) >= 0.005
                                    ? `${weight > 0 ? "+" : "−"}${num(Math.abs(weight), 2)}`
                                    : "—"}
                            </span>
                        </div>
                    );
                })}
            </div>

            {alerts.length ? (
                <div className="mt-2 flex flex-wrap items-start gap-1.5">
                    <span className="text-[10.5px] leading-relaxed" style={{ color: STATUS_META.epistasis_alert.color }}>
                        {alerts.map((entry) => `${entry.position}${entry.wt_residue}`).join(", ")} {alerts.length === 1 ? "is" : "are"} non-additive:
                    </span>
                    <span className="text-[10.5px] leading-relaxed ink-faint">
                        the effect of changing {alerts.length === 1 ? "it" : "them"} depends on which other positions are
                        mutated at the same time. A position-by-position model gets this wrong by construction, which is
                        why the loop runs a clash check instead of trusting the sum of single mutations.
                    </span>
                    <InfoTip label="Why non-additivity matters">
                        {STATUS_META.epistasis_alert.meaning} In this campaign the check flagged{" "}
                        <span className="mono">
                            {alerts.map((entry) => `P${entry.position}`).join(", ")}
                        </span>{" "}
                        against the additive expectation, which is the kind of residue pair where two separately-good
                        mutations combine badly.
                    </InfoTip>
                </div>
            ) : null}

            <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 border-t pt-1.5 text-[9.5px] ink-faint" style={{ borderColor: "var(--border)" }}>
                <span>the number under each residue is that position's signed prior weight</span>
                <InfoTip label="How the weight is computed">
                    The shrunk average effect of mutating that position, estimated from the campaign's own revealed
                    candidates and shrunk toward zero when there are few observations. A dim cell is a position the loop
                    has not seen enough of to form an opinion about.
                </InfoTip>
            </div>
        </section>
    );
}

/** The four states, exported so a legend elsewhere cannot drift from this table. */
export const RIBBON_LEGEND = STATUS_ORDER.map((status) => ({
    status,
    ...STATUS_META[status],
}));
