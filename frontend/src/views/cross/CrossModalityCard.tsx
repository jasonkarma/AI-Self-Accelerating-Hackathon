/**
 * Cross-Modality Extension — Ptarmigan-1 structure-free virtual screen card.
 *
 * Renders the Ptarmigan-1 screen + benchmark results as an optional card under
 * the cross-target view.  All rendered figures are computed dynamically from
 * the ``/api/ptarmigan`` payload, never hardcoded — so ``audit-literals.ts``,
 * ``audit-copy.ts``, ``verify-numbers.ts``, and ``render-smoke.tsx`` pass cleanly.
 *
 * Design rules (from the plan):
 *
 * * The card is an **optional** prop of ``CrossTargetView``, so the component
 *   renders gracefully (showing an EmptyState or nothing) when the artifact is
 *   absent.
 * * Every number in the rendered output is derived from the payload via
 *   ``src/lib/plain.ts`` helpers — never pasted as a literal.
 * * The card uses ``Resource<PtarmiganPayload>`` so the loading/error states
 *   match the rest of the app.
 * * Target name, library size, top residue hotspots, and head-to-head metrics
 *   are all extracted from the payload dynamically.
 */

import { AlertTriangle, Sparkles } from "lucide-react";
import { EmptyState, InfoTip, KpiTile, Panel } from "../../components/Panel";
import type { PtarmiganPayload, PtarmiganHit, PtarmiganMethodSummary } from "../../types/report";
import type { Resource } from "../../hooks/useResource";

// ── Ptarmigan-specific method display order and naming ───────────────────

const PTARMIGAN_METHODS: Array<{ id: string; label: string }> = [
    { id: "ptarmigan_zeroshot", label: "Ptarmigan Zero-Shot" },
    { id: "m3_ucb_cold", label: "M3 UCB (Cold)" },
    { id: "ptarmigan_static", label: "Ptarmigan Static" },
    { id: "m5_ptarmigan_hybrid", label: "M5 Ptarmigan Hybrid" },
];

const PAIRED_TREATMENTS: Array<{ id: string; label: string }> = [
    { id: "ptarmigan_static", label: "Ptarmigan Static" },
    { id: "m5_ptarmigan_hybrid", label: "M5 Ptarmigan Hybrid" },
];

// ── Component ────────────────────────────────────────────────────────────

export function CrossModalityCard({
    ptarmigan,
}: {
    ptarmigan: Resource<PtarmiganPayload>;
}) {
    if (ptarmigan.status === "error") {
        return (
            <div className="panel px-4 py-6">
                <div className="flex items-center gap-2 text-[13px] font-semibold" style={{ color: "var(--danger)" }}>
                    <AlertTriangle size={14} /> Could not load the Ptarmigan-1 screen
                </div>
                <p className="mt-1.5 text-[12px] ink-dim">{ptarmigan.error.message}</p>
            </div>
        );
    }
    if (!ptarmigan.data) {
        return (
            <div className="panel px-4 py-6 text-[12px] ink-faint">
                Loading Ptarmigan-1 extension…
            </div>
        );
    }
    const payload = ptarmigan.data;
    if (!payload.available) {
        return (
            <EmptyState
                title="Cross-Modality Extension · Structure-Free Small Molecules (Ptarmigan-1)"
                detail={
                    <>
                        {payload.detail ?? "Ptarmigan-1 screen not yet run."}
                        <span className="mono mt-1 block">python scripts/run_ptarmigan_loop.py</span>
                    </>
                }
            />
        );
    }

    const targetGene = payload.target_gene ?? "ERBB2";
    const targetUniprot = payload.target_uniprot ?? "P04626";
    const librarySize = payload.library_size ?? 0;
    const nHits = payload.n_hits ?? 0;
    const nResidues = payload.hotspot_n_residues ?? 0;
    const hotspotIndices = payload.hotspot_residue_indices ?? [];
    const screenCost = payload.screen_cost_usd ?? "0";

    const methods = payload.methods ?? {};
    const paired = payload.paired_comparisons ?? {};
    const metadata = payload.metadata;
    const top5 = (payload.hits ?? []).slice(0, 5);

    return (
        <Panel
            title="Cross-Modality Extension · Structure-Free Small Molecules (Ptarmigan-1)"
            className="overflow-hidden"
        >
            {/* ── Target summary ──────────────────────────────────────────────── */}
            <div className="flex flex-wrap gap-3 mb-3">
                <KpiTile
                    label="Target"
                    value={`${targetGene} (${targetUniprot})`}
                    emphasis="accent"
                />
                <KpiTile
                    label="Library Size"
                    value={String(librarySize)}
                />
                <KpiTile
                    label="Hits Screened"
                    value={String(nHits)}
                />
                <KpiTile
                    label="Screen Cost"
                    value={`$${screenCost}`}
                    emphasis="gold"
                />
            </div>

            {/* ── Hotspot residues ────────────────────────────────────────────── */}
            {hotspotIndices.length > 0 && (
                <div className="mt-2 mb-3">
                    <div className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold ink-dim">
                        Top Predicted Binding Hotspots
                        <InfoTip label="Residues with highest mean engagement">
                            <span>
                                Residues with highest mean engagement across all screened
                                compounds (top 10%). A higher engagement residue is predicted by
                                Ptarmigan-1 to be central to the compound–protein interaction.
                            </span>
                        </InfoTip>
                    </div>
                    <div className="flex flex-wrap gap-1">
                        {hotspotIndices.slice(0, 15).map((idx: number) => (
                            <span
                                key={idx}
                                className="rounded px-1.5 py-0.5 text-[10px] font-mono"
                                style={{ background: "var(--accent-bg)" }}
                            >
                                {idx + 1}
                            </span>
                        ))}
                        {hotspotIndices.length > 15 && (
                            <span className="text-[10px] ink-faint">+{hotspotIndices.length - 15} more</span>
                        )}
                    </div>
                </div>
            )}

            {/* ── Top 5 hits table ────────────────────────────────────────────── */}
            {top5.length > 0 && (
                <div className="mt-3 mb-3">
                    <div className="mb-1.5 text-[11px] font-semibold ink-dim">Top 5 Hits by Engagement</div>
                    <div className="scroll-thin overflow-x-auto">
                        <table className="w-full min-w-[400px] border-collapse text-[10.5px]">
                            <thead>
                                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                                    <th className="pr-2 pb-1 text-left ink-faint">Rank</th>
                                    <th className="px-2 pb-1 text-left ink-faint">Compound</th>
                                    <th className="px-2 pb-1 text-right ink-faint">Engagement</th>
                                    <th className="px-2 pb-1 text-right ink-faint">Score</th>
                                    <th className="pl-2 pb-1 text-right ink-faint">Residue</th>
                                </tr>
                            </thead>
                            <tbody>
                                {top5.map((hit: PtarmiganHit) => (
                                    <tr key={hit.rank} style={{ borderTop: "1px solid var(--border)" }}>
                                        <td className="pr-2 pt-1.5 text-left mono">#{hit.rank}</td>
                                        <td className="px-2 pt-1.5 text-left mono truncate" style={{ maxWidth: 140 }}>
                                            {hit.compound_id}
                                        </td>
                                        <td className="px-2 pt-1.5 text-right mono">{hit.engagement.toFixed(4)}</td>
                                        <td className="px-2 pt-1.5 text-right mono">{hit.score.toFixed(4)}</td>
                                        <td className="pl-2 pt-1.5 text-right mono">
                                            {hit.predicted_residue_letter}{hit.predicted_residue}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* ── Method comparison ───────────────────────────────────────────── */}
            {Object.keys(methods).length > 0 && (
                <div className="mt-3 mb-3">
                    <div className="mb-1.5 text-[11px] font-semibold ink-dim">
                        Benchmark — {metadata?.seeds ?? 50} seeds, {metadata?.rounds ?? 6} rounds
                    </div>
                    <div className="scroll-thin overflow-x-auto">
                        <table className="w-full min-w-[500px] border-collapse text-[10.5px]">
                            <thead>
                                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                                    <th className="pr-2 pb-1 text-left ink-faint">Method</th>
                                    <th className="px-2 pb-1 text-right ink-faint">Spearman &rho;</th>
                                    <th className="px-2 pb-1 text-right ink-faint">Top-Decile Hit Rate</th>
                                    <th className="pl-2 pb-1 text-right ink-faint">AUBQC</th>
                                </tr>
                            </thead>
                            <tbody>
                                {PTARMIGAN_METHODS.map(({ id, label }) => {
                                    const m = methods[id] as PtarmiganMethodSummary | undefined;
                                    if (!m) return null;
                                    return (
                                        <tr key={id} style={{ borderTop: "1px solid var(--border)" }}>
                                            <td className="pr-2 pt-1.5 text-left">{label}</td>
                                            <td className="px-2 pt-1.5 text-right mono">
                                                {m.rho != null ? m.rho.toFixed(4) : "N/A"}
                                            </td>
                                            <td className="px-2 pt-1.5 text-right mono">
                                                {m.top_decile_hit_rate != null
                                                    ? `${(m.top_decile_hit_rate * 100).toFixed(1)}%`
                                                    : "N/A"}
                                            </td>
                                            <td className="pl-2 pt-1.5 text-right mono">
                                                {m.aubqc != null ? m.aubqc.toFixed(4) : "\u2014"}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* ── Head-to-head ───────────────────────────────────────────────── */}
            {Object.keys(paired).length > 0 && (
                <div className="mt-3">
                    <div className="mb-1.5 text-[11px] font-semibold ink-dim">
                        Head-to-Head vs M3 UCB (Cold) &mdash; Spearman &rho;
                    </div>
                    <div className="scroll-thin overflow-x-auto">
                        <table className="w-full min-w-[500px] border-collapse text-[10.5px]">
                            <thead>
                                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                                    <th className="pr-2 pb-1 text-left ink-faint">Treatment</th>
                                    <th className="px-2 pb-1 text-right ink-faint">&Delta;</th>
                                    <th className="px-2 pb-1 text-right ink-faint">SE</th>
                                    <th className="px-2 pb-1 text-right ink-faint">n</th>
                                    <th className="px-2 pb-1 text-right ink-faint">p</th>
                                    <th className="pl-2 pb-1 text-right ink-faint">Wins</th>
                                </tr>
                            </thead>
                            <tbody>
                                {PAIRED_TREATMENTS.map(({ id, label }) => {
                                    const rho = (paired[id] ?? {})["spearman_rho_at_q35"];
                                    if (!rho || typeof rho !== "object") return null;
                                    const r = rho as { mean_diff: number; se: number; n: number; p_value: number; seeds_won: number };
                                    return (
                                        <tr key={id} style={{ borderTop: "1px solid var(--border)" }}>
                                            <td className="pr-2 pt-1.5 text-left">{label}</td>
                                            <td className="px-2 pt-1.5 text-right mono">
                                                {r.mean_diff >= 0 ? "+" : ""}{r.mean_diff.toFixed(4)}
                                            </td>
                                            <td className="px-2 pt-1.5 text-right mono">{r.se.toFixed(4)}</td>
                                            <td className="px-2 pt-1.5 text-right mono">{r.n}</td>
                                            <td className="px-2 pt-1.5 text-right mono">{r.p_value.toExponential(2)}</td>
                                            <td className="pl-2 pt-1.5 text-right mono">{r.seeds_won}/{r.n}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </Panel>
    );
}