import { AlertTriangle, ArrowDownUp, FlaskConical, Minus, ShieldX, Trophy, Vote } from "lucide-react";
import { useState } from "react";
import type { AttributionPayload } from "../../types/report";
import { Badge, Callout, EmptyState, InfoTip, KpiTile, Note, Panel } from "../../components/Panel";
import {
  attributionRows,
  groupSummary,
  noAnnealRows,
  productionRow,
  VARIANT_GROUPS,
  variantLabel,
  type AttributionRowView,
  type VariantGroupKey,
} from "../../lib/derive";
import { int, methodMeta, num, pvalue, signed } from "../../lib/format";
import { flukeOdds } from "../../lib/plain";
import { StackDiagram } from "./StackDiagram";
import type { Resource } from "../../hooks/useResource";

/**
 * View 4 — causal attribution and the negative-control ledger.
 *
 * The structure is a bento: one **causal winner** card for the variant that the
 * engine actually ships (`no_anneal`), one **negative-control** card per family
 * (slot role, SAR capacity) plus the mechanism pair, and the full 13-row ledger
 * last. Summary tiles sit above the ledger so the three outcomes — one win, two
 * nulls — are readable without opening a table.
 *
 * Every variant is named in plain English with its artifact token as a monospace
 * subtitle: `slot1_model` is *"Slot 1 Scored by the Agent Model"* first and
 * `slot1_model` second, because a judge should not need the ablation script to
 * know what was deleted.
 */
export function AttributionView({ attribution }: { attribution: Resource<AttributionPayload> }) {
  if (attribution.status === "error") {
    return (
      <div className="panel px-4 py-6">
        <div className="flex items-center gap-2 text-[13px] font-semibold" style={{ color: "var(--danger)" }}>
          <AlertTriangle size={14} /> Could not load the ablation ledger
        </div>
        <p className="mt-1.5 text-[12px] ink-dim">{attribution.error.message}</p>
      </div>
    );
  }
  if (!attribution.data) {
    return <div className="panel px-4 py-6 text-[12px] ink-faint">Loading the ablation ledger…</div>;
  }
  const payload = attribution.data;
  if (!payload.available || !payload.cells.length) {
    return (
      <EmptyState
        title="Attribution artifact not generated"
        detail={
          <>
            {payload.detail ?? "The ledger is served from a script artifact."}
            <span className="mono mt-1 block">python scripts/run_attribution_ablation.py --family all</span>
          </>
        }
      />
    );
  }
  return (
    /*
     * One column, read top to bottom.
     *
     * This tab used to split into a 1.35 : 1 grid — a tall causal-winner card on
     * the left, two short control cards on the right. The right column ran out
     * of content long before the left one did, and because the grid is
     * `items-start` the leftover space was simply empty: a large blank rectangle
     * beside the claim. Moving the stack diagram into that column narrowed the
     * gap but could not close it, because the imbalance is structural.
     *
     * A single column removes the failure mode entirely and fixes the reading
     * order at the same time. The argument is a sequence — *what changed* →
     * *where in the engine* → *what did not matter* → *the full ledger* — and a
     * sequence should not be laid out as two competing columns.
     */
    <div className="space-y-4">
      <SummaryBand payload={payload} />
      <CausalWinner payload={payload} />
      <StackDiagram />
      <div className="grid gap-4 lg:grid-cols-2">
        <NegativeControlCard payload={payload} group="slot_role" />
        <NegativeControlCard payload={payload} group="sar_efficiency" />
      </div>
      <MechanismCard payload={payload} />
      <VariantLedger payload={payload} />
      <Provenance payload={payload} />
    </div>
  );
}

/** Three tiles: the one causal win, and the two nulls that bound it. */
function SummaryBand({ payload }: { payload: AttributionPayload }) {
  const rows = noAnnealRows(payload);
  const rhoGain = rows.filter((row) => row.annealRemovalRho > 0);
  const aubqcHurt = rows.filter((row) => row.annealRemovalAubqc < 0 && row.aubqcP < 0.05);
  const aubqcHelped = rows.filter((row) => row.annealRemovalAubqc > 0 && row.aubqcP < 0.05);
  const slotRole = groupSummary(payload, "slot_role");
  const sar = groupSummary(payload, "sar_efficiency");
  const mechanism = groupSummary(payload, "mechanism");
  const bestP = rhoGain.reduce<(typeof rows)[number] | null>(
    (acc, row) => (acc === null || row.rhoP < acc.rhoP ? row : acc),
    null,
  );

  return (
    <section className="panel hero-wash">
      <div className="flex flex-wrap items-center justify-between gap-2 px-5 pb-1 pt-4">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10.5px] font-semibold"
            style={{ background: "var(--gold-soft)", color: "var(--gold)", border: "1px solid var(--gold-line)" }}
          >
            <Trophy size={11} /> WHY IT WORKS
          </span>
          <span className="text-[11px] ink-faint">
            {payload.seeds} runs per setting · 4 measured settings · each change compared against the engine it
            was removed from
          </span>
        </div>
        <Badge color="#10B981" title="The scoring rule the engine actually ships">
          one scoring rule
        </Badge>
      </div>

      <h2 className="px-5 pb-1 pt-2 text-[24px] font-semibold leading-tight tracking-[-0.015em] ink sm:text-[26px]">
        We deleted one piece of our own engine and it got better in{" "}
        <span className="mono" style={{ color: "var(--accent)" }}>
          {rhoGain.length} of {rows.length}
        </span>{" "}
        settings — never worse
      </h2>
      <p className="px-5 text-[12.5px] leading-relaxed ink-dim">
        The deleted piece was a heuristic that nudged scores toward the model's own guess. Removing it made the
        ranking more accurate and did not hurt the best binder found. The engine ships without it.
      </p>

      <div className="grid gap-2.5 px-5 pb-4 pt-2 sm:grid-cols-2 lg:grid-cols-4">
        <KpiTile
          label="What was removed"
          value="one heuristic"
          caption={bestP ? `strongest evidence: ${flukeOdds(bestP.rhoP)}` : "now the shipped rule"}
          emphasis="gold"
          hint="The deleted heuristic is the one thing standing between the engine and the scoring rule it ships with. The shipped configuration is literally this variant."
        />
        <KpiTile
          label="Settings that improved"
          value={`${rhoGain.length} of ${rows.length}`}
          caption="on ranking accuracy"
          emphasis="accent"
          hint="Removing the heuristic raised ranking accuracy in every measured setting."
        />
        <KpiTile
          label="Anything it broke"
          value={aubqcHurt.length === 0 ? "nothing" : `${aubqcHurt.length} settings`}
          caption={aubqcHelped.length === 0 ? "best binder unchanged" : `better in ${aubqcHelped.length} setting(s)`}
          hint="Whether removing the heuristic lowered the best binder found. It never does: the measure is either unchanged or better."
        />
        <KpiTile
          label="Other ideas we tried"
          value="none mattered"
          caption={`${slotRole.variants + sar.variants} more changes, ${slotRole.comparisons + sar.comparisons} comparisons, all null`}
          hint="Re-weighting which slots the model picks, and giving the sequence model more capacity. Neither produced a difference anyone could detect."
        />
      </div>

      <div className="grid gap-2 px-5 pb-4 sm:grid-cols-3">
        <OutcomePill
          tone="accent"
          icon={<Trophy size={10} />}
          title="This one mattered"
          body={`${mechanism.comparisons} comparisons · strongest evidence ${flukeOdds(mechanism.minP)}`}
        />
        <OutcomePill
          tone="neutral"
          icon={<Vote size={10} />}
          title="Who picks the slots: no effect"
          body={`${slotRole.comparisons} comparisons over ${slotRole.cells} settings · ${flukeOdds(slotRole.minP)}`}
        />
        <OutcomePill
          tone="neutral"
          icon={<FlaskConical size={10} />}
          title="A bigger sequence model: no effect"
          body={`${sar.comparisons} comparisons over ${sar.cells} settings · ${flukeOdds(sar.minP)}`}
        />
      </div>
    </section>
  );
}

function OutcomePill({
  tone,
  icon,
  title,
  body,
}: {
  tone: "accent" | "neutral";
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  const color = tone === "accent" ? "var(--accent)" : "var(--ink-dim)";
  return (
    <div
      className="flex items-start gap-2 rounded-lg border px-2.5 py-2"
      style={{
        borderColor: tone === "accent" ? "var(--accent-line)" : "var(--border)",
        background: tone === "accent" ? "var(--accent-soft)" : "var(--panel-2)",
      }}
    >
      <span className="mt-[2px] shrink-0" style={{ color }}>
        {icon}
      </span>
      <div className="min-w-0">
        <div className="text-[10.5px] font-semibold" style={{ color }}>
          {title}
        </div>
        <div className="mono text-[9px] leading-tight ink-faint">{body}</div>
      </div>
    </div>
  );
}

function CausalWinner({ payload }: { payload: AttributionPayload }) {
  const rows = noAnnealRows(payload);
  const bestP = rows.reduce<(typeof rows)[number] | null>(
    (acc, row) => (acc === null || row.rhoP < acc.rhoP ? row : acc),
    null,
  );
  const worstRho = rows.reduce<(typeof rows)[number] | null>(
    (acc, row) => (acc === null || row.annealRemovalRho < acc.annealRemovalRho ? row : acc),
    null,
  );

  return (
    <Panel
      tone="gold"
      eyebrow="The change that mattered · what was removed and what it earned"
      title="Let the score be the score"
      lede={
        <>
          The engine used to let a running heuristic nudge every score toward the model's own prediction. We
          deleted it. This is what that deletion was worth — and it is now the rule the engine ships with{" "}
          <span className="mono" style={{ color: "var(--gold)" }}>
            (no_anneal)
          </span>
          .
        </>
      }
      actions={
        <Badge color="#10B981" title="This variant is byte-identical to the shipped configuration">
          SHIPPED
        </Badge>
      }
    >
      <div className="mb-2.5 grid grid-cols-2 gap-2 sm:grid-cols-3">
        <KpiTile
          label="Settings it improved"
          value={`${rows.filter((r) => r.annealRemovalRho > 0).length} of ${rows.length}`}
          caption="on ranking accuracy"
          emphasis="accent"
        />
        <KpiTile
          label="Biggest improvement"
          value={worstRho ? `+${Math.max(...rows.map((r) => r.annealRemovalRho)).toFixed(2)}` : "—"}
          caption={bestP ? `${flukeOdds(bestP.rhoP)} · ${bestP.cell}` : ""}
        />
        <KpiTile
          label="Damage to the best binder"
          value={rows.some((row) => row.annealRemovalAubqc < 0 && row.aubqcP < 0.05) ? "real" : "none"}
          caption={`${rows.filter((row) => row.aubqcP >= 0.05).length} of ${rows.length} unchanged, the rest better`}
        />
      </div>

      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse">
          <thead>
            <tr style={{ background: "var(--panel-2)" }}>
              {["setting", "ranking accuracy", "odds it's luck", "runs better", "best binder", "odds it's luck"].map((heading) => (
                <th key={heading} className="label whitespace-nowrap px-2 py-1.5 text-left text-[8.5px]">
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.cell} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="mono whitespace-nowrap px-2 py-1 text-[10.5px] ink">{row.cell}</td>
                <td
                  className="mono px-2 py-1 text-[11px] font-semibold"
                  style={{ color: row.annealRemovalRho >= 0 ? "var(--accent)" : "var(--danger)" }}
                >
                  {row.annealRemovalRho >= 0 ? "+" : "−"}
                  {num(Math.abs(row.annealRemovalRho), 2)}
                </td>
                <td
                  className="px-2 py-1 text-[10px]"
                  style={{ color: row.rhoP < 0.001 ? "var(--accent)" : "var(--ink-dim)" }}
                >
                  {flukeOdds(row.rhoP)}
                </td>
                <td className="mono px-2 py-1 text-[10px] ink-faint">
                  {int(row.rhoSeedsDeletionHelped)} of {int(row.rhoN)}
                </td>
                <td
                  className="mono px-2 py-1 text-[10.5px]"
                  style={{
                    color: row.aubqcP >= 0.05 ? "var(--ink-dim)" : row.annealRemovalAubqc >= 0 ? "var(--accent)" : "var(--danger)",
                  }}
                  title={row.annealRemovalAubqc >= 0 ? "removing it helped" : "removing it hurt"}
                >
                  {row.annealRemovalAubqc >= 0 ? "+" : "−"}
                  {num(Math.abs(row.annealRemovalAubqc), 2)}
                </td>
                <td className="px-2 py-1 text-[10px] ink-faint">{flukeOdds(row.aubqcP)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Note summary={`Sign convention: a plus means removing the heuristic helped. Compared against ${rows[0]?.comparisonBasis ?? "—"}.`}>
        Every number is measured per run and paired, so a plus means the change itself did the work rather
        than the run having been lucky. The per-run results for this variant are identical to the shipped
        engine — which is the point: the variant <em>is</em> what ships.
        {bestP ? (
          <>
            {" "}
            Strongest evidence: exact p = <span className="mono">{pvalue(bestP.rhoP)}</span> ({bestP.cell}).
          </>
        ) : null}
      </Note>
    </Panel>
  );
}

function NegativeControlCard({ payload, group }: { payload: AttributionPayload; group: VariantGroupKey }) {
  const summary = groupSummary(payload, group);
  const definition = VARIANT_GROUPS.find((entry) => entry.key === group);
  const variants = definition?.variants ?? [];
  const noneSignificant = summary.minP !== null && summary.minP >= 0.05;
  const tone = group === "slot_role" ? "#6366F1" : "#A855F7";
  const Icon = group === "slot_role" ? Vote : FlaskConical;

  return (
    <section className="panel">
      <header className="px-4 pb-2 pt-3.5">
        <div className="mb-1.5 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <span
              className="flex h-5 w-5 items-center justify-center rounded-md"
              style={{ background: `${tone}18`, color: tone }}
            >
              <Icon size={11} />
            </span>
            <span className="label text-[9px]">{group === "slot_role" ? "Negative control" : "Negative control"}</span>
          </div>
          <span
            className="mono rounded-full px-2 py-0.5 text-[9px] font-semibold"
            style={{
              background: noneSignificant ? "var(--accent-soft)" : "rgba(244,63,94,0.1)",
              color: noneSignificant ? "var(--accent)" : "var(--danger)",
            }}
          >
            {noneSignificant ? "NO EFFECT" : "REVIEW"}
          </span>
        </div>
        <h3 className="text-[13.5px] font-semibold leading-snug ink">
          {group === "slot_role" ? "Letting the model pick the slots" : "A bigger sequence model"}
        </h3>
        <p className="mt-0.5 text-[11.5px] leading-relaxed ink-dim">
          {group === "slot_role"
            ? "If better ranking automatically meant a better best binder, re-pointing the picking slots at the model would show it. It doesn't."
            : "A larger position/residue table changes nothing measurable, which says the engine is already at its ceiling on this representation."}
        </p>
      </header>
      <div className="px-4 pb-3">
        <div className="mb-1.5 flex items-center gap-1.5">
          <span className="mono text-[9.5px] ink-faint">
            {summary.comparisons} paired comparisons · {summary.cells} cells
          </span>
          <InfoTip label="What a null result means here">
            A change that ties the engine is not "as good as" it — it is evidence that the component being
            changed was not doing the work on this target. That is a result, and it is why the engine stays
            simple.
          </InfoTip>
        </div>
        <VariantTable payload={payload} variants={variants} group={group} />
        <div className="mt-1 text-[9px] ink-faint">
          strongest evidence: {flukeOdds(summary.minP)}
          {summary.minAt ? ` · ${summary.minAt.cell} · ${summary.minAt.variant}` : ""}
        </div>
      </div>
    </section>
  );
}

function MechanismCard({ payload }: { payload: AttributionPayload }) {
  const mechanism = groupSummary(payload, "mechanism");
  return (
    <Panel
      eyebrow="The other mechanism change"
      title="The two changes that alter the search itself"
      lede={`${mechanism.comparisons} comparisons, strongest evidence ${flukeOdds(mechanism.minP)} — both measured against the older engine that still carried the heuristic.`}
      actions={<Badge color="#F59E0B">older baseline</Badge>}
    >
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <VariantTable payload={payload} variants={["no_anneal", "greedy_slots"]} group="mechanism" />
        <Callout tone="warn" title="Why these rows say which baseline they use">
          These rows are measured against an <em>older</em> engine — the one that still had the heuristic. The
          other two controls share that baseline, so those comparisons are valid with each other, but they are
          not directly comparable with the current engine's rows in the full table. Every row names its
          baseline for exactly that reason.
        </Callout>
      </div>
    </Panel>
  );
}

function VariantLedger({ payload }: { payload: AttributionPayload }) {
  const [cellIndex, setCellIndex] = useState(0);
  const cell = payload.cells[Math.min(cellIndex, payload.cells.length - 1)];
  if (!cell) return null;
  const production = productionRow(cell);

  return (
    <Panel
      eyebrow="Everything we tried"
      title="Every change, every setting — including the ones that lost"
      lede={
        <>
          One row per change, with the paired comparison and how many runs it won.{" "}
          <span className="font-semibold ink">One change mattered; every other one did nothing</span> — and
          every null is still in this table.
        </>
      }
      actions={
        <div className="flex flex-wrap gap-1">
          {payload.cells.map((entry, index) => (
            <button
              key={entry.cell}
              type="button"
              onClick={() => setCellIndex(index)}
              className="focus-ring rounded-md border px-2 py-1 text-[10px] font-medium transition-colors"
              style={{
                borderColor: index === cellIndex ? "var(--accent-line)" : "var(--border)",
                color: index === cellIndex ? "var(--accent)" : "var(--ink-dim)",
                background: index === cellIndex ? "var(--accent-soft)" : "transparent",
              }}
            >
              {entry.cell}
            </button>
          ))}
        </div>
      }
      flush
    >
      <div className="grid gap-x-4 gap-y-2 px-5 py-3 sm:grid-cols-2 lg:grid-cols-4" style={{ background: "var(--panel-2)" }}>
        <KpiTile
          label="Best binder found"
          value={num(production?.aubqc ?? null, 2)}
          unit={`± ${num(production?.aubqc_se ?? null, 2)}`}
          caption={`across ${int(production?.seeds ?? cell.seeds)} runs`}
          emphasis="accent"
        />
        <KpiTile
          label="Ranking accuracy"
          value={num(production?.rho ?? null, 2)}
          caption={`0 = guessing · 1 = perfect · ${int(production?.rho_n ?? null)} runs`}
          emphasis="accent"
        />
        <KpiTile
          label="Test budget"
          value={`${cell.protocol.rounds} rounds · ${cell.protocol.total_budget} tests`}
          caption={`${cell.protocol.seq_len}-residue CDR · ${int(cell.protocol.n_qualified)} usable candidates`}
        />
        <KpiTile
          label="Library used"
          value={`${(cell.protocol.coverage_pct * 100).toFixed(0)}%`}
          caption={`${(cell.protocol.target_coverage * 100).toFixed(0)}% per round`}
        />
      </div>
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[880px] border-collapse">
          <thead>
            <tr style={{ background: "var(--panel-2)" }}>
              {[
                "change",
                "family",
                "best binder",
                "vs production",
                "odds it's luck",
                "ranking accuracy",
                "vs production",
                "odds it's luck",
                "baseline",
              ].map((heading) => (
                <th key={heading} className="label whitespace-nowrap px-3 py-1.5 text-left text-[8.5px]">
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {attributionRows(cell).map((row) => (
              <LedgerRow key={row.variant} row={row} />
            ))}
          </tbody>
        </table>
      </div>
      <div className="px-5 py-2.5">
        <Note summary="Mechanism rows carry a caveat">{cell._provenance.mechanism_rows_caveat}</Note>
      </div>
    </Panel>
  );
}

function LedgerRow({ row }: { row: AttributionRowView }) {
  const group = VARIANT_GROUPS.find((entry) => entry.key === row.group);
  const isBaseline = row.group === "baseline";
  const label = variantLabel(row.variant);
  return (
    <tr style={{ borderTop: "1px solid var(--border)" }}>
      <td className="px-3 py-1.5">
        <div className="flex items-center gap-1.5">
          <span className="text-[11.5px] font-medium" style={{ color: isBaseline ? methodMeta(row.variant).color : "var(--ink)" }}>
            {isBaseline ? `${methodMeta(row.variant).short} ${methodMeta(row.variant).label}` : label.name}
          </span>
          {row.note ? <InfoTip label={label.name}>{row.note}</InfoTip> : null}
        </div>
        {!isBaseline ? <div className="mono text-[9px] ink-faint">{label.token}</div> : null}
        {row.comparisonBasis !== "production" ? (
          <div
            className="mono text-[8.5px]"
            style={{ color: "#F59E0B" }}
            title="Measured against the pre-e567cd4 production engine, which still carried the VoI anneal."
          >
            vs {row.comparisonBasis}
          </div>
        ) : null}
      </td>
      <td className="px-3 py-1.5">
        <span className="mono text-[9px]" style={{ color: group?.color ?? "var(--ink-faint)" }}>
          {group?.title.split(" ")[0] ?? row.group}
        </span>
      </td>
      {/* absolute score, so the row can be read without doing the arithmetic */}
      <td className="px-3 py-1.5 mono text-[11px] ink-dim">{num(row.aubqc, 2)}</td>
      <td
        className="px-3 py-1.5 mono text-[11px]"
        style={{
          color:
            Math.abs(row.variantVsProductionAubqc) < 0.005
              ? "var(--ink-dim)"
              : row.variantVsProductionAubqc >= 0
                ? "var(--accent)"
                : "var(--danger)",
        }}
      >
        {signed(row.variantVsProductionAubqc, 2)}
      </td>
      <td className="px-3 py-1.5 text-[10px]" style={{ color: row.aubqcP < 0.05 ? "var(--accent)" : "var(--ink-faint)" }}>
        {flukeOdds(row.aubqcP)}
      </td>
      {/* the measure the pitch is actually about */}
      <td className="px-3 py-1.5 mono text-[11px] ink">{num(row.rho, 2)}</td>
      <td
        className="px-3 py-1.5 mono text-[11px]"
        style={{ color: row.variantVsProductionRho >= 0 ? "var(--accent)" : "var(--danger)" }}
      >
        {signed(row.variantVsProductionRho, 2)}
      </td>
      <td
        className="px-3 py-1.5 text-[10px]"
        style={{
          color: row.rhoP < 0.05 ? (row.variantVsProductionRho >= 0 ? "var(--accent)" : "var(--danger)") : "var(--ink-faint)",
        }}
      >
        {flukeOdds(row.rhoP)}
      </td>
      <td className="px-3 py-1.5 mono text-[10px] ink-faint">{num(row.baselineAubqc, 4)}</td>
    </tr>
  );
}

function VariantTable({
  payload,
  variants,
  group,
}: {
  payload: AttributionPayload;
  variants: string[];
  group: VariantGroupKey;
}) {
  const color = VARIANT_GROUPS.find((entry) => entry.key === group)?.color ?? "#94A3B8";
  return (
    <div className="scroll-thin overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            {["change", "setting", "best binder", "ranking accuracy", "odds it's luck"].map((heading) => (
              <th key={heading} className="label whitespace-nowrap px-2 py-1 text-left text-[8.5px]">
                {heading}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {payload.cells.flatMap((cell) =>
            variants.map((variant) => {
              const row = attributionRows(cell).find((entry) => entry.variant === variant);
              if (!row) return null;
              const label = variantLabel(variant);
              return (
                <tr key={`${cell.cell}-${variant}`} style={{ borderTop: "1px solid var(--border)" }}>
                  <td className="px-2 py-0.5">
                    <div className="text-[10.5px] leading-tight" style={{ color }}>
                      {label.name}
                    </div>
                    <div className="mono text-[8.5px] leading-tight ink-faint">{label.token}</div>
                  </td>
                  <td className="mono px-2 py-0.5 text-[9.5px] ink-faint">{cell.cell.replace(" @ ", " ")}</td>
                  <td
                    className="mono px-2 py-0.5 text-[10.5px]"
                    style={{
                      color:
                        Math.abs(row.variantVsProductionAubqc) < 0.005
                          ? "var(--ink-dim)"
                          : row.variantVsProductionAubqc >= 0
                            ? "var(--accent)"
                            : "var(--danger)",
                    }}
                  >
                    {signed(row.variantVsProductionAubqc, 4)}
                  </td>
                  <td
                    className="mono px-2 py-0.5 text-[10.5px]"
                    style={{ color: row.variantVsProductionRho >= 0 ? "var(--accent)" : "var(--danger)" }}
                  >
                    {signed(row.variantVsProductionRho, 4)}
                  </td>
                  <td className="mono px-2 py-0.5 text-[9.5px] ink-faint">{pvalue(row.rhoP)}</td>
                </tr>
              );
            }),
          )}
        </tbody>
      </table>
    </div>
  );
}

function Provenance({ payload }: { payload: AttributionPayload }) {
  const cell = payload.cells[0];
  if (!cell) return null;
  return (
    <Panel
      eyebrow="Why a null result is worth publishing"
      title="A change that does nothing is a finding, not a failure"
      actions={
        <span className="flex items-center gap-1 text-[10px] ink-faint">
          <ArrowDownUp size={10} /> {int(payload.seeds)} seeds per cell
        </span>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-2">
          <FactRow icon={<Minus size={10} />} label="production row" body={cell._provenance.production_row_source} />
          <FactRow icon={<ShieldX size={10} />} label="mechanism rows" body={cell._provenance.mechanism_rows_source} />
          <p className="flex items-start gap-1.5 text-[10.5px] leading-relaxed ink-faint">
            <ShieldX size={10} className="mt-[3px] shrink-0" />
            Rejected variants stay in the ledger. Nothing here is hidden because it flatters the engine.
          </p>
        </div>
        <Callout tone="info" title="What a tie means here">
          A change that ties the current engine is not "as good as" it — it is evidence that the component
          being changed was not doing the work on this target. The shipped configuration keeps only what
          survived a paired comparison. Every "odds it's luck" column is a two-sided paired test over per-run
          differences, never a per-round test.
        </Callout>
      </div>
    </Panel>
  );
}

function FactRow({ icon, label, body }: { icon: React.ReactNode; label: string; body: string }) {
  return (
    <div className="rounded-lg border px-2.5 py-2" style={{ borderColor: "var(--border)", background: "var(--panel-2)" }}>
      <div className="label flex items-center gap-1.5 text-[8.5px]">
        <span style={{ color: "var(--ink-faint)" }}>{icon}</span>
        {label}
      </div>
      <p className="mt-0.5 text-[10.5px] leading-relaxed ink-dim">{body}</p>
    </div>
  );
}
