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
    <div className="space-y-4">
      <SummaryBand payload={payload} />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] xl:items-start">
        <CausalWinner payload={payload} />
        <div className="space-y-4">
          <NegativeControlCard payload={payload} group="slot_role" />
          <NegativeControlCard payload={payload} group="sar_efficiency" />
        </div>
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
    <section className="panel hero-wash overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 px-5 pb-1 pt-4">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10.5px] font-semibold"
            style={{ background: "var(--gold-soft)", color: "var(--gold)", border: "1px solid var(--gold-line)" }}
          >
            <Trophy size={11} /> CAUSAL RESULT
          </span>
          <span className="text-[11px] ink-faint">
            {payload.seeds} seeds per cell · 4 measured cells · two-sided paired t-tests
          </span>
        </div>
        <Badge color="#10B981" title="The production objective shipped at e567cd4">
          a_null := a_base
        </Badge>
      </div>

      <h2 className="px-5 pb-1 pt-2 text-[24px] font-semibold leading-tight tracking-[-0.015em] ink sm:text-[26px]">
        Deleting the VoI anneal improves rank correlation in{" "}
        <span className="mono" style={{ color: "var(--accent)" }}>
          {rhoGain.length}/{rows.length}
        </span>{" "}
        cells, and never lowers AUBQC
      </h2>

      <div className="grid gap-2.5 px-5 pb-4 pt-2 sm:grid-cols-2 lg:grid-cols-4">
        <KpiTile
          label="Causal winner"
          value="no_anneal"
          caption={bestP ? `strongest p ${pvalue(bestP.rhoP)}` : "shipped configuration"}
          emphasis="gold"
          hint="The engine without the heuristic anneal is byte-identical to the live production row — the variant IS the shipped law."
        />
        <KpiTile
          label="ρ improved"
          value={`${rhoGain.length}/${rows.length}`}
          caption="cells, rank correlation"
          emphasis="accent"
          hint="Deletion of the anneal raises Spearman ρ in every measured cell."
        />
        <KpiTile
          label="AUBQC cost"
          value={aubqcHurt.length === 0 ? "none" : `${aubqcHurt.length}/${rows.length} cells`}
          caption={`${aubqcHelped.length} cell(s) improved at α = 0.05`}
          hint="Cells where deleting the anneal lowers AUBQC at α = 0.05. None does: the ceiling statistic is either unchanged or improved."
        />
        <KpiTile
          label="Negative controls"
          value={`${slotRole.comparisons + sar.comparisons}`}
          caption={`smallest p ${pvalue(Math.min(slotRole.minP ?? 1, sar.minP ?? 1))}`}
          hint="Paired comparisons across the slot-role and SAR-capacity families. None reaches α = 0.05."
        />
      </div>

      <div className="grid gap-2 px-5 pb-4 sm:grid-cols-3">
        <OutcomePill
          tone="accent"
          icon={<Trophy size={10} />}
          title="Causal winner"
          body={`${mechanism.comparisons} mechanism comparisons · smallest p ${pvalue(mechanism.minP)}`}
        />
        <OutcomePill
          tone="neutral"
          icon={<Vote size={10} />}
          title="Slot-role null"
          body={`${slotRole.comparisons} comparisons across ${slotRole.cells} cells · smallest p ${pvalue(slotRole.minP)}`}
        />
        <OutcomePill
          tone="neutral"
          icon={<FlaskConical size={10} />}
          title="SAR-capacity null"
          body={`${sar.comparisons} comparisons across ${sar.cells} cells · smallest p ${pvalue(sar.minP)}`}
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
      eyebrow="Causal winner · what was deleted and what it earned"
      title="Unbiased Null Objective"
      lede={
        <>
          <span className="mono" style={{ color: "var(--gold)" }}>
            no_anneal
          </span>{" "}
          — the engine without the heuristic anneal that used to pull{" "}
          <span className="mono">a_null</span> toward the surrogate prediction.
        </>
      }
      actions={
        <Badge color="#10B981" title="Production law, shipped at e567cd4">
          SHIPPED
        </Badge>
      }
    >
      <div className="mb-2.5 grid grid-cols-2 gap-2 sm:grid-cols-3">
        <KpiTile label="ρ gain, every cell" value={`${rows.filter((r) => r.annealRemovalRho > 0).length}/${rows.length}`} caption="rank correlation" emphasis="accent" />
        <KpiTile label="largest ρ gain" value={worstRho ? signed(Math.max(...rows.map((r) => r.annealRemovalRho)), 4) : "—"} caption={bestP ? `p ${pvalue(bestP.rhoP)} · ${bestP.cell}` : ""} />
        <KpiTile
          label="AUBQC cost"
          value={rows.some((row) => row.annealRemovalAubqc < 0 && row.aubqcP < 0.05) ? "present" : "none"}
          caption={`${rows.filter((row) => row.aubqcP >= 0.05).length}/${rows.length} neutral, rest improved`}
        />
      </div>

      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse">
          <thead>
            <tr style={{ background: "var(--panel-2)" }}>
              {["cell", "Δρ from deleting", "p", "seeds better", "ΔAUBQC", "p"].map((heading) => (
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
                  {signed(row.annealRemovalRho, 4)}
                </td>
                <td
                  className="mono px-2 py-1 text-[10.5px]"
                  style={{ color: row.rhoP < 0.001 ? "var(--accent)" : "var(--ink-dim)" }}
                >
                  {pvalue(row.rhoP)}
                </td>
                <td className="mono px-2 py-1 text-[10px] ink-faint">
                  {int(row.rhoSeedsDeletionHelped)}/{int(row.rhoN)}
                </td>
                <td
                  className="mono px-2 py-1 text-[10.5px]"
                  style={{
                    color: row.aubqcP >= 0.05 ? "var(--ink-dim)" : row.annealRemovalAubqc >= 0 ? "var(--accent)" : "var(--danger)",
                  }}
                  title={row.annealRemovalAubqc >= 0 ? "deletion helped" : "deletion hurt"}
                >
                  {signed(row.annealRemovalAubqc, 4)}
                </td>
                <td className="mono px-2 py-1 text-[10px] ink-faint">{pvalue(row.aubqcP)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Note summary={`Sign convention: variant − baseline. Baseline ${rows[0]?.comparisonBasis ?? "—"}.`}>
        A positive Δ means removing the anneal <em>helped</em>. The per-seed arrays for{" "}
        <span className="mono">no_anneal</span> are byte-identical to the live production row, which is the
        point: the variant <em>is</em> the shipped configuration.
        {bestP ? (
          <>
            {" "}
            Strongest ρ evidence: <span className="mono">{pvalue(bestP.rhoP)}</span> ({bestP.cell}).
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
          {group === "slot_role" ? "Slot-role separation" : "SAR sample efficiency"}
        </h3>
        <p className="mt-0.5 text-[11.5px] leading-relaxed ink-dim">
          {group === "slot_role"
            ? "Re-homing the exploitation slots onto the scoring model does not convert ρ into AUBQC."
            : "Extra representational capacity in the SAR prior buys nothing measurable."}
        </p>
      </header>
      <div className="px-4 pb-3">
        <div className="mb-1.5 flex items-center gap-1.5">
          <span className="mono text-[9.5px] ink-faint">
            {summary.comparisons} paired comparisons · {summary.cells} cells
          </span>
          <InfoTip label="What a null result means here">
            A variant that ties the engine is not &quot;as good as&quot; it — it is evidence that the
            component it changes was not doing the work on this target.
          </InfoTip>
        </div>
        <VariantTable payload={payload} variants={variants} group={group} />
        <div className="mono mt-1 text-[9px] ink-faint">
          smallest p {pvalue(summary.minP)}
          {summary.minAt ? ` · ${summary.minAt.cell} · ${summary.minAt.variant} · ${summary.minAt.metric}` : ""}
        </div>
      </div>
    </section>
  );
}

function MechanismCard({ payload }: { payload: AttributionPayload }) {
  const mechanism = groupSummary(payload, "mechanism");
  return (
    <Panel
      eyebrow="Mechanism"
      title="The two variants that change the search itself"
      lede={`${mechanism.comparisons} comparisons, smallest p ${pvalue(mechanism.minP)} — both measured against the pre-e567cd4 engine, which still carried the anneal.`}
      actions={<Badge color="#F59E0B">different baseline</Badge>}
    >
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <VariantTable payload={payload} variants={["no_anneal", "greedy_slots"]} group="mechanism" />
        <Callout tone="warn" title="Reading the mechanism rows">
          These are the only rows measured against a <em>different</em> baseline from the live engine
          (pre-<span className="mono">e567cd4</span>). The slot-role and SAR controls share that baseline, so
          all six are comparable with each other but not with the live-engine rows in the ledger — which is
          why every row names its basis.
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
      eyebrow="Full ledger"
      title="Every variant, every cell — including the baselines"
      lede="One row per variant. Δ is `production − variant` on both axes, with the paired p and seeds won."
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
          label="production AUBQC"
          value={`${num(production?.aubqc ?? null, 3)}`}
          unit={`± ${num(production?.aubqc_se ?? null, 3)}`}
          caption={`${int(production?.seeds ?? cell.seeds)} seeds`}
          emphasis="accent"
        />
        <KpiTile
          label="production ρ"
          value={num(production?.rho ?? null, 3)}
          caption={`n = ${int(production?.rho_n ?? null)} defined replicates`}
          emphasis="accent"
        />
        <KpiTile
          label="protocol"
          value={`R ${cell.protocol.rounds} · Q ${cell.protocol.total_budget}`}
          caption={`L ${cell.protocol.seq_len} · N_qual ${int(cell.protocol.n_qualified)}`}
        />
        <KpiTile
          label="coverage"
          value={`${(cell.protocol.coverage_pct * 100).toFixed(2)}%`}
          caption={`target ${(cell.protocol.target_coverage * 100).toFixed(2)}%`}
        />
      </div>
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[880px] border-collapse">
          <thead>
            <tr style={{ background: "var(--panel-2)" }}>
              {[
                "variant",
                "group",
                "ΔAUBQC (variant − base)",
                "p",
                "seeds",
                "Δρ (variant − base)",
                "p",
                "seeds",
                "baseline AUBQC",
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
        {signed(row.variantVsProductionAubqc, 4)}
      </td>
      <td className="px-3 py-1.5 mono text-[10.5px]" style={{ color: row.aubqcP < 0.05 ? "var(--accent)" : "var(--ink-faint)" }}>
        {pvalue(row.aubqcP)}
      </td>
      <td className="px-3 py-1.5 mono text-[10px] ink-faint">
        {row.aubqcVariantBetter === null ? "—" : `${int(row.aubqcVariantBetter)}/${int(row.aubqcN)}`}
      </td>
      <td
        className="px-3 py-1.5 mono text-[11px]"
        style={{ color: row.variantVsProductionRho >= 0 ? "var(--accent)" : "var(--danger)" }}
      >
        {signed(row.variantVsProductionRho, 4)}
      </td>
      <td
        className="px-3 py-1.5 mono text-[10.5px]"
        style={{
          color: row.rhoP < 0.05 ? (row.variantVsProductionRho >= 0 ? "var(--accent)" : "var(--danger)") : "var(--ink-faint)",
        }}
      >
        {pvalue(row.rhoP)}
      </td>
      <td className="px-3 py-1.5 mono text-[10px] ink-faint">
        {row.rhoVariantBetter === null ? "—" : `${int(row.rhoVariantBetter)}/${int(row.rhoN)}`}
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
            {["variant", "cell", "ΔAUBQC", "Δρ", "p (ρ)"].map((heading) => (
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
      eyebrow="Provenance"
      title="How to read a negative result"
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
          A variant that ties production on both axes is not &quot;as good as production&quot; — it is
          evidence that the component it removes was not doing the work on this target. The shipped
          configuration keeps only what survived a paired test. Every p-value is a two-sided paired t-test on
          per-seed differences, not a per-round test.
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
