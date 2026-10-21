import { Crown, FlaskConical, ShieldCheck, Sparkles, Star } from "lucide-react";
import type { ConfigPayload, StatePayload } from "../../types/state";
import { AlignmentLegend, SequenceAlignment } from "../../components/SequenceAlignment";
import { InfoTip, Note } from "../../components/Panel";
import {
  deltaAgentSpread,
  harvestFromRound,
  slotRoles,
  slotSplit,
  type SlotRoleView,
} from "../../lib/derive";
import { num, signed } from "../../lib/format";

/**
 * The wet-lab plate — five rows, one per slot.
 *
 * A row layout rather than a card grid, because the comparison a reader has to
 * make is *across* slots: slot 1 vs slot 5 is the paired H1/H0 experiment, so
 * `a_h1` and `a_null` sit in the same column and the lift is visible without
 * arithmetic.
 *
 * `μ_bind ± σ_bind` is the surrogate prediction made *before* the assay;
 * `−log₁₀ K_D` is what the wet lab returned. Both are shown, side by side, in
 * separate columns — the difference between them is the residual the audit grades
 * the hypothesis on.
 */
export function PlatePanel({
  state,
  config,
  totalRounds,
}: {
  state: StatePayload;
  config: ConfigPayload | null;
  totalRounds: number;
}) {
  const revealed = new Map(state.pareto_data.map((point) => [point.candidate_id, point.binding] as const));
  const best = state.kpis.best_qualified_binding;
  const roles = slotRoles(state, revealed, best);
  const split = slotSplit(state);
  const spread = deltaAgentSpread(state);
  const wildType = config?.wild_type ?? "SRWGGDGCYAMDY";
  const falsified = new Set<number>();
  for (const hypothesis of state.hypotheses ?? []) {
    for (const position of hypothesis.evidence.inverted_positions) falsified.add(position);
  }
  const harvest = harvestFromRound(totalRounds);
  const graduated = state.voi_diagnostics.slot_graduation === true;
  const champion = roles.find((role) => role.isNewBest) ?? null;

  return (
    <section className="panel overflow-hidden">
      <header className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-4 pb-2.5 pt-3.5">
        <div className="min-w-0">
          <div className="label mb-1 flex flex-wrap items-center gap-x-1.5 text-[9.5px]">
            <span>5-slot plate · paired design · round {state.current_round}</span>
            <span style={{ color: "var(--border-strong)" }}>|</span>
            <span className="mono normal-case tracking-normal">
              WT <span style={{ color: "var(--ink-dim)" }}>{wildType}</span>
            </span>
          </div>
          <h2 className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[14px] font-semibold leading-snug ink">
            {graduated ? (
              <>
                Harvest phase — <span style={{ color: "var(--gold)" }}>5 exploit : 0 VoI : 0 control</span>
              </>
            ) : (
              <>
                <span style={{ color: "var(--accent)" }}>3 exploit</span>
                <span className="ink-faint">:</span>
                <span style={{ color: "#0EA5E9" }}>1 VoI probe</span>
                <span className="ink-faint">:</span>
                <span style={{ color: "#F59E0B" }}>1 H0 control</span>
              </>
            )}
          </h2>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span
            className="chip mono"
            style={{
              borderColor: graduated ? "var(--gold-line)" : "rgba(14,165,233,0.35)",
              color: graduated ? "var(--gold)" : "#0EA5E9",
              background: graduated ? "var(--gold-soft)" : "rgba(14,165,233,0.07)",
            }}
            title={`slot_graduation fires when round > ceil(2R/3) = ${harvest - 1}, or when spread(Δ_agent) < the floor.`}
          >
            {graduated ? "GRADUATED 5:0:0 HARVEST" : `HARVEST FROM R${harvest}`}
          </span>
        </div>
      </header>

      <div className="space-y-1.5 px-4 pb-3">
        {roles.map((role) => (
          <SlotRow
            key={role.slot.slot_id}
            role={role}
            wildType={wildType}
            falsified={[...falsified]}
            champion={champion !== null && champion.slot.slot_id === role.slot.slot_id}
          />
        ))}
      </div>

      <div
        className="flex flex-wrap items-center justify-between gap-2 border-t px-4 py-2"
        style={{ borderColor: "var(--border)" }}
      >
        <AlignmentLegend />
        <div className="flex items-center gap-2.5">
          <span className="mono text-[9.5px] ink-faint">
            {state.batch_slots.filter((slot) => slot.risk_passed).length}/{state.batch_slots.length} risk-qualified ·{" "}
            {roles.filter((role) => role.revealedBinding !== null).length} revealed
          </span>
          <InfoTip label="Slot graduation rule">
            Slots graduate when <span className="mono">round &gt; ceil(2R/3)</span> ({harvest - 1} for a{" "}
            {totalRounds}-round horizon) or when <span className="mono">spread(Δ_agent)</span> falls below the
            floor{spread !== null ? ` — here ${spread.toFixed(4)}` : ""}. After graduation slots 3 and 5 become
            pure exploitation picks, so their old rule names are no longer what selected them.
          </InfoTip>
          <Note summary={`${split.label} · split counted from the slots' own rules`}>
            <span className="mono">split.exploit</span> is the count of slots whose selection rule is an
            exploitation rule; <span className="mono">voi</span> and <span className="mono">h0</span> are the
            uncertainty probe and the paired negative control. Once slot graduation fires, all five are
            exploitation picks and the split reads 5:0:0.
          </Note>
        </div>
      </div>
    </section>
  );
}

function SlotRow({
  role,
  wildType,
  falsified,
  champion,
}: {
  role: SlotRoleView;
  wildType: string;
  falsified: number[];
  champion: boolean;
}) {
  const { slot } = role;
  const isH0 = slot.slot_id === 5;
  const isVoi = slot.slot_type === "Max Joint Uncertainty" && role.roleBucket === "voi";
  const lift = slot.a_h1 !== null && slot.a_null !== null ? slot.a_h1 - slot.a_null : null;
  const mutations = slot.sequence.split("").filter((residue, index) => residue !== wildType[index]).length;

  return (
    <article
      className={`grid items-center gap-x-3 gap-y-2 rounded-xl border px-3 py-2.5 transition-colors ${
        champion ? "gold-row" : ""
      }`}
      style={{
        borderColor: champion ? "var(--gold-line)" : "var(--border)",
        background: champion ? undefined : "var(--panel-2)",
        gridTemplateColumns: "minmax(0,1fr)",
      }}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 lg:flex-nowrap">
        {/* slot identity */}
        <div className="flex w-full shrink-0 items-center gap-2 lg:w-[154px]">
          <span
            className="mono flex h-8 w-8 items-center justify-center rounded-lg text-[13px] font-semibold"
            style={{
              background: champion ? "var(--gold-soft)" : `${role.roleColor}1a`,
              color: champion ? "var(--gold)" : role.roleColor,
              border: `1px solid ${champion ? "var(--gold-line)" : `${role.roleColor}44`}`,
            }}
          >
            {slot.slot_id}
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-1">
              <span className="text-[11.5px] font-semibold leading-tight ink">{role.roleLabel}</span>
              {champion ? <Star size={11} style={{ color: "var(--gold)" }} fill="var(--gold)" /> : null}
            </div>
            <div className="mono truncate text-[9.5px] leading-tight ink-faint" title={slot.slot_type}>
              {slot.slot_type}
            </div>
            {isH0 ? (
              <div className="mono text-[9px] leading-tight" style={{ color: "#F59E0B" }}>
                H0: Δ_agent = 0
              </div>
            ) : null}
            {isVoi ? (
              <div className="mono text-[9px] leading-tight" style={{ color: "#0EA5E9" }}>
                A_H1 + 0.5·VoI·σ_joint
              </div>
            ) : null}
          </div>
        </div>

        {/* candidate + alignment */}
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <span className="mono text-[11.5px] font-semibold" style={{ color: champion ? "var(--gold)" : "var(--ink)" }}>
              {slot.candidate_id}
            </span>
            <span className="mono text-[9.5px] ink-faint">{mutations} mutations</span>
            {champion ? (
              <span
                className="mono rounded px-1.5 py-[1px] text-[8.5px] font-semibold tracking-wide"
                style={{ background: "var(--gold-soft)", color: "var(--gold)", border: "1px solid var(--gold-line)" }}
              >
                BEST SO FAR
              </span>
            ) : null}
          </div>
          <SequenceAlignment sequence={slot.sequence} wildType={wildType} falsifiedPositions={falsified} />
        </div>

        {/* measured + predicted */}
        <div className="grid w-full shrink-0 grid-cols-2 gap-x-3 gap-y-1 lg:w-[300px] lg:grid-cols-4">
          <Cell label="μ ± σ">
            <span className="mono text-[11px] ink">
              {num(slot.mu_bind, 2)}
              <span className="ink-faint"> ± {num(slot.sigma_bind, 2)}</span>
            </span>
          </Cell>
          <Cell label="Δ_agent">
            <span
              className="mono text-[11px]"
              style={{ color: slot.delta_agent === null ? "var(--ink-faint)" : slot.delta_agent >= 0 ? "var(--accent)" : "var(--danger)" }}
            >
              {signed(slot.delta_agent, 3)}
            </span>
          </Cell>
          <Cell label="A_H1 / A_null">
            <span className="mono text-[11px] ink">{num(slot.a_h1, 3)}</span>
            <span className="mono text-[11px] ink-faint"> / {num(slot.a_null, 3)}</span>
            {lift !== null ? (
              <div className="mono text-[9px] leading-tight" style={{ color: lift >= 0 ? "var(--accent)" : "var(--danger)" }}>
                lift {signed(lift, 3)}
              </div>
            ) : null}
          </Cell>
          <Cell label="−log₁₀ K_D">
            {role.revealedBinding === null ? (
              <span className="mono flex items-center gap-1 text-[10px] ink-faint">
                <FlaskConical size={9} /> pending
              </span>
            ) : (
              <span
                className="mono flex items-center gap-1 text-[15px] font-semibold leading-none"
                style={{ color: champion ? "var(--gold)" : role.isNewBest ? "var(--accent)" : "var(--ink)" }}
              >
                {num(role.revealedBinding, 3)}
                {role.isNewBest ? (
                  champion ? (
                    <Crown size={11} style={{ color: "var(--gold)" }} />
                  ) : (
                    <Sparkles size={10} style={{ color: "var(--accent)" }} />
                  )
                ) : null}
              </span>
            )}
          </Cell>
        </div>

        {/* developability */}
        <div className="hidden w-[62px] shrink-0 flex-col items-end gap-0.5 xl:flex">
          <span className="mono text-[10.5px]" style={{ color: slot.risk_passed ? "var(--accent)" : "var(--danger)" }}>
            {num(slot.risk_score, 2)}
          </span>
          <span
            className="mono flex items-center gap-0.5 text-[8.5px] leading-none"
            style={{ color: slot.risk_passed ? "var(--ink-faint)" : "var(--danger)" }}
          >
            <ShieldCheck size={8} /> {slot.risk_passed ? "ok" : "fail"}
          </span>
        </div>
      </div>
    </article>
  );
}

function Cell({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="label mb-0.5 whitespace-nowrap text-[8.5px]">{label}</div>
      <div className="leading-none">{children}</div>
    </div>
  );
}
