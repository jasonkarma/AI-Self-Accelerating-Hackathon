import { Crown, FlaskConical, ShieldCheck, ShieldX, Sparkles, Star } from "lucide-react";
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
 * The batch of five candidates — one row per slot.
 *
 * A row layout rather than a card grid, because the comparison a reader makes is
 * *across* slots: slot 1 vs slot 5 is the paired hypothesis/control experiment,
 * so the two measured results sit in the same column and the difference is
 * visible without arithmetic.
 *
 * Two rules govern the labelling:
 *
 * * **The rule name is a role, not a formula.** "Tests the biggest unknown" and
 *   "control: must show nothing" are what the slots are for; the internal rule
 *   (`Max Joint Uncertainty`, `a_h1`, `Δ_agent`) stays in the row's tooltip.
 * * **The safety verdict travels with the candidate.** Each row carries an
 *   explicit pass/fail badge, because "this candidate would be rejected in a
 *   real development pipeline" is a fact the audience can read instantly.
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
            <span>the five candidates tested in round {state.current_round}</span>
            <span style={{ color: "var(--border-strong)" }}>|</span>
            <span className="mono normal-case tracking-normal">
              starting antibody <span style={{ color: "var(--ink-dim)" }}>{wildType}</span>
            </span>
          </div>
          <h2 className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[14px] font-semibold leading-snug ink">
            {graduated ? (
              <>
                Every slot goes after the <span style={{ color: "var(--gold)" }}>best predicted candidate</span>
              </>
            ) : (
              <>
                <span style={{ color: "var(--accent)" }}>3 to exploit</span>
                <span className="ink-faint">:</span>
                <span style={{ color: "#0EA5E9" }}>1 to resolve an unknown</span>
                <span className="ink-faint">:</span>
                <span style={{ color: "#F59E0B" }}>1 control</span>
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
            title={`After round ${harvest - 1} of ${totalRounds}, the loop stops spending slots on exploration and control.`}
          >
            {graduated ? `ALL-IN SINCE R${harvest} · no control slot` : `FULL EXPLORATION UNTIL R${harvest}`}
          </span>
          <InfoTip label="Why the mix changes">
            Early rounds reserve one slot to resolve the biggest open question and one to check the measure
            itself. Once the horizon passes two thirds, all five go to the best predicted candidates — because
            learning is worth less than finding.
          </InfoTip>
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
            {state.batch_slots.filter((slot) => slot.risk_passed).length}/{state.batch_slots.length} pass the safety filter ·{" "}
            {roles.filter((role) => role.revealedBinding !== null).length} measured
          </span>
          <InfoTip label="Slot graduation rule">
            Slots stop exploring when <span className="mono">round &gt; ceil(2R/3)</span> ({harvest - 1} for a{" "}
            {totalRounds}-round horizon) or when the agents' priors stop disagreeing
            <span className="mono"> (spread(Δ_agent))</span>
            {spread !== null ? ` — currently ${spread.toFixed(4)}` : ""}. After that, slots 3 and 5 become pure
            exploitation picks, so their original rule names no longer describe what selected them.
          </InfoTip>
          <Note summary={`${split.label} · counted from the slots' own rules`}>
            <span className="mono">split.exploit</span> counts slots whose rule is an exploitation rule;{" "}
            <span className="mono">voi</span> and <span className="mono">h0</span> are the uncertainty probe and
            the paired control. Once graduation fires, all five are exploitation picks and the split reads
            5:0:0.
          </Note>
        </div>
      </div>
    </section>
  );
}

/** What each selection rule is *for*, in the words a reader needs. */
function roleWords(role: SlotRoleView): { label: string; why: string } {
  if (role.roleBucket === "harvest") {
    return { label: "Best predicted candidate", why: "Learning is worth less than finding now." };
  }
  if (role.roleBucket === "voi") {
    return { label: "Resolves the biggest unknown", why: "Tests where the two models disagree most." };
  }
  if (role.roleBucket === "h0") {
    return { label: "Control — must show nothing", why: "If this beats its twin, the measurement is broken." };
  }
  return { label: "Strong predicted candidate", why: "Exploit what the model already believes." };
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
  const words = roleWords(role);
  const isH0 = slot.slot_id === 5;
  const isVoi = slot.slot_type === "Max Joint Uncertainty" && role.roleBucket === "voi";
  const lift = slot.a_h1 !== null && slot.a_null !== null ? slot.a_h1 - slot.a_null : null;
  const mutations = slot.sequence.split("").filter((residue, index) => residue !== wildType[index]).length;

  return (
    <article
      className={`grid items-center gap-x-3 gap-y-2 rounded-xl border px-3 py-2.5 transition-colors ${champion ? "gold-row" : ""
        }`}
      style={{
        borderColor: champion ? "var(--gold-line)" : "var(--border)",
        background: champion ? undefined : "var(--panel-2)",
        gridTemplateColumns: "minmax(0,1fr)",
      }}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 lg:flex-nowrap">
        {/* slot identity */}
        <div className="flex w-full shrink-0 items-center gap-2 lg:w-[176px]">
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
              <span className="text-[11.5px] font-semibold leading-tight ink">{words.label}</span>
              {champion ? <Star size={11} style={{ color: "var(--gold)" }} fill="var(--gold)" /> : null}
            </div>
            <div className="truncate text-[9.5px] leading-tight ink-faint" title={slot.slot_type}>
              {words.why}
            </div>
            <div className="mono flex flex-wrap items-center gap-1 text-[9px] leading-tight ink-faint">
              <InfoTip label="Internal rule name">
                <span className="mono">{slot.slot_type}</span>
                {isH0 ? (
                  <>
                    {" "}— the control arm: the agents' prior is forced to zero, so this candidate is picked by
                    the base rule alone.
                  </>
                ) : isVoi ? (
                  <>
                    {" "}— the exploration slot: scored as the hypothesis arm plus a bonus for how much the two
                    models disagree.
                  </>
                ) : null}
              </InfoTip>
              <span>{slot.slot_type}</span>
            </div>
          </div>
        </div>

        {/* candidate + alignment */}
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <span className="mono text-[11.5px] font-semibold" style={{ color: champion ? "var(--gold)" : "var(--ink)" }}>
              {slot.candidate_id}
            </span>
            <span className="mono text-[9.5px] ink-faint">{mutations} changes from the start</span>
            {champion ? (
              <span
                className="mono rounded px-1.5 py-[1px] text-[8.5px] font-semibold tracking-wide"
                style={{ background: "var(--gold-soft)", color: "var(--gold)", border: "1px solid var(--gold-line)" }}
              >
                BEST FOUND
              </span>
            ) : null}
          </div>
          <SequenceAlignment sequence={slot.sequence} wildType={wildType} falsifiedPositions={falsified} />
        </div>

        {/* predicted + measured + safety */}
        <div className="grid w-full shrink-0 grid-cols-2 gap-x-3 gap-y-1 lg:w-[352px] lg:grid-cols-4">
          <Cell label="Predicted">
            <span className="mono text-[11px] ink">
              {num(slot.mu_bind, 2)}
              <span className="ink-faint"> ± {num(slot.sigma_bind, 2)}</span>
            </span>
            <span className="mono block text-[8.5px] leading-tight ink-faint">before the test</span>
          </Cell>
          <Cell label="Measured">
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
            <span className="mono block text-[8.5px] leading-tight ink-faint">from the assay</span>
          </Cell>
          <Cell label="Hypothesis vs. control">
            <span className="mono inline-flex items-center gap-1.5 text-[11px] ink">
              {num(slot.a_h1, 2)}
              <span className="ink-faint">vs</span>
              {num(slot.a_null, 2)}
            </span>
            {lift !== null ? (
              <div
                className="mono inline-flex items-center gap-1 text-[9px] leading-tight"
                style={{ color: lift >= 0 ? "var(--accent)" : "var(--danger)" }}
              >
                {lift >= 0 ? "control held" : "control beat it"} ({signed(lift, 2)})
              </div>
            ) : null}
            <InfoTip label="The paired experiment">
              <span className="mono">a_h1</span> is the score with the agents' prior applied;{" "}
              <span className="mono">a_null</span> is the same candidate under the base rule alone. The two are
              measured in the same batch, which is what makes the comparison a real experiment.
            </InfoTip>
          </Cell>
          <Cell label="Safety filter">
            <span
              className="mono inline-flex items-center gap-1 rounded-full px-1.5 py-[1px] text-[9px] font-semibold"
              style={{
                color: slot.risk_passed ? "var(--accent)" : "var(--danger)",
                background: slot.risk_passed ? "var(--accent-soft)" : "rgba(244,63,94,0.1)",
                border: `1px solid ${slot.risk_passed ? "var(--accent-line)" : "rgba(244,63,94,0.35)"}`,
              }}
              title={`Developability risk ${num(slot.risk_score, 2)} — chemical liabilities that would sink a real drug candidate, independent of how well it binds.`}
            >
              {slot.risk_passed ? <ShieldCheck size={9} /> : <ShieldX size={9} />}
              {slot.risk_passed ? "passes" : "fails"}
            </span>
            <span className="mono block text-[8.5px] leading-tight ink-faint">
              risk {num(slot.risk_score, 2)}
            </span>
          </Cell>
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
