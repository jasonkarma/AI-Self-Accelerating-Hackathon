"""Six deterministic, JSON-serializable scientific tools."""
from __future__ import annotations

from collections.abc import Callable, Iterable
import numpy as np
import pandas as pd

from .constants import WT_SEQUENCE
from .developability import raw_components

_CANDIDATE_LOOKUP: dict[str, str] = {}


def configure_candidate_lookup(mapping: dict[str, str]) -> None:
    global _CANDIDATE_LOOKUP
    _CANDIDATE_LOOKUP = dict(mapping)


def scan_positional_sar(ledger_df: pd.DataFrame) -> dict:
    needed = {"candidate_id", "sequence", "y_bind"}
    if ledger_df.empty or not needed <= set(ledger_df.columns):
        return {"tool": "scan_positional_sar", "rows": [], "wt_binding": None}
    wt = ledger_df.loc[ledger_df.sequence == WT_SEQUENCE, "y_bind"]
    reference = float(wt.mean()) if len(wt) else float(ledger_df.y_bind.mean())
    records = []
    for position in range(13):
        for residue, group in ledger_df[ledger_df.sequence.str[position] != WT_SEQUENCE[position]].groupby(ledger_df.sequence.str[position]):
            mean = float(group.y_bind.mean())
            records.append({"position": position + 1, "residue": residue, "n": int(len(group)),
                            "mean_y_bind": mean, "delta_vs_wt": mean - reference,
                            "std": float(group.y_bind.std(ddof=1)) if len(group) > 1 else 0.0,
                            "candidate_ids": sorted(group.candidate_id.astype(str).tolist())})
    return {"tool": "scan_positional_sar", "wt_binding": reference, "rows": records}


def detect_epistasis_clashes(ledger_df: pd.DataFrame) -> dict:
    if ledger_df.empty or not {"sequence", "y_bind"} <= set(ledger_df.columns):
        return {"tool": "detect_epistasis_clashes", "clashes": []}
    base = float(ledger_df.loc[ledger_df.sequence == WT_SEQUENCE, "y_bind"].mean()) if (ledger_df.sequence == WT_SEQUENCE).any() else float(ledger_df.y_bind.mean())
    effects: dict[int, list[float]] = {i: [] for i in range(13)}
    for row in ledger_df.itertuples():
        positions = [i for i, (a, b) in enumerate(zip(row.sequence, WT_SEQUENCE)) if a != b]
        if len(positions) == 1:
            effects[positions[0]].append(float(row.y_bind) - base)
    residuals = []
    for row in ledger_df.itertuples():
        positions = [i for i, (a, b) in enumerate(zip(row.sequence, WT_SEQUENCE)) if a != b]
        if len(positions) == 2 and all(effects[p] for p in positions):
            expected = base + sum(np.mean(effects[p]) for p in positions)
            residuals.append((row, positions, float(row.y_bind) - expected))
    sigma = float(np.std([r[2] for r in residuals], ddof=1)) if len(residuals) > 1 else 0.0
    threshold = 1.5 * sigma
    clashes = [{"candidate_id": str(r.candidate_id), "positions": [p + 1 for p in ps], "residual": residual,
                "direction": "positive" if residual > 0 else "negative"}
               for r, ps, residual in residuals if abs(residual) > threshold and (sigma > 0 or residual != 0)]
    return {"tool": "detect_epistasis_clashes", "sigma": sigma, "threshold": threshold, "clashes": clashes}


def profile_developability_liabilities(candidate_ids: Iterable[str]) -> dict:
    rows = []
    for candidate_id in candidate_ids:
        if candidate_id not in _CANDIDATE_LOOKUP:
            raise KeyError(f"Unknown candidate: {candidate_id}")
        sequence = _CANDIDATE_LOOKUP[candidate_id]
        raw = raw_components(sequence)
        rows.append({"candidate_id": candidate_id, "sequence": sequence, "max_5mer_hydropathy": raw["window"],
                     "aromatic_fraction": raw["aromatic"], "net_charge_abs_fraction": raw["charge"],
                     "adjacent_charge_clusters": int(raw["cluster"]),
                     "deamidation_motifs": [m for m in ("NG", "NS") if m in sequence],
                     "isomerization_motifs": [m for m in ("DG", "DS") if m in sequence], "unpaired_cys": sequence.count("C")})
    return {"tool": "profile_developability_liabilities", "rows": rows}


def inspect_joint_uncertainty_hotspots(unqueried_df: pd.DataFrame, top_k: int = 15) -> dict:
    needed = {"candidate_id", "sigma_bind", "sigma_dev", "sigma_joint"}
    if not needed <= set(unqueried_df.columns):
        raise ValueError(f"Missing uncertainty columns: {sorted(needed - set(unqueried_df.columns))}")
    rows = unqueried_df.sort_values(["sigma_joint", "candidate_id"], ascending=[False, True]).head(top_k)
    return {"tool": "inspect_joint_uncertainty_hotspots", "rows": rows[list(needed)].to_dict("records")}


SUPPORTED = "SUPPORTED"
FALSIFIED = "FALSIFIED"
INCONCLUSIVE = "INCONCLUSIVE"

#: How far the H1 arm's mean residual may sit below the **control's** residual
#: before the round counts as refuted. The check is relative because the surrogate
#: is fitted on the revealed rows: a fresh batch it has not seen can be uniformly
#: over-predicted, which pushes every measured residual negative at once. Reading
#: that level shift as evidence against the rule refuted a hypothesis the paired
#: arms had just confirmed (seed 21, R5: arm +0.748 over control, stamped
#: FALSIFIED). Comparing arm to control cancels the shared shift.
RESIDUAL_LEVEL_TOLERANCE = 0.05

#: A paired lift this large is decisive on its own: the arm beat the control by
#: more than any plausible batch-level drift, so the hypothesis stands even if the
#: whole batch measured below its prediction.
DECISIVE_LIFT = 0.10


def _measured_bindings(frame: pd.DataFrame | None) -> dict[str, float]:
    """candidate_id -> revealed y_bind, tolerating an empty or partial ledger."""
    if frame is None or getattr(frame, "empty", True) or not {"candidate_id", "y_bind"} <= set(frame.columns):
        return {}
    return {
        str(cid): float(value)
        for cid, value in zip(frame.candidate_id, frame.y_bind)
        if pd.notna(value)
    }


#: Slot ids that make up the **H1 exploitation arm**. Slot 3 is the VoI
#: exploration probe and belongs to neither arm; slot 5 is the H0 control. The
#: batch is a 3-exploit / 2-experimental design, so "the H1 arm" is a *set* of
#: three measurements, not the single fourth-choice candidate slot 4 happens to
#: be left with.
H1_EXPLOIT_SLOT_IDS: tuple[int, ...] = (1, 2, 4)

#: Decision key carrying the measured ids of that arm. The campaign stamps it
#: once the batch is actually queried; the benchmark's M5 policy stamps the same
#: key through :func:`audit_past_hypotheses`, so both paths grade identically.
H1_ARM_KEY = "h1_arm_measured_ids"


def _tested_positions(decision: dict, sequence: str | None) -> set[int]:
    """Positions whose declared rule the *hypothesis candidate actually exercised*.

    A round can only falsify a rule it tested. Inverting every position in the
    prior list would blacklist most of the CDR3 after two or three rounds (the
    deterministic fallback declares a rule at nearly every mutated position),
    which is not self-correction — it is self-destruction. A position counts as
    tested only when the candidate carries the exact residue that rule preferred,
    because that is the substitution whose effect the paired arms actually
    measure.
    """
    if not sequence:
        return set()
    tested: set[int] = set()
    for prior in decision.get("positional_priors") or []:
        if not isinstance(prior, dict):
            continue
        position = prior.get("position")
        if not position or not (1 <= int(position) <= len(sequence)):
            continue
        residue = sequence[int(position) - 1]
        if residue in (prior.get("preferred_residues") or []):
            tested.add(int(position))
    return tested


def _h1_arm_ids(decision: dict) -> list[str]:
    """Measured ids of the H1 exploitation arm (slots 1, 2 and 4).

    ``h1_arm_measured_ids`` is written by the campaign's
    :meth:`core.campaign.Campaign._annotate_decision` and by the benchmark's M5
    policy, and already excludes the exploitation slots whose ``Δ_agent`` was
    negative (those are not exercising the hypothesis).

    Ledgers written before the arm was recorded only carry ``slot4_measured_id``.
    That single id is the arm's *narrowest* representative, which is exactly the
    4th-choice bias this function exists to remove — but it is the only evidence
    a historical ledger has, so it is used as the fallback rather than declaring
    every older round untestable.
    """
    explicit = decision.get(H1_ARM_KEY)
    if explicit:
        return [str(cid) for cid in explicit if cid]
    slot4 = decision.get("slot4_measured_id") or decision.get("slot4_hypothesis_target_id")
    return [str(slot4)] if slot4 else []


def _residual_of(
    candidate_id: str | None,
    measured: dict[str, float],
    expectations: dict[str, float],
) -> float | None:
    """``y_bind - mu_bind`` for one measured candidate, or ``None`` without a μ.

    ``mu_bind`` comes from the decision's ``pre_query_expectation`` — the
    surrogate mean recorded **before** the assay was run. Using the pre-query
    value rather than a refit is what makes this a genuine out-of-sample
    surprise; a post-hoc refit would absorb the very signal being tested.
    """
    if not candidate_id:
        return None
    key = str(candidate_id)
    if key not in measured or not expectations:
        return None
    mu = expectations.get(key)
    if mu is None:
        return None
    try:
        return float(measured[key]) - float(mu)
    except (TypeError, ValueError):
        return None


def _residual_gated_positions(
    positions: Iterable[int],
    *,
    priors: Iterable[dict],
    arm_ids: Iterable[str],
    sequences: dict[str, str],
    measured: dict[str, float],
    expectations: dict[str, float],
) -> tuple[set[int], dict[int, float]]:
    """Keep only tested positions whose **measured** residual was negative.

    The old rule inverted every position a losing hypothesis touched, which
    produced a ``SUPPORTED -> FALSIFIED -> SUPPORTED`` oscillation: a position
    whose variants measured *better* than the surrogate predicted was pushed
    down anyway, the next round's arm therefore lost, and its inversion flipped
    back. A position is now penalised only when the wet lab measured it
    **worse than the surrogate expected** (``y_bind - mu_bind < 0``).

    Positions with no residual evidence (no ``mu_bind`` on record, e.g. a
    hand-built fixture ledger) are **not** inverted: absence of evidence is not
    evidence against the rule, and the conservative branch is the one that
    cannot produce an oscillation.

    Returns ``(positions_to_invert, {position: mean_residual})``; the second
    element is the audit trail the skeptic agent reads.
    """
    preferred_by_position: dict[int, set[str]] = {}
    for prior in priors:
        if not isinstance(prior, dict):
            continue
        position = prior.get("position")
        if not position:
            continue
        preferred_by_position.setdefault(int(position), set()).update(
            str(residue) for residue in (prior.get("preferred_residues") or [])
        )

    residual_by_position: dict[int, float] = {}
    for position in sorted({int(p) for p in positions}):
        preferred = preferred_by_position.get(position, set())
        values: list[float] = []
        for candidate_id in arm_ids:
            sequence = sequences.get(str(candidate_id))
            if not sequence or len(sequence) < position:
                continue
            if preferred and sequence[position - 1] not in preferred:
                continue
            residual = _residual_of(candidate_id, measured, expectations)
            if residual is not None:
                values.append(residual)
        if values:
            residual_by_position[position] = float(np.mean(values))
    if not residual_by_position:
        return set(), {}

    # Absolute gate (the campaign's rule): a position may only be inverted if its
    # measured residual was negative. A position whose carriers beat the
    # surrogate is never touched, however the round was graded.
    #
    # Relative gate: it must also be negative *relative to the rest of the batch*.
    # A surrogate that systematically over-predicted one batch makes every
    # measured residual negative at once — that is a level shift in the model,
    # not positional evidence. On the live ledger the absolute gate alone inverted
    # 8 of 13 positions after a single batch, which is the same self-destruction
    # the tested-positions rule exists to stop.
    #
    # The reference is the **median**, not the mean. Arm candidates often carry
    # the same residue at several positions, so their residuals form a tight
    # cluster of near-identical values; comparing against the mean of that cluster
    # leaves half of it technically "below average" and re-creates the same
    # multi-position inversion (measured: 6 of 7 positions on one batch). The
    # median splits the cluster at the middle and is what makes "worse than the
    # typical measured substitution" mean something. With fewer than three
    # observed positions no median is meaningful and nothing is inverted.
    observed = np.asarray(list(residual_by_position.values()), dtype=float)
    if observed.size < 3:
        return set(), residual_by_position
    threshold = float(np.median(observed))
    return (
        {
            position
            for position, value in residual_by_position.items()
            if value < 0.0 and value < threshold
        },
        residual_by_position,
    )


def audit_past_hypotheses(ledger_df: pd.DataFrame, previous_decisions: list[dict]) -> dict:
    """Grade each completed round as a paired H1-vs-H0 experiment.

    The batch is a **3-exploit / 2-experimental** design, so the H1 arm is a set
    of three measurements — slots 1, 2 and 4 — not the single candidate slot 4
    happens to be left with. Slot 5 is the **H0 null control** (the best
    candidate the hypothesis does not boost); slot 3 is the VoI exploration probe
    and belongs to neither arm.

    Why the arm and not just slot 4: with ``SLOT4_AGENT_BOOST = 0.0``, slots 1, 2
    and 4 all rank by ``A_H1``, so slots 1-2 take the two strongest candidates,
    slot 3 takes the best VoI exploration pick, and slot 4 inherits the *fourth*
    choice. Grading only that leftover against slot 5 therefore stamped
    ``FALSIFIED`` in rounds where slot 1 or slot 2 found an elite binder, and the
    resulting weight inversion flipped the verdict back the next round — a
    ``SUPPORTED -> FALSIFIED -> SUPPORTED`` oscillation at the batch frequency.
    The arm's **best** measurement is what the hypothesis actually bought; its
    **mean** is reported alongside for depth.

    Rule order (deterministic):

    1. No arm candidate measured ⇒ ``INCONCLUSIVE``.
    2. The control beats the **best** arm measurement ⇒ ``FALSIFIED``.
    3. The best arm measurement beats the control by at least
       :data:`DECISIVE_LIFT` ⇒ ``SUPPORTED``, whatever the residuals say.
    4. The best arm measurement beats the control, but the arm's mean residual
       ``y_bind - mu_bind`` trails the **control's** residual by more than
       :data:`RESIDUAL_LEVEL_TOLERANCE` ⇒ ``FALSIFIED``. A positive lift bought by
       candidates the surrogate already predicted well is not evidence.
       Comparing against the control — not against zero — is what makes this a
       positional claim rather than a measurement of where the batch happened to
       land relative to the surrogate: a freshly fitted model can over-predict a
       whole batch, which drives every residual negative at once without saying
       anything about the rule.
    5. Otherwise the arm's lift stands ⇒ ``SUPPORTED``.
    6. Tie (or no control) ⇒ fall back to the round median, then to the stored
       pre-query surrogate expectation, so a round stays decidable.

    The returned ``falsified_positions`` is the self-correction signal, and it is
    **residual-gated**: a position is inverted only if the wet lab measured its
    carriers *worse than the surrogate expected* (``y_bind - mu_bind < 0``). A
    position whose measured residual was positive is never inverted, however the
    round was graded — that inversion was the negative-feedback half of the
    oscillation. A legacy round with no ``pre_query_expectation`` has no residual
    to read, so it falls back to the tested-positions rule rather than declaring
    nothing falsifiable.

    ``h1_arm_measured_ids`` / ``slot5_measured_id`` are the ids the quota actually
    queried (stamped by the campaign), not the planner's declared targets — which
    Slots 1-3 of the same batch routinely consume.
    """
    verdicts, falsified_positions = [], set()
    has_rounds = ledger_df is not None and "round_number" in getattr(ledger_df, "columns", ())
    # candidate_id -> sequence, used to decide which declared rules the arm's
    # candidates actually exercised. Sequences are free at t=0 (SPEC §2), so this
    # is not a label read.
    sequences: dict[str, str] = {}
    if ledger_df is not None and "sequence" in getattr(ledger_df, "columns", ()):
        sequences = dict(zip(ledger_df.candidate_id.astype(str), ledger_df.sequence.astype(str)))
    for decision in previous_decisions:
        round_no = decision.get("round_number")
        arm_ids = _h1_arm_ids(decision)
        slot4 = decision.get("slot4_measured_id") or decision.get("slot4_hypothesis_target_id")
        slot5 = decision.get("slot5_measured_id") or decision.get("slot5_falsification_target_id")

        round_rows = ledger_df[ledger_df.round_number == round_no] if has_rounds else ledger_df
        measured = _measured_bindings(round_rows)
        if not measured:
            # Reveals can carry a different round tag (cached seed re-query);
            # judge against whatever the ledger has measured rather than
            # declaring the round untestable.
            measured = _measured_bindings(ledger_df)

        expectations = decision.get("pre_query_expectation") or {}

        # The arm is every measured exploitation pick, so a round where slot 1
        # found the elite binder is graded on that find rather than on slot 4's
        # leftover.
        arm_measured = {cid: measured[cid] for cid in arm_ids if cid in measured}
        best_id = max(arm_measured, key=lambda cid: arm_measured[cid]) if arm_measured else None
        observed = arm_measured.get(best_id) if best_id else None
        arm_mean = float(np.mean(list(arm_measured.values()))) if arm_measured else None
        arm_residuals = [
            residual
            for cid in arm_measured
            if (residual := _residual_of(cid, measured, expectations)) is not None
        ]
        mean_residual = float(np.mean(arm_residuals)) if arm_residuals else None

        # The planner's *named* pair, when it survives in this ledger. Both members
        # were snapped into the prospective batch, so grading them is grading the
        # claim the planner actually made rather than the selector's leftover.
        declared_h1 = decision.get("declared_h1_id")
        declared_h0 = decision.get("declared_h0_id")
        declared_pair_measured = (
            bool(declared_h1)
            and bool(declared_h0)
            and str(declared_h1) in measured
            and str(declared_h0) in measured
            and str(declared_h1) != str(declared_h0)
        )
        if declared_pair_measured:
            h1_id, h0_id = str(declared_h1), str(declared_h0)
            observed = measured[h1_id]
            arm_mean = float(np.mean(list(arm_measured.values()))) if arm_measured else observed
            arm_residuals = [
                residual
                for cid in ({h1_id} | set(arm_measured))
                if (residual := _residual_of(cid, measured, expectations)) is not None
            ]
            mean_residual = float(np.mean(arm_residuals)) if arm_residuals else None
            best_id = h1_id
            control = measured[h0_id]
        else:
            h1_id, h0_id = None, None
            control = measured.get(slot5) if slot5 else None
        control_residual = _residual_of(h0_id or slot5, measured, expectations) if (h0_id or slot5) else None
        # The residual check below is **relative to the control**, not absolute.
        # The surrogate fits on the revealed rows, so a batch it has not seen yet
        # can be uniformly over-predicted: every measured residual goes negative
        # together, and an absolute `mean_residual >= 0` gate then refutes a
        # hypothesis the paired arms just confirmed. That is a level shift in the
        # model, not evidence against the rule — and it is exactly what happened
        # in R5 of the seed-21 run, where the arm beat the control by +0.748 and
        # was still stamped FALSIFIED.
        residual_tolerance = max(0.0, float(control_residual)) + RESIDUAL_LEVEL_TOLERANCE if control_residual is not None else None
        # A decisive win also overrides any residual doubt: if the arm beat the
        # control by a margin the assay's own noise cannot produce, the lift is
        # real regardless of where the batch sat relative to the surrogate.
        decisive_lift = control is not None and observed is not None and observed > control + DECISIVE_LIFT
        median = float(np.median(list(measured.values()))) if measured else float("nan")
        expected = expectations.get(best_id) if best_id else None
        paired = observed is not None and control is not None
        arm_label = "/".join(arm_ids) if arm_ids else "unrecorded"

        # How the round is described, so a UI never has to guess which comparison
        # produced the verdict. ``claimed_pair`` = the planner's own named
        # comparison was measured and graded; ``arm_vs_control`` = the fallback.
        design = "claimed pair (planner slot 4 vs slot 5)" if declared_pair_measured else "paired H1 (slots 1/2/4) vs H0 control (slot 5)"
        h1_label = "H1 candidate" if declared_pair_measured else "best H1 arm pick"
        h0_label = "H0 candidate" if declared_pair_measured else "H0 control"
        control_id = h0_id or slot5

        if observed is None:
            verdict = INCONCLUSIVE
            reason = f"no H1 arm candidate ({arm_label!r}) was measured this round"
        elif paired and control > observed:
            verdict = FALSIFIED
            reason = (f"{h0_label} {control_id} ({control:.3f}) beat {h1_label} "
                      f"{best_id} ({observed:.3f})")
        elif paired and observed > control and decisive_lift:
            verdict = SUPPORTED
            reason = (f"{h1_label} {best_id} ({observed:.3f}) beat {h0_label} "
                      f"{control_id} ({control:.3f}) by {observed - control:+.3f} — decisive "
                      f"(>= {DECISIVE_LIFT:.2f}), overriding residual doubt")
        elif (
            paired
            and observed > control
            and mean_residual is not None
            and residual_tolerance is not None
            and mean_residual < residual_tolerance
        ):
            verdict = FALSIFIED
            reason = (f"{h1_label} {best_id} ({observed:.3f}) beat {h0_label} {control_id} "
                      f"({control:.3f}) but the arm's mean residual ({mean_residual:+.3f}) "
                      f"trailed the control's ({control_residual:+.3f}) by more than "
                      f"{RESIDUAL_LEVEL_TOLERANCE:.2f}: the lift was already predicted")
        elif paired and observed > control:
            verdict = SUPPORTED
            reason = (f"{h1_label} {best_id} ({observed:.3f}) beat {h0_label} "
                      f"{control_id} ({control:.3f}); arm residual {mean_residual:+.3f} is "
                      f"level with the batch" if mean_residual is not None else
                      f"{h1_label} {best_id} ({observed:.3f}) beat {h0_label} "
                      f"{control_id} ({control:.3f})")
        elif not np.isnan(median) and observed >= median:
            verdict = SUPPORTED
            reason = (f"paired arms tied; {h1_label} {best_id} ({observed:.3f}) reached "
                      f"the round median ({median:.3f})")
        elif not np.isnan(median):
            verdict = FALSIFIED
            reason = (f"paired arms tied; {h1_label} {best_id} ({observed:.3f}) fell "
                      f"below the round median ({median:.3f})")
        elif expected is not None:
            hit = observed >= float(expected)
            verdict = SUPPORTED if hit else FALSIFIED
            reason = (f"{h1_label} {best_id} ({observed:.3f}) "
                      f"{'met' if hit else 'missed'} its pre-query expectation ({float(expected):.3f})")
        else:
            verdict = INCONCLUSIVE
            reason = "no control, median or pre-query expectation available"

        inverted: set[int] = set()
        residual_by_position: dict[int, float] = {}
        if verdict == FALSIFIED:
            tested: set[int] = set()
            for cid in (arm_ids or [slot4]):
                tested |= _tested_positions(decision, sequences.get(str(cid)) if cid else None)
            if expectations:
                # Residual gate: the wet lab is the arbiter, and a position whose
                # carriers measured *better* than predicted is never inverted.
                inverted, residual_by_position = _residual_gated_positions(
                    tested,
                    priors=decision.get("positional_priors") or [],
                    arm_ids=[cid for cid in (arm_ids or [slot4]) if cid],
                    sequences=sequences,
                    measured=measured,
                    expectations=expectations,
                )
            else:
                # Legacy ledger with no pre-query expectation: no residual exists,
                # so fall back to the tested-positions rule rather than silently
                # making the round uninvertable.
                inverted = tested
            falsified_positions |= inverted

        verdicts.append({
            "hypothesis_id": decision.get("hypothesis_id"),
            "round_number": round_no,
            "verdict": verdict,
            "reason": reason,
            "design": design,
            "compared": "claimed_pair" if declared_pair_measured else "arm_vs_control",
            "statement": decision.get("hypothesis_statement", ""),
            "h1_arm_ids": [str(cid) for cid in arm_ids],
            "h1_target_id": best_id if best_id is not None else slot4,
            "h0_control_id": control_id,
            "round_median": None if np.isnan(median) else round(median, 3),
            "h1_binding": None if observed is None else round(observed, 3),
            "h1_arm_mean_binding": None if arm_mean is None else round(arm_mean, 3),
            "h1_mean_residual": None if mean_residual is None else round(mean_residual, 3),
            "h0_binding": None if control is None else round(control, 3),
            # The residual comparison the verdict actually used: the arm's mean
            # residual against the control's, so a UI can show *why* a round with
            # a positive paired lift was still graded on residual grounds.
            "h0_residual": None if control_residual is None else round(control_residual, 3),
            "residual_tolerance": (
                None if residual_tolerance is None else round(residual_tolerance, 3)
            ),
            "decisive_lift": bool(decisive_lift),
            "paired_lift": (
                None if observed is None or control is None else round(observed - control, 3)
            ),
            "expected_binding": None if expected is None else round(float(expected), 3),
            # The ids the *hypothesis statement* names, and the ids this verdict
            # actually graded. The UI shows both so an ungradeable claim is visible
            # rather than implied.
            "claim_about": [cid for cid in (declared_h1, declared_h0) if cid],
            "graded_on": sorted({str(cid) for cid in (best_id, control_id) if cid}),
            "inverted_positions": sorted(inverted),
            "position_residuals": {
                str(position): round(value, 3)
                for position, value in sorted(residual_by_position.items())
            },
        })
    return {
        "tool": "audit_past_hypotheses",
        "design": "paired H1 (slots 1/2/4) / H0 (slot 5)",
        "verdicts": verdicts,
        "falsified_positions": sorted(falsified_positions),
    }


def compute_cost_and_effort_saved(
    ledger_df: pd.DataFrame,
    assay_cost_usd: int = 1500,
    full_pool: int = 419,
) -> dict:
    """Cost saved against an exhaustive screen of ``full_pool`` candidates.

    ``full_pool`` defaults to the frozen 1N8Z pool (419), so the live campaign's
    figure is unchanged. It is a parameter because a cross-target campaign that
    kept reporting "saved vs 419" would be quoting another dataset's denominator:
    aayl49_ML has 1,510 candidates, and its saving is 3.6x larger for the same
    spend.
    """
    queried = int(ledger_df.candidate_id.nunique()) if "candidate_id" in ledger_df else 0
    return {"tool": "compute_cost_and_effort_saved", "queries": queried,
            "spent_usd": queried * assay_cost_usd, "full_pool_cost_usd": full_pool * assay_cost_usd,
            "saved_vs_exhaustive_usd": max(0, (full_pool - queried) * assay_cost_usd),
            "assay_days_parallel_batches": int(np.ceil(queried / 5)),
            "full_pool": int(full_pool)}


TOOL_REGISTRY: dict[str, Callable] = {fn.__name__: fn for fn in (
    scan_positional_sar, detect_epistasis_clashes, profile_developability_liabilities,
    inspect_joint_uncertainty_hotspots, audit_past_hypotheses, compute_cost_and_effort_saved,
)}


# ── Option C: two-tier SOP baselines + state-triggered drills ────────────────
#: Tier labels. Rendered verbatim by the UI, so they are stable strings rather
#: than an enum: a new tier can be added without migrating a ledger.
TIER_SOP = "SOP Baseline"
TIER_DRILL = "State-Triggered Drill"

#: Which council seats each tool is routed to. The planner is the synthesizer and
#: receives every active tool's output; the scientists see only their own.
TOOL_ROUTING: dict[str, tuple[str, ...]] = {
    "scan_positional_sar": ("Affinity Scientist", "Experimental Planner"),
    "profile_developability_liabilities": ("Developability Scientist", "Experimental Planner"),
    "inspect_joint_uncertainty_hotspots": ("Scientific Skeptic", "Experimental Planner"),
    "audit_past_hypotheses": ("Scientific Skeptic", "Experimental Planner"),
    "detect_epistasis_clashes": ("Affinity Scientist", "Scientific Skeptic"),
    "compute_cost_and_effort_saved": ("Experimental Planner",),
}

#: Tier per tool. The two SOP baselines are the evidence every round is required
#: to gather; the four drills are diagnostics the campaign state has to justify.
TOOL_TIERS: dict[str, str] = {
    "scan_positional_sar": TIER_SOP,
    "profile_developability_liabilities": TIER_SOP,
    "inspect_joint_uncertainty_hotspots": TIER_DRILL,
    "audit_past_hypotheses": TIER_DRILL,
    "detect_epistasis_clashes": TIER_DRILL,
    "compute_cost_and_effort_saved": TIER_DRILL,
}

#: Display order for the tool panel, and the order ``select_active_tools``
#: returns. Stable so a UI can diff two rounds positionally and highlight the
#: rows that flipped.
ACTIVE_TOOL_ORDER: tuple[str, ...] = (
    "scan_positional_sar",
    "profile_developability_liabilities",
    "inspect_joint_uncertainty_hotspots",
    "audit_past_hypotheses",
    "detect_epistasis_clashes",
    "compute_cost_and_effort_saved",
)

#: ``sar_maturity`` at which the epistasis drill may open before the query count
#: does. A table this converged already carries the co-mutation evidence the drill
#: exists to find, so waiting for batch 2 solely on a query count would be
#: withholding a measurement the campaign has already earned.
EPISTASIS_MATURITY_TRIGGER = 0.30

#: ``VoI`` at or below which the cost drill opens: the campaign has entered its
#: payoff regime, so "what has this loop saved versus an exhaustive screen?" stops
#: being a forecast and becomes a reportable number.
COST_PAYOFF_VOI_TRIGGER = 0.35


def _sop_reason(round_number: int) -> tuple[bool, str]:
    """Tier 1 predicate: on for every planning round, off for the seed batch.

    Round 0 is the five-query seed draw. No council runs and no batch is designed,
    so claiming the SOP baselines were *routed* there would be false — the rounds
    the request names are 1..6.
    """
    if round_number < 1:
        return False, "Deactivated: seed round has no council planning step"
    return True, f"Active: mandatory SOP baseline (R{round_number})"


#: Reason emitted for every drill on the seed batch. Round 0 has no council, and
#: its ``voi``/``sar_maturity`` are ``empty_state`` placeholders rather than
#: measurements — ``voi = 0.0`` there would otherwise satisfy the cost drill's
#: ``voi <= 0.35`` clause and report a "payoff regime" for a campaign that has not
#: designed a single batch. A drill is a diagnostic the council chooses to run, so
#: no council means no drill.
_NO_COUNCIL = "Deactivated: seed round has no council (no batch designed yet)"


def _hotspot_reason(round_number: int, voi: float) -> tuple[bool, str]:
    """Tier 2: joint-uncertainty hotspots are worth inspecting only if VoI > 0.

    ``VoI == 0`` means no future query can be re-decided against what this batch
    would learn — the terminal harvest. Inspecting uncertainty there would rank
    candidates nothing can still act on.
    """
    if round_number < 1:
        return False, _NO_COUNCIL
    if voi > 0.0:
        return True, f"Active: VoI = {voi:.4f} > 0"
    return False, f"Deactivated: Terminal harvest (VoI = {voi:.4f})"


def _audit_reason(round_number: int) -> tuple[bool, str]:
    """Tier 2: there is nothing to audit before one round has completed."""
    if round_number < 1:
        return False, _NO_COUNCIL
    if round_number >= 2:
        return True, f"Active: {round_number - 1} past decision(s) to audit (R{round_number} >= R2)"
    return False, "Deactivated: no completed round to audit yet (R1 < R2)"


def _epistasis_reason(
    round_number: int, queries_spent: int, sar_maturity: float
) -> tuple[bool, str]:
    """Tier 2: co-mutation residuals need two observations per position.

    The gate is ``queries_spent >= 10``, which — counting the seed batch — is
    exactly ``round_number >= 2``, the round the request names. ``sar_maturity``
    is accepted and reported for context but deliberately **not** an alternative
    trigger: the golden ledger already reports ``sar_maturity = 0.65`` in Round 1,
    so an ``or sar_maturity >= 0.30`` clause would open the drill in the very round
    the request says it must be closed. What the drill actually consumes is
    co-mutation evidence, and that is a function of how many variants have been
    measured — not of how converged the table already is.
    """
    if round_number < 1:
        return False, _NO_COUNCIL
    if queries_spent >= 10:
        return True, (
            f"Active: {queries_spent} queries spent >= 10 (R{round_number} >= R2)"
        )
    return (
        False,
        f"Deactivated: only {queries_spent} queries spent < 10 "
        f"(R{round_number} < R2); SAR maturity {sar_maturity:.4f} is not a trigger",
    )


def _cost_reason(round_number: int, voi: float) -> tuple[bool, str]:
    """Tier 2: cost saved is a payoff-regime report, not a tuition-phase forecast."""
    if round_number < 1:
        return False, _NO_COUNCIL
    if round_number >= 3:
        return True, f"Active: payoff regime (R{round_number} >= R3)"
    if voi <= COST_PAYOFF_VOI_TRIGGER:
        return (
            True,
            f"Active: payoff regime (VoI = {voi:.4f} <= {COST_PAYOFF_VOI_TRIGGER:.2f})",
        )
    return (
        False,
        f"Deactivated: tuition regime (R{round_number} < R3 and VoI = {voi:.4f} "
        f"> {COST_PAYOFF_VOI_TRIGGER:.2f})",
    )


#: ``name -> predicate``. Each predicate returns ``(active, trigger_reason)`` and
#: is the *single* definition of that tool's trigger; the UI, the runner and the
#: tests all read the flag this produces, so a reason string can never disagree
#: with the boolean beside it.
TOOL_TRIGGERS: dict[str, Callable[..., tuple[bool, str]]] = {
    "scan_positional_sar": _sop_reason,
    "profile_developability_liabilities": _sop_reason,
    "inspect_joint_uncertainty_hotspots": _hotspot_reason,
    "audit_past_hypotheses": _audit_reason,
    "detect_epistasis_clashes": _epistasis_reason,
    "compute_cost_and_effort_saved": _cost_reason,
}


def select_active_tools(
    round_number: int,
    queries_spent: int,
    voi: float,
    sar_maturity: float,
) -> dict:
    """Option C: partition the six tools into SOP baselines and triggered drills.

    Deterministic and pure — the same four numbers always produce the same table,
    so the live ``/api/state`` and a ``/api/state?round=k`` reconstruction cannot
    disagree, and a test can recompute every flag from its own ``trigger_reason``.

    Two tiers:

    **Tier 1 — SOP baselines**, active in every planning round (1..6) and off only
    on the seed batch, which has no council::

        scan_positional_sar             -> Affinity Scientist, Experimental Planner
        profile_developability_liabilities
                                        -> Developability Scientist, Experimental Planner

    **Tier 2 — state-triggered drills**, each opened by a campaign-state predicate::

        inspect_joint_uncertainty_hotspots   voi > 0            (deactivates at the
                                                                 terminal harvest)
        audit_past_hypotheses                round_number >= 2
        detect_epistasis_clashes             queries_spent >= 10
                                             or sar_maturity >= 0.30
        compute_cost_and_effort_saved        round_number >= 3 or voi <= 0.35

    ``queries_spent`` must be the **planning prefix** — the ledger the council saw
    when it designed this round, i.e. the reveals through ``round_number - 1``, not
    through ``round_number``. Passing the round's own spend would open the
    epistasis drill one batch early in every campaign.

    ``voi`` and ``sar_maturity`` are expected pre-rounded to the precision the API
    exports them at (4 dp), because a raw ``-1e-17`` residual would otherwise
    deactivate the hotspot drill in the live view while the reconstructed view,
    reading a rounded ``0.0``, deactivated it by the terminal-harvest branch. One
    rounding rule, applied by the caller, keeps both paths byte-identical.
    """
    # Every predicate receives ``round_number`` first: the seed batch has no
    # council, and on it ``voi == 0.0`` is an ``empty_state`` placeholder rather
    # than a measured payoff signal. Without the guard that placeholder satisfies
    # the cost drill's ``voi <= 0.35`` clause and reports a payoff regime for a
    # campaign that has not designed a batch.
    arguments = {
        "scan_positional_sar": (int(round_number),),
        "profile_developability_liabilities": (int(round_number),),
        "inspect_joint_uncertainty_hotspots": (int(round_number), float(voi)),
        "audit_past_hypotheses": (int(round_number),),
        "detect_epistasis_clashes": (
            int(round_number), int(queries_spent), float(sar_maturity),
        ),
        "compute_cost_and_effort_saved": (int(round_number), float(voi)),
    }
    tools = []
    for name in ACTIVE_TOOL_ORDER:
        active, reason = TOOL_TRIGGERS[name](*arguments[name])
        tools.append({
            "name": name,
            "tier": TOOL_TIERS[name],
            "active": bool(active),
            "trigger_reason": reason,
            "routed_to": list(TOOL_ROUTING[name]),
        })
    return {
        "round_number": int(round_number),
        "queries_spent": int(queries_spent),
        "voi": float(voi),
        "sar_maturity": float(sar_maturity),
        "active": [tool["name"] for tool in tools if tool["active"]],
        "tools": tools,
    }


#: Decimal places ``/api/state`` exports ``voi`` and ``sar_maturity`` at. The
#: tool table is derived from the *exported* values, never the raw floats, so the
#: live view and a round-``k`` reconstruction cannot disagree: the live campaign
#: holds a raw ``voi`` of ~``-1e-17`` on the final batch while the reconstruction
#: reads the rounded ``0.0`` off the persisted batch design, and the hotspot drill
#: would otherwise be active in one and inactive in the other.
VOI_EXPORT_DIGITS = 4


def active_tools_for_state(
    *, round_number: int, queries_spent: int, voi: float, sar_maturity: float
) -> dict:
    """The exported ``active_tools_metadata`` block for one round.

    Single call site for the rounding rule documented on :data:`VOI_EXPORT_DIGITS`,
    so ``empty_state``, the live ``_refresh_state`` path and ``state_at`` all
    produce byte-identical tables for the same round.
    """
    return select_active_tools(
        round_number=int(round_number),
        queries_spent=int(queries_spent),
        voi=round(float(voi), VOI_EXPORT_DIGITS),
        sar_maturity=round(float(sar_maturity), VOI_EXPORT_DIGITS),
    )
