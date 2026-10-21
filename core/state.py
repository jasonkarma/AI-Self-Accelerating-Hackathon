"""Exact API-contract state builders."""
from __future__ import annotations
from .constants import ANCHOR_POSITIONS, WT_SEQUENCE

#: Positions the ribbon renders, in display order. A subset of the 13 HCDR3
#: positions: the four structural anchors (3/11/12/13) plus the positions the
#: council has historically argued about (1, 6, 8).
RIBBON_POSITIONS: tuple[int, ...] = (1, 3, 6, 8, 11, 12, 13)


def empty_voi_diagnostics() -> dict:
    """Batch-level view of the VoI-annealed objective, before any round has run.

    ``voi`` is the single number that blends early exploration with pure
    qualified-binding exploitation (``exploit_weight = 1 - 0.25 * VoI``, see
    :data:`core.acquisition.VOI_EXPLOIT_WEIGHT`); ``horizon`` is the
    remaining-horizon factor and ``sar_maturity`` the empirically corroborated
    fraction of the hierarchical SAR table. ``exploit_weight`` is exported
    explicitly so the UI can show the mix without re-deriving the formula.

    ``active_tools_metadata`` carries the Option C tool arsenal (SOP baselines
    versus state-triggered drills) for the same round. It lives *inside* this
    object rather than at the top level because ``/api/state``'s top-level key set
    is frozen by ``api_contract.json`` and asserted for exact equality in
    ``tests/test_core.py``; ``voi_diagnostics`` is the additive-only container, and
    VoI is what triggers half the drills. It is computed by
    :func:`core.tool_arsenal.select_active_tools` once the round's numbers exist;
    the empty table here is the seed-round shape.
    """
    from .tool_arsenal import active_tools_for_state

    return {
        "round_number": 0,
        "voi": 0.0,
        "horizon": 0.0,
        "sar_maturity": 0.0,
        "exploit_weight": 0.0,
        "exploration_weight": 0.0,
        "slot_split": "3-exploit / 2-experimental",
        "position_tolerance_effects": [],
        # First-principles provenance. The UI must never present ``a_null`` as
        # something ``exploit_weight`` steers: the VoI anneal was retired and the
        # objective is ``a_null := a_base`` on every path, so the weight is a
        # *diagnostic* of how much headroom the anneal would have had, not a
        # coefficient in the shipped objective. See
        # :func:`voi_objective_provenance` for why this is data and not a comment.
        **voi_objective_provenance(voi=0.0, sar_maturity=0.0),
        "active_tools_metadata": active_tools_for_state(
            round_number=0, queries_spent=0, voi=0.0, sar_maturity=0.0
        ),
    }


def voi_objective_provenance(
    *,
    voi: float,
    sar_maturity: float,
    protocol=None,
) -> dict:
    """The three coefficients that explain *where the objective comes from*.

    Added for the frontend showcase, where a chart that plots ``exploit_weight``
    next to the selected candidates invites the reading "this weight steers the
    picks". It does not: ``a_null`` was shipped as ``a_base`` at ``e567cd4`` and
    the ``no_anneal`` ablation is the evidence. So the weight is exported with an
    explicit ``steers_null_objective: False`` rather than left to inference.

    ``eta_sar`` is :func:`core.acquisition.additive_prior_trust` — how much the
    empirical SAR prior is trusted given this pool and budget — and
    ``sar_maturity_target`` is the denominator ``sar_maturity`` is measured
    against. Both are derived from ``t=0`` observables, which is what makes them
    reportable without leaking a label.

    ``slot_graduation`` is read from the *same* function the selector calls
    (:func:`core.acquisition.slot_graduation`), so the badge cannot disagree with
    the batch that was actually chosen. It needs the measured ``Δ_agent`` spread,
    which this builder does not have, so it is left ``None`` and filled in by the
    caller that does (``Campaign._voi_diagnostics``).
    """
    eta_sar = None
    target = None
    if protocol is not None:
        # ``Campaign.protocol`` is the *frozen JSON dict* from
        # ``config/frozen_protocol.json``, not a ``CampaignProtocol``: it carries
        # ``qualified_count`` / ``tau_dev`` but no ``seq_len`` or ``total_budget``,
        # so the two coefficients cannot be read off it directly. Coerce it into
        # the dataclass first, defaulting the protocol-shaped fields the frozen
        # 1N8Z file predates. Without this the badges are silently ``None``.
        try:
            from .acquisition import CampaignProtocol, REFERENCE_N_QUALIFIED

            if hasattr(protocol, "additive_prior_trust"):
                campaign_protocol = protocol
            else:
                campaign_protocol = CampaignProtocol(
                    n_qualified=int(
                        protocol.get("qualified_count") or REFERENCE_N_QUALIFIED
                    ),
                    seq_len=int(protocol.get("seq_len") or 13),
                )
            eta_sar = float(campaign_protocol.additive_prior_trust())
            target = float(campaign_protocol.sar_maturity_target)
        except Exception:  # noqa: BLE001 - a frozen protocol may not round-trip
            eta_sar, target = None, None
    return {
        "eta_sar": eta_sar,
        "sar_maturity_target": target,
        "slot_graduation": None,
        "steers_null_objective": False,
        "objective": "a_null := a_base",
        "objective_note": (
            "exploit_weight / exploration_weight report how much exploration the "
            "*anneal* would have done at this VoI. They are diagnostics: the "
            "shipped objective is a_null := a_base, and the no_anneal ablation is "
            "the evidence. Do not label them as coefficients of a_null."
        ),
    }


def hcdr3_ribbon(
    priors: list[dict] | list | None = None,
    falsified_positions: set[int] | frozenset[int] = frozenset(),
) -> list[dict]:
    """Per-position ribbon: declared prior weight plus its evidentiary status.

    The contract key ``hcdr3_ribbon`` used to be a constant — every weight was
    ``0.0`` and every non-anchor status ``neutral`` — so the UI could not show
    that a position had actually been argued about or refuted. It is derived here
    from the round's **positional priors** (what the council declared or what the
    data-derived set reached the selector) and the audit's ``falsified_positions``
    (the self-correction signal), which makes it a pure function of the ledger.

    ``status`` is one of the contract's four values: a structural anchor stays
    ``anchor_conserved``; a position the paired audit refuted is
    ``epistasis_alert``; a position a prior actively prefers is
    ``low_risk_swap``; anything else is ``neutral``.
    """
    declared: dict[int, float] = {}
    for prior in priors or []:
        if hasattr(prior, "model_dump"):
            prior = prior.model_dump()
        if not isinstance(prior, dict):
            continue
        position = prior.get("position")
        if not position:
            continue
        position = int(position)
        weight = float(prior.get("weight_delta") or 0.0)
        # One number per position. A position declared twice keeps its *strongest*
        # claim rather than their sum: summing would let a planner inflate a
        # position simply by restating the same rule.
        if abs(weight) > abs(declared.get(position, 0.0)):
            declared[position] = weight
    inverted = {int(p) for p in falsified_positions or ()}
    ribbon = []
    for position in RIBBON_POSITIONS:
        declared_weight = declared.get(position, 0.0)
        # An inverted rule is shown *with the sign it now carries*, because the
        # selector really does push those residues down.
        weight = -declared_weight if position in inverted else declared_weight
        if position in ANCHOR_POSITIONS:
            status = "anchor_conserved"
        elif position in inverted:
            status = "epistasis_alert"
        elif declared_weight != 0.0:
            status = "low_risk_swap"
        else:
            status = "neutral"
        ribbon.append({
            "position": position,
            "wt_residue": WT_SEQUENCE[position - 1],
            "soft_prior_weight": round(weight, 3),
            "status": status,
        })
    return ribbon


def empty_state(benchmark: dict | None = None) -> dict:
    # SPEC §7: the contract's required trio (m1/m3/m5) plus m2/m4 for the UI.
    # Missing keys degrade to None rather than raising, so a partially written
    # benchmark artifact cannot crash ``GET /api/state``.
    curves = []
    for row in (benchmark or {}).get("curves", []):
        curves.append({
            key: row.get(key)
            for key in ("query_count", "m1_random", "m2_greedy", "m3_ucb", "m4_fixed_prior", "m5_hybrid")
        })
    return {"current_round": 0, "phase": "Cold-Start Tuition", "kpis": {"best_qualified_binding": 0.0, "delta_vs_wt": 0.0, "query_budget_spent": 0, "total_budget": 35, "wet_lab_cost_saved_usd": 0, "hypotheses_supported": 0, "hypotheses_falsified": 0},
            "hcdr3_ribbon": hcdr3_ribbon(),
            #: The planner's decision record, or ``{}`` for the seed batch — which
            #: has no planning step, because five candidates are drawn at random
            #: before any evidence exists. Present from the first round onward.
            "planner": {},
            "agent_council": [], "batch_slots": [], "pareto_data": [], "benchmark_curves": curves,
            "voi_diagnostics": empty_voi_diagnostics()}
