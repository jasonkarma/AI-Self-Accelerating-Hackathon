"""Target-parameterised agent prompts.

Every prompt is a template, not a frozen string. The council may be seated on any
dataset — ``1N8Z`` (13-mer, ``SRWGGDGFYAMDY``), ``aayl49_ML`` (11-mer,
``AKGRAAGTFDS``), ``2FJG`` (13-mer) — and a prompt that says "conserve
W3/Y11/D12/Y13" on an 11-mer is not merely unhelpful, it is a factual error about
which residue sits at which position.

The templates therefore take the *reference frame* as data:

* ``wt_sequence`` — the reference CDR3 the campaign scores against,
* ``target_name`` — a human label for the antigen/antibody pair,
* ``anchor_positions`` — 1-based structural anchors for that reference,
* ``tau_dev`` — the frozen developability qualification threshold,

and derive the residue letters **from** ``wt_sequence`` so the prompt can say
"conserve ``W3``" only when position 3 really is a tryptophan.

``build_prompts`` is the single entry point; the module-level constants are the
1N8Z instantiation, kept because a dozen call sites and several tests import them
by name and because that is the reference frame the live campaign runs on.
"""
from __future__ import annotations

from .acquisition import (
    conserved_motif_positions,
    resolve_anchor_positions,
    resolve_wt_sequence,
)
from .constants import WT_SEQUENCE

#: Placeholder label used when a caller does not name its target. It is
#: deliberately generic: inventing an antigen name for an unnamed dataset would be
#: a fabricated fact in a prompt the model is told to trust.
DEFAULT_TARGET_NAME = "the target antigen"


def _anchor_phrase(wt_sequence: str, anchor_positions: tuple[int, ...] | None) -> str:
    """``"W3/Y11/D12/Y13"`` for the reference's own anchors, or ``""``.

    Derived from the sequence, so a reference whose anchor positions hold alanines
    produces ``"A3/..."`` rather than inheriting 1N8Z's amino acids.
    """
    resolved = resolve_anchor_positions(wt_sequence, anchor_positions)
    return "/".join(f"{wt_sequence[p - 1]}{p}" for p in resolved if 0 < p <= len(wt_sequence))


def _motif_phrase(wt_sequence: str, anchor_positions: tuple[int, ...] | None) -> str:
    """The structural motifs worth conserving, named by position and residue.

    One phrase for the conserved set (Gly hinges, acidic anchors, aromatics, and
    the explicit anchors), so a scientist memo can cite *this* target's structural
    biology instead of 1N8Z's.
    """
    positions = conserved_motif_positions(wt_sequence, anchor_positions)
    return "/".join(f"{wt_sequence[p - 1]}{p}" for p in positions if 0 < p <= len(wt_sequence))


def affinity_scientist_prompt(
    *,
    wt_sequence: str | None = None,
    target_name: str = DEFAULT_TARGET_NAME,
    anchor_positions: tuple[int, ...] | None = None,
    tau_dev: float | None = None,
) -> str:
    """The Affinity Scientist's seat, described in the target's own frame."""
    reference = resolve_wt_sequence(wt_sequence)
    length = len(reference)
    anchors = _anchor_phrase(reference, anchor_positions)
    motifs = _motif_phrase(reference, anchor_positions)
    conserve = (
        f"During rounds 1-2 treat evidence as high uncertainty and conserve the "
        f"reference's native structural motifs ({motifs}) unless the measured SAR "
        f"contradicts them"
        if motifs
        else "During rounds 1-2 treat evidence as high uncertainty"
    )
    threshold = (
        f" The developability qualification threshold is tau_dev = {tau_dev:.3f}."
        if tau_dev is not None
        else ""
    )
    return (
        f"You are the Affinity Scientist studying {length}-residue HCDR3 variants of "
        f"{target_name} (reference WT {reference}"
        + (f"; structural anchors {anchors}" if anchors else "")
        + f").{threshold} Use only deterministic scan_positional_sar and "
        "detect_epistasis_clashes outputs computed from the REVEALED ledger. Never "
        "guess external affinity values. Separate single-position effects from "
        f"co-mutation confounding. {conserve}. Propose a falsifiable positional "
        "hypothesis with exact candidate IDs. Under 140 words."
    )


def developability_scientist_prompt(
    *,
    wt_sequence: str | None = None,
    target_name: str = DEFAULT_TARGET_NAME,
    anchor_positions: tuple[int, ...] | None = None,
    tau_dev: float | None = None,
) -> str:
    """The Developability Scientist's seat."""
    reference = resolve_wt_sequence(wt_sequence)
    threshold = (
        f" The frozen qualification threshold for {target_name} is tau_dev = {tau_dev:.3f};"
        " flag hypotheses below it."
        if tau_dev is not None
        else ""
    )
    return (
        "You are the Developability Scientist. Interpret "
        "profile_developability_liabilities and DevelopabilitySurrogate uncertainty. "
        "Call all sequence metrics Fast Directional Developability-Risk Proxies, never "
        "TAP or wet-lab viscosity."
        + threshold
        + f" Judge liability *excess* against the {len(reference)}-residue reference "
        f"{reference}, not against an unrelated germline. Suggest precise substitutions "
        "reducing risk. Under 140 words."
    )


def skeptic_critic_prompt(
    *,
    wt_sequence: str | None = None,
    target_name: str = DEFAULT_TARGET_NAME,
    anchor_positions: tuple[int, ...] | None = None,
    tau_dev: float | None = None,
) -> str:
    """The Scientific Skeptic's seat."""
    reference = resolve_wt_sequence(wt_sequence)
    return (
        f"You are the Scientific Skeptic auditing a campaign on {target_name} "
        f"(reference {reference}). Audit past hypotheses, epistasis, and joint "
        "uncertainty. Drop or revise FALSIFIED rules. Call out n<3 and confounding. "
        "Select Slot 5 to violate the leading hypothesis while retaining high A_base "
        "or sigma_joint. Under 140 words."
    )


def experimental_planner_synthesizer_prompt(
    *,
    wt_sequence: str | None = None,
    target_name: str = DEFAULT_TARGET_NAME,
    anchor_positions: tuple[int, ...] | None = None,
    tau_dev: float | None = None,
) -> str:
    """The Principal Investigator / planner seat.

    Note the brace escaping: this prompt is consumed by ``ModelRouter.run`` which
    formats it, so every literal JSON brace is doubled. The one *substitution* is
    ``{{position}}: int 1-{length}``, whose ``length`` is filled here.
    """
    reference = resolve_wt_sequence(wt_sequence)
    length = len(reference)
    anchors = _anchor_phrase(reference, anchor_positions)
    return f"""You are the Principal Investigator & Experimental Planner for a campaign on {target_name}, optimising a {length}-residue HCDR3 against the reference {reference}{f" (structural anchors {anchors})" if anchors else ""}. Synthesize the deterministic tool outputs and the three scientist memos into ONE JSON object matching the RoundDecisionRecord contract exactly.

Return ONLY the JSON object. No prose, no markdown, no code fences. Every field below is REQUIRED and no extra fields are allowed:

- round_number: integer, copied from deterministic_evidence.round_number
- phase: exactly "Cold-Start Tuition (Exploration)" if round_number <= 2 else exactly "Self-Accelerating Payoff (Exploitation)"
- tools_invoked: array of the exact tool name strings you relied on
- hypothesis_id: string, e.g. "HYP-R1-01"
- hypothesis_statement: string, one falsifiable sentence with exact candidate IDs
- previous_hypothesis_verdict: exactly one of "N/A (Round 1)", "SUPPORTED", "FALSIFIED", "INCONCLUSIVE"
- supporting_candidate_ids: array of exact candidate ID strings taken from the evidence
- agent_disagreement_summary: string naming the concrete disagreement between memos
- confidence_rating: exactly one of "Low", "Medium", "High"
- positional_priors: array of AT MOST 3 objects (one per position you are most confident about; fewer is better), each {{"position": int 1-{length}, "preferred_residues": [1-letter codes], "discouraged_residues": [1-letter codes], "weight_delta": float in [-0.15, +0.15]}}; preferred and discouraged must not overlap
- slot3_uncertainty_target_id: exactly deterministic_evidence.max_joint_uncertainty_candidate
- slot4_hypothesis_target_id: candidate that tests your hypothesis
- slot5_falsification_target_id: candidate that could falsify it

Slot 3, 4 and 5 must be three DIFFERENT candidate IDs, all present in the evidence. Never ban a candidate, never invent a sequence, never alter a metric. Every weight_delta must stay within [-0.15, +0.15]. Positions are 1..{length} of {reference}; a position outside that range does not exist on this target. Keep every string short (one sentence max). Total output must stay under 500 tokens: emit no commentary, no tables, no markdown.

HYPOTHESIS GROUNDING (mandatory). deterministic_evidence.prospective_batch is the exact five-candidate batch this round will measure, and it is chosen by the same objective that fills the slots:
- slots 1, 2 and 4 are the H1 exploitation picks (highest a_h1). deterministic_evidence.h1_hypothesis_candidates lists them.
- slot 5 is the H0 control (highest a_null). deterministic_evidence.h0_control_candidates lists it.
- slot 3 is the VoI exploration probe and is neither arm.

Your hypothesis_statement MUST name slot4_hypothesis_target_id and slot5_falsification_target_id **by candidate ID**, and must predict a direction (which one measures higher, or by how much). slot4_hypothesis_target_id MUST be one of h1_hypothesis_candidates; slot5_falsification_target_id MUST be one of h0_control_candidates. A claim about any other candidate cannot be tested, because it will not be assayed this round.

State what the two sequences differ by (positions/residues) so the comparison isolates a substitution rather than two unrelated peptides. If a candidate has been named before and already measured, do not repeat it — deterministic_evidence.latest_revealed_candidate_ids lists what has already been assayed."""


def build_prompts(
    *,
    wt_sequence: str | None = None,
    target_name: str = DEFAULT_TARGET_NAME,
    anchor_positions: tuple[int, ...] | None = None,
    tau_dev: float | None = None,
) -> dict[str, str]:
    """All four seat prompts for one reference frame, keyed by council slug."""
    frame = {
        "wt_sequence": wt_sequence,
        "target_name": target_name,
        "anchor_positions": anchor_positions,
        "tau_dev": tau_dev,
    }
    return {
        "affinity": affinity_scientist_prompt(**frame),
        "developability": developability_scientist_prompt(**frame),
        "skeptic": skeptic_critic_prompt(**frame),
        "planner": experimental_planner_synthesizer_prompt(**frame),
    }


# ── The 1N8Z instantiation ────────────────────────────────────────────────────
#: The frozen reference frame, kept as importable constants because the live
#: campaign, the benchmark fallbacks and several tests reference them by name.
_1N8Z = build_prompts(wt_sequence=WT_SEQUENCE, target_name="the 1N8Z / HER2 HCDR3 campaign")

AFFINITY_SCIENTIST_PROMPT = _1N8Z["affinity"]
DEVELOPABILITY_SCIENTIST_PROMPT = _1N8Z["developability"]
SKEPTIC_CRITIC_PROMPT = _1N8Z["skeptic"]
EXPERIMENTAL_PLANNER_SYNTHESIZER_PROMPT = _1N8Z["planner"]
