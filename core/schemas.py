"""Validated records exchanged among tools, agents, campaign, and API."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import AMINO_ACIDS


#: Upper bound on a positional prior's 1-based position. Raised from the frozen
#: 13 to the longest CDR3 AbBiBench exposes (18) so a cross-target campaign can
#: declare a rule at its own positions. Every 1N8Z path is unaffected — its
#: sequences are 13 residues and `validate_sequence` still enforces that length at
#: the dataset boundary, so a 14th position can never reach a 1N8Z prior.
MAX_CDR3_LENGTH = 18


class PositionalSoftPrior(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: int = Field(ge=1, le=MAX_CDR3_LENGTH)
    preferred_residues: list[str] = Field(default_factory=list)
    discouraged_residues: list[str] = Field(default_factory=list)
    weight_delta: float = Field(ge=-0.15, le=0.15)

    @field_validator("preferred_residues", "discouraged_residues")
    @classmethod
    def canonical_residues(cls, value: list[str]) -> list[str]:
        normalized = [v.upper() for v in value]
        if any(len(v) != 1 or v not in AMINO_ACIDS for v in normalized):
            raise ValueError("All residues must be canonical one-letter amino-acid codes")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Residues must be unique")
        return normalized

    @model_validator(mode="after")
    def disjoint(self) -> "PositionalSoftPrior":
        if set(self.preferred_residues) & set(self.discouraged_residues):
            raise ValueError("Preferred and discouraged residues must be disjoint")
        return self


#: Ceiling on a round number. Mirrors :data:`core.acquisition.ROUNDS_MAX`; kept as
#: a literal here so ``core.schemas`` stays importable without the acquisition
#: module (which pulls in numpy/pandas). A dynamically sized campaign runs more
#: than six rounds, and ``le=6`` would have rejected its planner's own record.
MAX_ROUNDS = 60


class RoundDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round_number: int = Field(ge=1, le=MAX_ROUNDS)
    phase: Literal["Cold-Start Tuition (Exploration)", "Self-Accelerating Payoff (Exploitation)"]
    tools_invoked: list[str]
    hypothesis_id: str
    hypothesis_statement: str
    previous_hypothesis_verdict: Literal["N/A (Round 1)", "SUPPORTED", "FALSIFIED", "INCONCLUSIVE"]
    supporting_candidate_ids: list[str]
    agent_disagreement_summary: str
    confidence_rating: Literal["Low", "Medium", "High"]
    positional_priors: list[PositionalSoftPrior]
    slot3_uncertainty_target_id: str
    slot4_hypothesis_target_id: str
    slot5_falsification_target_id: str

    @model_validator(mode="after")
    def unique_targets(self) -> "RoundDecisionRecord":
        targets = [self.slot3_uncertainty_target_id, self.slot4_hypothesis_target_id, self.slot5_falsification_target_id]
        if len(set(targets)) != 3:
            raise ValueError("Slots 3-5 must target distinct candidates")
        return self


class AgentMemo(BaseModel):
    role: str
    tools_invoked: list[str]
    thinking_trace: str
    memo: str
    #: Provenance, so the stream proves which model actually answered.
    model: str | None = None
    provider: str | None = None
    thinking_budget: int | None = None
