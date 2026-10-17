"""Validated records exchanged among tools, agents, campaign, and API."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import AMINO_ACIDS


class PositionalSoftPrior(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: int = Field(ge=1, le=13)
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


class RoundDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round_number: int = Field(ge=1, le=6)
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
