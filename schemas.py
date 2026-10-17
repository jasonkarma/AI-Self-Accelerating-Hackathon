from pydantic import BaseModel, Field
from typing import List, Literal

class PositionalSoftPrior(BaseModel):
    position: int = Field(..., ge=1, le=13, description="1-based HCDR3 position (1..13)")
    preferred_residues: List[str] = Field(default_factory=list, description="Single-letter AA codes receiving soft bonus")
    discouraged_residues: List[str] = Field(default_factory=list, description="Single-letter AA codes receiving soft penalty")
    weight_delta: float = Field(..., ge=-0.15, le=0.15, description="Bounded soft prior adjustment in [-0.15, +0.15]")

class RoundDecisionRecord(BaseModel):
    round_number: int
    phase: Literal["Cold-Start Tuition (Exploration)", "Self-Accelerating Payoff (Exploitation)"]
    tools_invoked: List[str] = Field(..., description="Names of deterministic arsenal tools used in this round")
    hypothesis_id: str
    hypothesis_statement: str
    previous_hypothesis_verdict: Literal["N/A (Round 1)", "SUPPORTED", "FALSIFIED", "INCONCLUSIVE"]
    supporting_candidate_ids: List[str]
    agent_disagreement_summary: str
    confidence_rating: Literal["Low", "Medium", "High"]
    positional_priors: List[PositionalSoftPrior]
    slot3_uncertainty_target_id: str = Field(..., description="Unqueried candidate_id with highest joint uncertainty (sigma_joint)")
    slot4_hypothesis_target_id: str = Field(..., description="Unqueried candidate_id testing the leading hypothesis")
    slot5_falsification_target_id: str = Field(..., description="Unqueried candidate_id testing outside/against the hypothesis")
