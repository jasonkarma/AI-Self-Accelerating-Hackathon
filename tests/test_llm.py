"""Guards for cognitive diversity, gateway routing, and deliberation settings."""
from __future__ import annotations

import pytest

from core.agents import AGENT_DEFINITIONS, run_planner, thinking_configuration
from core.llm import (
    AGENT_MODELS,
    AnthropicReasoner,
    ModelRouter,
    OpenAICompatibleReasoner,
    resolve_reasoning_effort,
    resolve_thinking_budget,
    strict_json_schema,
)
from core.schemas import RoundDecisionRecord


def test_each_council_role_has_a_distinct_model_family():
    providers = {spec.role: spec.provider for spec in AGENT_MODELS.values()}
    assert providers["Affinity Scientist"] == "google"
    assert providers["Developability Scientist"] == "anthropic"
    assert providers["Scientific Skeptic"] == "anthropic"
    assert providers["Experimental Planner"] == "openai"
    assert len({spec.model for spec in AGENT_MODELS.values()}) == 4


def test_anthropic_uses_router_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.internal/v1")
    reasoner = AnthropicReasoner(model="claude-5.5-sonnet", api_key="test")
    assert reasoner.base_url == "https://gateway.internal/v1"
    assert reasoner.messages_url == "https://gateway.internal/v1/messages"


def test_anthropic_falls_back_to_configured_default(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(AnthropicReasoner, "default_base_url", "https://api.anthropic.com")
    reasoner = AnthropicReasoner(model="claude-5.5-sonnet", api_key="test")
    assert reasoner.messages_url == "https://api.anthropic.com/v1/messages"


def test_thinking_block_only_when_budget_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    thinking = AnthropicReasoner(model="claude-5.5-opus", api_key="t", thinking_budget=4096)
    payload = thinking.build_payload("system", {})
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert payload["max_tokens"] > 4096

    plain = AnthropicReasoner(model="claude-5.5-sonnet", api_key="t", thinking_budget=None)
    assert "thinking" not in plain.build_payload("system", {})


def test_skeptic_and_developability_think_with_asymmetric_budgets():
    budgets = {row["role"]: row["thinking_budget"] for row in thinking_configuration()}
    assert budgets["Scientific Skeptic"] == 4096
    assert budgets["Developability Scientist"] == 2048
    assert budgets["Affinity Scientist"] is None
    assert AGENT_DEFINITIONS["planner"].response_schema is RoundDecisionRecord


def test_planner_defaults_to_high_reasoning_effort():
    assert AGENT_MODELS["planner"].reasoning_effort == "high"
    assert resolve_reasoning_effort(AGENT_MODELS["planner"]) == "high"
    # Non-OpenAI seats never emit the field.
    assert resolve_reasoning_effort(AGENT_MODELS["skeptic"]) is None


def test_reasoning_effort_env_override(monkeypatch):
    planner = AGENT_MODELS["planner"]
    monkeypatch.setenv("PLANNER_REASONING_EFFORT", "medium")
    assert resolve_reasoning_effort(planner) == "medium"

    monkeypatch.setenv("PLANNER_REASONING_EFFORT", "off")
    assert resolve_reasoning_effort(planner) is None

    monkeypatch.setenv("PLANNER_REASONING_EFFORT", "turbo")
    with pytest.raises(ValueError):
        resolve_reasoning_effort(planner)


def test_planner_payload_carries_reasoning_effort():
    planner = OpenAICompatibleReasoner(
        base_url="https://api.openai.com/v1",
        api_key="t",
        model="gpt-6.1-sol",
        structured_outputs=True,
        token_param="max_completion_tokens",
        reasoning_effort="high",
    )
    assert planner.build_payload("system", {})["reasoning_effort"] == "high"


def test_agent_thinking_budget_can_be_tuned_by_environment(monkeypatch):
    spec = AGENT_MODELS["developability"]
    assert resolve_thinking_budget(spec) == 2048

    monkeypatch.setenv("DEVELOPABILITY_THINKING_BUDGET", "1024")
    assert resolve_thinking_budget(spec) == 1024

    monkeypatch.setenv("SKEPTIC_THINKING_BUDGET", "off")
    assert resolve_thinking_budget(AGENT_MODELS["skeptic"]) is None


def test_router_thinking_overrides_beat_env(monkeypatch):
    monkeypatch.setenv("SKEPTIC_THINKING_BUDGET", "1024")
    router = ModelRouter(thinking_budgets={"skeptic": 8192})
    assert router.spec_for("Scientific Skeptic").thinking_budget == 8192
    router.configure_thinking(skeptic=None)
    assert router.spec_for("Scientific Skeptic").thinking_budget is None


def test_planner_uses_strict_structured_outputs():
    planner = OpenAICompatibleReasoner(
        base_url="https://api.openai.com/v1",
        api_key="t",
        model="gpt-6.1-sol",
        structured_outputs=True,
        token_param="max_completion_tokens",
    )
    schema = strict_json_schema(RoundDecisionRecord.model_json_schema())
    payload = planner.build_payload("system", {}, schema=schema)
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert payload["max_completion_tokens"] > 0

    root = response_format["json_schema"]["schema"]
    assert root["additionalProperties"] is False
    assert set(root["required"]) == set(root["properties"])
    assert all("default" not in field for field in root["properties"].values())


def test_router_falls_back_to_deterministic_without_credentials(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    roster = ModelRouter().describe_roster()
    assert [row["live"] for row in roster] == [False, False, False, False]


@pytest.mark.asyncio
async def test_legacy_reasoner_without_schema_keyword_still_runs():
    class LegacyReasoner:
        async def complete(self, system, context):
            return "legacy memo"

    router = ModelRouter(LegacyReasoner())
    assert await router.run("Experimental Planner", "prompt", {}, schema={"type": "object"}) == (
        "legacy memo"
    )


@pytest.mark.asyncio
async def test_router_passes_strict_schema_to_structured_output_transport():
    class CapturingReasoner:
        def __init__(self):
            self.schema = None

        async def complete(self, system, context, *, schema=None):
            self.schema = schema
            return "{}"

    capturing = CapturingReasoner()
    await run_planner(ModelRouter(capturing), deterministic_evidence={"a": 1}, memos=[])
    assert capturing.schema is not None
    assert "slot3_uncertainty_target_id" in capturing.schema["properties"]


@pytest.mark.asyncio
async def test_anthropic_thinking_blocks_are_split_from_text(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    reasoner = AnthropicReasoner(model="claude-5.5-opus", api_key="t", thinking_budget=2048)
    body = {
        "content": [
            {"type": "thinking", "thinking": "weigh confounders"},
            {"type": "text", "text": "final memo"},
        ]
    }
    assert reasoner._extract(body) == "final memo"
    assert reasoner.last_thinking == "weigh confounders"
