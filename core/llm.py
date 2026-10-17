"""Frontier reasoning transport with deterministic offline fallback.

Cognitive diversity is enforced structurally rather than by prompt: every
council role is pinned to a distinct model family, and each transport is a thin
``httpx`` wrapper so a gateway/router can never silently swap the model behind a
role.  No vendor SDK is required, which keeps custom gateways (LiteLLM, OpenRouter,
internal proxies) working through the ``*_BASE_URL`` environment variables.

Roster (see ``AGENT_MODELS``):
    Affinity Scientist        -> Google    gemini-4-argon
    Developability Scientist  -> Anthropic claude-5.5-sonnet
    Scientific Skeptic        -> Anthropic claude-5.5-opus   (extended thinking)
    Experimental Planner      -> OpenAI    gpt-6.1-sol       (strict structured outputs)
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, replace
from typing import Any, Protocol

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ANTHROPIC_VERSION = "2023-06-01"

PROVIDER_API_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

PROVIDER_BASE_URL_ENVS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_BASE_URL",),
    "openai": ("OPENAI_BASE_URL",),
    "google": ("GEMINI_BASE_URL", "GOOGLE_BASE_URL"),
}

PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "anthropic": DEFAULT_ANTHROPIC_BASE_URL,
    "openai": DEFAULT_OPENAI_BASE_URL,
    "google": DEFAULT_GEMINI_BASE_URL,
}

# OpenAI structured outputs reject unknown keywords such as ``default``.
_STRICT_SCHEMA_DROPPED_KEYS = ("default",)


class Reasoner(Protocol):
    async def complete(self, system: str, context: dict, *, schema: dict | None = None) -> str: ...


def strict_json_schema(schema: dict) -> dict:
    """Normalize a JSON Schema for provider "strict" structured-output modes.

    Strict modes require every object to set ``additionalProperties: false`` and
    to list *all* of its properties under ``required`` (pydantic marks
    defaulted fields optional).  ``default`` is stripped because several
    providers reject it outright.
    """
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in _STRICT_SCHEMA_DROPPED_KEYS:
                node.pop(key, None)
            if isinstance(node.get("properties"), dict):
                node.setdefault("type", "object")
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in list(node.values()):
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    cloned = copy.deepcopy(schema)
    walk(cloned)
    return cloned


def _stringify(context: Any) -> str:
    if isinstance(context, str):
        return context
    try:
        return json.dumps(context, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(context)


def _env_int(name: str, default: int | None) -> int | None:
    """Read an integer env override; ``off``/``none``/``0`` disables a budget."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "0", "off", "none", "false", "disable", "disabled"}:
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative token budget")
    return value or None


@dataclass(frozen=True)
class AgentModel:
    """Immutable cognitive-diversity assignment for one council role.

    ``thinking_budget`` is the *code-declared* default.  Runtime tuning happens
    through ``<SLUG>_THINKING_BUDGET`` (see :func:`resolve_thinking_budget`), so a
    live test can dial the Skeptic down without editing this table.
    """

    slug: str
    role: str
    provider: str  # "google" | "anthropic" | "openai"
    model: str
    thinking_budget: int | None = None
    structured_outputs: bool = False
    max_tokens: int = 1024
    temperature: float | None = 0.0


def resolve_thinking_budget(spec: AgentModel) -> int | None:
    """Apply the env override for one agent at router-construction time.

    Concrete budgets live in ``AGENT_MODELS``; this function is what lets an
    operator write ``SKEPTIC_THINKING_BUDGET=8192`` (or ``=off``) and have it take
    effect without touching source.
    """
    return _env_int(f"{spec.slug.upper()}_THINKING_BUDGET", spec.thinking_budget)


def _agent(
    slug: str,
    role: str,
    provider: str,
    model: str,
    *,
    thinking_budget: int | None = None,
    structured_outputs: bool = False,
    max_tokens: int = 1024,
    temperature: float | None = 0.0,
) -> AgentModel:
    return AgentModel(
        slug=slug,
        role=role,
        provider=provider,
        model=model,
        thinking_budget=thinking_budget,
        structured_outputs=structured_outputs,
        max_tokens=max_tokens,
        temperature=temperature,
    )


#: Single source of truth for "which mind answers as which scientist".
AGENT_MODELS: dict[str, AgentModel] = {
    "affinity": _agent(
        "affinity", "Affinity Scientist", "google", "gemini-4-argon", max_tokens=1024
    ),
    "developability": _agent(
        "developability",
        "Developability Scientist",
        "anthropic",
        "claude-5.5-sonnet",
        thinking_budget=None,  # default execution: no extended-thinking block
        max_tokens=1024,
    ),
    "skeptic": _agent(
        "skeptic",
        "Scientific Skeptic",
        "anthropic",
        "claude-5.5-opus",
        thinking_budget=4096,  # high budget: adversarial audit warrants deliberation
        max_tokens=8192,
    ),
    "planner": _agent(
        "planner",
        "Experimental Planner",
        "openai",
        "gpt-6.1-sol",
        structured_outputs=True,
        max_tokens=2048,
        temperature=None,  # reasoning models reject non-default temperature
    ),
}


class DeterministicReasoner:
    """Offline stand-in used whenever a role has no credentials configured."""

    async def complete(self, system: str, context: dict, *, schema: dict | None = None) -> str:
        return context.get("fallback", "Insufficient revealed evidence; retain bounded exploration.")


class AnthropicReasoner:
    """Anthropic Messages API client with optional extended-thinking budget.

    ``base_url`` resolution order: explicit argument -> ``ANTHROPIC_BASE_URL``
    environment variable -> the standard Anthropic endpoint.  This lets a
    gateway/router sit in front of the vendor without changing call sites.
    """

    default_base_url = DEFAULT_ANTHROPIC_BASE_URL

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        thinking_budget: int | None = None,
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        timeout: float = 90.0,
        base_url_env: str = "ANTHROPIC_BASE_URL",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url_env = base_url_env
        self.base_url = (base_url or os.getenv(base_url_env) or self.default_base_url).rstrip("/")
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.timeout = timeout
        # Anthropic requires max_tokens > budget_tokens when thinking is enabled.
        self.max_tokens = max(max_tokens, thinking_budget + 1024) if thinking_budget else max_tokens
        self.last_thinking: str | None = None

    @property
    def messages_url(self) -> str:
        """Tolerate gateways that want the ``/v1`` prefix included in the base URL."""
        return f"{self.base_url}/messages" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/messages"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def build_payload(self, system: str, context: dict) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": _stringify(context)}],
        }
        if self.thinking_budget:
            # Extended thinking: inject the explicit block, and keep the
            # provider-required default temperature (1.0).
            payload["thinking"] = {"type": "enabled", "budget_tokens": int(self.thinking_budget)}
            payload["temperature"] = 1.0
        elif self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    async def complete(self, system: str, context: dict, *, schema: dict | None = None) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.messages_url, headers=self.headers, json=self.build_payload(system, context)
            )
            response.raise_for_status()
            return self._extract(response.json())

    def _extract(self, body: dict[str, Any]) -> str:
        text, thinking = [], []
        for block in body.get("content", []):
            if block.get("type") == "text":
                text.append(block.get("text", ""))
            elif block.get("type") in {"thinking", "redacted_thinking"}:
                thinking.append(block.get("thinking", ""))
        self.last_thinking = "\n".join(filter(None, thinking)) or None
        return "\n".join(filter(None, text))


class OpenAICompatibleReasoner:
    """OpenAI-compatible chat endpoint (OpenAI, Gemini OpenAI-compat, gateways)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        structured_outputs: bool = False,
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        timeout: float = 90.0,
        token_param: str = "max_tokens",
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.structured_outputs = structured_outputs
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.token_param = token_param
        self.reasoning_effort = reasoning_effort

    def build_payload(self, system: str, context: dict, *, schema: dict | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": _stringify(context)},
            ],
            self.token_param: self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if schema is not None and self.structured_outputs:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title", "agent_response"),
                    "strict": True,
                    "schema": strict_json_schema(schema),
                },
            }
        return payload

    async def complete(self, system: str, context: dict, *, schema: dict | None = None) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self.build_payload(system, context, schema=schema),
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            return message.get("content") or ""


def _env_for(spec: AgentModel, suffix: str, provider_envs: dict[str, tuple[str, ...]]) -> str | None:
    """Role-specific env wins over the provider-wide default."""
    candidates = (
        f"{spec.slug.upper()}_{suffix}",
        f"{spec.role.upper().replace(' ', '_')}_{suffix}",
        *provider_envs.get(spec.provider, ()),
    )
    for name in candidates:
        value = os.getenv(name)
        if value:
            return value
    return None


class ModelRouter:
    """Route each council role to its pinned, cognitively distinct model."""

    def __init__(
        self,
        reasoner: Reasoner | None = None,
        *,
        agents: dict[str, AgentModel] | None = None,
        thinking_budgets: dict[str, int | None] | None = None,
    ) -> None:
        self.reasoner = reasoner
        self.agents = dict(agents or AGENT_MODELS)
        self._thinking_overrides = dict(thinking_budgets or {})
        self._clients: dict[str, Reasoner] = {}

    # -- introspection ---------------------------------------------------
    @property
    def models(self) -> dict[str, str]:
        """Backwards-compatible slug -> model id view."""
        return {slug: spec.model for slug, spec in self.agents.items()}

    def slug_for(self, role: str) -> str | None:
        needle = " ".join(role.strip().lower().split())
        for slug, spec in self.agents.items():
            if needle in {slug, spec.role.strip().lower()}:
                return slug
        first = needle.split()[0] if needle else ""
        return first if first in self.agents else None

    def spec_for(self, role: str) -> AgentModel | None:
        """Resolve a role to its assignment, applying thinking-budget overrides.

        Precedence: explicit ``thinking_budgets`` argument > ``<SLUG>_THINKING_BUDGET``
        env var > the code-declared default in ``AGENT_MODELS``.
        """
        slug = self.slug_for(role)
        if slug is None:
            return None
        spec = replace(
            self.agents[slug], thinking_budget=resolve_thinking_budget(self.agents[slug])
        )
        for key in (slug, spec.role, spec.role.lower()):
            if key in self._thinking_overrides:
                return replace(spec, thinking_budget=self._thinking_overrides[key])
        return spec

    def configure_thinking(self, **budgets: int | None) -> None:
        """Retune thinking budgets at runtime, e.g. ``configure_thinking(skeptic=8192)``."""
        unknown = set(budgets) - set(self.agents)
        if unknown:
            raise KeyError(f"Unknown agent slug(s): {sorted(unknown)}")
        self._thinking_overrides.update(budgets)
        # Clients embed their budget in the request payload, so drop stale ones.
        self._clients.clear()

    # -- client construction ---------------------------------------------
    def _build(self, spec: AgentModel) -> Reasoner | None:
        api_key = _env_for(spec, "API_KEY", PROVIDER_API_KEY_ENVS)
        if not api_key:
            return None
        base_url = (
            _env_for(spec, "BASE_URL", PROVIDER_BASE_URL_ENVS)
            or PROVIDER_DEFAULT_BASE_URLS[spec.provider]
        )
        if spec.provider == "anthropic":
            return AnthropicReasoner(
                model=spec.model,
                api_key=api_key,
                base_url=base_url,
                thinking_budget=spec.thinking_budget,
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
            )
        return OpenAICompatibleReasoner(
            base_url=base_url,
            api_key=api_key,
            model=spec.model,
            structured_outputs=spec.structured_outputs,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
            token_param="max_completion_tokens" if spec.provider == "openai" else "max_tokens",
        )

    def client_for(self, role: str) -> Reasoner:
        spec = self.spec_for(role)
        if spec is None:
            return DeterministicReasoner()
        if spec.slug not in self._clients:
            self._clients[spec.slug] = self._build(spec) or DeterministicReasoner()
        return self._clients[spec.slug]

    def thinking_trace(self, role: str) -> str | None:
        """Return the last extended-thinking transcript for a role, if any.

        Only Anthropic models expose a separate reasoning stream; transports
        without one simply return ``None`` so callers can fall back to the memo.
        An injected reasoner is honored here too, so tests and offline runs never
        accidentally construct a live client.
        """
        transport = self.reasoner or self.client_for(role)
        return getattr(transport, "last_thinking", None) or None

    def describe_roster(self) -> list[dict[str, Any]]:
        """Pre-flight audit: prove the council is genuinely multi-model."""
        report: list[dict[str, Any]] = []
        for spec in self.agents.values():
            client = self.client_for(spec.role)
            report.append(
                {
                    "role": spec.role,
                    "provider": spec.provider,
                    "model": spec.model,
                    "live": not isinstance(client, DeterministicReasoner),
                    "base_url": getattr(client, "base_url", None),
                    "thinking_budget": getattr(client, "thinking_budget", None),
                    "structured_outputs": spec.structured_outputs,
                }
            )
        return report

    # -- execution --------------------------------------------------------
    async def run(
        self, role: str, prompt: str, context: dict, *, schema: dict | None = None
    ) -> str:
        transport = self.reasoner or self.client_for(role)
        if schema is None:
            return await transport.complete(prompt, context)
        try:
            return await transport.complete(prompt, context, schema=schema)
        except TypeError:
            # Injected/legacy reasoners may implement the older two-argument protocol.
            return await transport.complete(prompt, context)
