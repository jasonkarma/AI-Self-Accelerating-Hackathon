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

from .env import load_env_file

#: Read ``.env`` from the repo root once, before anything resolves credentials.
#: Real environment variables keep precedence, so CI/ops exports still win.
load_env_file()

#: Standard vendor endpoints, used only when no base URL is configured.  Put a
#: gateway/router in ``.env`` via ANTHROPIC_BASE_URL / OPENAI_BASE_URL /
#: GEMINI_BASE_URL (or LLM_BASE_URL for a single all-in-one gateway) instead of
#: editing this file — that keeps the "works out of the box" default intact.
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ANTHROPIC_VERSION = "2023-06-01"

PROVIDER_API_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

#: Names of the env vars that override ``PROVIDER_DEFAULT_BASE_URLS``.
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

    ``thinking_budget`` (Anthropic) and ``reasoning_effort`` (OpenAI) are the
    *code-declared* defaults.  Runtime tuning happens through
    ``<SLUG>_THINKING_BUDGET`` / ``<SLUG>_REASONING_EFFORT`` (see
    :func:`resolve_thinking_budget` and :func:`resolve_reasoning_effort`), so a
    live test can dial deliberation up or down without editing this table.

    ``fallbacks`` is an ordered escape hatch: if the primary provider has no
    credentials (or is intentionally disabled), the seat re-homes to the next
    entry instead of silently degrading to the deterministic stub.
    """

    slug: str
    role: str
    provider: str  # "google" | "anthropic" | "openai"
    model: str
    thinking_budget: int | None = None
    #: Structured-output flavour: "json_schema" (OpenAI strict), "json_object"
    #: (broadly supported), or None. Resolved from <SLUG>_RESPONSE_FORMAT.
    structured_outputs: bool = False
    response_format: str | None = None
    max_tokens: int = 1024
    temperature: float | None = 0.0
    #: OpenAI-style deliberation hint: "low" | "medium" | "high". Anthropic uses
    #: ``thinking_budget`` instead; Google ignores both.
    reasoning_effort: str | None = None
    #: Per-request HTTP timeout in seconds. High reasoning effort needs headroom.
    timeout: float = 90.0
    fallbacks: tuple["AgentModel", ...] = ()

    def rehomed(self, provider: str, model: str) -> "AgentModel":
        """Clone for another provider/model, dropping provider-specific flags."""
        normalized = normalize_provider(provider)
        return replace(
            self,
            provider=normalized,
            model=model,
            # Thinking blocks, strict json_schema and reasoning_effort are all
            # provider features; a re-homed seat must not carry stale settings.
            thinking_budget=self.thinking_budget if normalized == "anthropic" else None,
            structured_outputs=self.structured_outputs,
            response_format=self.response_format,
            reasoning_effort=self.reasoning_effort if normalized == "openai" else None,
            # OpenAI reasoning models reject any non-default temperature.
            temperature=None if normalized == "openai" else self.temperature,
        )


def resolve_thinking_budget(spec: AgentModel) -> int | None:
    """Apply the env override for one agent at router-construction time.

    Concrete budgets live in ``AGENT_MODELS``; this function is what lets an
    operator write ``SKEPTIC_THINKING_BUDGET=8192`` (or ``=off``) and have it take
    effect without touching source.
    """
    return _env_int(f"{spec.slug.upper()}_THINKING_BUDGET", spec.thinking_budget)


#: Accepted OpenAI deliberation levels; anything else is treated as "unset".
REASONING_EFFORTS = ("low", "medium", "high")
_EFFORT_DISABLED = {"", "0", "off", "none", "false", "disable", "disabled", "default"}


def resolve_reasoning_effort(spec: AgentModel) -> str | None:
    """Resolve the OpenAI deliberation hint for a seat.

    Reads ``<SLUG>_REASONING_EFFORT`` (``off``/``none`` disables it).  Only
    meaningful for OpenAI-transported seats, so other providers always get
    ``None`` rather than sending a field their API would reject.
    """
    if normalize_provider(spec.provider) != "openai":
        return None
    raw = os.getenv(f"{spec.slug.upper()}_REASONING_EFFORT")
    value = (spec.reasoning_effort if raw is None else raw).strip().lower()
    if value in _EFFORT_DISABLED:
        return None
    if value not in REASONING_EFFORTS:
        raise ValueError(
            f"{spec.slug.upper()}_REASONING_EFFORT must be one of {REASONING_EFFORTS} "
            f"or 'off', got {value!r}"
        )
    return value


#: Accepted structured-output flavours.
RESPONSE_FORMATS = ("json_schema", "json_object", "off")


def resolve_timeout(spec: AgentModel) -> float:
    """Per-seat HTTP timeout, override with ``<SLUG>_TIMEOUT`` (seconds).

    Reasoning models bill deliberation against the same wall clock, so a seat
    that resolved onto one gets a floor of 240s even if its seat was configured
    for a fast non-reasoning model.
    """
    raw = os.getenv(f"{spec.slug.upper()}_TIMEOUT")
    configured = float(raw) if raw else spec.timeout
    if resolve_reasoning_effort(spec) and normalize_provider(spec.provider) == "openai":
        return max(configured, 240.0)
    return configured


def resolve_response_format(spec: AgentModel) -> str | None:
    """Resolve how structured output is requested for a seat.

    ``json_schema`` is the strongest guarantee but many OpenAI-compatible
    gateways reject it or leak the schema into the completion.  ``json_object``
    is far more widely supported: the model still emits JSON, and the pydantic
    validation downstream is the real contract.  ``off`` omits the field.
    """
    raw = os.getenv(f"{spec.slug.upper()}_RESPONSE_FORMAT")
    value = (spec.response_format or raw or "").strip().lower()
    if not value:
        return "json_schema" if spec.structured_outputs else None
    if value in {"off", "none", "0", "false", "disable", "disabled"}:
        return None
    if value not in {"json_schema", "json_object"}:
        raise ValueError(
            f"{spec.slug.upper()}_RESPONSE_FORMAT must be one of {RESPONSE_FORMATS}, got {value!r}"
        )
    return value


def resolve_model(spec: AgentModel) -> AgentModel:
    """Apply ``<SLUG>_PROVIDER`` / ``<SLUG>_MODEL`` overrides to a seat."""
    provider = (os.getenv(f"{spec.slug.upper()}_PROVIDER") or spec.provider).strip().lower()
    model = (os.getenv(f"{spec.slug.upper()}_MODEL") or spec.model).strip()
    if provider == spec.provider and model == spec.model:
        return spec
    return spec.rehomed(provider, model)


def _env_fallback(spec: AgentModel, provider_env: str, model_env: str) -> AgentModel | None:
    provider = (os.getenv(provider_env) or "").strip().lower()
    model = (os.getenv(model_env) or "").strip()
    if not provider or not model:
        return None
    return spec.rehomed(provider, model)


#: Ordered "who else can cover this seat" ladders.  Used only when the primary
#: provider has no credentials or is disabled, so the council keeps speaking
#: instead of falling silent.  XAI/DeepSeek/etc. can be added here or via
#: ``<SLUG>_FALLBACK_PROVIDER`` / ``<SLUG>_FALLBACK_MODEL``.
AGENT_FALLBACK_CHAINS: dict[str, tuple[tuple[str, str], ...]] = {
    "affinity": (("openai", "gpt-6.1-sol"), ("anthropic", "claude-5.5-sonnet")),
    "developability": (("openai", "gpt-6.1-sol"), ("google", "gemini-4-argon")),
    "skeptic": (("openai", "gpt-6.1-sol"), ("google", "gemini-4-argon")),
    "planner": (("anthropic", "claude-5.5-sonnet"), ("google", "gemini-4-argon")),
}


def fallback_ladder(spec: AgentModel) -> tuple[AgentModel, ...]:
    """Build a seat's fallback ladder, evaluated fresh on every call.

    Order: ``<SLUG>_FALLBACK_PROVIDER``/``_MODEL`` -> the code chain in
    :data:`AGENT_FALLBACK_CHAINS` -> ``LLM_FALLBACK_PROVIDER``/``_MODEL``
    (a global catch-all such as a local LiteLLM gateway).
    """
    ladder: list[AgentModel] = []
    env_override = _env_fallback(
        spec, f"{spec.slug.upper()}_FALLBACK_PROVIDER", f"{spec.slug.upper()}_FALLBACK_MODEL"
    )
    if env_override:
        ladder.append(env_override)
    ladder.extend(spec.fallbacks)
    global_override = _env_fallback(spec, "LLM_FALLBACK_PROVIDER", "LLM_FALLBACK_MODEL")
    if global_override:
        ladder.append(global_override)

    seen = {(spec.provider, spec.model)}
    unique: list[AgentModel] = []
    for candidate in ladder:
        key = (candidate.provider, candidate.model)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return tuple(unique)


def _agent(
    slug: str,
    role: str,
    provider: str,
    model: str,
    *,
    thinking_budget: int | None = None,
    structured_outputs: bool = False,
    response_format: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = None,
    timeout: float = 90.0,
) -> AgentModel:
    return AgentModel(
        slug=slug,
        role=role,
        provider=provider,
        model=model,
        thinking_budget=thinking_budget,
        structured_outputs=structured_outputs,
        response_format=response_format,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        fallbacks=tuple(
            AgentModel(slug=slug, role=role, provider=p, model=m)
            for p, m in AGENT_FALLBACK_CHAINS.get(slug, ())
        ),
    )


#: Single source of truth for "which mind answers as which scientist".
AGENT_MODELS: dict[str, AgentModel] = {
    "affinity": _agent(
        "affinity",
        "Affinity Scientist",
        "google",
        "gemini-4-argon",
        # Re-homed onto a reasoning model the budget covers *thinking* too,
        # so 1024 produced empty completions. Keep headroom for real prose.
        max_tokens=4096,
    ),
    "developability": _agent(
        "developability",
        "Developability Scientist",
        "anthropic",
        "claude-5.5-sonnet",
        thinking_budget=2048,  # moderate: enough to weigh liability trade-offs
        max_tokens=4096,
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
        # json_object: widely supported by gateways, unlike json_schema, which
        # several routers reject outright or echo back into the completion.
        response_format="json_object",
        # ~600 reasoning tokens + a few hundred JSON tokens; 8192 leaves room
        # for a verbose decision without silently truncating the object.
        max_tokens=8192,
        temperature=None,  # reasoning models reject non-default temperature
        reasoning_effort="high",  # final JSON synthesis must not break schema
        # High effort plus a large evidence payload regularly exceeds 90s.
        timeout=240.0,
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
        self.last_raw: str | None = None

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
            if response.status_code >= 400:
                # Surface the gateway's own explanation: a 403 usually means the
                # token has no access to *this model id*, which is otherwise
                # indistinguishable from a bad key.
                raise httpx.HTTPStatusError(
                    f"{response.status_code} for {self.model} at {self.messages_url}: "
                    f"{response.text[:400]}",
                    request=response.request,
                    response=response,
                )
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
        response_format: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        #: "json_schema" | "json_object" | None. Gateways frequently reject
        #: ``json_schema`` (or leak the schema into the completion), so make the
        #: flavour configurable instead of assuming the strict variant works.
        self.response_format = response_format or ("json_schema" if structured_outputs else None)
        self.structured_outputs = self.response_format is not None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.token_param = token_param
        self.reasoning_effort = reasoning_effort
        self.last_raw: str | None = None

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
        if schema is not None and self.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif schema is not None and self.response_format == "json_schema":
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
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{response.status_code} from {self.base_url}: {response.text[:400]}",
                    request=response.request,
                    response=response,
                )
            choice = response.json()["choices"][0]
            content = choice["message"].get("content") or ""
            self.last_raw = content
            if choice.get("finish_reason") == "length":
                # Reasoning tokens are billed against this budget, so a
                # high-effort model can spend every token thinking and return
                # ``content: null``. Fail loudly instead of returning "".
                raise ValueError(
                    f"{self.model} hit the {self.token_param} limit ({self.max_tokens}) "
                    f"({'empty completion' if not content else 'truncated output'}); "
                    "raise max_tokens or lower reasoning_effort"
                )
            if not content.strip():
                raise ValueError(f"{self.model} returned an empty completion")
            return content


PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
    "googleai": "google",
    "google-ai": "google",
    "claude": "anthropic",
    "oai": "openai",
    "openai-compatible": "openai",
}


def normalize_provider(provider: str) -> str:
    key = provider.strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def api_key_for(spec: AgentModel) -> str | None:
    """Resolve a credential: seat-specific, then provider-wide, then global.

    Unknown providers (``openrouter``, ``groq``, ``litellm``, ...) are supported
    through their own ``<PROVIDER>_API_KEY`` variable, and ``LLM_API_KEY`` covers
    any single-gateway setup.
    """
    provider = normalize_provider(spec.provider)
    return _first_env(
        f"{spec.slug.upper()}_API_KEY",
        f"{spec.role.upper().replace(' ', '_')}_API_KEY",
        *PROVIDER_API_KEY_ENVS.get(provider, ()),
        f"{provider.upper()}_API_KEY",
        "LLM_API_KEY",
    )


def base_url_for(spec: AgentModel) -> str | None:
    """Seat-specific URL, then provider-wide, then global, then vendor default."""
    provider = normalize_provider(spec.provider)
    explicit = _first_env(
        f"{spec.slug.upper()}_BASE_URL",
        *PROVIDER_BASE_URL_ENVS.get(provider, ()),
        f"{provider.upper()}_BASE_URL",
        "LLM_BASE_URL",
    )
    return explicit or PROVIDER_DEFAULT_BASE_URLS.get(provider)


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
        #: slug -> the seat that actually answers (primary or re-homed fallback)
        self._effective: dict[str, AgentModel | None] = {}

    # -- introspection ---------------------------------------------------
    @property
    def models(self) -> dict[str, str]:
        """Backwards-compatible slug -> model id view (env overrides applied)."""
        return {slug: resolve_model(spec).model for slug, spec in self.agents.items()}

    def slug_for(self, role: str) -> str | None:
        needle = " ".join(role.strip().lower().split())
        for slug, spec in self.agents.items():
            if needle in {slug, spec.role.strip().lower()}:
                return slug
        first = needle.split()[0] if needle else ""
        return first if first in self.agents else None

    def spec_for(self, role: str) -> AgentModel | None:
        """Resolve a role's declared assignment, applying runtime overrides.

        Precedence: ``<SLUG>_PROVIDER``/``<SLUG>_MODEL`` > code-declared seat;
        ``thinking_budgets`` argument > ``<SLUG>_THINKING_BUDGET`` > code default.
        This is the *intended* seat, before credential-driven fallback.
        """
        slug = self.slug_for(role)
        if slug is None:
            return None
        spec = resolve_model(self.agents[slug])
        spec = replace(
            spec,
            thinking_budget=resolve_thinking_budget(spec),
            reasoning_effort=resolve_reasoning_effort(spec),
        )
        for key in (slug, spec.role, spec.role.lower()):
            if key in self._thinking_overrides:
                return replace(spec, thinking_budget=self._thinking_overrides[key])
        return spec

    def effective_spec_for(self, role: str) -> AgentModel | None:
        """The seat actually serving requests, after fallback re-homing."""
        self.client_for(role)  # populates _effective
        slug = self.slug_for(role)
        return self._effective.get(slug) if slug else None

    def configure_thinking(self, **budgets: int | None) -> None:
        """Retune thinking budgets at runtime, e.g. ``configure_thinking(skeptic=8192)``."""
        unknown = set(budgets) - set(self.agents)
        if unknown:
            raise KeyError(f"Unknown agent slug(s): {sorted(unknown)}")
        self._thinking_overrides.update(budgets)
        self.reset()

    def reset(self) -> None:
        """Drop cached clients so credential/env changes take effect."""
        self._clients.clear()
        self._effective.clear()

    # -- client construction ---------------------------------------------
    def _build(self, spec: AgentModel) -> Reasoner | None:
        """Build one transport, or ``None`` when it cannot be configured."""
        provider = normalize_provider(spec.provider)
        if provider in {"none", "off", "disabled"}:
            return None
        candidate = replace(spec, provider=provider)
        api_key = api_key_for(candidate)
        if not api_key:
            return None
        base_url = base_url_for(candidate)
        if not base_url:
            return None
        if provider == "anthropic":
            return AnthropicReasoner(
                model=spec.model,
                api_key=api_key,
                base_url=base_url,
                thinking_budget=spec.thinking_budget,
                    max_tokens=spec.max_tokens,
                    temperature=spec.temperature,
                    timeout=spec.timeout,
                )
        return OpenAICompatibleReasoner(
            base_url=base_url,
            api_key=api_key,
            model=spec.model,
            structured_outputs=spec.structured_outputs,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
            token_param="max_completion_tokens" if provider == "openai" else "max_tokens",
            # Re-homed seats resolve against their own slug's env var, so a
            # fallback onto OpenAI still honors AFFINITY_REASONING_EFFORT.
            reasoning_effort=resolve_reasoning_effort(candidate),
            response_format=resolve_response_format(candidate),
            timeout=resolve_timeout(candidate),
        )

    def _resolve(self, spec: AgentModel) -> tuple[AgentModel | None, Reasoner | None]:
        """Pick the first seat in the ladder whose transport can actually be built.

        Fallback fires only for *missing credentials*.  Setting
        ``<SLUG>_PROVIDER=off`` is an explicit opt-out and stays deterministic.
        """
        if normalize_provider(spec.provider) in {"none", "off", "disabled"}:
            return None, None
        for candidate in (spec, *fallback_ladder(spec)):
            client = self._build(candidate)
            if client is not None:
                return candidate, client
        return None, None

    def client_for(self, role: str) -> Reasoner:
        spec = self.spec_for(role)
        if spec is None:
            return DeterministicReasoner()
        if spec.slug not in self._clients:
            effective, client = self._resolve(spec)
            self._effective[spec.slug] = effective
            self._clients[spec.slug] = client or DeterministicReasoner()
        return self._clients[spec.slug]

    def is_live(self, role: str) -> bool:
        """True when this seat resolved a real transport rather than the stub."""
        return not isinstance(self.client_for(role), DeterministicReasoner)

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
        """Pre-flight audit: prove the council is genuinely multi-model.

        Reports both the *requested* seat and the *effective* one, so a silent
        fallback (Gemini key missing -> Affinity answers on OpenAI) is visible.
        """
        report: list[dict[str, Any]] = []
        for declared in self.agents.values():
            # Report the *configured* intent (env overrides applied), not the
            # hard-coded default, so the audit matches what .env actually asks for.
            spec = self.spec_for(declared.role) or declared
            client = self.client_for(spec.role)
            effective = self._effective.get(spec.slug)
            report.append(
                {
                    "role": spec.role,
                    "provider": spec.provider,
                    "model": spec.model,
                    "effective_provider": effective.provider if effective else None,
                    "effective_model": effective.model if effective else None,
                    "fallback_used": bool(
                        effective and (effective.provider, effective.model) != (spec.provider, spec.model)
                    ),
                    "live": not isinstance(client, DeterministicReasoner),
                    "base_url": getattr(client, "base_url", None),
                    "thinking_budget": getattr(client, "thinking_budget", None),
                    "reasoning_effort": getattr(client, "reasoning_effort", None),
                    "response_format": getattr(client, "response_format", None),
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
