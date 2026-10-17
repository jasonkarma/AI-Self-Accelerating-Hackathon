"""Frontier reasoning transport with deterministic offline fallback."""
from __future__ import annotations
import os
import httpx
from typing import Protocol

class Reasoner(Protocol):
    async def complete(self, system: str, context: dict) -> str: ...

class DeterministicReasoner:
    async def complete(self, system: str, context: dict) -> str:
        return context.get("fallback", "Insufficient revealed evidence; retain bounded exploration.")

class OpenAICompatibleReasoner:
    """Calls any OpenAI-compatible frontier endpoint with explicit thinking mode."""
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 90):
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout

    async def complete(self, system: str, context: dict) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json={"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": str(context)}], "temperature": 0, "reasoning_effort": "high", "thinking_mode": True})
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

class ModelRouter:
    models = {"affinity": "gemini-4-argon", "developability": "claude-opus-5.5", "skeptic": "gpt-6.1-sol", "planner": "claude-opus-5.5"}
    def __init__(self, reasoner: Reasoner | None = None): self.reasoner = reasoner
    async def run(self, role: str, prompt: str, context: dict) -> str:
        if self.reasoner:
            return await self.reasoner.complete(prompt, context)
        prefix = role.upper().replace(" ", "_")
        key, endpoint = os.getenv(f"{prefix}_API_KEY"), os.getenv(f"{prefix}_BASE_URL")
        if key and endpoint:
            return await OpenAICompatibleReasoner(base_url=endpoint, api_key=key, model=self.models.get(role.lower().split()[0], self.models["planner"])).complete(prompt, context)
        return await DeterministicReasoner().complete(prompt, context)
