"""Test isolation: never read the developer's real ``.env``.

Without this, running the suite on a machine with live credentials would build
real HTTP clients and fire network calls from unit tests. ``core.env`` honors
``LLM_ENV_FILE``, so pointing it at a path that does not exist keeps every
transport offline and deterministic.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ["LLM_ENV_FILE"] = str(Path(__file__).parent / ".env.disabled-for-tests")

# Belt and braces: strip credentials that the outer shell may have exported.
for _name in (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "LLM_API_KEY",
):
    os.environ.pop(_name, None)
