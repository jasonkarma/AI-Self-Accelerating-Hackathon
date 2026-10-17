"""Zero-dependency ``.env`` loader.

The app reads configuration exclusively from environment variables, but typing
``export ANTHROPIC_API_KEY=...`` before every uvicorn/script invocation is easy
to forget.  Drop a ``.env`` file at the repository root instead; it is loaded
once when :mod:`core.llm` is imported.

Precedence: real environment variables always win over the file, so
``OPENAI_MODEL=x python -m scripts.check_council`` still overrides ``.env``.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = ROOT / ".env"

_QUOTES = {'"', "'"}


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    key, sep, value = line.partition("=")
    if not sep:
        return None
    key, value = key.strip(), value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] in _QUOTES and value[-1] == value[0]:
        value = value[1:-1]
    else:
        # Strip trailing inline comments on unquoted values only.
        value = value.split(" #", 1)[0].strip()
    return key, value


def load_env_file(path: Path | str | None = None, *, override: bool = False) -> list[str]:
    """Load ``KEY=VALUE`` pairs without clobbering existing variables.

    Returns the names actually applied, which makes the loader easy to test and
    easy to log at startup.
    """
    target = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not target.is_file():
        return []

    applied: list[str] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(raw)
        if parsed is None:
            continue
        key, value = parsed
        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
        applied.append(key)
    return applied
