#!/usr/bin/env python3
"""Regenerate ``api_contract.json`` from the real backend, not from memory.

The committed ``api_contract.json`` is what the separately-built React frontend
codes against. It historically carried **handwritten** numbers — four benchmark
checkpoints instead of seven, three methods instead of five, and a
``best_qualified_binding`` of ``9.38`` that no artifact ever produced. A frontend
built on it renders a benchmark that does not exist, and because the file *is* the
contract, nothing downstream can catch the lie.

This script boots the real FastAPI app through ``TestClient``, captures the four
endpoints the showcase renders, and writes them out verbatim. Every number in the
resulting file therefore came from an artifact that a script produced — if the
backend changes, regenerating is the only way to get a different answer.

Usage
-----
    python scripts/generate_api_contract.py            # write api_contract.json
    python scripts/generate_api_contract.py --check    # exit 1 if it is stale

``--check`` is what CI and ``tests/test_api_sync.py`` use: it regenerates into
memory and compares, so a stale contract fails rather than rotting silently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "api_contract.json"

#: Endpoints captured into the contract, in the order the file lists them.
#: ``/api/state`` is the live golden campaign; the other three are reports.
ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("state", "/api/state"),
    ("benchmark", "/api/benchmark"),
    ("cross_target", "/api/cross-target"),
    ("attribution", "/api/attribution"),
)


def _strip_volatile(payload):
    """Drop values that change per machine and must not pin the contract.

    Absolute paths differ between checkouts, and hashing them into the contract
    would make ``--check`` fail on every clone for no scientific reason. The
    *basename* is kept, because "which file is this" is part of the contract.
    """
    if isinstance(payload, dict):
        cleaned = {}
        for key, value in payload.items():
            if key in {"path", "csv_path", "protocol_path"} and isinstance(value, str):
                cleaned[key] = Path(value).name
            else:
                cleaned[key] = _strip_volatile(value)
        return cleaned
    if isinstance(payload, list):
        return [_strip_volatile(item) for item in payload]
    return payload


def build_contract() -> dict:
    """Boot the app and capture every showcase endpoint."""
    from fastapi.testclient import TestClient

    from app.server import app

    contract: dict = {
        "_meta": {
            "generated_by": "scripts/generate_api_contract.py",
            "note": (
                "Auto-generated from live TestClient payloads. Do not hand-edit: "
                "run the generator, which is also what `--check` verifies. "
                "Paths are reduced to basenames so the file is machine-independent."
            ),
            "endpoints": {key: url for key, url in ENDPOINTS},
        }
    }
    with TestClient(app) as client:
        for key, url in ENDPOINTS:
            response = client.get(url)
            response.raise_for_status()
            contract[key] = _strip_volatile(response.json())
    return contract


def serialize(contract: dict) -> str:
    return json.dumps(contract, indent=2, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true",
        help="verify the committed contract matches the backend; exit 1 if stale",
    )
    parser.add_argument("--out", default=str(CONTRACT_PATH))
    args = parser.parse_args(argv)

    rendered = serialize(build_contract())
    out = Path(args.out)

    if args.check:
        if not out.exists():
            print(f"MISSING: {out} does not exist", file=sys.stderr)
            return 1
        on_disk = out.read_text()
        if on_disk == rendered:
            print(f"ok: {out.name} matches the live backend")
            return 0
        try:
            expected = json.loads(rendered)
            actual = json.loads(on_disk)
            differing = sorted(
                key for key in expected
                if expected.get(key) != actual.get(key)
            )
            print(
                f"STALE: {out.name} differs from the backend in {differing}. "
                f"Run `python scripts/generate_api_contract.py`.",
                file=sys.stderr,
            )
        except json.JSONDecodeError as exc:
            print(f"STALE: {out.name} is not valid JSON ({exc})", file=sys.stderr)
        return 1

    out.write_text(rendered)
    size_kb = len(rendered) / 1024
    print(f"wrote {out} ({size_kb:.1f} KiB)")
    for key, url in ENDPOINTS:
        print(f"  {key:<14} <- {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
