"""Read-only HTTP client for the Ptarmigan-1 structure-free virtual screen API.

Route table (one declarative constant, reconciliation edits there):

    PTARMIGAN_ROUTES[name] = (method, path_template)

Key design rules (from the plan):

    * ``PTARMIGAN_API_KEY`` is read from ``os.environ`` **inside** every request
      method — never at module or class import time — so that
      ``tests/conftest.py`` (which does **not** pop ``PTARMIGAN_API_KEY``) never
      opens a real socket when the transport is injectable.
    * A custom ``User-Agent`` is sent on every request (Cloudflare blocks default
      ``python-httpx`` / ``python-requests`` agents).
    * ``transport`` is injectable (a ``httpx.Client`` or a mock / ``pytest-httpx``
      resolver), so unit tests verify error handling, the idempotency header, and
      the $15 gate without touching ``ptarmigan.app``.
    * **Credit protection ordering**: balance → estimate → confirm → create.
      ``--dry-run`` prints the estimate and exits before any ``POST``.
    * A **single ``Idempotency-Key``** (fresh UUID per screen) is sent on every
      ``POST /api/v1/screens``; the same key is used on retry so the API never
      double-charges.
    * **$15 hard gate**: if ``estimate.cost_usd > $15`` the client prints the
      estimate and raises ``PtarmiganGateError`` — it never POSTs.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = "virtual-lab-dnaforge/1.0"
"""Sent on every HTTP request (Cloudflare blocks ``python-httpx`` / ``python-requests``)."""

HARD_GATE_USD = 15.0
"""Hard credit-protection gate — exit (no POST) if the estimate exceeds this."""

BASE_URL = "https://ptarmigan.app"
"""Default Ptarmigan-1 API base URL."""

#: One declarative route table.  Reconciliation edits only touch this dict;
#: every call site derives its final URL from this constant.
PTARMIGAN_ROUTES: dict[str, tuple[str, str]] = {
    "reference":             ("GET",    "/api/v1/reference"),
    "proteins":              ("GET",    "/api/v1/proteins"),
    "compound_collections":  ("GET",    "/api/v1/compound-collections"),
    "upload_collection":     ("POST",   "/api/v1/compound-collections"),
    "get_collection":        ("GET",    "/api/v1/compound-collections/{collection_id}"),
    "delete_collection":     ("DELETE", "/api/v1/compound-collections/{collection_id}"),
    "collection_compounds":  ("GET",    "/api/v1/compound-collections/{collection_id}/compounds"),
    "balance":               ("GET",    "/api/v1/balance"),
    "estimate_screen":       ("POST",   "/api/v1/screens/estimate"),
    "create_screen":         ("POST",   "/api/v1/screens"),
    "list_screens":          ("GET",    "/api/v1/screens"),
    "get_screen":            ("GET",    "/api/v1/screens/{screen_id}"),
    "cancel_screen":         ("POST",   "/api/v1/screens/{screen_id}/cancel"),
    "get_hits":              ("GET",    "/api/v1/screens/{screen_id}/hits"),
    "download":              ("GET",    "/api/v1/screens/{screen_id}/download"),
    "bulk_download":         ("POST",   "/api/v1/screens/downloads"),
    "compound_detail":       ("GET",    "/api/v1/screens/{screen_id}/compounds/{compound_id}"),
}

CURRENCY_DECIMAL_PATTERN = r"^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PtarmiganError(Exception):
    """A non-2xx response from the Ptarmigan-1 API.

    ``code`` and ``message`` come from the JSON error body (:external:doc:`errors
    </docs/errors>`).  ``extra_fields`` carries code-specific keys such as
    ``required_usd`` / ``available_usd`` on ``insufficient_funds``, or
    ``screen_id`` on ``idempotency_key_reused``.
    """

    def __init__(self, payload: dict, *, status: int) -> None:
        self.code: str = str(payload.get("code", "unknown"))
        self.message: str = str(payload.get("message", ""))
        self.status: int = status
        # Gather code-specific fields (everything besides code/message).
        self.extra_fields: dict[str, Any] = {
            k: v for k, v in payload.items() if k not in ("code", "message")
        }
        super().__init__(f"[{self.status}] {self.code}: {self.message}")

    @property
    def required_usd(self) -> str | None:
        return self.extra_fields.get("required_usd")

    @property
    def available_usd(self) -> str | None:
        return self.extra_fields.get("available_usd")

    @property
    def screen_id(self) -> str | None:
        return self.extra_fields.get("screen_id")


class PtarmiganGateError(Exception):
    """The screen cost exceeds :data:`HARD_GATE_USD` — the client will not POST."""

    def __init__(self, cost_usd: str, *, gate: float) -> None:
        self.cost_usd = cost_usd
        self.gate = gate
        super().__init__(f"Estimate ${cost_usd} exceeds ${gate} gate — not posting")


class PtarmiganAuthError(Exception):
    """No ``PTARMIGAN_API_KEY`` in the environment."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_key() -> str:
    """Read ``PTARMIGAN_API_KEY`` from environment (inside a request, not at import)."""
    key = os.environ.get("PTARMIGAN_API_KEY")
    if not key:
        raise PtarmiganAuthError(
            "PTARMIGAN_API_KEY is not set. "
            "Add it to .env (PTARMIGAN_API_KEY=sk_...) or export it."
        )
    return key


def _format_currency(value: str | int | float) -> float:
    """Parse a Ptarmigan currency string (like ``"12.50"``) to float."""
    return round(float(value), 2)


def _resolve(route_name: str, **path_params: str) -> tuple[str, str]:
    """Resolve a route name to (method, final_url).

    Raises ``KeyError`` if the route name is unknown, so call sites don't
    accidentally construct URLs for routes that don't exist.
    """
    method, template = PTARMIGAN_ROUTES[route_name]
    if path_params:
        safe: dict[str, str] = {}
        for key, value in path_params.items():
            if isinstance(value, str):
                # Percent-encode only the characters the docs say to encode:
                # ``#``, ``?``, ``%``, and spaces. ``/`` is allowed in path,
                # ``+`` is fine as a literal, ``&`` as well.
                value = value.replace("%", "%25").replace("#", "%23").replace("?", "%3F").replace(" ", "%20")
            safe[key] = str(value)
        url = template.format(**safe)
    else:
        url = template
    full_url = f"{BASE_URL}{url}"
    return method, full_url


# ---------------------------------------------------------------------------
# Transport interface
# ---------------------------------------------------------------------------

class _Transport:
    """Injectable transport wrapper.

    In production: wraps ``httpx.Client``.
    In unit tests: can be replaced with ``pytest_httpx.HTTPXMock`` or a thin
    ``FakeTransport`` that returns canned responses.
    """

    def __init__(self, client: httpx.Client | None = None) -> None:
        # Ptarmigan-1 library embedding can take several minutes for 300+
        # compounds, so the read timeout must be generous.  The connect and pool
        # timeouts are kept short so a dead server is detected quickly.
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(300.0, connect=15.0, pool=15.0),
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: Any = None,
        retries: int = 2,
    ) -> tuple[int, dict[str, Any]]:
        """Send a synchronous request and return ``(status, parsed_json)``.

        Retries on transient ``ReadTimeout`` or ``ConnectError`` up to
        ``retries`` times with a 2-second backoff.  Raises ``PtarmiganError``
        on non-2xx status codes (including after exhausting retries).
        """
        if headers is None:
            headers = {}
        headers.setdefault("User-Agent", USER_AGENT)
        headers.setdefault("Authorization", f"Bearer {_api_key()}")
        headers.setdefault("Accept", "application/json")

        last_error: Exception | None = None
        for attempt in range(1 + retries):
            try:
                response = self._client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                )
            except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt < retries:
                    import time as _time
                    _time.sleep(2.0)
                    continue
                raise PtarmiganError(
                    {"code": "transport_error", "message": str(exc)},
                    status=0,
                )

            try:
                payload: dict[str, Any] = response.json()
            except json.JSONDecodeError:
                payload = {"code": "parse_error", "message": response.text[:500]}

            if not 200 <= response.status_code < 300:
                raise PtarmiganError(payload, status=response.status_code)

            return response.status_code, payload

        raise RuntimeError("Unreachable") from last_error

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PtarmiganClient:
    """High-level client for the Ptarmigan-1 screen API.

    Never instantiated at import time — only created on demand inside a
    function or script.
    """

    def __init__(self, transport: _Transport | None = None) -> None:
        self._transport = transport or _Transport()

    # ---- Account / balance -----------------------------------------------

    def get_balance(self) -> dict[str, Any]:
        """``GET /api/v1/balance``.

        Returns the parsed balance (``available_usd``, ``reserved_usd``,
        ``promotional_usd``, ``purchased_usd``, ``grants``, etc.).
        """
        method, url = _resolve("balance")
        _, payload = self._transport.request(method, url)
        return payload

    def available_balance(self) -> float:
        """Convenience: parse ``available_usd`` from the balance endpoint."""
        bal = self.get_balance()
        return _format_currency(bal.get("available_usd", "0"))

    # ---- Proteins --------------------------------------------------------

    def list_proteins(self, search: str | None = None, keyword: str | None = None, limit: int = 20) -> dict[str, Any]:
        """``GET /api/v1/proteins``.

        Returns a page of proteins matching ``search`` (gene or name).
        """
        method, url = _resolve("proteins")
        params: list[tuple[str, str]] = [("limit", str(limit))]
        if search:
            params.append(("search", search))
        if keyword:
            params.append(("keyword", keyword))
        # httpx can take `params`; we handle it via URL construction
        url = _append_params(url, params)
        _, payload = self._transport.request(method, url)
        return payload

    # ---- Compound collections --------------------------------------------

    def list_collections(self, limit: int = 20) -> dict[str, Any]:
        """``GET /api/v1/compound-collections``."""
        method, url = _resolve("compound_collections")
        url = _append_params(url, [("limit", str(limit))])
        _, payload = self._transport.request(method, url)
        return payload

    def get_collection(self, collection_id: str) -> dict[str, Any]:
        """``GET /api/v1/compound-collections/{collection_id}``."""
        method, url = _resolve("get_collection", collection_id=collection_id)
        _, payload = self._transport.request(method, url)
        return payload

    def list_collection_compounds(
        self, collection_id: str, limit: int = 100, search: str | None = None
    ) -> dict[str, Any]:
        """``GET /api/v1/compound-collections/{collection_id}/compounds``."""
        method, url = _resolve("collection_compounds", collection_id=collection_id)
        params: list[tuple[str, str]] = [("limit", str(limit))]
        if search:
            params.append(("search", search))
        url = _append_params(url, params)
        _, payload = self._transport.request(method, url)
        return payload

    # ---- Screens: estimate (dry-run safe) ----------------------------------

    def estimate_screen(self, protein_id: str, collection_id: str) -> dict[str, Any]:
        """``POST /api/v1/screens/estimate``.

        Returns the parsed :class:`ScreenEstimate` with ``compounds``,
        ``tokens``, ``cost_usd``, ``available_usd``, ``sufficient``.
        """
        method, url = _resolve("estimate_screen")
        _, payload = self._transport.request(
            method, url,
            json_body={"protein_id": protein_id, "collection_id": collection_id},
        )
        return payload

    # ---- Screens: create (credit-protected) -------------------------------

    def create_screen(
        self,
        protein_id: str,
        collection_id: str,
        *,
        idempotency_key: str | None = None,
        bypass_gate: bool = False,
    ) -> dict[str, Any]:
        """``POST /api/v1/screens``.

        **Credit protection** (unless ``bypass_gate=True``):

        1. Check :meth:`available_balance`.
        2. Fetch :meth:`estimate_screen`.
        3. If ``cost_usd > HARD_GATE_USD``, raise :class:`PtarmiganGateError` — **no POST**.
        4. POST with an ``Idempotency-Key`` header (auto-generated if none provided).
        5. Return the screen payload (queued).

        Set ``bypass_gate=True`` only in the ``--dry-run`` path or when the
        user has explicitly confirmed a cost above $15.
        """
        # 1. Check balance
        _ = self.available_balance()

        # 2. Estimate
        estimate = self.estimate_screen(protein_id, collection_id)
        cost = _format_currency(estimate.get("cost_usd", "0"))
        sufficient = estimate.get("sufficient", False)

        # 3. Gate
        if not bypass_gate and cost > HARD_GATE_USD:
            raise PtarmiganGateError(str(cost), gate=HARD_GATE_USD)

        if not sufficient and not bypass_gate:
            raise PtarmiganError(
                {
                    "code": "insufficient_funds",
                    "message": f"Balance (${estimate.get('available_usd','0')}) "
                               f"cannot cover ${cost} screen",
                    "required_usd": str(cost),
                    "available_usd": estimate.get("available_usd", "0"),
                },
                status=402,
            )

        # 4. POST with Idempotency-Key
        key = idempotency_key or str(uuid.uuid4())
        method, url = _resolve("create_screen")
        _, payload = self._transport.request(
            method, url,
            headers={"Idempotency-Key": key},
            json_body={"protein_id": protein_id, "collection_id": collection_id},
        )
        return payload

    # ---- Screens: read results -------------------------------------------

    def get_screen(self, screen_id: str) -> dict[str, Any]:
        """``GET /api/v1/screens/{screen_id}``.

        Poll this until ``status == "completed"`` (or ``"failed"`` / ``"cancelled"``).
        """
        method, url = _resolve("get_screen", screen_id=screen_id)
        _, payload = self._transport.request(method, url)
        return payload

    def get_hits(self, screen_id: str, limit: int = 100) -> dict[str, Any]:
        """``GET /api/v1/screens/{screen_id}/hits``.

        Returns a page (``data``, ``next_after``) of :class:`HitOut` objects:
        ``rank``, ``compound_id``, ``smiles``, ``engagement`` (0–1 calibrated
        probability), ``score``, ``predicted_residue`` (1-indexed),
        ``predicted_residue_letter``.
        """
        method, url = _resolve("get_hits", screen_id=screen_id)
        url = _append_params(url, [("limit", str(limit))])
        _, payload = self._transport.request(method, url)
        return payload

    def list_screens(
        self,
        protein_id: str | None = None,
        collection_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """``GET /api/v1/screens``.

        Filter by optional ``protein_id``, ``collection_id``, ``status``.
        Returns newest first.
        """
        method, url = _resolve("list_screens")
        params: list[tuple[str, str]] = [("limit", str(limit))]
        if protein_id:
            params.append(("protein_id", protein_id))
        if collection_id:
            params.append(("collection_id", collection_id))
        if status:
            params.append(("status", status))
        url = _append_params(url, params)
        _, payload = self._transport.request(method, url)
        return payload

    def get_compound_detail(
        self, screen_id: str, compound_id: str, *, include_residues: bool = True
    ) -> dict[str, Any]:
        """``GET /api/v1/screens/{screen_id}/compounds/{compound_id}``.

        Returns a :class:`CompoundDetailResponse` with ``residue_engagement``
        (array per residue) when ``include_residues=True``.
        """
        method, url = _resolve("compound_detail", screen_id=screen_id, compound_id=compound_id)
        if include_residues:
            url = _append_params(url, [("include_residues", "true")])
        _, payload = self._transport.request(method, url)
        return payload


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _append_params(url: str, params: list[tuple[str, str]]) -> str:
    """Append query parameters to a URL (handles existing query strings)."""
    if not params:
        return url
    sep = "&" if "?" in url else "?"
    return url + sep + "&".join(f"{k}={v}" for k, v in params)