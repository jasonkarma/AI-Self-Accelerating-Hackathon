"""Append-only SQLite reveal and decision ledger with a SHA-256 hash chain."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any
import pandas as pd

from config.settings import SETTINGS

#: Columns copied verbatim when a ledger is rewound to an earlier round.
_ENTRY_COLUMNS = (
    "sequence_no", "event_type", "candidate_id", "round_number", "actor",
    "batch_tag", "payload", "previous_hash", "entry_hash", "created_at",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
  sequence_no INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL, candidate_id TEXT, round_number INTEGER,
  actor TEXT NOT NULL, batch_tag TEXT, payload TEXT NOT NULL,
  previous_hash TEXT NOT NULL, entry_hash TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS entries_no_update BEFORE UPDATE ON entries
BEGIN SELECT RAISE(ABORT, 'immutable ledger: UPDATE forbidden'); END;
CREATE TRIGGER IF NOT EXISTS entries_no_delete BEFORE DELETE ON entries
BEGIN SELECT RAISE(ABORT, 'immutable ledger: DELETE forbidden'); END;
"""

#: The genesis link of every hash chain.
GENESIS_HASH = "0" * 64


def _connect(path: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(Path(path))
    con.row_factory = sqlite3.Row
    return con


def read_entries(path: Path | str, *, max_sequence_no: int | None = None) -> list[dict[str, Any]]:
    """Every entry up to ``max_sequence_no``, in append order, as plain dicts.

    Rows are returned **verbatim** including their hash links, so a prefix can be
    written into a new file without re-hashing anything: the surviving chain is
    the chain that was already verified.
    """
    query = "SELECT * FROM entries"
    args: tuple[Any, ...] = ()
    if max_sequence_no is not None:
        query += " WHERE sequence_no <= ?"
        args = (int(max_sequence_no),)
    query += " ORDER BY sequence_no"
    with _connect(path) as con:
        return [dict(row) for row in con.execute(query, args)]


def read_metadata(path: Path | str) -> dict[str, Any]:
    with _connect(path) as con:
        return {row[0]: json.loads(row[1]) for row in con.execute("SELECT key,value FROM metadata")}


def write_ledger_file(
    destination: Path | str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Materialize a complete ledger file, atomically replacing ``destination``.

    Used by the rewind path, where deleting rows in place is impossible *by
    design* (the ``entries_no_delete`` trigger). Rebuilding the file from the
    surviving prefix is the only way to go backwards that cannot leave a
    half-truncated chain behind: the replacement is a single ``os.replace``.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    if staging.exists():
        staging.unlink()
    con = sqlite3.connect(staging)
    try:
        con.executescript(_SCHEMA)
        con.executemany(
            f"INSERT INTO entries({','.join(_ENTRY_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(_ENTRY_COLUMNS))})",
            [tuple(row.get(column) for column in _ENTRY_COLUMNS) for row in rows],
        )
        for key, value in (metadata or {}).items():
            con.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                (key, json.dumps(value, sort_keys=True)),
            )
        con.commit()
    finally:
        con.close()
    os.replace(staging, destination)
    return destination


def copy_ledger(
    source: Path | str,
    destination: Path | str,
    *,
    max_sequence_no: int | None = None,
) -> int:
    """Copy ``source`` to ``destination``, optionally stopping at a sequence no.

    Returns the number of entries written. The metadata table travels with the
    entries so the restored file is a *ledger*, not just a table: a campaign
    rebuilt from it still knows which policy mode stamped it.
    """
    rows = read_entries(source, max_sequence_no=max_sequence_no)
    write_ledger_file(destination, rows, read_metadata(source))
    return len(rows)


def ledger_version(path: Path | str) -> tuple[int, int]:
    """``(max sequence_no, entry count)`` — a cheap content fingerprint.

    Caches keyed on this survive every append (the pair changes) and every rewind
    (it changes back), without stat-ing mtimes that a copy would preserve.
    """
    try:
        with _connect(path) as con:
            row = con.execute("SELECT COALESCE(MAX(sequence_no), 0), COUNT(*) FROM entries").fetchone()
    except sqlite3.OperationalError:
        return (0, 0)
    return (int(row[0]), int(row[1]))


def is_ledger(path: Path | str) -> bool:
    """True when ``path`` is a readable ledger file with a hash chain."""
    try:
        version = ledger_version(path)
    except sqlite3.DatabaseError:
        return False
    return version[1] > 0


class ImmutableLedger:
    def __init__(self, path: Path | str = SETTINGS.ledger_path, *, metadata: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize(metadata or {})

    def connect(self) -> sqlite3.Connection:
        return _connect(self.path)

    def _initialize(self, metadata: dict[str, Any]) -> None:
        with self.connect() as con:
            con.executescript(_SCHEMA)
            for key, value in metadata.items():
                con.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES (?,?)", (key, json.dumps(value, sort_keys=True)))

    # ── Time travel ──────────────────────────────────────────────────────────
    def version(self) -> tuple[int, int]:
        """Content fingerprint used to memoize ledger-derived frames."""
        return ledger_version(self.path)

    def cutoff_for_round(self, round_number: int) -> int:
        """Highest ``sequence_no`` written at or before the end of ``round_number``.

        ``round_number == 0`` is the seed batch, so the cutoff is the last seed
        reveal. Entries that carry no round tag (none are written today, but the
        column is nullable) are always kept: they are not round-scoped
        bookkeeping, so dropping them would lose history rather than truncate it.
        """
        with self.connect() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) FROM entries "
                "WHERE round_number IS NULL OR round_number <= ?",
                (int(round_number),),
            ).fetchone()
        return int(row[0])

    def truncate_to_round(self, round_number: int) -> dict[str, int]:
        """Rewrite the file so it ends at the close of ``round_number``.

        Rewind is *structural*, not a filter applied at read time: the ledger on
        disk becomes the shorter one, so a subsequent ``execute_round`` appends to
        the truncated chain exactly as if the later rounds had never run. The
        immutability triggers are honoured rather than defeated — no row is ever
        updated or deleted; the whole file is rebuilt from a verified prefix and
        swapped in atomically.
        """
        cutoff = self.cutoff_for_round(round_number)
        before = self.version()[1]
        copy_ledger(self.path, self.path, max_sequence_no=cutoff)
        after = self.version()[1]
        return {
            "round": int(round_number),
            "cutoff_sequence_no": cutoff,
            "entries_before": before,
            "entries_after": after,
            "entries_dropped": before - after,
        }

    def append(self, event_type: str, payload: dict[str, Any], *, actor: str,
               candidate_id: str | None = None, round_number: int | None = None,
               batch_tag: str | None = None) -> dict[str, Any]:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with self.connect() as con:
            last = con.execute("SELECT entry_hash FROM entries ORDER BY sequence_no DESC LIMIT 1").fetchone()
            previous = last[0] if last else "0" * 64
            material = "|".join(map(str, (previous, event_type, candidate_id or "", round_number or "", actor, batch_tag or "", serialized)))
            digest = hashlib.sha256(material.encode()).hexdigest()
            cursor = con.execute(
                "INSERT INTO entries(event_type,candidate_id,round_number,actor,batch_tag,payload,previous_hash,entry_hash) VALUES (?,?,?,?,?,?,?,?)",
                (event_type, candidate_id, round_number, actor, batch_tag, serialized, previous, digest),
            )
            sequence_no = int(cursor.lastrowid)
        return {"sequence_no": sequence_no, "entry_hash": digest, **payload}

    def find_reveal(self, candidate_id: str, objective: str) -> dict[str, Any] | None:
        with self.connect() as con:
            rows = con.execute("SELECT payload FROM entries WHERE event_type='oracle_reveal' AND candidate_id=? ORDER BY sequence_no", (candidate_id,)).fetchall()
        for row in rows:
            payload = json.loads(row[0])
            if payload.get("objective") == objective:
                return payload
        return None

    def dataframe(self, event_type: str | None = None) -> pd.DataFrame:
        query, args = "SELECT * FROM entries", ()
        if event_type:
            query, args = query + " WHERE event_type=?", (event_type,)
        query += " ORDER BY sequence_no"
        with self.connect() as con:
            frame = pd.read_sql_query(query, con, params=args)
        if not frame.empty:
            expanded = pd.json_normalize(frame.pop("payload").map(json.loads))
            frame = pd.concat([frame.reset_index(drop=True), expanded], axis=1)
        return frame

    def metadata(self) -> dict[str, Any]:
        with self.connect() as con:
            return {row[0]: json.loads(row[1]) for row in con.execute("SELECT key,value FROM metadata")}

    def set_metadata(self, key: str, value: Any) -> None:
        with self.connect() as con:
            con.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", (key, json.dumps(value, sort_keys=True)))

    def verify(self) -> bool:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM entries ORDER BY sequence_no").fetchall()
        previous = "0" * 64
        for row in rows:
            material = "|".join(map(str, (previous, row["event_type"], row["candidate_id"] or "", row["round_number"] or "", row["actor"], row["batch_tag"] or "", row["payload"])))
            digest = hashlib.sha256(material.encode()).hexdigest()
            if row["previous_hash"] != previous or row["entry_hash"] != digest:
                return False
            previous = digest
        return True
