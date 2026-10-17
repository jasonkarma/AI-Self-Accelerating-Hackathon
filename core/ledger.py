"""Append-only SQLite reveal and decision ledger with a SHA-256 hash chain."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any
import pandas as pd

from config.settings import SETTINGS


class ImmutableLedger:
    def __init__(self, path: Path | str = SETTINGS.ledger_path, *, metadata: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize(metadata or {})

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _initialize(self, metadata: dict[str, Any]) -> None:
        with self.connect() as con:
            con.executescript("""
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
            """)
            for key, value in metadata.items():
                con.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES (?,?)", (key, json.dumps(value, sort_keys=True)))

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
