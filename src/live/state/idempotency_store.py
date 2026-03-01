from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class IdempotencyRecord:
    idempotency_key: str
    intent_id: str
    client_order_id: str
    state: str  # READY/ACCEPTED/FILLED/ATTACHED/OPEN_VERIFIED/CLOSED/FAILED_FINAL

    attempt_no: int = 1
    broker_order_id: Optional[int] = None
    broker_deal_id: Optional[int] = None
    broker_position_id: Optional[int] = None

    # Optional analytics fields (added v1.1)
    entry_equity: Optional[float] = None
    entry_ts_utc: str = ""

    created_ts_utc: str = ""
    updated_ts_utc: str = ""


class IdempotencyStore:
    """
    Persistent idempotency store (SQLite).
    Purpose:
      - prevent duplicate submissions on restart / unknown outcome
      - keep mapping intent_id <-> broker tickets
      - state machine durability
      - store minimal analytics fields (e.g., entry_equity) for fallback finalizers
    """

    TABLE = "idempotency"

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # durability & concurrency helpers
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    idempotency_key TEXT PRIMARY KEY,
                    intent_id TEXT NOT NULL,
                    client_order_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt_no INTEGER NOT NULL DEFAULT 1,
                    broker_order_id INTEGER,
                    broker_deal_id INTEGER,
                    broker_position_id INTEGER,
                    entry_equity REAL,
                    entry_ts_utc TEXT,
                    created_ts_utc TEXT NOT NULL,
                    updated_ts_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_intent_id ON {self.TABLE}(intent_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_state ON {self.TABLE}(state)")
            conn.commit()

            # Lightweight migration for existing DBs (adds missing columns safely)
            self._ensure_column(conn, "entry_equity", "REAL")
            self._ensure_column(conn, "entry_ts_utc", "TEXT")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, col: str, col_type: str) -> None:
        """
        Adds column if missing. Safe for existing DBs; no data loss.
        """
        table = "idempotency"
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = {row["name"] for row in cur.fetchall()}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            conn.commit()

    def get(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {self.TABLE} WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            return dict(row) if row else None

    def upsert_new(self, record: IdempotencyRecord) -> None:
        now = _now_utc_iso()
        created = record.created_ts_utc or now
        updated = record.updated_ts_utc or now
        entry_ts = record.entry_ts_utc or None

        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.TABLE} (
                    idempotency_key,intent_id,client_order_id,state,
                    attempt_no,broker_order_id,broker_deal_id,broker_position_id,
                    entry_equity,entry_ts_utc,
                    created_ts_utc,updated_ts_utc
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (
                    record.idempotency_key,
                    record.intent_id,
                    record.client_order_id,
                    record.state,
                    int(record.attempt_no),
                    record.broker_order_id,
                    record.broker_deal_id,
                    record.broker_position_id,
                    record.entry_equity,
                    entry_ts,
                    created,
                    updated,
                ),
            )
            conn.commit()

    def mark_state(
        self,
        idempotency_key: str,
        *,
        state: str,
        attempt_no: Optional[int] = None,
        broker_order_id: Optional[int] = None,
        broker_deal_id: Optional[int] = None,
        broker_position_id: Optional[int] = None,
    ) -> None:
        now = _now_utc_iso()
        fields = ["state = ?", "updated_ts_utc = ?"]
        params: list[Any] = [state, now]

        if attempt_no is not None:
            fields.append("attempt_no = ?")
            params.append(int(attempt_no))
        if broker_order_id is not None:
            fields.append("broker_order_id = ?")
            params.append(int(broker_order_id))
        if broker_deal_id is not None:
            fields.append("broker_deal_id = ?")
            params.append(int(broker_deal_id))
        if broker_position_id is not None:
            fields.append("broker_position_id = ?")
            params.append(int(broker_position_id))

        params.append(idempotency_key)

        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE {self.TABLE} SET {', '.join(fields)} WHERE idempotency_key = ?",
                tuple(params),
            )
            if cur.rowcount == 0:
                raise KeyError(f"idempotency_key not found: {idempotency_key}")
            conn.commit()

    def update_fields(self, idempotency_key: str, **fields: Any) -> None:
        """
        Generic partial update for extra columns (e.g., entry_equity, entry_ts_utc).
        Only updates known columns; always updates updated_ts_utc.
        """
        if not fields:
            return

        # whitelist allowed columns to prevent accidental schema drift
        allowed = {
            "intent_id",
            "client_order_id",
            "state",
            "attempt_no",
            "broker_order_id",
            "broker_deal_id",
            "broker_position_id",
            "entry_equity",
            "entry_ts_utc",
        }

        bad = [k for k in fields.keys() if k not in allowed]
        if bad:
            raise ValueError(f"update_fields contains unknown columns: {bad}")

        now = _now_utc_iso()
        set_parts: list[str] = []
        params: list[Any] = []

        for k, v in fields.items():
            set_parts.append(f"{k} = ?")
            params.append(v)

        set_parts.append("updated_ts_utc = ?")
        params.append(now)

        params.append(idempotency_key)

        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE {self.TABLE} SET {', '.join(set_parts)} WHERE idempotency_key = ?",
                tuple(params),
            )
            if cur.rowcount == 0:
                raise KeyError(f"idempotency_key not found: {idempotency_key}")
            conn.commit()

    def list_recent(self, limit: int = 50) -> list[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {self.TABLE} ORDER BY updated_ts_utc DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]