from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


INCIDENTS_SCHEMA_VERSION = "incidents_v1"

INCIDENTS_HEADER: List[str] = [
    "schema_version",
    "run_id",
    "run_mode",
    "portfolio_mode",
    "ts_utc",
    "severity",          # WARN|ERROR|CRITICAL
    "type",              # ASSOC_AMBIGUOUS, UNASSOCIATED_POSITION, SLTP_ATTACH_FAILED, etc.
    "action_taken",      # HALT, FORCE_CLOSE, FORCE_CANCEL, IGNORE
    "ftmo_day_id",
    "intent_id",
    "strategy_id",
    "instrument",
    "broker_order_id",
    "broker_position_id",
    "details",
]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _norm(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


class IncidentsLog:
    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path)
        _ensure_parent(self.path)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(INCIDENTS_HEADER)

    def append(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("schema_version", INCIDENTS_SCHEMA_VERSION)
        row.setdefault("ts_utc", _now_utc_iso())
        out = [_norm(row.get(col)) for col in INCIDENTS_HEADER]
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(out)

    def log(
        self,
        *,
        run_id: str,
        run_mode: str,
        portfolio_mode: str,
        severity: str,
        type: str,
        action_taken: str,
        ftmo_day_id: str,
        instrument: str,
        details: str,
        intent_id: str = "",
        strategy_id: str = "",
        broker_order_id: Optional[int] = None,
        broker_position_id: Optional[int] = None,
    ) -> None:
        self.append(
            {
                "run_id": run_id,
                "run_mode": run_mode,
                "portfolio_mode": portfolio_mode,
                "severity": severity,
                "type": type,
                "action_taken": action_taken,
                "ftmo_day_id": ftmo_day_id,
                "intent_id": intent_id,
                "strategy_id": strategy_id,
                "instrument": instrument,
                "broker_order_id": broker_order_id,
                "broker_position_id": broker_position_id,
                "details": details,
            }
        )