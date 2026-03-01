from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict, List


ORDERS_SCHEMA_VERSION = "orders_v1"

ORDERS_HEADER: List[str] = [
    "schema_version",
    "run_id",
    "run_mode",
    "portfolio_mode",
    "ts_utc",
    "ftmo_day_id",
    "intent_id",
    "strategy_id",
    "strategy_version",
    "instrument",
    "timeframe",
    "side",
    "order_type",
    "time_in_force",
    "risk_mode",
    "risk_value_pct",
    "risk_value_decimal",
    "requested_sl_pips",
    "requested_tp_pips",
    "max_hold_min",
    "valid_to_utc",
    "idempotency_key",
    "client_order_id",
    "broker_symbol",
    "broker_order_id",
    "broker_position_id",
    "event_type",
    "event_state",
    "event_reason",
    "requested_qty",
    "filled_qty",
    "request_price",
    "fill_price",
    "fill_ts_utc",
    "sl_price",
    "tp_price",
    "broker_comment",
    "broker_magic",
    "last_error_code",
    "last_error_msg",
    "latency_ms",
    "attempt_no",
]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


class OrdersLog:
    """
    Append-only CSV logger for orders lifecycle events.
    Safe:
      - writes header if file not exists
      - writes only known columns, unknown fields ignored
    """

    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path)
        _ensure_parent(self.path)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(ORDERS_HEADER)

    def append(self, row: Dict[str, Any]) -> None:
        """
        `row` can contain extra keys; only header keys are written.
        Missing keys are written as empty.
        """
        # Enforce schema version and ts if caller forgot
        row = dict(row)
        row.setdefault("schema_version", ORDERS_SCHEMA_VERSION)
        row.setdefault("ts_utc", _now_utc_iso())

        out = [_normalize_value(row.get(col)) for col in ORDERS_HEADER]

        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(out)

    def append_event(
        self,
        *,
        run_id: str,
        run_mode: str,
        portfolio_mode: str,
        ftmo_day_id: str,
        intent_id: str,
        strategy_id: str,
        strategy_version: str,
        instrument: str,
        timeframe: str,
        side: str,
        order_type: str,
        time_in_force: str,
        risk_mode: str,
        risk_value_pct: float,
        requested_sl_pips: float,
        requested_tp_pips: float,
        max_hold_min: int,
        valid_to_utc: str,
        idempotency_key: str,
        client_order_id: str,
        event_type: str,
        event_state: str,
        event_reason: str,
        # Optional broker & fill details
        broker_symbol: Optional[str] = None,
        broker_order_id: Optional[int] = None,
        broker_position_id: Optional[int] = None,
        requested_qty: Optional[float] = None,
        filled_qty: Optional[float] = None,
        request_price: Optional[float] = None,
        fill_price: Optional[float] = None,
        fill_ts_utc: Optional[str] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        broker_comment: Optional[str] = None,
        broker_magic: Optional[int] = None,
        last_error_code: Optional[int] = None,
        last_error_msg: Optional[str] = None,
        latency_ms: Optional[int] = None,
        attempt_no: Optional[int] = None,
    ) -> None:
        risk_value_decimal = float(risk_value_pct) / 100.0

        self.append(
            {
                "run_id": run_id,
                "run_mode": run_mode,
                "portfolio_mode": portfolio_mode,
                "ftmo_day_id": ftmo_day_id,
                "intent_id": intent_id,
                "strategy_id": strategy_id,
                "strategy_version": strategy_version,
                "instrument": instrument,
                "timeframe": timeframe,
                "side": side,
                "order_type": order_type,
                "time_in_force": time_in_force,
                "risk_mode": risk_mode,
                "risk_value_pct": risk_value_pct,
                "risk_value_decimal": risk_value_decimal,
                "requested_sl_pips": requested_sl_pips,
                "requested_tp_pips": requested_tp_pips,
                "max_hold_min": max_hold_min,
                "valid_to_utc": valid_to_utc,
                "idempotency_key": idempotency_key,
                "client_order_id": client_order_id,
                "broker_symbol": broker_symbol or instrument,
                "broker_order_id": broker_order_id,
                "broker_position_id": broker_position_id,
                "event_type": event_type,
                "event_state": event_state,
                "event_reason": event_reason,
                "requested_qty": requested_qty,
                "filled_qty": filled_qty,
                "request_price": request_price,
                "fill_price": fill_price,
                "fill_ts_utc": fill_ts_utc,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "broker_comment": broker_comment,
                "broker_magic": broker_magic,
                "last_error_code": last_error_code,
                "last_error_msg": last_error_msg,
                "latency_ms": latency_ms,
                "attempt_no": attempt_no,
            }
        )