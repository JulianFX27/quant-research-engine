from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.execution.paper_engine import Trade


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def append_log(path: str, line: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def _write_csv_header_if_needed(path: str, fieldnames: list[str]) -> None:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_trade_csv(path: str, trade: Trade, day_id_ftmo: str, dd_at_entry_pct: float, dd_at_exit_pct: float, daily_pnl_pct_at_entry: float) -> None:
    # Flatten trade dataclass
    row = {
        "trade_id": trade.trade_id,
        "intent_id": trade.intent_id,
        "direction": trade.direction,
        "entry_time_utc": trade.entry_time_utc.isoformat().replace("+00:00", "Z"),
        "exit_time_utc": trade.exit_time_utc.isoformat().replace("+00:00", "Z") if trade.exit_time_utc else None,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "sl_price": trade.sl_price,
        "tp_price": trade.tp_price,
        "exit_reason": trade.exit_reason,
        "risk_base_pct": trade.risk_base_pct,
        "risk_multiplier": trade.risk_multiplier,
        "risk_effective_pct": trade.risk_effective_pct,
        "risk_price": trade.risk_price,
        "pnl": trade.pnl,
        "R": trade.R,
        "day_id_ftmo": day_id_ftmo,
        "dd_at_entry_pct": dd_at_entry_pct,
        "dd_at_exit_pct": dd_at_exit_pct,
        "daily_pnl_pct_at_entry": daily_pnl_pct_at_entry,
        "meta_json": json.dumps(trade.meta, ensure_ascii=False),
    }

    fieldnames = list(row.keys())
    _write_csv_header_if_needed(path, fieldnames)
    with Path(path).open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)


def append_equity_csv(path: str, ts_utc: datetime, equity: float, equity_peak: float, dd_pct: float, day_id_ftmo: str) -> None:
    row = {
        "ts_utc": ts_utc.isoformat().replace("+00:00", "Z"),
        "equity": equity,
        "equity_peak": equity_peak,
        "dd_pct": dd_pct,
        "day_id_ftmo": day_id_ftmo,
    }
    fieldnames = list(row.keys())
    _write_csv_header_if_needed(path, fieldnames)
    with Path(path).open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)


def _read_daily_summary(path: str) -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["day_id_ftmo"]] = row
    return out


def _write_daily_summary(path: str, rows: Dict[str, Dict[str, Any]]) -> None:
    # normalize header
    fieldnames = [
        "day_id_ftmo",
        "equity_start_day",
        "equity_end_day",
        "daily_pnl_pct",
        "max_intraday_dd_pct",
        "trades_taken",
        "daily_stop_triggered",
        "hard_stop_triggered",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for day_id in sorted(rows.keys()):
            w.writerow(rows[day_id])


def upsert_daily_summary(
    path: str,
    *,
    day_id_ftmo: str,
    equity_start_day: float,
    equity_end_day: float,
    daily_pnl_pct: float,
    max_intraday_dd_pct: float,
    trades_taken: int,
    daily_stop_triggered: bool,
    hard_stop_triggered: bool,
) -> None:
    rows = _read_daily_summary(path)
    rows[day_id_ftmo] = {
        "day_id_ftmo": day_id_ftmo,
        "equity_start_day": equity_start_day,
        "equity_end_day": equity_end_day,
        "daily_pnl_pct": daily_pnl_pct,
        "max_intraday_dd_pct": max_intraday_dd_pct,
        "trades_taken": trades_taken,
        "daily_stop_triggered": str(bool(daily_stop_triggered)),
        "hard_stop_triggered": str(bool(hard_stop_triggered)),
    }
    _write_daily_summary(path, rows)
