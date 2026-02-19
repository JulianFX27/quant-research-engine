import json
import os
from pathlib import Path


def write_json(path: str, obj: dict) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def base_state() -> dict:
    return {
        "schema_version": "1.0.0",
        "policy_name": "FTMO_Anchor_ExecutionPolicy_v1",
        "account_mode": "challenge",
        "timezone_day_rollover": "Europe/Prague",
        "day_rollover_time": "00:00:00",
        "initial_balance": 1.0,
        "equity_current": 1.0,
        "equity_peak": 1.0,
        "current_day_id": None,
        "equity_start_day": None,
        "daily_pnl_pct": 0.0,
        "dd_current_pct": 0.0,
        "hard_stop_dd_triggered": False,
        "daily_stop_triggered": False,
        "trading_enabled": True,
        "trades_taken_today": 0,
        "max_trades_per_day": 1,
        "last_update_utc": None,
        "last_run_id": None,
        "audit_tail": [],
    }
