from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    ZoneInfo = None  # type: ignore

from .io_utils import json_read, atomic_json_write


REQUIRED_KEYS = [
    "schema_version",
    "policy_name",
    "account_mode",
    "timezone_day_rollover",
    "day_rollover_time",
    "initial_balance",
    "equity_current",
    "equity_peak",
    "current_day_id",
    "equity_start_day",
    "daily_pnl_pct",
    "dd_current_pct",
    "hard_stop_dd_triggered",
    "daily_stop_triggered",
    "trading_enabled",
    "trades_taken_today",
    "max_trades_per_day",
    "last_update_utc",
    "last_run_id",
    "audit_tail",
]


@dataclass(frozen=True)
class RiskConfig:
    tz_name: str = "Europe/Prague"
    rollover_time: str = "00:00:00"  # informational; day_id is by local date
    daily_stop_pct: float = -1.5
    hard_stop_dd_pct: float = 8.0
    max_audit: int = 200


class RiskStateManager:
    def __init__(self, state_path: str, override_path: str, config: Optional[RiskConfig] = None):
        self.state_path = state_path
        self.override_path = override_path
        self.cfg = config or RiskConfig()

        if ZoneInfo is None:
            raise RuntimeError("Python >= 3.9 required (zoneinfo).")

        self._tz = ZoneInfo(self.cfg.tz_name)

    # ----------------------------
    # IO
    # ----------------------------
    def load_state(self) -> Dict[str, Any]:
        state = json_read(self.state_path)
        self._validate_schema(state)
        return state

    def save_state(self, state: Dict[str, Any]) -> None:
        atomic_json_write(self.state_path, state)

    def load_override(self) -> Dict[str, Any]:
        try:
            override = json_read(self.override_path)
        except FileNotFoundError:
            return {"allow_reset_hard_stop": False, "reason": None, "approved_by": None, "ts_utc": None}
        return override

    # ----------------------------
    # Time helpers
    # ----------------------------
    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def current_day_id(self, now_utc: Optional[datetime] = None) -> str:
        now_utc = now_utc or self._now_utc()
        local_dt = now_utc.astimezone(self._tz)
        return local_dt.date().isoformat()

    def is_new_day(self, state: Dict[str, Any], now_utc: Optional[datetime] = None) -> bool:
        return state.get("current_day_id") != self.current_day_id(now_utc)

    # ----------------------------
    # Rollover
    # ----------------------------
    def rollover_if_needed(self, state: Dict[str, Any], run_id: str, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        now_utc = now_utc or self._now_utc()
        if not self.is_new_day(state, now_utc):
            return state

        state["current_day_id"] = self.current_day_id(now_utc)
        state["equity_start_day"] = state["equity_current"]
        state["daily_pnl_pct"] = 0.0
        state["daily_stop_triggered"] = False
        state["trades_taken_today"] = 0

        # enabled depends on hard stop
        state["trading_enabled"] = not bool(state.get("hard_stop_dd_triggered", False))

        state["last_update_utc"] = now_utc.isoformat().replace("+00:00", "Z")
        state["last_run_id"] = run_id

        self._audit(state, now_utc, run_id, event="DAY_ROLLOVER")
        return state

    # ----------------------------
    # Core update after trade close
    # ----------------------------
    def on_trade_closed(self, state: Dict[str, Any], equity_after: float, run_id: str, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        now_utc = now_utc or self._now_utc()

        # Ensure day initialized
        state = self.rollover_if_needed(state, run_id=run_id, now_utc=now_utc)

        state["equity_current"] = float(equity_after)
        state["equity_peak"] = float(max(state["equity_peak"], state["equity_current"]))

        # Drawdown from peak
        peak = state["equity_peak"]
        dd = (peak - state["equity_current"]) / peak if peak > 0 else 0.0
        state["dd_current_pct"] = float(dd * 100.0)

        # Daily PnL
        day_start = state.get("equity_start_day")
        if day_start is None or day_start <= 0:
            state["equity_start_day"] = state["equity_current"]
            day_start = state["equity_start_day"]

        daily_pnl = (state["equity_current"] - day_start) / day_start
        state["daily_pnl_pct"] = float(daily_pnl * 100.0)

        # trades today
        state["trades_taken_today"] = int(state.get("trades_taken_today", 0) + 1)

        # Apply internal gates
        if state["daily_pnl_pct"] <= self.cfg.daily_stop_pct:
            state["daily_stop_triggered"] = True
            state["trading_enabled"] = False

        if state["dd_current_pct"] >= self.cfg.hard_stop_dd_pct:
            state["hard_stop_dd_triggered"] = True
            state["trading_enabled"] = False

        if state["trades_taken_today"] >= state.get("max_trades_per_day", 1):
            state["trading_enabled"] = False

        state["last_update_utc"] = now_utc.isoformat().replace("+00:00", "Z")
        state["last_run_id"] = run_id

        self._audit(state, now_utc, run_id, event="TRADE_CLOSED")
        return state

    # ----------------------------
    # Permission check
    # ----------------------------
    def can_trade(self, state: Dict[str, Any], run_id: str, now_utc: Optional[datetime] = None) -> bool:
        now_utc = now_utc or self._now_utc()
        state = self.rollover_if_needed(state, run_id=run_id, now_utc=now_utc)

        if not bool(state.get("trading_enabled", True)):
            return False
        if bool(state.get("hard_stop_dd_triggered", False)):
            return False
        if bool(state.get("daily_stop_triggered", False)):
            return False
        if int(state.get("trades_taken_today", 0)) >= int(state.get("max_trades_per_day", 1)):
            return False
        return True

    # ----------------------------
    # Manual reset
    # ----------------------------
    def try_manual_reset_hard_stop(self, state: Dict[str, Any], run_id: str, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        now_utc = now_utc or self._now_utc()
        override = self.load_override()

        if not override.get("allow_reset_hard_stop", False):
            return state

        state["hard_stop_dd_triggered"] = False

        # Re-enable only if other gates aren't active
        if not bool(state.get("daily_stop_triggered", False)) and int(state.get("trades_taken_today", 0)) < int(state.get("max_trades_per_day", 1)):
            state["trading_enabled"] = True

        state["last_update_utc"] = now_utc.isoformat().replace("+00:00", "Z")
        state["last_run_id"] = run_id

        self._audit(state, now_utc, run_id, event="MANUAL_RESET_HARD_STOP", extra={"override": override})
        return state

    # ----------------------------
    # Schema + Audit
    # ----------------------------
    def _validate_schema(self, state: Dict[str, Any]) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in state]
        if missing:
            raise ValueError(f"Risk state missing keys: {missing}")

    def _audit(self, state: Dict[str, Any], now_utc: datetime, run_id: str, event: str, extra: Optional[Dict[str, Any]] = None) -> None:
        rec: Dict[str, Any] = {
            "ts_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "run_id": run_id,
            "event": event,
            "equity_current": state.get("equity_current"),
            "equity_peak": state.get("equity_peak"),
            "dd_current_pct": state.get("dd_current_pct"),
            "equity_start_day": state.get("equity_start_day"),
            "daily_pnl_pct": state.get("daily_pnl_pct"),
            "trades_taken_today": state.get("trades_taken_today"),
            "trading_enabled": state.get("trading_enabled"),
        }
        if extra is not None:
            rec["extra"] = extra

        tail = list(state.get("audit_tail", []))
        tail.append(rec)

        if len(tail) > self.cfg.max_audit:
            tail = tail[-self.cfg.max_audit :]

        state["audit_tail"] = tail
