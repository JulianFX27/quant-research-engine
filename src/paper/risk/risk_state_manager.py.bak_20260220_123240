from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone, time, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    ZoneInfo = None  # type: ignore

from .io_utils import json_read, atomic_json_write


# ----------------------------
# Schema
# ----------------------------
SCHEMA_VERSION = 2

# v1 keys (legacy) + v2 keys (FTMO-style loss tracking)
REQUIRED_KEYS_V1 = [
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

# New in v2 (FTMO-style)
REQUIRED_KEYS_V2_EXTRA = [
    "daily_loss_pct",       # positive number, e.g. 2.1 means -2.1% daily
    "overall_loss_pct",     # positive number from initial_balance
    "dd_from_peak_pct",     # positive number from equity_peak (analytics)
]

DEFAULT_OVERRIDE = {"allow_reset_hard_stop": False, "reason": None, "approved_by": None, "ts_utc": None}


@dataclass(frozen=True)
class RiskConfig:
    tz_name: str = "Europe/Prague"
    rollover_time: str = "00:00:00"  # day boundary in local time (FTMO: 00:00:00 CET/CEST)

    # IMPORTANT:
    # These are interpreted as LOSS LIMITS (positive percentages), FTMO-style.
    # Example: max_daily_loss_pct=5.0 means stop when daily_loss_pct >= 5.0
    max_daily_loss_pct: float = 5.0
    max_overall_loss_pct: float = 10.0

    max_audit: int = 200


class RiskStateManager:
    def __init__(self, state_path: str, override_path: str, config: Optional[RiskConfig] = None):
        self.state_path = state_path
        self.override_path = override_path
        self.cfg = config or RiskConfig()

        if ZoneInfo is None:
            raise RuntimeError("Python >= 3.9 required (zoneinfo).")

        self._tz = ZoneInfo(self.cfg.tz_name)
        self._rollover_local_time = self._parse_rollover_time(self.cfg.rollover_time)

    # ----------------------------
    # IO
    # ----------------------------
    def load_state(self) -> Dict[str, Any]:
        state = json_read(self.state_path)
        state = self._upgrade_and_validate(state)
        return state

    def save_state(self, state: Dict[str, Any]) -> None:
        atomic_json_write(self.state_path, state)

    def load_override(self) -> Dict[str, Any]:
        try:
            override = json_read(self.override_path)
        except FileNotFoundError:
            return dict(DEFAULT_OVERRIDE)
        return override

    # ----------------------------
    # Time helpers
    # ----------------------------
    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_rollover_time(s: str) -> time:
        """
        Parse 'HH:MM:SS' -> datetime.time
        """
        try:
            parts = s.strip().split(":")
            if len(parts) != 3:
                raise ValueError
            hh, mm, ss = (int(parts[0]), int(parts[1]), int(parts[2]))
            if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
                raise ValueError
            return time(hour=hh, minute=mm, second=ss)
        except Exception as e:
            raise ValueError(f"Invalid rollover_time '{s}'. Expected 'HH:MM:SS'.") from e

    def _local_dt(self, now_utc: datetime) -> datetime:
        return now_utc.astimezone(self._tz)

    def current_day_id(self, now_utc: Optional[datetime] = None) -> str:
        """
        FTMO-style day_id with configurable local rollover boundary.

        Definition:
          - Convert now_utc -> local_dt (Europe/Prague)
          - The "trading day" starts at local rollover_time (cfg.rollover_time).
          - If local time is BEFORE rollover_time, it belongs to previous calendar date.
          - Else, it belongs to current calendar date.

        Example (rollover_time=00:00:00): day_id is simply local calendar date.
        Example (rollover_time=23:00:00): times 00:00..22:59 belong to previous day_id.
        """
        now_utc = now_utc or self._now_utc()
        local_dt = self._local_dt(now_utc)

        # Compare local time to boundary; if before boundary -> previous day.
        local_t = local_dt.timetz().replace(tzinfo=None)  # pure time comparison
        boundary = self._rollover_local_time

        if local_t < boundary:
            day = (local_dt.date() - timedelta(days=1)).isoformat()
        else:
            day = local_dt.date().isoformat()
        return day

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
        state["equity_start_day"] = float(state["equity_current"])

        # Reset daily metrics
        state["daily_pnl_pct"] = 0.0
        state["daily_loss_pct"] = 0.0

        state["daily_stop_triggered"] = False
        state["trades_taken_today"] = 0

        # Enabled depends on hard stop
        state["trading_enabled"] = not bool(state.get("hard_stop_dd_triggered", False))

        state["last_update_utc"] = now_utc.isoformat().replace("+00:00", "Z")
        state["last_run_id"] = run_id

        self._audit(state, now_utc, run_id, event="DAY_ROLLOVER")
        return state

    # ----------------------------
    # Trade lifecycle hooks
    # ----------------------------
    def on_trade_opened(self, state: Dict[str, Any], run_id: str, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Count trades at OPEN (or intent fill), not at close.
        This aligns max_trades_per_day with actual positions taken.
        """
        now_utc = now_utc or self._now_utc()
        state = self.rollover_if_needed(state, run_id=run_id, now_utc=now_utc)

        state["trades_taken_today"] = int(state.get("trades_taken_today", 0) + 1)

        # If reached max trades, disable further trading for the day.
        if int(state["trades_taken_today"]) >= int(state.get("max_trades_per_day", 1)):
            state["trading_enabled"] = False

        state["last_update_utc"] = now_utc.isoformat().replace("+00:00", "Z")
        state["last_run_id"] = run_id

        self._audit(state, now_utc, run_id, event="TRADE_OPENED")
        return state

    def on_trade_closed(self, state: Dict[str, Any], equity_after: float, run_id: str, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update equity + FTMO-style daily/overall loss.
        Also keeps dd_from_peak_pct as analytics (NOT used for FTMO hard stop).
        """
        now_utc = now_utc or self._now_utc()

        # Ensure day initialized (uses rollover boundary)
        state = self.rollover_if_needed(state, run_id=run_id, now_utc=now_utc)

        state["equity_current"] = float(equity_after)
        state["equity_peak"] = float(max(float(state["equity_peak"]), float(state["equity_current"])))

        # --- Analytics: DD from peak (not FTMO)
        peak = float(state["equity_peak"])
        dd_peak = (peak - float(state["equity_current"])) / peak if peak > 0 else 0.0
        dd_from_peak_pct = float(dd_peak * 100.0)
        state["dd_from_peak_pct"] = dd_from_peak_pct
        state["dd_current_pct"] = dd_from_peak_pct  # backward-compat field used in logs/plots

        # --- Daily PnL (signed)
        day_start = state.get("equity_start_day")
        if day_start is None or float(day_start) <= 0:
            state["equity_start_day"] = float(state["equity_current"])
            day_start = state["equity_start_day"]

        day_start_f = float(day_start)
        daily_ret = (float(state["equity_current"]) - day_start_f) / day_start_f if day_start_f > 0 else 0.0
        state["daily_pnl_pct"] = float(daily_ret * 100.0)

        # --- FTMO-style DAILY LOSS (positive number)
        # daily_loss_pct = max(0, (equity_start_day - equity_current)/equity_start_day * 100)
        daily_loss = (day_start_f - float(state["equity_current"])) / day_start_f if day_start_f > 0 else 0.0
        state["daily_loss_pct"] = float(max(0.0, daily_loss) * 100.0)

        # --- FTMO-style OVERALL LOSS from initial balance (positive number)
        init_bal = float(state.get("initial_balance") or 0.0)
        overall_loss = (init_bal - float(state["equity_current"])) / init_bal if init_bal > 0 else 0.0
        state["overall_loss_pct"] = float(max(0.0, overall_loss) * 100.0)

        # Apply account-level gates (FTMO-style loss limits)
        if float(state["daily_loss_pct"]) >= float(self.cfg.max_daily_loss_pct):
            state["daily_stop_triggered"] = True
            state["trading_enabled"] = False

        if float(state["overall_loss_pct"]) >= float(self.cfg.max_overall_loss_pct):
            state["hard_stop_dd_triggered"] = True
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
        if (not bool(state.get("daily_stop_triggered", False))) and int(state.get("trades_taken_today", 0)) < int(state.get("max_trades_per_day", 1)):
            state["trading_enabled"] = True

        state["last_update_utc"] = now_utc.isoformat().replace("+00:00", "Z")
        state["last_run_id"] = run_id

        self._audit(state, now_utc, run_id, event="MANUAL_RESET_HARD_STOP", extra={"override": override})
        return state

    # ----------------------------
    # Schema upgrade + Audit
    # ----------------------------
    def _upgrade_and_validate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Validate at least v1 shape
        missing_v1 = [k for k in REQUIRED_KEYS_V1 if k not in state]
        if missing_v1:
            raise ValueError(f"Risk state missing keys (v1): {missing_v1}")

        # Upgrade schema_version
        try:
            sv = int(state.get("schema_version", 1))
        except Exception:
            sv = 1

        if sv < SCHEMA_VERSION:
            # Add v2 keys with safe defaults
            state.setdefault("daily_loss_pct", 0.0)
            state.setdefault("overall_loss_pct", 0.0)
            state.setdefault("dd_from_peak_pct", float(state.get("dd_current_pct", 0.0)))
            state["schema_version"] = SCHEMA_VERSION

        # Validate v2 extras exist (after upgrade)
        missing_v2 = [k for k in REQUIRED_KEYS_V2_EXTRA if k not in state]
        if missing_v2:
            raise ValueError(f"Risk state missing keys (v2): {missing_v2}")

        return state

    def _audit(self, state: Dict[str, Any], now_utc: datetime, run_id: str, event: str, extra: Optional[Dict[str, Any]] = None) -> None:
        rec: Dict[str, Any] = {
            "ts_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "run_id": run_id,
            "event": event,
            "equity_current": state.get("equity_current"),
            "equity_peak": state.get("equity_peak"),
            "dd_from_peak_pct": state.get("dd_from_peak_pct"),
            "daily_loss_pct": state.get("daily_loss_pct"),
            "overall_loss_pct": state.get("overall_loss_pct"),
            "equity_start_day": state.get("equity_start_day"),
            "daily_pnl_pct": state.get("daily_pnl_pct"),
            "trades_taken_today": state.get("trades_taken_today"),
            "trading_enabled": state.get("trading_enabled"),
            "daily_stop_triggered": state.get("daily_stop_triggered"),
            "hard_stop_dd_triggered": state.get("hard_stop_dd_triggered"),
        }
        if extra is not None:
            rec["extra"] = extra

        tail = list(state.get("audit_tail", []))
        tail.append(rec)

        if len(tail) > self.cfg.max_audit:
            tail = tail[-self.cfg.max_audit :]

        state["audit_tail"] = tail
