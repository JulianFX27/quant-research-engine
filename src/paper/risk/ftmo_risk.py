# src/paper/risk/ftmo_risk.py

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
from datetime import datetime, date, timezone


CEST_TZ = ZoneInfo("Europe/Prague")  # FTMO usa CET/CEST; Prague es estándar y DST-aware.


@dataclass(frozen=True)
class ThrottleStep:
    dd_from: float          # inclusive, in fraction (e.g. 0.02)
    dd_to: float            # exclusive
    multiplier: float       # e.g. 0.75


@dataclass(frozen=True)
class FTMOConfig:
    phase: str  # "challenge" | "funded"
    base_risk_challenge: float  # 0.005
    base_risk_funded: float     # 0.004

    max_daily_loss: float  # 0.015 (fraction)
    max_total_loss: float  # 0.08  (fraction)
    hard_stop_total_loss: float  # 0.08 (fraction) - por claridad, igual a max_total_loss

    trades_per_day: int  # 1

    throttle_steps: Tuple[ThrottleStep, ...]  # sorted by dd_from asc

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FTMOConfig":
        steps = tuple(
            ThrottleStep(
                dd_from=float(s["dd_from"]),
                dd_to=float(s["dd_to"]),
                multiplier=float(s["multiplier"]),
            )
            for s in d["throttle"]["steps"]
        )
        return FTMOConfig(
            phase=str(d["phase"]).lower(),
            base_risk_challenge=float(d["sizing"]["challenge_risk"]),
            base_risk_funded=float(d["sizing"]["funded_risk"]),
            max_daily_loss=float(d["limits"]["max_daily_loss"]),
            max_total_loss=float(d["limits"]["max_total_loss"]),
            hard_stop_total_loss=float(d["limits"].get("hard_stop_total_loss", d["limits"]["max_total_loss"])),
            trades_per_day=int(d["limits"]["trades_per_day"]),
            throttle_steps=steps,
        )

    def base_risk(self) -> float:
        if self.phase == "challenge":
            return self.base_risk_challenge
        if self.phase == "funded":
            return self.base_risk_funded
        raise ValueError(f"Unknown FTMO phase: {self.phase!r}")


@dataclass
class FTMORiskState:
    # Identity
    run_id: str

    # Equity anchors
    start_equity: float
    day_start_equity: float

    # Time anchors (CEST day)
    current_cest_day: str  # YYYY-MM-DD

    # Running stats
    peak_equity: float
    current_equity: float

    # Control
    trades_taken_today: int = 0
    stop_trading: bool = False
    stop_reason: Optional[str] = None

    # Last update
    last_update_utc: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "FTMORiskState":
        d = json.loads(s)
        return FTMORiskState(**d)


class FTMORiskManager:
    """
    Pure risk gate + sizing policy for FTMO-aware paper trading.

    Responsibilities:
    - CEST day rollover
    - MDL & ML tracking (daily/total loss limits)
    - Throttle sizing by current DD
    - 1 trade/day cap
    - Persist/restore state
    """

    def __init__(self, cfg: FTMOConfig, state_path: Path):
        self.cfg = cfg
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state: Optional[FTMORiskState] = None

    # ---------- time helpers ----------
    @staticmethod
    def _utc_to_cest_day(dt_utc: datetime) -> date:
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        return dt_utc.astimezone(CEST_TZ).date()

    @staticmethod
    def _date_str(d: date) -> str:
        return d.isoformat()

    # ---------- state I/O ----------
    def load_or_init(self, *, run_id: str, start_equity: float, now_utc: datetime) -> FTMORiskState:
        if self.state_path.exists():
            self.state = FTMORiskState.from_json(self.state_path.read_text(encoding="utf-8"))
            return self.state

        cest_day = self._utc_to_cest_day(now_utc)
        st = FTMORiskState(
            run_id=run_id,
            start_equity=float(start_equity),
            day_start_equity=float(start_equity),
            current_cest_day=self._date_str(cest_day),
            peak_equity=float(start_equity),
            current_equity=float(start_equity),
            trades_taken_today=0,
            stop_trading=False,
            stop_reason=None,
            last_update_utc=now_utc.replace(tzinfo=timezone.utc).isoformat(),
        )
        self.state = st
        self.flush(now_utc)
        return st

    def flush(self, now_utc: datetime) -> None:
        assert self.state is not None
        self.state.last_update_utc = now_utc.replace(tzinfo=timezone.utc).isoformat()
        self.state_path.write_text(self.state.to_json(), encoding="utf-8")

    # ---------- core computations ----------
    def _rollover_if_needed(self, now_utc: datetime) -> None:
        assert self.state is not None
        cest_day = self._utc_to_cest_day(now_utc)
        cest_day_str = self._date_str(cest_day)
        if cest_day_str != self.state.current_cest_day:
            # New CEST day → reset daily counters and day_start_equity anchor.
            self.state.current_cest_day = cest_day_str
            self.state.day_start_equity = float(self.state.current_equity)
            self.state.trades_taken_today = 0
            # Note: stop_trading should remain if ML was hit; but if stop was only MDL, reset it.
            if self.state.stop_trading and self.state.stop_reason == "HIT_MDL":
                self.state.stop_trading = False
                self.state.stop_reason = None

    def _update_peak_and_dd(self) -> float:
        """
        Returns drawdown fraction from peak: (peak - current) / peak.
        """
        assert self.state is not None
        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = float(self.state.current_equity)
        peak = self.state.peak_equity
        cur = self.state.current_equity
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - cur) / peak)

    def _total_loss_frac(self) -> float:
        assert self.state is not None
        start = self.state.start_equity
        cur = self.state.current_equity
        if start <= 0:
            return 0.0
        return max(0.0, (start - cur) / start)

    def _daily_loss_frac(self) -> float:
        assert self.state is not None
        day0 = self.state.day_start_equity
        cur = self.state.current_equity
        if day0 <= 0:
            return 0.0
        return max(0.0, (day0 - cur) / day0)

    def _check_limits_and_flag(self) -> None:
        assert self.state is not None
        # Total loss (ML)
        tl = self._total_loss_frac()
        if tl >= self.cfg.hard_stop_total_loss:
            self.state.stop_trading = True
            self.state.stop_reason = "HIT_ML"
            return

        # Daily loss (MDL)
        dl = self._daily_loss_frac()
        if dl >= self.cfg.max_daily_loss:
            self.state.stop_trading = True
            self.state.stop_reason = "HIT_MDL"
            return

    # ---------- public API ----------
    def on_mark_to_market(self, *, equity: float, now_utc: datetime) -> None:
        """
        Call on every bar/tick step after updating paper equity.
        """
        assert self.state is not None
        self._rollover_if_needed(now_utc)
        self.state.current_equity = float(equity)
        self._update_peak_and_dd()
        self._check_limits_and_flag()
        self.flush(now_utc)

    def allowed_to_trade(self, now_utc: datetime) -> Tuple[bool, Optional[str]]:
        assert self.state is not None
        self._rollover_if_needed(now_utc)
        self._check_limits_and_flag()

        if self.state.stop_trading:
            return False, self.state.stop_reason

        if self.state.trades_taken_today >= self.cfg.trades_per_day:
            return False, "TRADE_CAP_TODAY"

        return True, None

    def risk_fraction(self, now_utc: datetime) -> float:
        """
        Returns throttle-adjusted risk fraction for position sizing.
        If stop_trading is True, returns 0.
        """
        assert self.state is not None
        self._rollover_if_needed(now_utc)
        self._check_limits_and_flag()
        if self.state.stop_trading:
            return 0.0

        base = self.cfg.base_risk()
        dd = self._update_peak_and_dd()

        mult = 1.0
        for step in self.cfg.throttle_steps:
            if dd >= step.dd_from and dd < step.dd_to:
                mult = step.multiplier
                break

        # If dd >= max_total_loss, we should stop (safety)
        if self._total_loss_frac() >= self.cfg.max_total_loss:
            self.state.stop_trading = True
            self.state.stop_reason = "HIT_ML"
            return 0.0

        return base * mult

    def on_trade_opened(self, now_utc: datetime) -> None:
        assert self.state is not None
        self._rollover_if_needed(now_utc)
        self.state.trades_taken_today += 1
        self.flush(now_utc)

    def snapshot(self) -> Dict[str, Any]:
        assert self.state is not None
        dd = self._update_peak_and_dd()
        return {
            "phase": self.cfg.phase,
            "equity": self.state.current_equity,
            "peak_equity": self.state.peak_equity,
            "dd_frac": dd,
            "total_loss_frac": self._total_loss_frac(),
            "daily_loss_frac": self._daily_loss_frac(),
            "trades_taken_today": self.state.trades_taken_today,
            "stop_trading": self.state.stop_trading,
            "stop_reason": self.state.stop_reason,
            "cest_day": self.state.current_cest_day,
        }
