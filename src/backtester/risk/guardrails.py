from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class GuardrailsConfig:
    # Positions
    max_concurrent_positions: int = 1  # 1 = single-position engine

    # Time window gating (UTC). If disabled, ignored.
    time_window_enabled: bool = False
    window_start_utc: Optional[str] = None  # "HH:MM"
    window_end_utc: Optional[str] = None    # "HH:MM"

    # Max holding bars (0 disables)
    max_holding_bars: int = 0

    # NEW: no-hold-weekend guardrail (UTC)
    weekend_exit_enabled: bool = False
    weekend_exit_cutoff_utc: Optional[str] = None  # "HH:MM" on Friday
    weekend_exit_weekday: int = 4  # Friday = 4 (Mon=0..Sun=6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_concurrent_positions": self.max_concurrent_positions,
            "time_window_enabled": self.time_window_enabled,
            "window_start_utc": self.window_start_utc,
            "window_end_utc": self.window_end_utc,
            "max_holding_bars": self.max_holding_bars,
            "weekend_exit_enabled": self.weekend_exit_enabled,
            "weekend_exit_cutoff_utc": self.weekend_exit_cutoff_utc,
            "weekend_exit_weekday": self.weekend_exit_weekday,
        }


def _parse_hhmm(s: str) -> Tuple[int, int]:
    if not isinstance(s, str) or ":" not in s:
        raise ValueError(f"Invalid HH:MM time: {s!r}")
    hh, mm = s.split(":", 1)
    h = int(hh)
    m = int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid HH:MM time: {s!r}")
    return h, m


def _time_in_window_utc(ts_utc: pd.Timestamp, start_hm: Tuple[int, int], end_hm: Tuple[int, int]) -> bool:
    """
    Window semantics:
      - Times are interpreted in UTC (ts_utc must be tz-aware UTC).
      - Supports wrap-around windows (e.g., 22:00 -> 02:00).
    """
    if ts_utc.tzinfo is None:
        raise ValueError("Timestamp must be tz-aware UTC for time window checks")
    ts_utc = ts_utc.tz_convert("UTC")

    h, m = ts_utc.hour, ts_utc.minute
    cur = h * 60 + m

    start = start_hm[0] * 60 + start_hm[1]
    end = end_hm[0] * 60 + end_hm[1]

    if start == end:
        # Degenerate => treat as closed window (never allow)
        return False

    if start < end:
        return start <= cur < end
    # wrap-around
    return (cur >= start) or (cur < end)


def _is_weekend_cutoff(ts_utc: pd.Timestamp, *, cutoff_hm: Tuple[int, int], weekday: int = 4) -> bool:
    """
    No-hold-weekend policy (deterministic, UTC):
      - If ts_utc is on the specified weekday (default Friday=4)
      - and time >= cutoff_hm
      => force exit (close on that bar).
    """
    if ts_utc.tzinfo is None:
        raise ValueError("Timestamp must be tz-aware UTC for weekend checks")
    t = ts_utc.tz_convert("UTC")
    if int(t.weekday()) != int(weekday):
        return False
    cur = int(t.hour) * 60 + int(t.minute)
    cutoff = int(cutoff_hm[0]) * 60 + int(cutoff_hm[1])
    return cur >= cutoff


class Guardrails:
    """
    Stateless guardrail evaluator + stateful counters for a single run.

    Integrates with engine by checking:
      - allow_entry(bar_time_utc, active_positions)
      - should_force_exit(holding_bars, bar_time_utc=None)
    """

    def __init__(self, cfg: Dict[str, Any] | None):
        cfg = cfg or {}
        self.cfg = GuardrailsConfig(
            max_concurrent_positions=int(cfg.get("max_concurrent_positions", 1)),
            time_window_enabled=bool(cfg.get("time_window_enabled", False)),
            window_start_utc=cfg.get("window_start_utc"),
            window_end_utc=cfg.get("window_end_utc"),
            max_holding_bars=int(cfg.get("max_holding_bars", 0) or 0),
            weekend_exit_enabled=bool(cfg.get("weekend_exit_enabled", False)),
            weekend_exit_cutoff_utc=cfg.get("weekend_exit_cutoff_utc"),
            weekend_exit_weekday=int(cfg.get("weekend_exit_weekday", 4) if cfg.get("weekend_exit_weekday") is not None else 4),
        )
        if self.cfg.max_concurrent_positions < 1:
            raise ValueError("max_concurrent_positions must be >= 1")

        # Time window parsing
        if self.cfg.time_window_enabled:
            if not self.cfg.window_start_utc or not self.cfg.window_end_utc:
                raise ValueError("time_window_enabled=true requires window_start_utc and window_end_utc")
            self._start_hm = _parse_hhmm(self.cfg.window_start_utc)
            self._end_hm = _parse_hhmm(self.cfg.window_end_utc)
        else:
            self._start_hm = None
            self._end_hm = None

        # Weekend cutoff parsing
        if self.cfg.weekend_exit_enabled:
            if not self.cfg.weekend_exit_cutoff_utc:
                raise ValueError("weekend_exit_enabled=true requires weekend_exit_cutoff_utc='HH:MM'")
            self._weekend_cutoff_hm = _parse_hhmm(str(self.cfg.weekend_exit_cutoff_utc))
        else:
            self._weekend_cutoff_hm = None

        # Audit counters
        self.blocked = {
            "by_max_concurrent_positions": 0,
            "by_time_window": 0,
        }
        self.forced_exits = {
            "by_max_holding_bars": 0,
            "by_weekend_no_hold": 0,
        }

    def allow_entry(self, bar_time_utc: pd.Timestamp, active_positions: int) -> Tuple[bool, Optional[str]]:
        # 1) max concurrent positions
        if active_positions >= self.cfg.max_concurrent_positions:
            self.blocked["by_max_concurrent_positions"] += 1
            return False, "by_max_concurrent_positions"

        # 2) time window (UTC)
        if self.cfg.time_window_enabled:
            assert self._start_hm is not None and self._end_hm is not None
            if not _time_in_window_utc(bar_time_utc, self._start_hm, self._end_hm):
                self.blocked["by_time_window"] += 1
                return False, "by_time_window"

        return True, None

    def should_force_exit(self, holding_bars: int, bar_time_utc: Optional[pd.Timestamp] = None) -> Tuple[bool, Optional[str]]:
        # 0) weekend no-hold (time-based, UTC)
        if self.cfg.weekend_exit_enabled and (bar_time_utc is not None):
            assert self._weekend_cutoff_hm is not None
            if _is_weekend_cutoff(bar_time_utc, cutoff_hm=self._weekend_cutoff_hm, weekday=self.cfg.weekend_exit_weekday):
                self.forced_exits["by_weekend_no_hold"] += 1
                return True, "by_weekend_no_hold"

        # 1) max holding bars
        if self.cfg.max_holding_bars and holding_bars >= self.cfg.max_holding_bars:
            self.forced_exits["by_max_holding_bars"] += 1
            return True, "by_max_holding_bars"

        return False, None

    def report(self) -> Dict[str, Any]:
        return {
            "guardrails_cfg": self.cfg.to_dict(),
            "blocked": dict(self.blocked),
            "forced_exits": dict(self.forced_exits),
        }
