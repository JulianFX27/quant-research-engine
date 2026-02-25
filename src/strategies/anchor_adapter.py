from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
from datetime import time as dtime
from zoneinfo import ZoneInfo
from datetime import datetime

from src.runner.interfaces import Bar, StrategyContext, OrderIntent, Direction


NY_TZ = ZoneInfo("America/New_York")
WEEKEND_CUTOFF_NY = dtime(16, 55, 0)  # Friday 16:55 NY


@dataclass
class AnchorAdapterConfig:
    # Core
    anchor_col: str = "ny_open"
    entry_threshold_pips: float = 8.0
    exit_threshold_pips: float = 0.3
    sl_pips: float = 10.0
    tp_pips: float = 15.0
    warmup_bars: int = 0

    # IMPORTANT: This remains in "bars" for config compatibility,
    # but TIME_STOP will be enforced in wall-clock minutes using bar_minutes.
    max_hold_bars: int = 96

    tag: str = "ANCHOR_ADAPTER_V1"

    # Research alignment: event conditioning
    require_event: bool = True
    event_col: str = "shock_z"
    event_z_threshold: float = 2.0

    # Research alignment: trade only within a post-event window (bars after the event)
    event_window_bars: int = 96
    one_trade_per_event: bool = True

    # Weekend guard (avoid FORCE_WEEKEND distortions)
    guard_friday_entries: bool = True
    bar_minutes: int = 5
    friday_buffer_minutes: int = 10


class AnchorReversionAdapter:
    """
    Research-aligned adapter.

    Event conditioning (DAY-SCOPED, deterministic):
      - Define event when abs(shock_z) >= z_threshold.
      - Track event within NY-day using a day-local bar index (not global i).
      - Allow entries only within event_window_bars after the latest event.
      - Optional: only 1 trade per event.

    Entry (signal):
      dist = close - anchor
      dist >= entry_thr -> SHORT
      dist <= -entry_thr -> LONG

    Exit:
      - TIME_STOP when held >= hold_budget_minutes (wall-clock minutes, computed at fill)
      - ANCHOR_TOUCH if abs(close-anchor) <= exit_thr
      - SL/TP by engine intrabar

    Engine fills at NEXT bar open, so SL/TP must be recomputed at fill time.
    """

    name = "AnchorReversionAdapter"
    version = "0.3.2"  # bumped due to TIME_STOP semantics fix
    instrument = "EURUSD"

    def __init__(self, cfg: Optional[AnchorAdapterConfig] = None):
        self.cfg = cfg or AnchorAdapterConfig()

        self._i = -1  # global bar counter (for debugging only; NOT authoritative for TIME_STOP)

        # Runner/engine state via hooks
        self._pending = False
        self._in_pos = False
        self._side: Optional[Direction] = None

        # Day-scoped bar index (resets each NY-day)
        self._day_id: Optional[str] = None
        self._day_i: int = -1  # bar index within NY-day

        # Event state (NY-day scoped)
        self._event_day: Optional[str] = None
        self._event_i_day: Optional[int] = None
        self._event_ts_utc: Optional[Any] = None  # datetime
        self._traded_event_i_day: Optional[int] = None

        # Entry index within NY-day at fill time (used only to compute hold budget at entry)
        self._entry_i_day: Optional[int] = None

        # Fill-time anchors (authoritative for TIME_STOP)
        self._entry_ts_utc: Optional[datetime] = None

        # Fixed hold budget computed at entry fill time (bars/minutes). Authoritative for TIME_STOP.
        self._hold_budget_bars: Optional[int] = None
        self._hold_budget_minutes: Optional[int] = None

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _to_float_or_none(x: Any) -> Optional[float]:
        """
        Robust float parsing:
        - None -> None
        - '' / whitespace -> None
        - non-numeric -> None
        - NaN -> None
        """
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        try:
            v = float(x)
        except Exception:
            return None
        if v != v:  # NaN
            return None
        return v

    def _day_id_ny(self, ts_utc) -> str:
        ny = ts_utc.astimezone(NY_TZ)
        return ny.strftime("%Y-%m-%d")

    def _minutes_to_friday_cutoff(self, ts_utc) -> Optional[int]:
        ny = ts_utc.astimezone(NY_TZ)
        if ny.weekday() != 4:  # Friday
            return None
        cutoff_minutes = WEEKEND_CUTOFF_NY.hour * 60 + WEEKEND_CUTOFF_NY.minute
        now_minutes = ny.hour * 60 + ny.minute
        return cutoff_minutes - now_minutes

    def _roll_day_if_needed(self, bar: Bar) -> None:
        """
        Maintain a day-local bar counter for NY days.

        day-local counters are for event conditioning only.
        TIME_STOP must NOT rely on day-local counters (they reset), and must NOT rely on
        global bar counts either if the dataset is session-filtered (gaps).
        """
        day = self._day_id_ny(bar.ts_utc)
        if self._day_id is None or day != self._day_id:
            self._day_id = day
            self._day_i = 0

            # Reset NY-day scoped event tracking
            self._event_day = day
            self._event_i_day = None
            self._event_ts_utc = None
            self._traded_event_i_day = None

            # Reset entry index within day ONLY if no open position
            if not self._in_pos:
                self._entry_i_day = None
        else:
            self._day_i += 1

    def _update_event_state(self, bar: Bar) -> None:
        """
        Update latest event within current NY-day using day-local index.
        Latest event wins.
        """
        cfg = self.cfg
        if not cfg.require_event:
            return

        day = self._day_id_ny(bar.ts_utc)
        if self._event_day is None or day != self._event_day:
            self._event_day = day
            self._event_i_day = None
            self._event_ts_utc = None
            self._traded_event_i_day = None

        extras = getattr(bar, "extras", None) or {}
        z = self._to_float_or_none(extras.get(cfg.event_col, None))
        if z is None:
            return

        if abs(z) >= float(cfg.event_z_threshold):
            self._event_i_day = int(self._day_i)
            self._event_ts_utc = bar.ts_utc

    def _event_allows_entry(self) -> bool:
        cfg = self.cfg
        if not cfg.require_event:
            return True
        if self._event_i_day is None:
            return False

        if cfg.event_window_bars > 0:
            if (int(self._day_i) - int(self._event_i_day)) > int(cfg.event_window_bars):
                return False

        if cfg.one_trade_per_event:
            if self._traded_event_i_day is not None and self._traded_event_i_day == self._event_i_day:
                return False

        return True

    def _effective_hold_bars(self) -> int:
        """
        Bound hold bars so a trade cannot survive beyond the event window.

        This returns a FIXED bar budget computed at entry fill time.
        TIME_STOP enforcement itself is wall-clock minutes (hold_budget_minutes),
        but we keep bars for alignment/audit and to derive minutes.
        """
        cfg = self.cfg
        base = int(cfg.max_hold_bars) if cfg.max_hold_bars > 0 else 0

        if (not cfg.require_event) or (self._event_i_day is None) or (cfg.event_window_bars <= 0):
            return base

        if self._entry_i_day is None:
            return base

        elapsed_since_event_at_entry = int(self._entry_i_day) - int(self._event_i_day)
        remaining = int(cfg.event_window_bars) - int(elapsed_since_event_at_entry)
        if remaining <= 0:
            return 0

        if base <= 0:
            return remaining

        return min(base, remaining)

    def _bars_to_minutes(self, bars: int) -> int:
        bm = int(self.cfg.bar_minutes) if int(self.cfg.bar_minutes) > 0 else 5
        return int(max(0, int(bars)) * bm)

    def _friday_entry_guard_ok(self, bar: Bar) -> bool:
        cfg = self.cfg
        if not cfg.guard_friday_entries:
            return True
        rem = self._minutes_to_friday_cutoff(bar.ts_utc)
        if rem is None:
            return True

        hold_bars = int(cfg.max_hold_bars) if cfg.max_hold_bars > 0 else 0
        need_min = self._bars_to_minutes(hold_bars) + int(cfg.friday_buffer_minutes)
        if hold_bars <= 0:
            need_min = int(cfg.friday_buffer_minutes)

        return rem >= need_min

    # -----------------------------
    # Main logic
    # -----------------------------
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> Optional[OrderIntent]:
        self._i += 1
        cfg = self.cfg

        if self._i < int(cfg.warmup_bars):
            return None

        self._roll_day_if_needed(bar)
        self._update_event_state(bar)

        pip = float(ctx.pip_size)
        entry_thr = float(cfg.entry_threshold_pips) * pip
        exit_thr = float(cfg.exit_threshold_pips) * pip

        c = float(bar.close)

        extras = getattr(bar, "extras", None) or {}
        anchor = self._to_float_or_none(extras.get(cfg.anchor_col, None))

        # --------------------
        # EXIT logic
        # --------------------
        if self._in_pos:
            # TIME_STOP must be WALL-CLOCK to be robust to session-filtered datasets / gaps.
            if self._entry_ts_utc is not None and self._hold_budget_minutes is not None and int(self._hold_budget_minutes) > 0:
                held_min = (bar.ts_utc - self._entry_ts_utc).total_seconds() / 60.0
                if held_min >= float(self._hold_budget_minutes):
                    return OrderIntent(
                        intent_id=f"EXIT_{bar.ts_utc.isoformat()}",
                        ts_utc=bar.ts_utc,
                        action="EXIT",
                        exit_reason="TIME_STOP",
                        meta={
                            "tag": cfg.tag,
                            "held_min": float(held_min),
                            "hold_budget_min": int(self._hold_budget_minutes),
                            "hold_budget_bars": int(self._hold_budget_bars or 0),
                        },
                    )

            if anchor is not None and abs(c - anchor) <= exit_thr:
                return OrderIntent(
                    intent_id=f"EXIT_{bar.ts_utc.isoformat()}",
                    ts_utc=bar.ts_utc,
                    action="EXIT",
                    exit_reason="ANCHOR_TOUCH",
                    meta={"tag": cfg.tag},
                )

            return None

        # --------------------
        # No entries if pending
        # --------------------
        if self._pending:
            return None

        # --------------------
        # ENTRY filters
        # --------------------
        if cfg.require_event and (not self._event_allows_entry()):
            return None

        if not self._friday_entry_guard_ok(bar):
            return None

        if anchor is None:
            return None

        dist = c - anchor

        if dist >= entry_thr:
            direction: Direction = "SHORT"
            sl = c + float(cfg.sl_pips) * pip
            tp = c - float(cfg.tp_pips) * pip
            return OrderIntent(
                intent_id=f"ENT_{bar.ts_utc.isoformat()}",
                ts_utc=bar.ts_utc,
                action="ENTER",
                direction=direction,
                sl_price=sl,
                tp_price=tp,
                meta={
                    "tag": cfg.tag,
                    "anchor": anchor,
                    "dist": dist,
                    "event_i_day": self._event_i_day,
                    "event_day": self._event_day,
                    "event_ts_utc": self._event_ts_utc.isoformat() if self._event_ts_utc is not None else None,
                    "_sl_pips": float(cfg.sl_pips),
                    "_tp_pips": float(cfg.tp_pips),
                    "_pip_size": float(pip),
                    "_recalc_sl_tp_at_fill": True,
                },
            )

        if dist <= -entry_thr:
            direction = "LONG"
            sl = c - float(cfg.sl_pips) * pip
            tp = c + float(cfg.tp_pips) * pip
            return OrderIntent(
                intent_id=f"ENT_{bar.ts_utc.isoformat()}",
                ts_utc=bar.ts_utc,
                action="ENTER",
                direction=direction,
                sl_price=sl,
                tp_price=tp,
                meta={
                    "tag": cfg.tag,
                    "anchor": anchor,
                    "dist": dist,
                    "event_i_day": self._event_i_day,
                    "event_day": self._event_day,
                    "event_ts_utc": self._event_ts_utc.isoformat() if self._event_ts_utc is not None else None,
                    "_sl_pips": float(cfg.sl_pips),
                    "_tp_pips": float(cfg.tp_pips),
                    "_pip_size": float(pip),
                    "_recalc_sl_tp_at_fill": True,
                },
            )

        return None

    # -----------------------------
    # Runner hooks
    # -----------------------------
    def on_intent_submitted(self, direction: Direction) -> None:
        self._pending = True
        self._side = direction

    def on_trade_opened_confirmed(self, entry_ts_utc: datetime) -> None:
        """
        Called after the engine confirms the trade entry fill.

        Critical:
        - Capture ENTRY TIME (fill time) for wall-clock TIME_STOP.
        - Capture day-local entry index only to compute fixed budgets at entry.
        - Persist hold budgets so day rollover cannot disable TIME_STOP.
        """
        self._in_pos = True
        self._pending = False
        self._entry_ts_utc = entry_ts_utc
        self._entry_i_day = int(self._day_i)

        # Freeze budgets at entry (bounded by event window)
        self._hold_budget_bars = int(self._effective_hold_bars())
        self._hold_budget_minutes = self._bars_to_minutes(int(self._hold_budget_bars))

        if self.cfg.require_event and self.cfg.one_trade_per_event and self._event_i_day is not None:
            self._traded_event_i_day = self._event_i_day

    def on_intent_cancelled(self) -> None:
        self._pending = False
        self._side = None

    def on_trade_closed_reset(self) -> None:
        self._in_pos = False
        self._pending = False
        self._entry_i_day = None
        self._entry_ts_utc = None
        self._hold_budget_bars = None
        self._hold_budget_minutes = None
        self._side = None