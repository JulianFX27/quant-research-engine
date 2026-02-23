from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
from datetime import time as dtime
from zoneinfo import ZoneInfo

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
      - TIME_STOP when held >= effective_hold_bars (bounded by remaining event window)
      - ANCHOR_TOUCH if abs(close-anchor) <= exit_thr
      - SL/TP by engine intrabar

    Engine fills at NEXT bar open, so SL/TP must be recomputed at fill time.
    """

    name = "AnchorReversionAdapter"
    version = "0.3.1"
    instrument = "EURUSD"

    def __init__(self, cfg: Optional[AnchorAdapterConfig] = None):
        self.cfg = cfg or AnchorAdapterConfig()

        self._i = -1  # global bar counter (kept for debugging only)

        # Runner/engine state via hooks
        self._pending = False
        self._in_pos = False
        self._entry_i_global: Optional[int] = None
        self._side: Optional[Direction] = None

        # Day-scoped bar index (resets each NY-day)
        self._day_id: Optional[str] = None
        self._day_i: int = -1  # bar index within NY-day

        # Event state (NY-day scoped)
        self._event_day: Optional[str] = None
        self._event_i_day: Optional[int] = None
        self._event_ts_utc: Optional[Any] = None  # datetime
        self._traded_event_i_day: Optional[int] = None

        # Entry index within NY-day at fill time (for hold bounds)
        self._entry_i_day: Optional[int] = None

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _is_nan(x: Any) -> bool:
        try:
            return float(x) != float(x)
        except Exception:
            return False

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

            # Reset entry index within day (position should already be closed by weekend logic,
            # but keep deterministic reset anyway)
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

        # ensure day counter is up to date
        day = self._day_id_ny(bar.ts_utc)
        if self._event_day is None or day != self._event_day:
            # day rollover handling is centralized in _roll_day_if_needed,
            # but keep safety here as well.
            self._event_day = day
            self._event_i_day = None
            self._event_ts_utc = None
            self._traded_event_i_day = None

        z_raw = bar.extras.get(cfg.event_col, None)
        if z_raw is None or self._is_nan(z_raw):
            return
        try:
            z = float(z_raw)
        except Exception:
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
        Bound hold time so a trade cannot survive beyond the event window.
        Uses day-local indices to be coherent with day-scoped event definition.
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

    def _friday_entry_guard_ok(self, bar: Bar) -> bool:
        cfg = self.cfg
        if not cfg.guard_friday_entries:
            return True
        rem = self._minutes_to_friday_cutoff(bar.ts_utc)
        if rem is None:
            return True

        hold_bars = int(cfg.max_hold_bars) if cfg.max_hold_bars > 0 else 0
        need_min = hold_bars * int(cfg.bar_minutes) + int(cfg.friday_buffer_minutes)
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

        # Maintain day-local index + reset day-scoped state
        self._roll_day_if_needed(bar)

        # Update event state (NY-day scoped)
        self._update_event_state(bar)

        pip = float(ctx.pip_size)
        entry_thr = float(cfg.entry_threshold_pips) * pip
        exit_thr = float(cfg.exit_threshold_pips) * pip

        c = float(bar.close)
        anchor_raw = bar.extras.get(cfg.anchor_col, None)
        anchor = float(anchor_raw) if anchor_raw is not None and (not self._is_nan(anchor_raw)) else None

        # --------------------
        # EXIT logic (only when position OPEN)
        # --------------------
        if self._in_pos:
            eff_hold = self._effective_hold_bars()
            if eff_hold > 0 and self._entry_i_day is not None:
                held = int(self._day_i) - int(self._entry_i_day)
                if held >= eff_hold:
                    return OrderIntent(
                        intent_id=f"EXIT_{bar.ts_utc.isoformat()}",
                        ts_utc=bar.ts_utc,
                        action="EXIT",
                        exit_reason="TIME_STOP",
                        meta={"tag": cfg.tag},
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
        # ENTRY filters (research alignment)
        # --------------------
        if cfg.require_event and (not self._event_allows_entry()):
            return None

        if not self._friday_entry_guard_ok(bar):
            return None

        if anchor is None:
            return None

        dist = c - anchor

        # We still populate sl_price/tp_price for interface completeness,
        # but the engine will recompute at fill time via meta.
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
    # Runner hooks (state transitions)
    # -----------------------------
    def on_intent_submitted(self, direction: Direction) -> None:
        self._pending = True
        self._side = direction

    def on_trade_opened_confirmed(self) -> None:
        self._in_pos = True
        self._pending = False
        self._entry_i_global = self._i
        self._entry_i_day = int(self._day_i)

        if self.cfg.require_event and self.cfg.one_trade_per_event and self._event_i_day is not None:
            self._traded_event_i_day = self._event_i_day

    def on_intent_cancelled(self) -> None:
        self._pending = False
        self._side = None

    def on_trade_closed_reset(self) -> None:
        self._in_pos = False
        self._pending = False
        self._entry_i_global = None
        self._entry_i_day = None
        self._side = None