from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import Any, Dict, List, Optional

import pandas as pd

from backtester.core.contracts import OrderIntent
from backtester.strategies.base import Strategy


FRIDAY_CUTOFF_HOUR = 16
FRIDAY_CUTOFF_MINUTE = 55


@dataclass(frozen=True)
class AnchorReversionAdapterBTConfig:
    qty: float = 1.0

    anchor_col: str = "ny_open"
    entry_threshold_pips: float = 8.0
    sl_pips: float = 10.0
    tp_pips: float = 15.0
    warmup_bars: int = 0
    max_hold_bars: int = 96
    tag: str = "ANCHOR_ADAPTER_BT"

    require_event: bool = True
    event_col: str = "shock_z"
    event_z_threshold: float = 2.0
    event_window_bars: int = 96
    one_trade_per_event: bool = True

    guard_friday_entries: bool = True
    bar_minutes: int = 5
    friday_buffer_minutes: int = 10

    pip_size: float = 0.0001
    col_close: str = "close"


class AnchorReversionAdapterBT(Strategy):
    name = "AnchorReversionAdapterBT"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.cfg = AnchorReversionAdapterBTConfig(**(params or {}))

        self._global_i = -1
        self._day_id: Optional[str] = None
        self._day_i: int = -1

        self._event_day: Optional[str] = None
        self._event_i_day: Optional[int] = None
        self._traded_event_i_day: Optional[int] = None

    def on_bar(self, i: int, df: pd.DataFrame, context: Dict[str, Any]) -> List[OrderIntent]:
        cfg = self.cfg
        out: List[OrderIntent] = []

        self._global_i += 1
        row = df.iloc[i]

        if cfg.col_close not in df.columns:
            raise ValueError(f"AnchorReversionAdapterBT requires column '{cfg.col_close}'")
        if cfg.anchor_col not in df.columns:
            return out

        ts = df.index[i]
        self._roll_day_if_needed(ts)
        self._update_event_state(row)

        if i < int(cfg.warmup_bars):
            return out

        close = _safe_float(row.get(cfg.col_close), default=None)
        anchor = _safe_float(row.get(cfg.anchor_col), default=None)
        if close is None or anchor is None:
            return out

        if cfg.require_event and (not self._event_allows_entry()):
            return out

        if cfg.guard_friday_entries and _blocked_by_friday_guard(
            ts=ts,
            max_hold_bars=cfg.max_hold_bars,
            bar_minutes=cfg.bar_minutes,
            friday_buffer_minutes=cfg.friday_buffer_minutes,
        ):
            return out

        pip = float(cfg.pip_size)
        thr = float(cfg.entry_threshold_pips) * pip
        dist = float(close) - float(anchor)

        if dist >= thr:
            side = "SELL"
            sl = float(close) + float(cfg.sl_pips) * pip
            tp = float(close) - float(cfg.tp_pips) * pip
        elif dist <= -thr:
            side = "BUY"
            sl = float(close) - float(cfg.sl_pips) * pip
            tp = float(close) + float(cfg.tp_pips) * pip
        else:
            return out

        if cfg.one_trade_per_event and cfg.require_event and self._event_i_day is not None:
            self._traded_event_i_day = self._event_i_day

        out.append(
            OrderIntent(
                side=side,
                qty=float(cfg.qty),
                sl_price=float(sl),
                tp_price=float(tp),
                tag=f"{cfg.tag}__evt_{self._event_i_day if self._event_i_day is not None else 'NA'}",
            )
        )
        return out

    def _roll_day_if_needed(self, ts: pd.Timestamp) -> None:
        day_id = _day_id_ny(ts)

        if self._day_id is None or day_id != self._day_id:
            self._day_id = day_id
            self._day_i = 0

            self._event_day = day_id
            self._event_i_day = None
            self._traded_event_i_day = None
        else:
            self._day_i += 1

    def _update_event_state(self, row: pd.Series) -> None:
        cfg = self.cfg
        if not cfg.require_event:
            return
        if cfg.event_col not in row.index:
            return

        z = _safe_float(row.get(cfg.event_col), default=None)
        if z is None:
            return

        if abs(float(z)) >= float(cfg.event_z_threshold):
            self._event_i_day = int(self._day_i)

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


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _day_id_ny(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        return ts.strftime("%Y-%m-%d")
    ny = ts.tz_convert("America/New_York")
    return ny.strftime("%Y-%m-%d")


def _blocked_by_friday_guard(
    ts: pd.Timestamp,
    max_hold_bars: int,
    bar_minutes: int,
    friday_buffer_minutes: int,
) -> bool:
    if ts.tzinfo is None:
        return False

    ny = ts.tz_convert("America/New_York")
    if ny.weekday() != 4:
        return False

    cutoff_total = FRIDAY_CUTOFF_HOUR * 60 + FRIDAY_CUTOFF_MINUTE
    now_total = ny.hour * 60 + ny.minute
    remaining = cutoff_total - now_total

    need = max(0, int(max_hold_bars)) * max(1, int(bar_minutes)) + max(0, int(friday_buffer_minutes))
    return remaining < need