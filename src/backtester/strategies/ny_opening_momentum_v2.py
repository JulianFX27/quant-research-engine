from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtester.core.contracts import OrderIntent
from backtester.strategies.base import Strategy


def ny_open_utc(ts_like) -> pd.Timestamp:
    """
    Return NY cash open in UTC (09:30 America/New_York) for the civil date of ts_like.
    """
    if isinstance(ts_like, pd.Timestamp):
        d = ts_like.date()
    else:
        d = pd.Timestamp(ts_like).date()

    ny_tz = ZoneInfo("America/New_York")
    dt_local = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ny_tz)
    dt_utc = dt_local.astimezone(timezone.utc)
    return pd.Timestamp(dt_utc)


class NYOpeningMomentumV2(Strategy):
    """
    NY Opening Momentum v2
    ----------------------

    Research-faithful implementation for the backtester:
      - Measure impulse from NY open to NY open + 30m
      - If ret_30m is extreme (crosses rolling quantile threshold) and impulse efficiency is high,
        enter in the direction of the impulse at NY open + entry_delay_min
      - Exit via time-stop using engine risk.max_holding_bars (no SL/TP price logic)

    Important:
      - Designed for M5 data
      - Emits an OrderIntent only once per day, on the exact entry bar
      - Uses historical ret_30m distribution up to the prior completed day (no same-day leakage)
      - Time-stop behavior must be enforced by config:
            risk.max_holding_bars
            risk.allow_missing_sl = true
            risk.risk_proxy_price > 0
    """

    def on_bar(self, i: int, df, context: Dict[str, Any]) -> List[OrderIntent]:
        threshold_q = float(self.params.get("threshold_q", 0.80))
        impulse_efficiency_min = float(self.params.get("impulse_efficiency_min", 0.70))
        entry_delay_min = int(self.params.get("entry_delay_min", 30))
        holding_min = int(self.params.get("holding_min", 30))
        qty = float(self.params.get("qty", 1.0))

        if i < max(self.warmup_bars, 250):
            return []

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("NYOpeningMomentumV2 expects df.index to be a DatetimeIndex")

        ts = df.index[i]
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        if i < 1:
            return []

        prev_ts = df.index[i - 1]
        if prev_ts.tz is None:
            prev_ts = prev_ts.tz_localize("UTC")
        else:
            prev_ts = prev_ts.tz_convert("UTC")

        bar_minutes = int((ts - prev_ts).total_seconds() // 60)
        if bar_minutes <= 0:
            return []

        if bar_minutes != 5:
            return []

        if entry_delay_min % bar_minutes != 0 or holding_min % bar_minutes != 0:
            raise ValueError("entry_delay_min and holding_min must align with bar size")

        current_day = ts.date()
        t0 = ny_open_utc(current_day)
        signal_time = t0 + pd.Timedelta(minutes=30)
        entry_time = t0 + pd.Timedelta(minutes=entry_delay_min)

        if ts != entry_time:
            return []

        try:
            loc_t0 = df.index.get_loc(t0)
            loc_signal = df.index.get_loc(signal_time)
            loc_entry = df.index.get_loc(entry_time)
        except KeyError:
            return []

        if loc_entry != i:
            return []

        if loc_signal <= loc_t0:
            return []

        open_0 = float(df.iloc[loc_t0]["open"])
        close_30 = float(df.iloc[loc_signal]["close"])

        window_30 = df.iloc[loc_t0 : loc_signal + 1]
        if window_30.empty:
            return []

        ret_30m = (close_30 - open_0) / open_0
        range_30m = (float(window_30["high"].max()) - float(window_30["low"].min())) / open_0

        if range_30m <= 0:
            return []

        impulse_efficiency = abs(ret_30m) / range_30m
        if impulse_efficiency < impulse_efficiency_min:
            return []

        # Historical benchmark using completed prior days only
        rets_history: list[float] = []
        dates_seen = pd.Index(df.index[: i + 1].date).unique()

        for d in dates_seen:
            if d >= current_day:
                continue

            d_t0 = ny_open_utc(d)
            d_signal = d_t0 + pd.Timedelta(minutes=30)

            try:
                d_loc_t0 = df.index.get_loc(d_t0)
                d_loc_signal = df.index.get_loc(d_signal)
            except KeyError:
                continue

            d_open = float(df.iloc[d_loc_t0]["open"])
            d_close30 = float(df.iloc[d_loc_signal]["close"])
            d_ret30 = (d_close30 - d_open) / d_open
            rets_history.append(d_ret30)

        if len(rets_history) < 100:
            return []

        q_low = float(np.quantile(rets_history, 1 - threshold_q))
        q_high = float(np.quantile(rets_history, threshold_q))

        tag_suffix = (
            f"_tq{threshold_q}"
            f"_eff{impulse_efficiency_min}"
            f"_ed{entry_delay_min}"
            f"_h{holding_min}"
        )

        if ret_30m > q_high:
            return [
                OrderIntent(
                    side="BUY",
                    qty=qty,
                    tag=f"nyomv2_long{tag_suffix}",
                )
            ]

        if ret_30m < q_low:
            return [
                OrderIntent(
                    side="SELL",
                    qty=qty,
                    tag=f"nyomv2_short{tag_suffix}",
                )
            ]

        return []