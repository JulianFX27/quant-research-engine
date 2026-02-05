from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtester.core.contracts import OrderIntent
from backtester.strategies.base import Strategy


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    return np.maximum(tr1, np.maximum(tr2, tr3))


def _sma(arr: np.ndarray, n: int) -> np.ndarray:
    # Simple moving average via convolution, "valid" (no lookahead)
    if n <= 0:
        raise ValueError("n must be > 0")
    kernel = np.ones(n, dtype=float) / float(n)
    return np.convolve(arr, kernel, mode="valid")


class BaselineV4VolBucketBuy(Strategy):
    """
    Baseline BUY fijo (1 trade por día: primer bar UTC del día) con gating por volatilidad.

    Volatilidad: ATR(n) sobre M5, bucket por cuantiles globales del ATR (sobre toda la historia disponible hasta i).

    Params:
      - qty: float
      - sl_pips: float
      - tp_pips: float
      - warmup_bars: int (heredado de base)
      - atr_n: int (default 96)
      - q_low: float (default 0.33)
      - q_high: float (default 0.66)
      - vol_bucket: "LOW"|"MED"|"HIGH"|"ALL"  (default "ALL")
      - tag: str
    """

    def on_bar(self, i: int, df: pd.DataFrame, context: Dict[str, Any]) -> List[OrderIntent]:
        qty = float(self.params.get("qty", 1.0))
        sl_pips = float(self.params.get("sl_pips", 10.0))
        tp_pips = float(self.params.get("tp_pips", 20.0))

        atr_n = int(self.params.get("atr_n", 96))
        q_low = float(self.params.get("q_low", 0.33))
        q_high = float(self.params.get("q_high", 0.66))

        vol_bucket = str(self.params.get("vol_bucket", "ALL")).upper()
        if vol_bucket not in ("LOW", "MED", "HIGH", "ALL"):
            raise ValueError(f"Invalid vol_bucket={vol_bucket}. Use LOW|MED|HIGH|ALL")

        tag = str(self.params.get("tag", "BASELINE_V4_VOL_BUCKET_BUY"))

        instrument = (context.get("instrument") or {})
        pip_size = float(instrument.get("pip_size", 0.0) or 0.0)
        if pip_size <= 0:
            raise ValueError("BaselineV4VolBucketBuy requires instrument.pip_size > 0 when using sl_pips/tp_pips")

        # Need warmup + atr_n to compute ATR series (TR->ATR)
        if i < max(self.warmup_bars, atr_n + 2):
            return []

        # Only first bar of the UTC day
        if i > 0 and df.index[i].date() == df.index[i - 1].date():
            return []

        # Build ATR series up to i (no lookahead)
        hist = df.iloc[: i + 1]
        high = hist["high"].to_numpy(dtype=float)
        low = hist["low"].to_numpy(dtype=float)
        close = hist["close"].to_numpy(dtype=float)

        tr = _true_range(high, low, close)
        if len(tr) < atr_n:
            return []

        atr_series = _sma(tr, atr_n)  # length = len(tr) - atr_n + 1
        if len(atr_series) < 50:
            # minimal history for quantiles stability
            return []

        # Align last ATR value to current bar i
        atr_now = float(atr_series[-1])

        # Quantiles computed on atr_series up to now
        p_low = float(np.quantile(atr_series, q_low))
        p_high = float(np.quantile(atr_series, q_high))

        if atr_now <= p_low:
            bucket = "LOW"
        elif atr_now <= p_high:
            bucket = "MED"
        else:
            bucket = "HIGH"

        # Gating (the actual fix: strict vol_bucket contract)
        if vol_bucket != "ALL" and bucket != vol_bucket:
            return []

        close_px = float(df.iloc[i]["close"])
        sl = close_px - sl_pips * pip_size
        tp = close_px + tp_pips * pip_size

        return [
            OrderIntent(
                side="BUY",
                qty=qty,
                sl_price=sl,
                tp_price=tp,
                tag=f"{tag}|VOL_{bucket}",
            )
        ]
