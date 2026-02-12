from __future__ import annotations

import pandas as pd

from .thresholds import Thresholds


def classify_raw_regime(er: pd.Series, dp: pd.Series, thr: Thresholds) -> pd.Series:
    """
    Default rules:
      TREND if er >= er_trend AND dp >= dp_trend
      RANGE if er <= er_range AND dp <= dp_range
      else NO_TRADE
    NaNs -> NO_TRADE
    """
    idx = er.index
    out = pd.Series(["NO_TRADE"] * len(idx), index=idx, dtype="object")

    mask_trend = (er >= thr.er_trend) & (dp >= thr.dp_trend)
    mask_range = (er <= thr.er_range) & (dp <= thr.dp_range)

    # NaNs => False in comparisons; remain NO_TRADE
    out.loc[mask_trend] = "TREND"
    out.loc[mask_range] = "RANGE"
    return out
