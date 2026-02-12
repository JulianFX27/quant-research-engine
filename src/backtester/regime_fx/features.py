from __future__ import annotations

import numpy as np
import pandas as pd


def compute_er(close: pd.Series, lookback: int, warmup: str = "nan") -> pd.Series:
    """
    Efficiency Ratio (ER):
      change = abs(close[t] - close[t-lookback])
      volatility = sum_{i=0..lookback-1} abs(close[t-i] - close[t-i-1])
      er = change / volatility (if volatility>0 else 0)
    """
    c = close.astype("float64")
    change = (c - c.shift(lookback)).abs()

    step = c.diff().abs()
    volatility = step.rolling(window=lookback, min_periods=lookback).sum()

    er = change / volatility.replace(0.0, np.nan)
    er = er.fillna(0.0)  # volatility==0 => 0, warmup handled below

    if warmup == "nan":
        er.iloc[:lookback] = np.nan
    elif warmup == "zero":
        er.iloc[:lookback] = 0.0
    else:
        raise ValueError("warmup must be 'nan' or 'zero'")

    return er.astype("float64")


def compute_dp(close: pd.Series, lookback: int, warmup: str = "nan") -> pd.Series:
    """
    Directional Persistence (DP):
      r_i = close[i] - close[i-1]
      s_i = sign(r_i) in {-1, 0, +1}
      dp = abs(sum(s_i)) / lookback  over window lookback
    """
    c = close.astype("float64")
    r = c.diff()
    s = np.sign(r).astype("float64")  # -1,0,1

    rolling_sum = pd.Series(s, index=c.index).rolling(window=lookback, min_periods=lookback).sum()
    dp = rolling_sum.abs() / float(lookback)

    if warmup == "nan":
        dp.iloc[:lookback] = np.nan
    elif warmup == "zero":
        dp.iloc[:lookback] = 0.0
    else:
        raise ValueError("warmup must be 'nan' or 'zero'")

    return dp.astype("float64")
