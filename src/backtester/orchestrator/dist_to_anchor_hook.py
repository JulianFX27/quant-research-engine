# src/backtester/orchestrator/dist_to_anchor_hook.py
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _norm_int(x: Any, default: int) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, str) and x.strip() == "":
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _norm_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, str) and x.strip() == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def attach_dist_to_anchor_if_enabled(
    df: pd.DataFrame,
    cfg_resolved: Dict[str, Any],
    *,
    pip_size: float,
) -> Tuple[pd.DataFrame, Dict[str, Any] | None]:
    """
    Deterministic feature hook.

    Adds column:
      - dist_to_anchor_pips: (close - anchor) / pip_size

    Anchor options:
      - SMA(close, lookback)
      - EMA(close, lookback)

    Config (run YAML):
      dist_to_anchor:
        enabled: true
        close_col: "close"
        out_col: "dist_to_anchor_pips"
        anchor_mode: "ema"        # "ema" or "sma"
        anchor_lookback: 288
        warmup_bars: 300
        abs_value: true           # optional: store abs(distance) instead of signed
    """
    dcfg = (cfg_resolved.get("dist_to_anchor") or {}) if isinstance(cfg_resolved, dict) else {}
    if not bool(dcfg.get("enabled", False)):
        return df, None

    if pip_size is None or float(pip_size) <= 0:
        raise ValueError("dist_to_anchor requires instrument.pip_size > 0")

    close_col = str(dcfg.get("close_col") or "close")
    out_col = str(dcfg.get("out_col") or "dist_to_anchor_pips")
    mode = str(dcfg.get("anchor_mode") or "ema").strip().lower()
    lookback = _norm_int(dcfg.get("anchor_lookback"), default=288)
    warmup = _norm_int(dcfg.get("warmup_bars"), default=max(lookback + 5, 300))
    abs_value = bool(dcfg.get("abs_value", False))

    if out_col in df.columns:
        payload = {
            "enabled": True,
            "status": "already_present",
            "out_col": out_col,
        }
        return df, payload

    if close_col not in df.columns:
        raise ValueError(f"dist_to_anchor requires column '{close_col}' in df")

    if lookback < 2:
        raise ValueError(f"dist_to_anchor.anchor_lookback must be >= 2, got {lookback}")
    if warmup < lookback:
        raise ValueError(f"dist_to_anchor.warmup_bars must be >= lookback ({lookback}), got {warmup}")
    if mode not in ("ema", "sma"):
        raise ValueError("dist_to_anchor.anchor_mode must be 'ema' or 'sma'")

    close = pd.to_numeric(df[close_col], errors="coerce")
    if close.isna().any():
        raise ValueError("dist_to_anchor: close contains NaN after coercion")

    if mode == "sma":
        anchor = close.rolling(lookback, min_periods=lookback).mean()
    else:
        # EMA deterministic; min_periods emulado con warmup
        anchor = close.ewm(span=lookback, adjust=False).mean()

    dist_pips = (close - anchor) / float(pip_size)

    if abs_value:
        dist_pips = dist_pips.abs()

    # warmup -> NaN
    if warmup > 0:
        dist_pips = dist_pips.copy()
        dist_pips.iloc[:warmup] = np.nan

    df2 = df.copy()
    df2[out_col] = dist_pips.astype("float64")

    payload = {
        "enabled": True,
        "status": "computed",
        "close_col": close_col,
        "out_col": out_col,
        "anchor_mode": mode,
        "anchor_lookback": int(lookback),
        "warmup_bars": int(warmup),
        "abs_value": bool(abs_value),
        "pip_size": float(pip_size),
    }
    return df2, payload
