# src/backtester/orchestrator/shock_z_hook.py
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _sha256_of_file(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


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


def attach_shock_z_if_enabled(
    df: pd.DataFrame,
    cfg_resolved: Dict[str, Any],
    *,
    dataset_id: str,
) -> Tuple[pd.DataFrame, Dict[str, Any] | None]:
    """
    Deterministic feature hook.

    Adds column:
      - shock_z: rolling z-score of log returns over horizon_bars

    Formula:
      r_t = log(close_t / close_{t-h})
      shock_z_t = (r_t - mean(r_{t-lookback+1:t})) / std(r_{t-lookback+1:t})
      std uses ddof=0; std_floor avoids division blow-ups.

    Config (run YAML):
      shock_z:
        enabled: true
        close_col: "close"
        horizon_bars: 1
        lookback: 288
        warmup_bars: 300
        std_floor: 1e-9
        clip_z: 10.0   # optional

    Notes:
      - Does NOT overwrite existing shock_z.
      - No randomness.
      - Assumes df index already aligned to time (hook does not change index).
    """
    if not isinstance(cfg_resolved, dict):
        return df, None

    sz_cfg = cfg_resolved.get("shock_z") or {}
    if not isinstance(sz_cfg, dict) or not bool(sz_cfg.get("enabled", False)):
        return df, None

    if "shock_z" in df.columns:
        payload = {
            "enabled": True,
            "dataset_id": dataset_id,
            "status": "already_present",
            "column": "shock_z",
        }
        return df, payload

    close_col = str(sz_cfg.get("close_col") or "close")
    if close_col not in df.columns:
        raise ValueError(f"shock_z requires column '{close_col}' in df")

    horizon = _norm_int(sz_cfg.get("horizon_bars"), default=1)
    lookback = _norm_int(sz_cfg.get("lookback"), default=288)
    warmup = _norm_int(sz_cfg.get("warmup_bars"), default=max(lookback + horizon + 5, 300))
    std_floor = _norm_float(sz_cfg.get("std_floor"), default=1e-9)

    clip_z = sz_cfg.get("clip_z", None)
    clip_z_f = None
    if clip_z is not None:
        clip_z_f = _norm_float(clip_z, default=10.0)

    if horizon < 1:
        raise ValueError(f"shock_z.horizon_bars must be >= 1, got {horizon}")
    if lookback < 5:
        raise ValueError(f"shock_z.lookback must be >= 5, got {lookback}")
    if warmup < (lookback + horizon + 1):
        raise ValueError(
            f"shock_z.warmup_bars too small: warmup={warmup} must be >= lookback+horizon+1={lookback+horizon+1}"
        )
    if std_floor <= 0:
        raise ValueError(f"shock_z.std_floor must be > 0, got {std_floor}")

    # Coerce close to float and validate
    close = pd.to_numeric(df[close_col], errors="coerce").astype("float64")
    if close.isna().any():
        n_bad = int(close.isna().sum())
        raise ValueError(f"shock_z: close contains {n_bad} NaN values after coercion")

    # log return over horizon: log(close_t / close_{t-h})
    denom = close.shift(horizon)
    ratio = close / denom
    ratio = ratio.where(ratio > 0.0, np.nan)  # guards against non-positive values
    r = np.log(ratio)

    # rolling mean/std
    mu = r.rolling(lookback, min_periods=lookback).mean()
    sd = r.rolling(lookback, min_periods=lookback).std(ddof=0)

    # floor std to avoid division blow-ups
    sd_eff = sd.where(sd > std_floor, std_floor)
    shock_z = (r - mu) / sd_eff

    # optional deterministic clipping
    if clip_z_f is not None and clip_z_f > 0:
        shock_z = shock_z.clip(lower=-clip_z_f, upper=clip_z_f)

    # enforce warmup: null out early values deterministically
    if warmup > 0:
        shock_z = shock_z.copy()
        shock_z.iloc[:warmup] = pd.NA

    df2 = df.copy()
    df2["shock_z"] = shock_z

    payload: Dict[str, Any] = {
        "enabled": True,
        "dataset_id": dataset_id,
        "status": "computed",
        "column": "shock_z",
        "close_col": close_col,
        "horizon_bars": int(horizon),
        "lookback": int(lookback),
        "warmup_bars": int(warmup),
        "std_floor": float(std_floor),
        "clip_z": float(clip_z_f) if clip_z_f is not None else None,
        "impl": "shock_z_hook_v1",
    }

    # Optional: include file hash if caller passes it somewhere else (not available here).
    return df2, payload
