from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


_REQUIRED_COLS: List[str] = ["time", "open", "high", "low", "close"]
_OPTIONAL_COLS: List[str] = ["volume", "spread"]


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    bad: List[str] = []
    for c in cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            bad.append(c)
    if bad:
        raise ValueError(f"Failed to coerce columns to numeric: {bad}")

    nan_cols = [c for c in cols if df[c].isna().any()]
    if nan_cols:
        # Show a small sample of offending rows for debugging
        sample = df[df[nan_cols].isna().any(axis=1)].head(5)
        raise ValueError(
            "Found non-numeric / missing values after coercion in columns: "
            f"{nan_cols}. Sample rows:\n{sample.to_string(index=False)}"
        )


def _validate_ohlc_integrity(df: pd.DataFrame) -> None:
    # high must be >= open/close, low must be <= open/close
    bad_high = df["high"] < df[["open", "close"]].max(axis=1)
    bad_low = df["low"] > df[["open", "close"]].min(axis=1)

    if bad_high.any() or bad_low.any():
        bad_idx = df.index[bad_high | bad_low]
        sample = df.loc[bad_idx].head(5).reset_index()
        raise ValueError(
            "Invalid OHLC integrity detected. Expected: high >= max(open,close) and low <= min(open,close). "
            f"Sample offending rows:\n{sample.to_string(index=False)}"
        )

    # Also ensure high >= low
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        sample = df[bad_hl].head(5).reset_index()
        raise ValueError(
            "Invalid OHLC integrity detected: high < low. "
            f"Sample offending rows:\n{sample.to_string(index=False)}"
        )


def load_bars_csv(path: str) -> pd.DataFrame:
    """
    Load normalized OHLCV bars.

    Required columns:
      - time (parseable datetime; timezone-naive assumed UTC)
      - open, high, low, close (numeric)

    Optional columns:
      - volume (numeric)
      - spread (numeric; in price units if provided)

    Output:
      - DataFrame indexed by UTC Timestamp (index name: 'time')
      - Sorted ascending by time
      - No duplicate timestamps (duplicates raise error to avoid silent data issues)
      - Canonical columns kept: open, high, low, close, volume?, spread?
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Data file not found: {path}")

    df = pd.read_csv(p)

    _ensure_columns(df, _REQUIRED_COLS)

    # Parse time -> UTC
    # If input is timezone-naive, pandas with utc=True assumes it's UTC; that's acceptable for our contract.
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if df["time"].isna().any():
        sample = df[df["time"].isna()].head(5)
        raise ValueError(
            "Failed to parse some 'time' values as datetime. Sample rows:\n"
            f"{sample.to_string(index=False)}"
        )

    # Sort
    df = df.sort_values("time")

    # Duplicates: raise (prefer explicit data cleaning over silent dropping)
    dup_mask = df["time"].duplicated(keep=False)
    if dup_mask.any():
        sample = df.loc[dup_mask, ["time"]].head(10)
        raise ValueError(
            "Duplicate timestamps found in 'time' column. "
            "Please de-duplicate upstream deterministically. Sample duplicates:\n"
            f"{sample.to_string(index=False)}"
        )

    # Coerce numerics (required + optional if present)
    numeric_cols = ["open", "high", "low", "close"] + [c for c in _OPTIONAL_COLS if c in df.columns]
    _coerce_numeric(df, numeric_cols)

    # Set index
    df = df.set_index("time")
    df.index.name = "time"

    # Validate OHLC integrity
    _validate_ohlc_integrity(df)

    # Keep canonical columns only
    keep = ["open", "high", "low", "close"] + [c for c in _OPTIONAL_COLS if c in df.columns]
    return df[keep].copy()
