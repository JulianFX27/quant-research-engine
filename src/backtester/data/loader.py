# src/backtester/data/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

from backtester.data.dataset_fingerprint import DatasetMetadata, build_dataset_metadata


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
        sample = df[df[nan_cols].isna().any(axis=1)].head(5)
        raise ValueError(
            "Found non-numeric / missing values after coercion in columns: "
            f"{nan_cols}. Sample rows:\n{sample.to_string(index=False)}"
        )


def _validate_ohlc_integrity(df: pd.DataFrame) -> None:
    bad_high = df["high"] < df[["open", "close"]].max(axis=1)
    bad_low = df["low"] > df[["open", "close"]].min(axis=1)

    if bad_high.any() or bad_low.any():
        bad_idx = df.index[bad_high | bad_low]
        sample = df.loc[bad_idx].head(5).reset_index()
        raise ValueError(
            "Invalid OHLC integrity detected. Expected: high >= max(open,close) and low <= min(open,close). "
            f"Sample offending rows:\n{sample.to_string(index=False)}"
        )

    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        sample = df[bad_hl].head(5).reset_index()
        raise ValueError(
            "Invalid OHLC integrity detected: high < low. "
            f"Sample offending rows:\n{sample.to_string(index=False)}"
        )


def load_bars_csv(
    path: str,
    *,
    return_fingerprint: bool = False,
    dataset_id: Optional[str] = None,
    keep_extra_cols: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, DatasetMetadata]]:
    """
    Load normalized OHLCV bars.

    Required columns:
      - time (parseable datetime; timezone-naive assumed UTC)
      - open, high, low, close (numeric)

    Optional columns:
      - volume (numeric)
      - spread (numeric; in price units if provided)

    keep_extra_cols:
      - False (default): return only canonical OHLC(+optional) columns.
      - True: return all columns from source CSV, but still validates OHLC integrity and sorts/dedups by time.
        (Fingerprint is still computed ONLY from canonical OHLC columns.)

    If return_fingerprint=True:
      - requires dataset_id
      - returns (df_out, DatasetMetadata) where DatasetMetadata is computed from canonical OHLC only.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Data file not found: {path}")

    df = pd.read_csv(p)

    _ensure_columns(df, _REQUIRED_COLS)

    # Parse time -> UTC
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if df["time"].isna().any():
        sample = df[df["time"].isna()].head(5)
        raise ValueError(
            "Failed to parse some 'time' values as datetime. Sample rows:\n"
            f"{sample.to_string(index=False)}"
        )

    # Sort
    df = df.sort_values("time")

    # Duplicates: raise
    dup_mask = df["time"].duplicated(keep=False)
    if dup_mask.any():
        sample = df.loc[dup_mask, ["time"]].head(10)
        raise ValueError(
            "Duplicate timestamps found in 'time' column. "
            "Please de-duplicate upstream deterministically. Sample duplicates:\n"
            f"{sample.to_string(index=False)}"
        )

    # Coerce required OHLC + optional columns if present
    numeric_cols = ["open", "high", "low", "close"] + [c for c in _OPTIONAL_COLS if c in df.columns]
    _coerce_numeric(df, numeric_cols)

    # Set index
    df = df.set_index("time")
    df.index.name = "time"

    # Validate OHLC integrity (on full df)
    _validate_ohlc_integrity(df)

    # Canonical slice (always exists)
    canonical_cols = ["open", "high", "low", "close"] + [c for c in _OPTIONAL_COLS if c in df.columns]
    df_canon = df[canonical_cols].copy()

    # Output df (either canonical-only or full)
    df_out = df.copy() if keep_extra_cols else df_canon

    if not return_fingerprint:
        return df_out

    if not dataset_id:
        raise ValueError("dataset_id is required when return_fingerprint=True")

    dataset_meta = build_dataset_metadata(
        df_canon,  # IMPORTANT: fingerprint only on canonical OHLC
        dataset_id=dataset_id,
        source_path=str(p),
        include_file_hash=True,
        time_col=None,
        price_cols=("open", "high", "low", "close"),
        sort_by_time=True,
    )

    return df_out, dataset_meta
