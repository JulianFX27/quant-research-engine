# src/backtester/data/dataset_fingerprint.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple
import hashlib
import json

import numpy as np
import pandas as pd


DEFAULT_PRICE_COLS = ("open", "high", "low", "close")
DEFAULT_TIME_COL_CANDIDATES = ("timestamp", "time", "ts", "datetime")


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    fingerprint_sha256: str
    schema_sha256: str
    rows: int
    start_ts: str
    end_ts: str
    time_col: str
    price_cols: Tuple[str, str, str, str]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["price_cols"] = list(self.price_cols)  # tuple -> list for JSON friendliness
        return d

    @property
    def fingerprint_short(self) -> str:
        hexpart = self.fingerprint_sha256.split("sha256:", 1)[-1]
        return hexpart[:8]


def _pick_time_col(df: pd.DataFrame, time_col: Optional[str]) -> str:
    if time_col is not None:
        if time_col not in df.columns:
            raise KeyError(f"time_col='{time_col}' not found in columns={list(df.columns)}")
        return time_col

    for c in DEFAULT_TIME_COL_CANDIDATES:
        if c in df.columns:
            return c

    raise KeyError(
        f"Could not infer time column. Provide time_col. "
        f"Tried={DEFAULT_TIME_COL_CANDIDATES}, columns={list(df.columns)}"
    )


def _validate_required_cols(df: pd.DataFrame, time_col: str, price_cols: Sequence[str]) -> None:
    missing = [c for c in (time_col, *price_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available={list(df.columns)}")


def _ensure_utc_ns(series: pd.Series) -> np.ndarray:
    """
    Returns int64 nanoseconds since epoch UTC (deterministic).

    Accepts:
      - datetime strings parseable by pandas
      - tz-aware / tz-naive datetimes (tz-naive treated as UTC)
      - integers already representing nanoseconds (if dtype is integer-like)

    IMPORTANT:
      Some pandas builds may hold tz-aware datetime as datetime64[us, UTC] (microseconds).
      If we cast to int64 directly, we'd get microseconds. We must normalize to ns.
    """
    s = series

    # Integer-like: assume already ns (caller responsibility). Deterministic.
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.to_numpy(dtype="int64", copy=False)

    # Parse/normalize to UTC tz-aware
    dt_utc = pd.to_datetime(s, utc=True, errors="raise")

    # Convert to tz-naive UTC then force ns resolution
    # This avoids the "datetime64[us]" vs "datetime64[ns]" ambiguity.
    dt_naive = dt_utc.dt.tz_convert("UTC").dt.tz_localize(None)

    # Force numpy datetime64[ns] and then view as int64 ns
    dt64_ns = dt_naive.to_numpy(dtype="datetime64[ns]")
    return dt64_ns.view("int64")


def _schema_hash(df: pd.DataFrame, time_col: str, price_cols: Sequence[str]) -> str:
    schema_obj = {
        "time_col": time_col,
        "price_cols": list(price_cols),
        "dtypes": {c: str(df[c].dtype) for c in (time_col, *price_cols)},
    }
    payload = json.dumps(schema_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_bytes(
    df: pd.DataFrame,
    time_col: str,
    price_cols: Sequence[str],
    sort_by_time: bool,
) -> Tuple[bytes, int, str, str]:
    """
    Canonical representation:
      - timestamps as int64 ns UTC
      - OHLC as float64
      - optionally sorted by timestamp (recommended True)
    Returns: (bytes, rows, start_ts_iso, end_ts_iso)
    """
    ts_ns = _ensure_utc_ns(df[time_col])

    # enforce finite OHLC
    ohlc = df.loc[:, list(price_cols)].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(ohlc).all():
        bad = np.where(~np.isfinite(ohlc))
        raise ValueError(f"Non-finite OHLC detected at indices={bad}")

    if sort_by_time:
        order = np.argsort(ts_ns, kind="mergesort")  # stable deterministic
        ts_ns = ts_ns[order]
        ohlc = ohlc[order]

    # duplicates are not allowed
    if ts_ns.size > 1:
        dup_mask = ts_ns[1:] == ts_ns[:-1]
        if dup_mask.any():
            i = int(np.where(dup_mask)[0][0])
            raise ValueError(f"Duplicate timestamps detected at sorted positions {i} and {i+1}")

    rows = int(ts_ns.shape[0])
    if rows == 0:
        raise ValueError("Empty dataset (0 rows)")

    start_iso = pd.to_datetime(ts_ns[0], utc=True, unit="ns").isoformat()
    end_iso = pd.to_datetime(ts_ns[-1], utc=True, unit="ns").isoformat()

    out = np.empty(
        rows,
        dtype=[("ts", "<i8"), ("o", "<f8"), ("h", "<f8"), ("l", "<f8"), ("c", "<f8")],
    )
    out["ts"] = ts_ns.astype("<i8", copy=False)
    out["o"] = ohlc[:, 0].astype("<f8", copy=False)
    out["h"] = ohlc[:, 1].astype("<f8", copy=False)
    out["l"] = ohlc[:, 2].astype("<f8", copy=False)
    out["c"] = ohlc[:, 3].astype("<f8", copy=False)

    return out.tobytes(order="C"), rows, start_iso, end_iso


def build_dataset_id(
    instrument: str,
    timeframe: str,
    start_ts: str,
    end_ts: str,
    source: Optional[str] = None,
) -> str:
    base = f"{instrument}_{timeframe}_{start_ts[:10]}__{end_ts[:10]}"
    return f"{base}__{source}" if source else base


def build_dataset_metadata(
    df: pd.DataFrame,
    dataset_id: str,
    time_col: Optional[str] = None,
    price_cols: Sequence[str] = DEFAULT_PRICE_COLS,
    sort_by_time: bool = True,
) -> DatasetMetadata:
    time_col = _pick_time_col(df, time_col)
    price_cols = tuple(price_cols)
    if len(price_cols) != 4:
        raise ValueError("price_cols must have 4 columns: (open, high, low, close)")

    _validate_required_cols(df, time_col=time_col, price_cols=price_cols)

    schema_sha = _schema_hash(df, time_col=time_col, price_cols=price_cols)
    payload, rows, start_iso, end_iso = _canonical_bytes(
        df=df,
        time_col=time_col,
        price_cols=price_cols,
        sort_by_time=sort_by_time,
    )
    fp = "sha256:" + hashlib.sha256(payload).hexdigest()

    return DatasetMetadata(
        dataset_id=dataset_id,
        fingerprint_sha256=fp,
        schema_sha256=schema_sha,
        rows=rows,
        start_ts=start_iso,
        end_ts=end_iso,
        time_col=time_col,
        price_cols=price_cols,
    )
