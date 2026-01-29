from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import hashlib
import json

import numpy as np
import pandas as pd


FINGERPRINT_VERSION = "v1.1"

DEFAULT_PRICE_COLS = ("open", "high", "low", "close")
DEFAULT_TIME_COL_CANDIDATES = ("timestamp", "time", "ts", "datetime")
_INDEX_SENTINEL = "__index__"


@dataclass(frozen=True)
class DatasetMetadata:
    # Semantic identity (not content hash)
    dataset_id: str

    # Canonical (content-based) identity
    fingerprint_sha256: str
    schema_sha256: str

    # Raw file identity (audit / provenance)
    file_sha256: Optional[str] = None
    file_bytes: Optional[int] = None
    source_path: Optional[str] = None

    # Basic stats
    rows: int = 0
    start_ts: str = ""
    end_ts: str = ""
    time_col: str = _INDEX_SENTINEL
    price_cols: Tuple[str, str, str, str] = DEFAULT_PRICE_COLS

    # Versioning
    fingerprint_version: str = FINGERPRINT_VERSION

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["price_cols"] = list(self.price_cols)
        return d

    @property
    def fingerprint_short(self) -> str:
        hexpart = self.fingerprint_sha256.split("sha256:", 1)[-1]
        return hexpart[:8]


def _sha256_file(path: Union[str, Path], chunk_bytes: int = 1024 * 1024) -> Tuple[str, int]:
    p = Path(path)
    h = hashlib.sha256()
    total = 0
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            total += len(b)
            h.update(b)
    return "sha256:" + h.hexdigest(), total


def _pick_time_source(df: pd.DataFrame, time_col: Optional[str]) -> str:
    if time_col is not None:
        if time_col == _INDEX_SENTINEL:
            return _INDEX_SENTINEL
        if time_col not in df.columns:
            raise KeyError(f"time_col='{time_col}' not found in columns={list(df.columns)}")
        return time_col

    for c in DEFAULT_TIME_COL_CANDIDATES:
        if c in df.columns:
            return c

    idx_name = getattr(df.index, "name", None)
    if idx_name in DEFAULT_TIME_COL_CANDIDATES:
        return _INDEX_SENTINEL

    raise KeyError(
        f"Could not infer time column. Provide time_col or set df.index.name='time'. "
        f"Tried columns={DEFAULT_TIME_COL_CANDIDATES} and index.name={idx_name}. "
        f"Available columns={list(df.columns)}"
    )


def _validate_required_cols(df: pd.DataFrame, time_source: str, price_cols: Sequence[str]) -> None:
    missing_prices = [c for c in price_cols if c not in df.columns]
    if missing_prices:
        raise KeyError(f"Missing required OHLC columns: {missing_prices}. Available={list(df.columns)}")

    if time_source != _INDEX_SENTINEL and time_source not in df.columns:
        raise KeyError(f"Missing required time column: '{time_source}'. Available={list(df.columns)}")


def _get_time_series(df: pd.DataFrame, time_source: str) -> Union[pd.Series, pd.Index]:
    if time_source == _INDEX_SENTINEL:
        return df.index
    return df[time_source]


def _ensure_utc_ns(x: Union[pd.Series, pd.Index]) -> np.ndarray:
    if isinstance(x, pd.Index):
        if pd.api.types.is_integer_dtype(x.dtype):
            return x.to_numpy(dtype="int64", copy=False)

        dt_utc = pd.to_datetime(x, utc=True, errors="raise")
        dt_naive = dt_utc.tz_convert("UTC").tz_localize(None)
        dt64_ns = dt_naive.to_numpy(dtype="datetime64[ns]")
        return dt64_ns.view("int64")

    s = x
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.to_numpy(dtype="int64", copy=False)

    dt_utc = pd.to_datetime(s, utc=True, errors="raise")
    dt_naive = dt_utc.dt.tz_convert("UTC").dt.tz_localize(None)
    dt64_ns = dt_naive.to_numpy(dtype="datetime64[ns]")
    return dt64_ns.view("int64")


def _schema_hash(df: pd.DataFrame, time_source: str, price_cols: Sequence[str]) -> str:
    dtypes: Dict[str, str] = {}

    if time_source == _INDEX_SENTINEL:
        dtypes[_INDEX_SENTINEL] = str(df.index.dtype)
    else:
        dtypes[time_source] = str(df[time_source].dtype)

    for c in price_cols:
        dtypes[c] = str(df[c].dtype)

    schema_obj = {
        "version": FINGERPRINT_VERSION,
        "time_col": time_source,
        "price_cols": list(price_cols),
        "dtypes": dtypes,
    }
    payload = json.dumps(schema_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_bytes(
    df: pd.DataFrame,
    time_source: str,
    price_cols: Sequence[str],
    sort_by_time: bool,
) -> Tuple[bytes, int, str, str]:
    ts_src = _get_time_series(df, time_source)
    ts_ns = _ensure_utc_ns(ts_src)

    ohlc = df.loc[:, list(price_cols)].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(ohlc).all():
        bad = np.where(~np.isfinite(ohlc))
        raise ValueError(f"Non-finite OHLC detected at indices={bad}")

    if sort_by_time:
        order = np.argsort(ts_ns, kind="mergesort")
        ts_ns = ts_ns[order]
        ohlc = ohlc[order]

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
    *,
    source_path: Optional[str] = None,
    include_file_hash: bool = False,
    time_col: Optional[str] = None,
    price_cols: Sequence[str] = DEFAULT_PRICE_COLS,
    sort_by_time: bool = True,
) -> DatasetMetadata:
    price_cols = tuple(price_cols)
    if len(price_cols) != 4:
        raise ValueError("price_cols must have 4 columns: (open, high, low, close)")

    time_source = _pick_time_source(df, time_col)
    _validate_required_cols(df, time_source=time_source, price_cols=price_cols)

    schema_sha = _schema_hash(df, time_source=time_source, price_cols=price_cols)
    payload, rows, start_iso, end_iso = _canonical_bytes(
        df=df,
        time_source=time_source,
        price_cols=price_cols,
        sort_by_time=sort_by_time,
    )
    fp = "sha256:" + hashlib.sha256(payload).hexdigest()

    file_sha = None
    file_bytes = None
    if include_file_hash and source_path:
        file_sha, file_bytes = _sha256_file(source_path)

    return DatasetMetadata(
        dataset_id=dataset_id,
        fingerprint_sha256=fp,
        schema_sha256=schema_sha,
        file_sha256=file_sha,
        file_bytes=file_bytes,
        source_path=str(source_path) if source_path else None,
        rows=rows,
        start_ts=start_iso,
        end_ts=end_iso,
        time_col=time_source,
        price_cols=price_cols,
        fingerprint_version=FINGERPRINT_VERSION,
    )
