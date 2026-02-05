from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .dataset_fingerprint import fingerprint_dataset  # assumes you already have this
from .loader import load_ohlc_dataset  # assumes your loader returns a DataFrame + maybe meta


# -----------------------------
# Types
# -----------------------------

@dataclass(frozen=True)
class DatasetEntry:
    dataset_id: str
    fingerprint: str
    instrument: str
    timeframe: str
    source: str
    timezone: str
    start_ts: str
    end_ts: str
    n_rows: int
    notes: str
    created_at: str
    path_hint: Optional[str] = None  # optional, useful for humans


@dataclass(frozen=True)
class Registry:
    version: int
    datasets: Dict[str, DatasetEntry]


# -----------------------------
# JSON Registry IO
# -----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_registry(registry_path: str | Path) -> Registry:
    p = Path(registry_path)
    if not p.exists():
        # initialize empty registry
        return Registry(version=1, datasets={})

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    version = int(raw.get("version", 1))
    raw_datasets = raw.get("datasets", {}) or {}

    datasets: Dict[str, DatasetEntry] = {}
    for dataset_id, d in raw_datasets.items():
        datasets[dataset_id] = DatasetEntry(
            dataset_id=dataset_id,
            fingerprint=str(d["fingerprint"]),
            instrument=str(d["instrument"]),
            timeframe=str(d["timeframe"]),
            source=str(d.get("source", "")),
            timezone=str(d.get("timezone", "UTC")),
            start_ts=str(d["start_ts"]),
            end_ts=str(d["end_ts"]),
            n_rows=int(d["n_rows"]),
            notes=str(d.get("notes", "")),
            created_at=str(d.get("created_at", "")),
            path_hint=d.get("path_hint"),
        )

    return Registry(version=version, datasets=datasets)


def save_registry(registry_path: str | Path, registry: Registry) -> None:
    p = Path(registry_path)
    payload = {
        "version": registry.version,
        "datasets": {
            dataset_id: {
                "fingerprint": e.fingerprint,
                "instrument": e.instrument,
                "timeframe": e.timeframe,
                "source": e.source,
                "timezone": e.timezone,
                "start_ts": e.start_ts,
                "end_ts": e.end_ts,
                "n_rows": e.n_rows,
                "notes": e.notes,
                "created_at": e.created_at,
                "path_hint": e.path_hint,
            }
            for dataset_id, e in registry.datasets.items()
        },
    }
    _atomic_write_json(p, payload)


# -----------------------------
# Core operations
# -----------------------------

def compute_dataset_stats(dataset_path: str | Path) -> Tuple[str, str, int]:
    """
    Uses the canonical loader to ensure the same parsing rules as backtests.
    Returns (start_ts_isoZ, end_ts_isoZ, n_rows).
    """
    df = load_ohlc_dataset(str(dataset_path))

    # Expect df has a datetime-like index or a 'time' column
    if hasattr(df, "index") and getattr(df.index, "dtype", None) is not None:
        # index case
        times = df.index
    elif "time" in df.columns:
        times = df["time"]
    else:
        raise ValueError("Dataset must have a datetime index or a 'time' column for registry stats.")

    # Convert to python datetimes in UTC ISO
    t0 = times.iloc[0] if hasattr(times, "iloc") else times[0]
    t1 = times.iloc[-1] if hasattr(times, "iloc") else times[-1]

    # Normalize to ISO string; assume loader already UTC
    def _to_isoz(t) -> str:
        if isinstance(t, str):
            return t
        try:
            dt = t.to_pydatetime()
        except Exception:
            dt = t
        if getattr(dt, "tzinfo", None) is None:
            # treat as UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    start_ts = _to_isoz(t0)
    end_ts = _to_isoz(t1)
    n_rows = int(len(df))
    return start_ts, end_ts, n_rows


def register_dataset(
    registry_path: str | Path,
    dataset_id: str,
    dataset_path: str | Path,
    *,
    instrument: str,
    timeframe: str,
    source: str,
    notes: str = "",
    timezone_name: str = "UTC",
    path_hint: Optional[str] = None,
) -> DatasetEntry:
    """
    Registers a dataset_id -> fingerprint + metadata.
    If dataset_id already exists:
      - if fingerprint matches: no-op (returns existing entry)
      - if fingerprint differs: raises (forces versioning discipline)
    """
    reg = load_registry(registry_path)

    fp = fingerprint_dataset(str(dataset_path))
    start_ts, end_ts, n_rows = compute_dataset_stats(dataset_path)

    if dataset_id in reg.datasets:
        existing = reg.datasets[dataset_id]
        if existing.fingerprint != fp:
            raise ValueError(
                f"Registry conflict for dataset_id='{dataset_id}': "
                f"existing fingerprint={existing.fingerprint} vs new fingerprint={fp}. "
                f"Create a new dataset_id (version bump) instead."
            )
        return existing

    entry = DatasetEntry(
        dataset_id=dataset_id,
        fingerprint=fp,
        instrument=instrument,
        timeframe=timeframe,
        source=source,
        timezone=timezone_name,
        start_ts=start_ts,
        end_ts=end_ts,
        n_rows=n_rows,
        notes=notes,
        created_at=_utc_now_iso(),
        path_hint=path_hint or str(dataset_path),
    )

    datasets = dict(reg.datasets)
    datasets[dataset_id] = entry
    reg2 = Registry(version=reg.version, datasets=datasets)
    save_registry(registry_path, reg2)
    return entry


def enforce_dataset(
    registry_path: str | Path,
    dataset_id: str,
    actual_fingerprint: str,
) -> DatasetEntry:
    """
    Ensures dataset_id exists in registry and fingerprint matches the actual file.
    Returns the registry entry if OK.
    """
    reg = load_registry(registry_path)
    if dataset_id not in reg.datasets:
        raise ValueError(
            f"Dataset '{dataset_id}' is not registered in {registry_path}. "
            f"Register it before running with strict_registry=true."
        )
    entry = reg.datasets[dataset_id]
    if entry.fingerprint != actual_fingerprint:
        raise ValueError(
            f"Dataset fingerprint mismatch for '{dataset_id}': "
            f"registry={entry.fingerprint} vs actual={actual_fingerprint}. "
            f"Your file changed; bump dataset_id and re-register."
        )
    return entry
