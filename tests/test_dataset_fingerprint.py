# tests/test_dataset_fingerprint.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtester.data.dataset_fingerprint import build_dataset_metadata


def test_dataset_metadata_includes_version_and_file_identity(tmp_path: Path) -> None:
    # Create a tiny deterministic dataset
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
            "open": [1.0, 1.1, 1.2],
            "high": [1.01, 1.11, 1.21],
            "low": [0.99, 1.09, 1.19],
            "close": [1.005, 1.105, 1.205],
        }
    ).set_index("time")
    df.index.name = "time"

    # Persist as CSV to test file hashing
    p = tmp_path / "x.csv"
    df.reset_index().to_csv(p, index=False)

    # Reload similarly to loader canonical output (index time + OHLC cols)
    df2 = pd.read_csv(p)
    df2["time"] = pd.to_datetime(df2["time"], utc=True)
    df2 = df2.set_index("time")
    df2.index.name = "time"

    meta = build_dataset_metadata(
        df2,
        dataset_id="EURUSD_M5_2026-01-01__2026-01-01__csv_unit",
        source_path=str(p),
        include_file_hash=True,
        time_col=None,  # infer from index.name == 'time'
        price_cols=("open", "high", "low", "close"),
        sort_by_time=True,
    )

    assert meta.fingerprint_version == "v1.1"
    assert meta.fingerprint_sha256.startswith("sha256:")
    assert meta.schema_sha256.startswith("sha256:")

    assert meta.file_sha256 is not None and meta.file_sha256.startswith("sha256:")
    assert meta.file_bytes is not None and meta.file_bytes > 0
    assert meta.source_path is not None and meta.source_path.endswith("x.csv")
    assert meta.time_col == "__index__"
    assert meta.rows == 3
