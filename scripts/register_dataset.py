from __future__ import annotations

import argparse
from pathlib import Path

from src.backtester.data.registry import register_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Register a dataset into datasets.json registry.")
    ap.add_argument("--registry", required=True, help="Path to datasets.json registry file")
    ap.add_argument("--dataset-id", required=True, help="Stable dataset identifier")
    ap.add_argument("--path", required=True, help="Path to dataset CSV/parquet")
    ap.add_argument("--instrument", required=True, help="Instrument, e.g., EURUSD")
    ap.add_argument("--timeframe", required=True, help="Timeframe, e.g., M5")
    ap.add_argument("--source", required=True, help="Data source, e.g., mt5_export")
    ap.add_argument("--notes", default="", help="Optional notes")
    args = ap.parse_args()

    entry = register_dataset(
        registry_path=Path(args.registry),
        dataset_id=args.dataset_id,
        dataset_path=Path(args.path),
        instrument=args.instrument,
        timeframe=args.timeframe,
        source=args.source,
        notes=args.notes,
        timezone_name="UTC",
        path_hint=args.path,
    )

    print("REGISTERED OK")
    print(f"dataset_id:   {entry.dataset_id}")
    print(f"fingerprint:  {entry.fingerprint}")
    print(f"instrument:   {entry.instrument}")
    print(f"timeframe:    {entry.timeframe}")
    print(f"start_ts:     {entry.start_ts}")
    print(f"end_ts:       {entry.end_ts}")
    print(f"n_rows:       {entry.n_rows}")


if __name__ == "__main__":
    main()
