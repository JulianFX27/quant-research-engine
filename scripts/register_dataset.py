# scripts/register_dataset.py
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from backtester.data.dataset_fingerprint import build_dataset_id
from backtester.data.dataset_registry import register_or_validate_dataset
from backtester.data.loader import load_bars_csv


_UNKNOWN_SENTINEL = "unknown__unknown"


def _ts_to_ymd(ts: Any) -> str:
    """
    Normalize a timestamp-ish (iso string, pandas Timestamp, datetime) to YYYY-MM-DD.
    Mirrors orchestrator behavior to keep dataset_id stable.
    """
    try:
        t = pd.to_datetime(ts, utc=True, errors="raise")
        return t.strftime("%Y-%m-%d")
    except Exception:
        s = str(ts).strip()
        if "T" in s:
            return s.split("T", 1)[0]
        if " " in s:
            return s.split(" ", 1)[0]
        return s


def _assert_final_dataset_id(dsid: str) -> None:
    if not str(dsid or "").strip():
        raise SystemExit("ERROR: dataset_id is empty after parsing.")
    if _UNKNOWN_SENTINEL in dsid:
        raise SystemExit(
            "ERROR: refusing to use provisional dataset_id containing 'unknown__unknown'. "
            "Provide a FINAL dataset_id or omit --dataset-id to auto-compute it."
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Register/validate a dataset using Dataset Contract v1.1 (datasets_latest.json + datasets.jsonl)."
    )
    ap.add_argument("--path", required=True, help="Path to dataset CSV (normalized columns time,open,high,low,close,...)")

    # dataset identity inputs (semantic)
    ap.add_argument("--instrument", required=True, help="Instrument symbol, e.g., EURUSD")
    ap.add_argument("--timeframe", required=True, help="Timeframe, e.g., M5")
    ap.add_argument("--source", default="csv", help="Data source suffix, e.g., mt5_export (default: csv)")

    # registry controls (match orchestrator)
    ap.add_argument("--registry-dir", default="data/registry", help="Registry directory (default: data/registry)")
    ap.add_argument(
        "--allow-override",
        action="store_true",
        help="Allow updating an existing dataset_id with a new fingerprint (writes UPDATE_FINGERPRINT_ALLOWED event).",
    )
    ap.add_argument(
        "--override-reason",
        default="",
        help="Required if --allow-override. Human explanation for audit trail.",
    )
    ap.add_argument(
        "--append-match-event",
        action="store_true",
        help="Also append MATCH events to datasets.jsonl when fingerprint matches.",
    )

    # optional: force dataset_id (advanced, must be FINAL; NEVER provisional)
    ap.add_argument(
        "--dataset-id",
        default="",
        help="Optional explicit dataset_id to register (must be FINAL; if omitted, computed from start/end dates).",
    )

    args = ap.parse_args()

    data_path = Path(args.path)
    if not data_path.exists():
        raise SystemExit(f"ERROR: data file not found: {data_path}")

    if args.allow_override and not str(args.override_reason or "").strip():
        raise SystemExit(
            "ERROR: --allow-override requires --override-reason.\n"
            "Example: --allow-override --override-reason \"Re-export MT5 with corrected DST parsing\""
        )

    symbol = str(args.instrument).strip().upper()
    timeframe = str(args.timeframe).strip().replace(" ", "").upper()
    source = str(args.source).strip()

    # Provisional id only for fingerprint computation (never registered)
    dataset_id_prov = build_dataset_id(
        instrument=symbol,
        timeframe=timeframe,
        start_ts="unknown",
        end_ts="unknown",
        source=source,
    )

    # Load + compute fingerprint/meta (includes file sha + bytes)
    df, meta = load_bars_csv(str(data_path), return_fingerprint=True, dataset_id=dataset_id_prov)

    # Build stable FINAL dataset_id from meta.start_ts/end_ts
    start_ymd = _ts_to_ymd(meta.start_ts)
    end_ymd = _ts_to_ymd(meta.end_ts)

    dataset_id_default = build_dataset_id(
        instrument=symbol,
        timeframe=timeframe,
        start_ts=start_ymd,
        end_ts=end_ymd,
        source=source,
    )

    forced = str(args.dataset_id or "").strip()
    dataset_id_final = forced if forced else dataset_id_default
    _assert_final_dataset_id(dataset_id_final)

    # Stamp final dataset_id into metadata before registry call
    meta_final = replace(meta, dataset_id=dataset_id_final)

    is_new, msg = register_or_validate_dataset(
        meta_final,
        registry_dir=str(args.registry_dir),
        allow_new_fingerprint=bool(args.allow_override),
        override_reason=str(args.override_reason or ""),
        append_match_event=bool(args.append_match_event),
    )

    print("OK")
    print(f"dataset_id:       {meta_final.dataset_id}")
    print(f"fingerprint_sha:  {meta_final.fingerprint_sha256}")
    print(f"schema_sha:       {meta_final.schema_sha256}")
    print(f"fp8:              {meta_final.fingerprint_short}")
    print(f"rows:             {meta_final.rows}")
    print(f"start_ts_utc:      {meta_final.start_ts}")
    print(f"end_ts_utc:        {meta_final.end_ts}")
    print(f"file_sha256:       {meta_final.file_sha256}")
    print(f"file_bytes:        {meta_final.file_bytes}")
    print(f"source_path:       {meta_final.source_path}")
    print(f"registry_dir:      {args.registry_dir}")
    print(f"is_new:            {is_new}")
    print(f"message:           {msg}")


if __name__ == "__main__":
    main()
