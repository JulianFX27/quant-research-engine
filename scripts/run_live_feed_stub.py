from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from src.live.state_store import load_state, save_state


FEATURE_FIELDS = [
    "time",
    "open","high","low","close",
    "tick_volume",
    "atr_14","shock_log_ret","shock_z",
    "daily_open","london_open","ny_open",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_rows(csv_path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> int:
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    file_exists = p.exists() and p.stat().st_size > 0
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()
    return len(rows)


def fetch_new_feature_rows(symbol: str, since_time_utc_iso: str | None) -> List[Dict[str, str]]:
    """
    TODO: Connect to MT5 and compute features for CLOSED M5 bars.
    Must return rows with:
      - time (UTC ISO Z)
      - open/high/low/close
      - atr_14, shock_log_ret, shock_z
      - daily_open/london_open/ny_open
    """
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--poll_seconds", type=float, default=5.0)
    args = ap.parse_args()

    st = load_state(args.state)
    last_time = st.get("last_time")

    print(f"[{utc_now_iso()}] live_feed start symbol={args.symbol} out={args.out_csv}")

    while True:
        rows = fetch_new_feature_rows(args.symbol, last_time)
        if rows:
            n = append_rows(args.out_csv, rows, FEATURE_FIELDS)
            last_time = rows[-1]["time"]
            st["last_time"] = last_time
            st["updated_at_utc"] = utc_now_iso()
            save_state(args.state, st)
            print(f"[{utc_now_iso()}] appended={n} last_time={last_time}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()