# analysis/roll12m.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def dd_and_streak(r: np.ndarray) -> tuple[float, int]:
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0.0

    streak = 0
    max_streak = 0
    for x in r:
        if x < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return max_dd, int(max_streak)


def compute_roll12m(df: pd.DataFrame, min_trades: int = 30) -> pd.DataFrame:
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_time", "R"]).sort_values("entry_time").reset_index(drop=True)

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "window_start", "window_end", "n_trades",
            "expectancy_R", "total_R", "max_dd_R", "max_losing_streak"
        ])

    start = df["entry_time"].min().normalize()
    end = df["entry_time"].max().normalize()

    rows = []
    cur = start

    while cur <= end:
        w_end = cur + pd.DateOffset(months=12)
        w = df[(df["entry_time"] >= cur) & (df["entry_time"] < w_end)]

        if len(w) >= min_trades:
            r = w["R"].to_numpy(float)
            max_dd, max_streak = dd_and_streak(r)
            rows.append({
                "window_start": cur.date().isoformat(),
                "window_end": (w_end - pd.Timedelta(seconds=1)).date().isoformat(),
                "n_trades": int(len(w)),
                "expectancy_R": float(np.mean(r)),
                "total_R": float(np.sum(r)),
                "max_dd_R": float(max_dd),
                "max_losing_streak": int(max_streak),
            })

        cur = cur + pd.DateOffset(months=1)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("RUN_ID", nargs="?", help="Run id under results/runs/<RUN_ID> (optional if --trades_csv is given)")
    ap.add_argument("--trades_csv", default=None, help="Path to trades csv (e.g., results/runs/<RID>/trades.csv or trades_gated.csv)")
    ap.add_argument("--out_path", default=None, help="Where to write roll12m.csv (default: next to trades_csv or under run dir)")
    ap.add_argument("--min_trades", type=int, default=30)
    args = ap.parse_args()

    trades_path: Path
    if args.trades_csv:
        trades_path = Path(args.trades_csv)
    else:
        if not args.RUN_ID:
            raise SystemExit("Usage: python analysis/roll12m.py <RUN_ID> OR --trades_csv <path>")
        run_dir = Path("results/runs") / args.RUN_ID
        trades_path = run_dir / "trades.csv"

    if not trades_path.exists():
        raise FileNotFoundError(trades_path)

    df = pd.read_csv(trades_path, parse_dates=["entry_time"])
    out = compute_roll12m(df, min_trades=args.min_trades)

    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = trades_path.parent / "roll12m.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("WROTE:", out_path)
    print("N_WINDOWS:", len(out))

    if len(out) > 0:
        print("\nROLL12M SUMMARY")
        print(
            out[["expectancy_R", "max_dd_R", "n_trades"]]
            .describe(percentiles=[.05, .25, .5, .75, .95])
            .to_string()
        )


if __name__ == "__main__":
    main()
