from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def qbucket(series: pd.Series, n: int = 5) -> pd.Series:
    valid = series.dropna()
    if valid.nunique() < n:
        return pd.Series(index=series.index, dtype="object")
    out = pd.qcut(valid, q=n, labels=[f"Q{i}" for i in range(1, n + 1)], duplicates="drop")
    full = pd.Series(index=series.index, dtype="object")
    full.loc[valid.index] = out.astype(str)
    return full


def summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_col):
        r = sub["net_ret"]
        rows.append(
            {
                group_col: key,
                "n_trades": len(sub),
                "mean_return": r.mean(),
                "median_return": r.median(),
                "std_return": r.std(),
                "winrate": (r > 0).mean(),
                "sharpe_per_trade": (r.mean() / r.std()) if r.std() and r.std() > 0 else np.nan,
                "final_equity_ret": r.sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe_per_trade", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    trades_path = Path(args.trades)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(trades_path)
    if df.empty:
        raise ValueError("Trades file is empty")

    # tipos
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"])
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])

    # features de régimen
    # ya deben venir del strategy runner actual
    required = [
        "overnight_ret_to_ny_open",
        "ret_30m",
        "range_30m",
        "impulse_efficiency",
        "net_ret",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in trades.csv: {missing}")

    df["overnight_abs"] = df["overnight_ret_to_ny_open"].abs()
    df["ret30_abs"] = df["ret_30m"].abs()

    df["overnight_bucket"] = qbucket(df["overnight_abs"], n=5)
    df["ret30_bucket"] = qbucket(df["ret30_abs"], n=5)
    df["range30_bucket"] = qbucket(df["range_30m"], n=5)
    df["eff_bucket"] = qbucket(df["impulse_efficiency"], n=5)

    overnight_dir = np.sign(df["overnight_ret_to_ny_open"]).replace(0, np.nan)
    impulse_dir = np.sign(df["ret_30m"]).replace(0, np.nan)

    df["overnight_vs_impulse"] = np.where(
        overnight_dir == impulse_dir,
        "aligned",
        "opposed",
    )

    # tablas
    overnight_tbl = summarize(df.dropna(subset=["overnight_bucket"]), "overnight_bucket")
    ret30_tbl = summarize(df.dropna(subset=["ret30_bucket"]), "ret30_bucket")
    range30_tbl = summarize(df.dropna(subset=["range30_bucket"]), "range30_bucket")
    eff_tbl = summarize(df.dropna(subset=["eff_bucket"]), "eff_bucket")
    align_tbl = summarize(df.dropna(subset=["overnight_vs_impulse"]), "overnight_vs_impulse")

    overnight_tbl.to_csv(out_dir / "overnight_bucket_summary.csv", index=False)
    ret30_tbl.to_csv(out_dir / "ret30_bucket_summary.csv", index=False)
    range30_tbl.to_csv(out_dir / "range30_bucket_summary.csv", index=False)
    eff_tbl.to_csv(out_dir / "eff_bucket_summary.csv", index=False)
    align_tbl.to_csv(out_dir / "overnight_vs_impulse_summary.csv", index=False)

    meta = {
        "n_trades": int(len(df)),
        "columns": list(df.columns),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== REGIME ANALYSIS — OVERNIGHT BUCKET ===\n")
    print(overnight_tbl.to_string(index=False))

    print("\n=== REGIME ANALYSIS — RET30 BUCKET ===\n")
    print(ret30_tbl.to_string(index=False))

    print("\n=== REGIME ANALYSIS — RANGE30 BUCKET ===\n")
    print(range30_tbl.to_string(index=False))

    print("\n=== REGIME ANALYSIS — EFFICIENCY BUCKET ===\n")
    print(eff_tbl.to_string(index=False))

    print("\n=== REGIME ANALYSIS — OVERNIGHT VS IMPULSE ===\n")
    print(align_tbl.to_string(index=False))

    print(f"\nSaved outputs in: {out_dir}")


if __name__ == "__main__":
    main()
    