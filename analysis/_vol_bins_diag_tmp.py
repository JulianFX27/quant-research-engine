import pandas as pd
import numpy as np
from pathlib import Path

rid = r"20260217_194934_071784_f40f3ffb"
trades_path = Path("results/runs")/rid/"trades.csv"
features_path = Path("data/anchor_reversion_fx/data/eurusd_m5_features.csv")

trades = pd.read_csv(trades_path, parse_dates=["entry_time"])
features = pd.read_csv(features_path, parse_dates=["time"])

# Fix TZ mismatch -> UTC naive
trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(None)
features["time"] = pd.to_datetime(features["time"], utc=True).dt.tz_convert(None)

df = trades.merge(features, left_on="entry_time", right_on="time", how="left")

print("\nMERGE CHECK")
print("trades rows:", len(trades), "| features rows:", len(features), "| merged rows:", len(df))
print("merged NaN rate on close:", float(df["close"].isna().mean()) if "close" in df.columns else "N/A")

# Candidate "volatility / risk" columns available in your dataset
candidates = [
    "atr_14",
    "abs_log_ret",
    "abs_log_ret_roll",
    "shock_z",
    "soec_z",
    "spread_daily_open_atr",
    "spread_london_open_atr",
    "spread_ny_open_atr",
    "tick_volume",  # keep but treat as activity proxy
]

present = [c for c in candidates if c in df.columns]
print("\nVOL/RISK CANDIDATES PRESENT:", present)

def max_dd_from_r(r: np.ndarray) -> float:
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(dd.max())

def summarize_by_bucket(tmp: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    g = tmp.groupby(bucket_col)["R"].agg(
        n_trades="count",
        expectancy_R="mean",
        total_R="sum",
        winrate=lambda x: float((x > 0).mean()),
    ).reset_index()

    # max_dd needs ordered-by-time within bucket
    # (still approximate, but better than nothing; for true DD you want rolling windows, which we already did)
    dd_rows = []
    for b, w in tmp.sort_values("entry_time").groupby(bucket_col):
        r = w["R"].to_numpy(float)
        dd_rows.append((b, max_dd_from_r(r)))
    dd = pd.DataFrame(dd_rows, columns=[bucket_col, "max_dd_R"])
    out = g.merge(dd, on=bucket_col, how="left")
    return out

def qbucket(s: pd.Series, q: int = 5) -> pd.Series:
    # robust quantile binning; duplicates='drop' handles constant-ish series
    return pd.qcut(s, q=q, duplicates="drop")

MIN_TRADES_PER_BUCKET = 25
Q = 5

for col in present:
    sub = df[["entry_time", "R", col]].dropna()
    if len(sub) < MIN_TRADES_PER_BUCKET * 2:
        print(f"\n[{col}] skipped: too few non-NaN rows ({len(sub)})")
        continue

    # Build quantile buckets
    try:
        sub["bucket"] = qbucket(sub[col], q=Q)
    except Exception as e:
        print(f"\n[{col}] bucket error: {e}")
        continue

    # Drop tiny buckets
    counts = sub["bucket"].value_counts()
    keep = counts[counts >= MIN_TRADES_PER_BUCKET].index
    sub = sub[sub["bucket"].isin(keep)].copy()

    if sub["bucket"].nunique() < 2:
        print(f"\n[{col}] skipped: not enough buckets after min-trade filter")
        continue

    out = summarize_by_bucket(sub, "bucket").sort_values("expectancy_R", ascending=False)

    print(f"\n=== BY QUANTILE BUCKET: {col} (Q={Q}, min_bucket_trades={MIN_TRADES_PER_BUCKET}) ===")
    print(out.to_string(index=False))

print("\nDONE.")
