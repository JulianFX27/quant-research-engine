import pandas as pd
import numpy as np
from pathlib import Path

rid = r"20260217_194934_071784_f40f3ffb"

trades_path = Path("results/runs")/rid/"trades.csv"
features_path = Path("data/anchor_reversion_fx/data/eurusd_m5_features.csv")

trades = pd.read_csv(trades_path, parse_dates=["entry_time"])
features = pd.read_csv(features_path, parse_dates=["time"])

# --- FIX TIMEZONE MISMATCH ---
trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(None)
features["time"] = pd.to_datetime(features["time"], utc=True).dt.tz_convert(None)

df = trades.merge(features, left_on="entry_time", right_on="time", how="left")

print("\nCOLUMNS AVAILABLE:")
print(df.columns.tolist())

vol_cols = [c for c in df.columns if "vol" in c.lower()]
print("\nVOL CANDIDATE COLUMNS:", vol_cols)

if vol_cols:
    col = vol_cols[0]
    print(f"\nUSING COLUMN: {col}")

    def max_dd(x):
        eq = np.cumsum(x)
        peak = np.maximum.accumulate(eq)
        return float((peak - eq).max()) if len(eq) else 0.0

    g = df.groupby(col)["R"].agg(
        n_trades="count",
        expectancy_R="mean",
        total_R="sum",
        winrate=lambda x: (x>0).mean(),
        max_dd_R=max_dd
    ).reset_index()

    print("\nRESULT BY VOL BUCKET:")
    print(g.sort_values("expectancy_R", ascending=False).to_string(index=False))
else:
    print("\n?? No volatility column detected automatically.")
