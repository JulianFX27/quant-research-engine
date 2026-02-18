import pandas as pd, numpy as np
from pathlib import Path

rid = r"20260217_194934_071784_f40f3ffb"
p = Path("results/runs")/rid/"trades.csv"

df = pd.read_csv(p, parse_dates=["entry_time","exit_time"])
df = df.sort_values("entry_time").reset_index(drop=True)

start = df["entry_time"].min().normalize()
end = df["entry_time"].max().normalize()

def dd_and_streak(r):
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0.0
    streak=0; max_streak=0
    for x in r:
        if x < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_dd, int(max_streak)

rows=[]
cur = start
while cur <= end:
    w_end = cur + pd.DateOffset(months=12)
    w = df[(df["entry_time"]>=cur) & (df["entry_time"]<w_end)]
    if len(w) >= 30:
        r = w["R"].to_numpy(float)
        max_dd, max_streak = dd_and_streak(r)
        rows.append({
            "window_start": str(cur.date()),
            "window_end": str((w_end - pd.Timedelta(seconds=1)).date()),
            "n_trades": int(len(w)),
            "expectancy_R": float(np.mean(r)),
            "total_R": float(np.sum(r)),
            "max_dd_R": max_dd,
            "max_losing_streak": max_streak,
        })
    cur = cur + pd.DateOffset(months=1)

out = pd.DataFrame(rows)
out_path = Path("results/runs")/rid/"roll12m.csv"
out.to_csv(out_path, index=False)

print("WROTE:", out_path)
print("N_WINDOWS:", len(out))

if len(out)==0:
    print("NO WINDOWS PRODUCED")
else:
    print("\nROLL12M SUMMARY")
    print(out[["expectancy_R","max_dd_R","n_trades"]].describe(percentiles=[.05,.25,.5,.75,.95]).to_string())
