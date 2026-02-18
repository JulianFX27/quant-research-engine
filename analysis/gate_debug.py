import sys
from pathlib import Path
import pandas as pd
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python analysis/gate_debug.py <RID>")
    raise SystemExit(1)

RID = sys.argv[1]
lo, hi = -3.067, 2.664

trades_path = Path("results/runs") / RID / "trades.csv"
feats_path  = Path("data/anchor_reversion_fx/data/eurusd_m5_features.csv")

if not trades_path.exists():
    raise FileNotFoundError(trades_path)
if not feats_path.exists():
    raise FileNotFoundError(feats_path)

t = pd.read_csv(trades_path, parse_dates=["entry_time"])
f = pd.read_csv(feats_path,  parse_dates=["time"])

t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce").dt.tz_convert(None)
f["time"]       = pd.to_datetime(f["time"],       utc=True, errors="coerce").dt.tz_convert(None)

cols = ["time","spread_ny_open_atr","shock_z","atr_14","hour","dow"]
cols = [c for c in cols if c in f.columns]

m = t.merge(f[cols], left_on="entry_time", right_on="time", how="left")

x = pd.to_numeric(m["spread_ny_open_atr"], errors="coerce")
keep  = (x <= lo) | (x > hi)
block = (x > lo) & (x <= hi)

def dd_streak(r):
    r = np.asarray(r, float)
    if len(r) == 0:
        return 0.0, 0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0.0
    s = mx = 0
    for v in r:
        if v < 0:
            s += 1
            mx = max(mx, s)
        else:
            s = 0
    return max_dd, mx

def summ(name, mask):
    r = m.loc[mask, "R"].to_numpy(float)
    max_dd, st = dd_streak(r)
    print(name,
          "n", int(len(r)),
          "expR", float(r.mean()) if len(r) else None,
          "sumR", float(r.sum()) if len(r) else None,
          "winR", float((r > 0).mean()) if len(r) else None,
          "maxDD", max_dd,
          "maxLS", int(st))

print("RID:", RID)
print("merged_rows:", len(m))
print("nan_rate_spread_ny_open_atr:", float(m["spread_ny_open_atr"].isna().mean()))
print("block_rate:", float(block.mean()))

qs = [0,0.05,0.1,0.25,0.5,0.75,0.9,0.95,1.0]
print("spread_quantiles:", {p: float(x.quantile(p)) for p in qs})

summ("KEEP", keep)
summ("BLOCK", block)

b = m.loc[block].copy()
for c in ["shock_z","atr_14","hour","dow"]:
    if c in b.columns:
        v = pd.to_numeric(b[c], errors="coerce")
        print("BLOCK", c, "mean", float(v.mean()), "p10", float(v.quantile(0.10)), "p90", float(v.quantile(0.90)))
