import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def qbin(x, k, labels=None):
    x = pd.to_numeric(x, errors="coerce")
    if labels is None:
        labels = [f"Q{i+1}" for i in range(k)]
    try:
        return pd.qcut(x, k, labels=labels, duplicates="drop")
    except Exception:
        return pd.Series(["NA"]*len(x), index=x.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--features-path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pip-size", type=float, default=0.0001)
    ap.add_argument("--time-bucket-min", type=int, default=30)
    ap.add_argument("--shock-bins", type=int, default=4)
    ap.add_argument("--vol-bins", type=int, default=3)
    ap.add_argument("--shock-col", default="shock_z")
    ap.add_argument("--shock-ret-col", default="shock_log_ret")
    ap.add_argument("--atr-col", default="atr_14")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir)
    (outdir / "matrices").mkdir(parents=True, exist_ok=True)
    (outdir / "diagnostics").mkdir(parents=True, exist_ok=True)

    run_id = run_dir.name
    trades = pd.read_csv(run_dir / "trades.csv")
    feats_path = Path(args.features_path)

    header = list(pd.read_csv(feats_path, nrows=1).columns)
    usecols = [c for c in ["time", args.shock_col, args.shock_ret_col, args.atr_col] if c in header]
    if "time" not in usecols:
        raise SystemExit("features must contain 'time' column")

    feats = pd.read_csv(feats_path, usecols=usecols).reset_index(drop=True)
    feats["__idx"] = feats.index.astype(int)

    m = trades.merge(feats, how="left", left_on="entry_idx", right_on="__idx")

    # time bucket
    t = pd.to_datetime(m["time"], utc=True, errors="coerce")
    m["session_bucket"] = ((t.dt.hour*60 + t.dt.minute)//args.time_bucket_min).astype("Int64")

    # shock sign: prefer shock_log_ret sign; fallback shock_z sign
    if args.shock_ret_col in m.columns:
        sret = pd.to_numeric(m[args.shock_ret_col], errors="coerce")
        m["shock_sign"] = np.where(sret>0, "+", np.where(sret<0, "-", "0"))
    elif args.shock_col in m.columns:
        sz = pd.to_numeric(m[args.shock_col], errors="coerce")
        m["shock_sign"] = np.where(sz>0, "+", np.where(sz<0, "-", "0"))
    else:
        m["shock_sign"] = "NA"

    # shock magnitude bins on abs(shock_z)
    if args.shock_col in m.columns:
        sz = pd.to_numeric(m[args.shock_col], errors="coerce").abs()
        m["shock_mag_bin"] = qbin(sz, args.shock_bins)
    else:
        m["shock_mag_bin"] = "NA"

    # vol bucket on atr
    if args.atr_col in m.columns:
        atr = pd.to_numeric(m[args.atr_col], errors="coerce")
        labels = ["VOL_LOW","VOL_MED","VOL_HIGH"] if args.vol_bins==3 else None
        m["vol_bucket"] = qbin(atr, args.vol_bins, labels=labels)
    else:
        m["vol_bucket"] = "NA"

    # exit proxy
    m["anchor_touch"] = (m["exit_reason"].astype(str)=="ANCHOR_TOUCH").astype(float)
    m["time_stop"] = (m["exit_reason"].astype(str)=="TIME_STOP").astype(float)

    # aggregates
    keys = ["shock_sign","shock_mag_bin","vol_bucket","session_bucket","side"]
    r_col = "R" if "R" in m.columns else None

    def profit_factor_R(s):
        s = pd.to_numeric(s, errors="coerce")
        gains = s.clip(lower=0).sum()
        losses = -s.clip(upper=0).sum()
        return (gains / losses) if losses > 0 else np.nan

    g = m.groupby(keys, dropna=False).agg(
        n_trades=("side","size"),
        winrate_R=(r_col, lambda s: (pd.to_numeric(s, errors="coerce")>0).mean()) if r_col else ("side", lambda s: np.nan),
        expectancy_R=(r_col, lambda s: pd.to_numeric(s, errors="coerce").mean()) if r_col else ("side", lambda s: np.nan),
        profit_factor_R=(r_col, profit_factor_R) if r_col else ("side", lambda s: np.nan),
        anchor_touch_rate=("anchor_touch","mean"),
        time_stop_rate=("time_stop","mean"),
    ).reset_index()

    mat_path = outdir / "matrices" / f"som_v3_features__run_{run_id}.csv"
    diag_path = outdir / "diagnostics" / f"diag_v3_features__run_{run_id}.json"
    g.to_csv(mat_path, index=False)

    diag = {
        "run_id": run_id,
        "features_path": str(feats_path),
        "features_cols_used": usecols,
        "keys": keys,
        "shock_col": args.shock_col,
        "shock_ret_col": args.shock_ret_col,
        "atr_col": args.atr_col,
        "shock_bins": args.shock_bins,
        "vol_bins": args.vol_bins,
        "time_bucket_min": args.time_bucket_min,
    }
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)

    print("OK")
    print("MATRIX:", mat_path)
    print("DIAG:", diag_path)

if __name__ == "__main__":
    main()
