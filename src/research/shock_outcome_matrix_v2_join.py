import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--features-path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pip-size", type=float, default=0.0001)
    ap.add_argument("--levels", default="4,6,8,10,12")
    ap.add_argument("--time-bucket-min", type=int, default=30)
    ap.add_argument("--shock-mag-bins", type=int, default=4)  # qcut bins
    ap.add_argument("--cols", default="time,shock_sign,shock_magnitude,vol_bucket,regime,event_flag,state_id")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "matrices").mkdir(parents=True, exist_ok=True)
    (outdir / "diagnostics").mkdir(parents=True, exist_ok=True)
    (outdir / "artifacts").mkdir(parents=True, exist_ok=True)

    run_id = run_dir.name

    trades_path = run_dir / "trades.csv"
    if not trades_path.exists():
        raise SystemExit(f"Missing trades.csv: {trades_path}")
    trades = pd.read_csv(trades_path)

    # Required for join
    if "entry_idx" not in trades.columns:
        raise SystemExit("trades.csv missing entry_idx (required for join).")

    # Read features columns (best-effort)
    feats_path = Path(args.features_path)
    if not feats_path.exists():
        raise SystemExit(f"Missing features file: {feats_path}")

    requested = [c.strip() for c in args.cols.split(",") if c.strip()]
    header = list(pd.read_csv(feats_path, nrows=1).columns)
    cols = [c for c in requested if c in header]

    # Ensure we can timestamp-bucket
    time_col = None
    for c in ["time","timestamp","datetime","time_utc","date_time"]:
        if c in header:
            time_col = c
            if c not in cols:
                cols.append(c)
            break

    if len(cols) == 0:
        raise SystemExit("No requested columns found in features file. Adjust --cols.")

    feats = pd.read_csv(feats_path, usecols=cols)

    # Join on entry_idx
    feats = feats.reset_index(drop=True)  # make sure index is 0..N-1
    feats["__idx"] = feats.index.astype(int)

    merged = trades.merge(feats, how="left", left_on="entry_idx", right_on="__idx")

    # Derived: session bucket
    if time_col and time_col in merged.columns:
        t = pd.to_datetime(merged[time_col], utc=True, errors="coerce")
        bucket = (t.dt.hour*60 + t.dt.minute)//args.time_bucket_min
        merged["session_bucket"] = bucket.astype("Int64")
    else:
        merged["session_bucket"] = pd.Series([pd.NA]*len(merged), dtype="Int64")

    # MFE/MAE: not in trades -> cannot compute without intrabar. Keep NA.
    # Reach levels: since we don't have MFE, reach_* cannot be computed. We'll output NA.
    # However: for Anchor MR Pure 8p, we can approximate reach_8p by exit_reason == ANCHOR_TOUCH (if that's equivalent to hitting threshold).
    # We'll do both: reach_* as NA and proxy_reach_8p as exit_reason==ANCHOR_TOUCH.

    merged["proxy_reach_8p"] = (merged.get("exit_reason", "").astype(str) == "ANCHOR_TOUCH").astype(float)

    # Shock magnitude bins
    if "shock_magnitude" in merged.columns:
        sm = pd.to_numeric(merged["shock_magnitude"], errors="coerce")
        if sm.notna().any():
            try:
                merged["shock_mag_bin"] = pd.qcut(sm, args.shock_mag_bins, labels=[f"Q{i+1}" for i in range(args.shock_mag_bins)])
            except Exception:
                merged["shock_mag_bin"] = "NA"
        else:
            merged["shock_mag_bin"] = "NA"
    else:
        merged["shock_mag_bin"] = "NA"

    # Keys
    keys = []
    for k in ["shock_sign","shock_mag_bin","vol_bucket","regime","event_flag","session_bucket","side"]:
        if k in merged.columns:
            keys.append(k)
    if not keys:
        keys = ["side"]

    # Aggregate
    r_col = "R" if "R" in merged.columns else None
    agg = {
        "n_trades": ("side","size"),
        "winrate_R": (r_col, lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean()) if r_col else ("side", lambda s: pd.NA),
        "expectancy_R": (r_col, lambda s: pd.to_numeric(s, errors="coerce").mean()) if r_col else ("side", lambda s: pd.NA),
        "profit_factor_R": (r_col, lambda s: (
            pd.to_numeric(s, errors="coerce").clip(lower=0).sum() /
            (-pd.to_numeric(s, errors="coerce").clip(upper=0).sum())
        ) if (-pd.to_numeric(s, errors="coerce").clip(upper=0).sum()) > 0 else pd.NA) if r_col else ("side", lambda s: pd.NA),
        "proxy_reach_8p": ("proxy_reach_8p","mean"),
        "time_stop_rate": ("exit_reason", lambda s: (s.astype(str)=="TIME_STOP").mean()) if "exit_reason" in merged.columns else ("side", lambda s: pd.NA),
        "anchor_touch_rate": ("exit_reason", lambda s: (s.astype(str)=="ANCHOR_TOUCH").mean()) if "exit_reason" in merged.columns else ("side", lambda s: pd.NA),
    }

    g = merged.groupby(keys, dropna=False).agg(**agg).reset_index()

    mat_path = outdir / "matrices" / f"som_v2_join__run_{run_id}.csv"
    diag_path = outdir / "diagnostics" / f"diag_v2_join__run_{run_id}.json"

    g.to_csv(mat_path, index=False)

    diag = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "features_path": str(feats_path),
        "trades_rows": int(len(trades)),
        "features_rows": int(len(feats)),
        "joined_null_rate_any": float(merged[keys].isna().any(axis=1).mean()) if keys else None,
        "keys": keys,
        "time_col_used": time_col,
        "cols_requested": requested,
        "cols_found_in_features": cols,
        "notes": [
            "This v2 is shock-aware via join on entry_idx -> features row index.",
            "MFE/MAE not available in trades.csv; reach levels cannot be computed without intrabar path reconstruction.",
            "proxy_reach_8p uses exit_reason==ANCHOR_TOUCH as a strategy-specific proxy."
        ]
    }
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)

    print("OK")
    print("MATRIX:", mat_path)
    print("DIAG:", diag_path)

if __name__ == "__main__":
    main()
