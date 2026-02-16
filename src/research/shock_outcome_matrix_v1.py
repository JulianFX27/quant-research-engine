import argparse, json
from pathlib import Path
import pandas as pd

def infer_side(df: pd.DataFrame) -> pd.Series:
    # intenta inferir lado si existe alguna columna típica
    for c in ["side", "direction", "trade_side", "pos_side"]:
        if c in df.columns:
            s = df[c].astype(str).str.upper()
            s = s.replace({"BUY":"LONG","SELL":"SHORT"})
            return s
    return pd.Series(["NA"]*len(df), index=df.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pip-size", type=float, default=0.0001)
    ap.add_argument("--levels", default="4,6,8,10,12")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trades_path = run_dir / "trades.csv"
    if not trades_path.exists():
        raise SystemExit(f"Missing trades.csv at {trades_path}")

    df = pd.read_csv(trades_path)

    # --- columnas esperadas (best-effort) ---
    # PnL en R si existe
    r_col = None
    for c in ["R", "pnl_R", "r_multiple", "r", "R_multiple"]:
        if c in df.columns:
            r_col = c
            break

    # MFE/MAE en pips si existe; si está en precio, convertimos
    mfe_pips = None
    mae_pips = None

    # candidatos comunes
    mfe_candidates = ["mfe_pips", "MFE_pips", "mfe"]
    mae_candidates = ["mae_pips", "MAE_pips", "mae"]

    def pick(colnames):
        for c in colnames:
            if c in df.columns:
                return c
        return None

    mfe_c = pick(mfe_candidates)
    mae_c = pick(mae_candidates)

    # si mfe/mae están en precio (poco probable), intenta convertir
    if mfe_c is not None:
        mfe_pips = df[mfe_c].astype(float)
        # heurística: si valores son muy pequeños (<0.2) podría ser precio
        if mfe_pips.abs().median() < 0.2:
            mfe_pips = mfe_pips / args.pip_size
    else:
        mfe_pips = pd.Series([pd.NA]*len(df))

    if mae_c is not None:
        mae_pips = df[mae_c].astype(float)
        if mae_pips.abs().median() < 0.2:
            mae_pips = mae_pips / args.pip_size
    else:
        mae_pips = pd.Series([pd.NA]*len(df))

    # shock fields si existen
    shock_sign = df["shock_sign"] if "shock_sign" in df.columns else pd.Series(["NA"]*len(df))
    shock_mag = df["shock_magnitude"] if "shock_magnitude" in df.columns else pd.Series([pd.NA]*len(df))
    vol_bucket = df["vol_bucket"] if "vol_bucket" in df.columns else pd.Series(["NA"]*len(df))

    # time bucket derivable
    tcol = None
    for c in ["entry_time_utc","entry_time","entry_ts","entry_datetime"]:
        if c in df.columns:
            tcol = c
            break
    if tcol:
        t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        # bucket de 30 minutos
        session_bucket = (t.dt.hour*60 + t.dt.minute)//30
        session_bucket = session_bucket.astype("Int64")
    else:
        session_bucket = pd.Series([pd.NA]*len(df), dtype="Int64")

    side = infer_side(df)

    # bins shock magnitude (quartiles si hay datos)
    if shock_mag.notna().any():
        sm = pd.to_numeric(shock_mag, errors="coerce")
        try:
            bins = pd.qcut(sm, 4, labels=["Q1","Q2","Q3","Q4"])
        except Exception:
            bins = pd.Series(["NA"]*len(df))
    else:
        bins = pd.Series(["NA"]*len(df))

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    reach = {f"reach_{L}p": (pd.to_numeric(mfe_pips, errors="coerce") >= L).astype("float") for L in levels}

    # métricas base
    out = df.copy()
    out["mfe_pips__derived"] = pd.to_numeric(mfe_pips, errors="coerce")
    out["mae_pips__derived"] = pd.to_numeric(mae_pips, errors="coerce")
    out["shock_sign__used"] = shock_sign
    out["shock_mag_bin__used"] = bins
    out["vol_bucket__used"] = vol_bucket
    out["session_bucket_30m__used"] = session_bucket
    out["side__used"] = side
    for k,v in reach.items():
        out[k] = v

    keys = ["shock_sign__used","shock_mag_bin__used","vol_bucket__used","session_bucket_30m__used","side__used"]

    agg = {
        "n_trades": ("side__used","size"),
        "winrate_R": (r_col, lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean()) if r_col else ("side__used", lambda s: pd.NA),
        "expectancy_R": (r_col, lambda s: pd.to_numeric(s, errors="coerce").mean()) if r_col else ("side__used", lambda s: pd.NA),
        "avg_MFE_pips": ("mfe_pips__derived","mean"),
        "p50_MFE_pips": ("mfe_pips__derived","median"),
        "avg_MAE_pips": ("mae_pips__derived","mean"),
    }
    for L in levels:
        agg[f"reach_{L}p"] = (f"reach_{L}p","mean")

    g = out.groupby(keys, dropna=False).agg(**agg).reset_index()

    # persistencia
    run_id = run_dir.name
    mat_path = outdir / "matrices" / f"som_v1__run_{run_id}.csv"
    diag_path = outdir / "diagnostics" / f"diag_v1__run_{run_id}.json"
    reach_path = outdir / "artifacts" / f"reach_levels_v1__run_{run_id}.csv"

    g.to_csv(mat_path, index=False)
    g[["reach_4p","reach_6p","reach_8p","reach_10p","reach_12p"] if all([f"reach_{L}p" in g.columns for L in levels]) else [c for c in g.columns if c.startswith("reach_")]] \
        .describe(include="all").to_csv(reach_path)

    diag = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "trades_rows": int(len(df)),
        "columns_present": list(df.columns),
        "R_column_used": r_col,
        "mfe_source_col": mfe_c,
        "mae_source_col": mae_c,
        "pip_size": args.pip_size,
        "keys": keys,
        "levels_pips": levels,
        "notes": [
            "Matrix is best-effort based on columns present in trades.csv.",
            "If shock_* or vol_bucket missing, they will be 'NA' and segmentation collapses.",
        ],
    }
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)

    print("OK")
    print("MATRIX:", mat_path)
    print("DIAG:", diag_path)
    print("REACH:", reach_path)

if __name__ == "__main__":
    main()
