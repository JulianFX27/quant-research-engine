from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


NY_TZ = "America/New_York"


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "run_manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"run_manifest.json not found in {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_trades(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "trades.csv"
    if not p.exists():
        raise FileNotFoundError(f"trades.csv not found in {run_dir}")
    df = pd.read_csv(p)
    if df.empty:
        return df

    if "entry_time" not in df.columns:
        raise ValueError("trades.csv missing required column: entry_time")

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df.get("exit_time"), utc=True, errors="coerce")

    bad = df["entry_time"].isna().sum()
    if bad > 0:
        raise ValueError(f"trades.csv has {bad} rows with invalid entry_time")

    if "R" in df.columns:
        df["R"] = pd.to_numeric(df["R"], errors="coerce")
    else:
        df["R"] = pd.NA

    if "pnl" in df.columns:
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    else:
        df["pnl"] = pd.NA

    if "hold_minutes" in df.columns:
        df["hold_minutes"] = pd.to_numeric(df["hold_minutes"], errors="coerce")
    else:
        df["hold_minutes"] = pd.NA

    if "exit_reason" not in df.columns:
        df["exit_reason"] = ""

    return df


def _load_features_from_manifest(run_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    data_path = manifest.get("data_path")
    if not data_path:
        raise ValueError("run_manifest.json missing data_path")

    p = Path(data_path)
    if not p.is_absolute():
        p = Path.cwd() / p

    if not p.exists():
        raise FileNotFoundError(f"features/data file not found: {p}")

    cols = ["time", "atr_14"]
    df = pd.read_csv(p, usecols=cols)

    if "time" not in df.columns or "atr_14" not in df.columns:
        raise ValueError("features csv must contain time and atr_14")

    df["entry_time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["atr_14"] = pd.to_numeric(df["atr_14"], errors="coerce")
    df = df.drop(columns=["time"])

    bad = df["entry_time"].isna().sum()
    if bad > 0:
        df = df[df["entry_time"].notna()].copy()

    df = df.sort_values("entry_time").drop_duplicates(subset=["entry_time"], keep="last").reset_index(drop=True)
    return df


def _compute_basic_stats(g: pd.DataFrame) -> dict[str, Any]:
    n = int(len(g))
    r = pd.to_numeric(g["R"], errors="coerce").dropna()
    pnl = pd.to_numeric(g["pnl"], errors="coerce").dropna()
    hold = pd.to_numeric(g["hold_minutes"], errors="coerce").dropna()

    wins_r = r[r > 0]
    losses_r = r[r < 0]

    wins_pnl = pnl[pnl > 0]
    losses_pnl = pnl[pnl < 0]

    expectancy_r = float(r.mean()) if len(r) > 0 else None
    winrate_r = float((r > 0).mean()) if len(r) > 0 else None
    avg_win_r = float(wins_r.mean()) if len(wins_r) > 0 else None
    avg_loss_r = float(losses_r.mean()) if len(losses_r) > 0 else None

    gross_profit = float(wins_pnl.sum()) if len(wins_pnl) > 0 else 0.0
    gross_loss_abs = float(abs(losses_pnl.sum())) if len(losses_pnl) > 0 else 0.0
    if gross_loss_abs > 0:
        pf_pnl = gross_profit / gross_loss_abs
    elif gross_profit > 0:
        pf_pnl = float("inf")
    else:
        pf_pnl = None

    forced_ratio = float((g["exit_reason"].astype(str).str.startswith("FORCE_")).mean()) if n > 0 else None
    force_max_hold_ratio = float((g["exit_reason"].astype(str) == "FORCE_MAX_HOLD").mean()) if n > 0 else None

    max_consec_losses_r = 0
    cur = 0
    for x in r.tolist():
        if x < 0:
            cur += 1
            max_consec_losses_r = max(max_consec_losses_r, cur)
        else:
            cur = 0

    return {
        "n_trades": n,
        "expectancy_R": expectancy_r,
        "winrate_R": winrate_r,
        "avg_win_R": avg_win_r,
        "avg_loss_R": avg_loss_r,
        "profit_factor_pnl": pf_pnl,
        "total_pnl": float(pnl.sum()) if len(pnl) > 0 else 0.0,
        "avg_hold_minutes": float(hold.mean()) if len(hold) > 0 else None,
        "median_hold_minutes": float(hold.median()) if len(hold) > 0 else None,
        "forced_exit_ratio": forced_ratio,
        "force_max_hold_ratio": force_max_hold_ratio,
        "max_consecutive_losses_R": int(max_consec_losses_r),
    }


def _make_atr_quantiles(df: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    out = df.copy()
    valid = out["atr_14"].notna()

    if valid.sum() == 0:
        out["atr_quantile"] = pd.NA
        out["atr_quantile_label"] = pd.NA
        return out

    ranks = out.loc[valid, "atr_14"].rank(method="first")
    out.loc[valid, "atr_quantile"] = pd.qcut(ranks, q=q, labels=False, duplicates="drop") + 1

    out["atr_quantile"] = out["atr_quantile"].astype("Int64")
    out["atr_quantile_label"] = out["atr_quantile"].map(
        {
            1: "Q1_LOWEST_VOL",
            2: "Q2_LOW_VOL",
            3: "Q3_MID_VOL",
            4: "Q4_HIGH_VOL",
            5: "Q5_HIGHEST_VOL",
        }
    )
    return out


def _make_atr_fixed_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s = out["atr_14"]

    def bucket(v: Any) -> str | None:
        x = _safe_float(v)
        if x is None:
            return None
        if x <= 0.00022:
            return "VOL_LOW"
        if x <= 0.00030:
            return "VOL_MED"
        return "VOL_HIGH"

    out["atr_bucket"] = s.map(bucket)
    return out


def _group_table(df: pd.DataFrame, group_col: str, extra_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for key, g in df.groupby(group_col, dropna=True):
        stats = _compute_basic_stats(g)
        row = {group_col: key}
        for c in extra_cols:
            if c in g.columns:
                vals = g[c].dropna().unique().tolist()
                row[c] = vals[0] if len(vals) == 1 else vals[0] if vals else None
        row.update(stats)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if "n_trades" in out.columns:
        out = out.sort_values([group_col]).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build volatility adversity maps from a run directory.")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    manifest = _load_manifest(run_dir)
    trades = _load_trades(run_dir)
    if trades.empty:
        raise ValueError("trades.csv is empty")

    feats = _load_features_from_manifest(run_dir, manifest)

    merged = trades.merge(feats, on="entry_time", how="left", validate="many_to_one")

    merged["entry_time_ny"] = merged["entry_time"].dt.tz_convert(NY_TZ)
    merged["entry_hour_ny"] = merged["entry_time_ny"].dt.hour
    merged["entry_weekday_ny"] = merged["entry_time_ny"].dt.weekday
    merged["entry_weekday_label_ny"] = merged["entry_weekday_ny"].map(
        {
            0: "Mon",
            1: "Tue",
            2: "Wed",
            3: "Thu",
            4: "Fri",
            5: "Sat",
            6: "Sun",
        }
    )

    merged = _make_atr_quantiles(merged, q=5)
    merged = _make_atr_fixed_buckets(merged)

    by_atr_quantile = _group_table(
        merged,
        "atr_quantile",
        ["atr_quantile_label"],
    )

    by_atr_bucket = _group_table(
        merged,
        "atr_bucket",
        [],
    )

    by_hour_ny_x_atr_bucket_rows: list[dict[str, Any]] = []
    for (hour, bucket), g in merged.groupby(["entry_hour_ny", "atr_bucket"], dropna=True):
        row = {
            "entry_hour_ny": int(hour),
            "atr_bucket": bucket,
        }
        row.update(_compute_basic_stats(g))
        by_hour_ny_x_atr_bucket_rows.append(row)

    by_hour_ny_x_atr_bucket = pd.DataFrame(by_hour_ny_x_atr_bucket_rows)
    if not by_hour_ny_x_atr_bucket.empty:
        by_hour_ny_x_atr_bucket = by_hour_ny_x_atr_bucket.sort_values(
            ["entry_hour_ny", "atr_bucket"]
        ).reset_index(drop=True)

    out1 = run_dir / "volatility_adversity_by_atr_quantile.csv"
    out2 = run_dir / "volatility_adversity_by_atr_bucket.csv"
    out3 = run_dir / "volatility_adversity_by_hour_ny_x_atr_bucket.csv"

    by_atr_quantile.to_csv(out1, index=False)
    by_atr_bucket.to_csv(out2, index=False)
    by_hour_ny_x_atr_bucket.to_csv(out3, index=False)

    result = {
        "run_id": run_dir.name,
        "name": manifest.get("name"),
        "symbol": manifest.get("symbol"),
        "timeframe": manifest.get("timeframe"),
        "n_trades_loaded": int(len(trades)),
        "outputs": {
            "by_atr_quantile": out1.name,
            "by_atr_bucket": out2.name,
            "by_hour_ny_x_atr_bucket": out3.name,
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()