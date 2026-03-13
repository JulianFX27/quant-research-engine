from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def ny_open_utc(ts_like) -> pd.Timestamp:
    if isinstance(ts_like, pd.Timestamp):
        d = ts_like.date()
    else:
        d = pd.Timestamp(ts_like).date()

    ny_tz = ZoneInfo("America/New_York")
    dt_local = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ny_tz)
    dt_utc = dt_local.astimezone(timezone.utc)
    return pd.Timestamp(dt_utc)


def qbucket(series: pd.Series, n: int = 5) -> pd.Series:
    valid = series.dropna()
    if len(valid) == 0 or valid.nunique() < min(n, 2):
        return pd.Series(index=series.index, dtype="object")

    try:
        out = pd.qcut(valid, q=n, labels=[f"Q{i}" for i in range(1, n + 1)], duplicates="drop")
    except Exception:
        return pd.Series(index=series.index, dtype="object")

    full = pd.Series(index=series.index, dtype="object")
    full.loc[valid.index] = out.astype(str)
    return full


def summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_col, dropna=True):
        if pd.isna(key):
            continue
        r = sub["pnl_proxy_R"]
        rows.append(
            {
                group_col: key,
                "n_trades": int(len(sub)),
                "mean_R": float(r.mean()),
                "median_R": float(r.median()),
                "std_R": float(r.std()) if len(r) > 1 else np.nan,
                "winrate_R": float((r > 0).mean()),
                "profit_factor_proxy": float(
                    r[r > 0].sum() / abs(r[r < 0].sum())
                ) if (r[r < 0].sum() < 0) else np.nan,
                "sum_R": float(r.sum()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and "mean_R" in out.columns:
        out = out.sort_values("mean_R", ascending=False).reset_index(drop=True)
    return out


def load_derived_slice(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    df["date"] = df["time"].dt.date
    return df


def build_daily_event_table(df: pd.DataFrame, entry_delay_min: int = 30, hold_min: int = 30) -> pd.DataFrame:
    """
    Build one event row per NY session day from a fold slice.

    Important:
    - signal_time = NY open + 30m
    - expected_strategy_bar = NY open + entry_delay_min
    - actual backtester trade entry with fill_mode=next_open will occur on the NEXT bar
    """
    px = df.set_index("time").sort_index()
    dates = sorted(df["date"].unique())

    rows = []
    for d in dates:
        t0 = ny_open_utc(d)
        signal_time = t0 + pd.Timedelta(minutes=30)
        strategy_entry_time = t0 + pd.Timedelta(minutes=entry_delay_min)
        actual_trade_entry_time = strategy_entry_time + pd.Timedelta(minutes=5)
        exit_time_from_actual_entry = actual_trade_entry_time + pd.Timedelta(minutes=hold_min)

        try:
            open_0 = float(px.loc[t0]["open"])
            close_30 = float(px.loc[signal_time]["close"])
            strategy_entry_open = float(px.loc[strategy_entry_time]["open"])
            actual_entry_open = float(px.loc[actual_trade_entry_time]["open"])
            exit_close = float(px.loc[exit_time_from_actual_entry]["close"])
        except KeyError:
            continue

        window_30 = px.loc[t0:signal_time]
        if window_30.empty:
            continue

        ret_30m = (close_30 - open_0) / open_0
        range_30m = (float(window_30["high"].max()) - float(window_30["low"].min())) / open_0
        impulse_efficiency = abs(ret_30m) / range_30m if range_30m > 0 else np.nan

        rows.append(
            {
                "session_date_utc": pd.Timestamp(actual_trade_entry_time).normalize(),
                "ny_open_time": t0,
                "signal_time": signal_time,
                "strategy_entry_time": strategy_entry_time,
                "actual_trade_entry_time": actual_trade_entry_time,
                "actual_trade_exit_time": exit_time_from_actual_entry,
                "ret_30m": ret_30m,
                "range_30m": range_30m,
                "impulse_efficiency": impulse_efficiency,
                "ret_fwd_30m_from_actual_entry": (exit_close - actual_entry_open) / actual_entry_open,
                "ret_fwd_30m_from_strategy_bar": (exit_close - strategy_entry_open) / strategy_entry_open,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wf-dir", required=True, help="WF run directory containing folds.csv and leaderboard.csv")
    parser.add_argument("--risk-proxy-price", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    wf_dir = Path(args.wf_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_path = wf_dir / "folds.csv"
    lb_path = wf_dir / "leaderboard.csv"

    if not folds_path.exists():
        raise FileNotFoundError(f"Missing folds.csv: {folds_path}")
    if not lb_path.exists():
        raise FileNotFoundError(f"Missing leaderboard.csv: {lb_path}")

    folds = pd.read_csv(folds_path)
    merged_trade_rows = []

    for _, row in folds.iterrows():
        fold = int(row["fold"])
        run_dir = Path(row["run_dir"])
        slice_path = Path(row["slice_path"])

        trades_path = run_dir / "trades.csv"
        if not trades_path.exists() or not slice_path.exists():
            continue

        trades = pd.read_csv(trades_path)
        if trades.empty:
            continue

        if "entry_time" not in trades.columns:
            continue

        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce") if "exit_time" in trades.columns else pd.NaT
        trades = trades.dropna(subset=["entry_time"]).copy()

        # backtester fills at next_open; merge by UTC session date to avoid exact timestamp mismatch
        trades["session_date_utc"] = trades["entry_time"].dt.normalize()

        derived = load_derived_slice(slice_path)
        events = build_daily_event_table(derived, entry_delay_min=30, hold_min=30)
        if events.empty:
            continue

        trades = trades.merge(
            events[
                [
                    "session_date_utc",
                    "ny_open_time",
                    "signal_time",
                    "strategy_entry_time",
                    "actual_trade_entry_time",
                    "actual_trade_exit_time",
                    "ret_30m",
                    "range_30m",
                    "impulse_efficiency",
                    "ret_fwd_30m_from_actual_entry",
                    "ret_fwd_30m_from_strategy_bar",
                ]
            ],
            how="left",
            on="session_date_utc",
        )

        # proxy return from realized engine trade prices
        if {"entry_price", "exit_price", "side"}.issubset(trades.columns):
            def side_ret(r):
                ep = float(r["entry_price"])
                xp = float(r["exit_price"])
                side = str(r["side"]).upper()
                if side == "BUY":
                    return (xp - ep) / ep
                return (ep - xp) / ep

            trades["net_ret_proxy"] = trades.apply(side_ret, axis=1)
        else:
            trades["net_ret_proxy"] = np.nan

        trades["pnl_proxy_R"] = trades["net_ret_proxy"] / float(args.risk_proxy_price)
        trades["fold"] = fold

        merged_trade_rows.append(trades)

    if not merged_trade_rows:
        raise ValueError("No trades merged from walk-forward runs.")

    all_trades = pd.concat(merged_trade_rows, ignore_index=True)

    # keep only trades where event features were successfully recovered
    all_trades = all_trades.dropna(subset=["ret_30m", "range_30m", "impulse_efficiency", "pnl_proxy_R"]).copy()

    if all_trades.empty:
        raise ValueError("Merged trades exist, but all regime features are missing after alignment.")

    all_trades["ret30_abs"] = all_trades["ret_30m"].abs()
    all_trades["ret30_bucket"] = qbucket(all_trades["ret30_abs"], n=5)
    all_trades["range30_bucket"] = qbucket(all_trades["range_30m"], n=5)
    all_trades["efficiency_bucket"] = qbucket(all_trades["impulse_efficiency"], n=5)

    ret30_tbl = summarize(all_trades.dropna(subset=["ret30_bucket"]), "ret30_bucket")
    range30_tbl = summarize(all_trades.dropna(subset=["range30_bucket"]), "range30_bucket")
    eff_tbl = summarize(all_trades.dropna(subset=["efficiency_bucket"]), "efficiency_bucket")

    by_fold_rows = []
    for fold, sub in all_trades.groupby("fold"):
        for group_col in ["ret30_bucket", "range30_bucket", "efficiency_bucket"]:
            for key, g in sub.groupby(group_col, dropna=True):
                if pd.isna(key):
                    continue
                r = g["pnl_proxy_R"]
                by_fold_rows.append(
                    {
                        "fold": int(fold),
                        "group_type": group_col,
                        "bucket": key,
                        "n_trades": int(len(g)),
                        "mean_R": float(r.mean()),
                        "winrate_R": float((r > 0).mean()),
                        "sum_R": float(r.sum()),
                    }
                )

    by_fold = pd.DataFrame(by_fold_rows)

    all_trades.to_csv(out_dir / "wf_trades_with_regimes.csv", index=False)
    ret30_tbl.to_csv(out_dir / "ret30_bucket_summary.csv", index=False)
    range30_tbl.to_csv(out_dir / "range30_bucket_summary.csv", index=False)
    eff_tbl.to_csv(out_dir / "efficiency_bucket_summary.csv", index=False)
    by_fold.to_csv(out_dir / "bucket_summary_by_fold.csv", index=False)

    meta = {
        "wf_dir": str(wf_dir),
        "risk_proxy_price": float(args.risk_proxy_price),
        "n_trades": int(len(all_trades)),
        "folds_included": sorted([int(x) for x in all_trades["fold"].dropna().unique().tolist()]),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== RET30 BUCKET SUMMARY ===\n")
    print(ret30_tbl.to_string(index=False))

    print("\n=== RANGE30 BUCKET SUMMARY ===\n")
    print(range30_tbl.to_string(index=False))

    print("\n=== EFFICIENCY BUCKET SUMMARY ===\n")
    print(eff_tbl.to_string(index=False))

    print(f"\nSaved outputs in: {out_dir}")


if __name__ == "__main__":
    main()