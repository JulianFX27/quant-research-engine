# analysis/gating_compare_v4.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Core metrics in R-space
# -----------------------------
@dataclass
class RSeriesMetrics:
    n: int
    expectancy_R: float
    total_R: float
    winrate: float
    max_dd_R: float
    max_losing_streak: int


def _dd_and_streak(r: np.ndarray) -> Tuple[float, int]:
    if len(r) == 0:
        return 0.0, 0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(np.max(dd)) if len(dd) else 0.0

    streak = 0
    max_streak = 0
    for x in r:
        if x < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_dd, int(max_streak)


def summarize_r_series(r: np.ndarray) -> RSeriesMetrics:
    r = np.asarray(r, dtype=float)
    n = int(len(r))
    if n == 0:
        return RSeriesMetrics(
            n=0,
            expectancy_R=0.0,
            total_R=0.0,
            winrate=0.0,
            max_dd_R=0.0,
            max_losing_streak=0,
        )
    max_dd, max_streak = _dd_and_streak(r)
    return RSeriesMetrics(
        n=n,
        expectancy_R=float(np.mean(r)),
        total_R=float(np.sum(r)),
        winrate=float(np.mean(r > 0)),
        max_dd_R=float(max_dd),
        max_losing_streak=int(max_streak),
    )


# -----------------------------
# Monte Carlo (same philosophy as monte_carlo_robustness.py)
# -----------------------------
@dataclass
class MCResult:
    total_R: float
    expectancy_R: float
    max_dd_R: float
    max_losing_streak: int


def equity_and_metrics_from_R(r: np.ndarray) -> Tuple[np.ndarray, MCResult]:
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(np.max(dd)) if len(dd) else 0.0

    streak = 0
    max_streak = 0
    for x in r:
        if x < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    res = MCResult(
        total_R=float(eq[-1]) if len(eq) else 0.0,
        expectancy_R=float(np.mean(r)) if len(r) else 0.0,
        max_dd_R=max_dd,
        max_losing_streak=int(max_streak),
    )
    return eq, res


def iid_bootstrap(trade_R: np.ndarray, n_sims: int, seed: int) -> list[MCResult]:
    rng = np.random.default_rng(seed)
    n = len(trade_R)
    out: list[MCResult] = []
    for _ in range(n_sims):
        sample = rng.choice(trade_R, size=n, replace=True)
        _, res = equity_and_metrics_from_R(sample)
        out.append(res)
    return out


def block_bootstrap(trade_R: np.ndarray, block_size: int, n_sims: int, seed: int) -> list[MCResult]:
    rng = np.random.default_rng(seed)
    n = len(trade_R)
    if n == 0:
        return []
    starts = np.arange(0, n - block_size + 1)
    if len(starts) == 0:
        return iid_bootstrap(trade_R, n_sims=n_sims, seed=seed)

    out: list[MCResult] = []
    for _ in range(n_sims):
        chunks = []
        while sum(len(c) for c in chunks) < n:
            s = int(rng.choice(starts))
            chunks.append(trade_R[s : s + block_size])
        sample = np.concatenate(chunks)[:n]
        _, res = equity_and_metrics_from_R(sample)
        out.append(res)
    return out


def daily_bootstrap(df: pd.DataFrame, n_sims: int, seed: int, block_days: int = 1) -> list[MCResult]:
    rng = np.random.default_rng(seed)

    d0 = df.copy()
    d0["entry_time"] = pd.to_datetime(d0["entry_time"], utc=True)
    d0["entry_date"] = d0["entry_time"].dt.floor("D")

    daily = d0.groupby("entry_date", as_index=False)["R"].sum().sort_values("entry_date")
    daily_R = daily["R"].to_numpy(dtype=float)
    n_days = len(daily_R)
    if n_days == 0:
        return []

    starts = np.arange(0, n_days - block_days + 1)
    if len(starts) == 0:
        starts = np.array([0])

    out: list[MCResult] = []
    for _ in range(n_sims):
        chunks = []
        while sum(len(c) for c in chunks) < n_days:
            s = int(rng.choice(starts))
            chunks.append(daily_R[s : s + block_days])
        sample_daily = np.concatenate(chunks)[:n_days]
        _, res = equity_and_metrics_from_R(sample_daily)
        out.append(res)
    return out


def summarize_mc(results: list[MCResult]) -> Dict[str, float]:
    if not results:
        return {}
    total_R = np.array([r.total_R for r in results], dtype=float)
    max_dd = np.array([r.max_dd_R for r in results], dtype=float)
    streak = np.array([r.max_losing_streak for r in results], dtype=int)
    expR = np.array([r.expectancy_R for r in results], dtype=float)

    def pct(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p))

    prob_dd_ge_5 = float(np.mean(max_dd >= 5.0))
    prob_dd_ge_8 = float(np.mean(max_dd >= 8.0))
    prob_dd_ge_12 = float(np.mean(max_dd >= 12.0))

    return {
        "n_sims": float(len(results)),
        "total_R_p5": pct(total_R, 5),
        "total_R_p50": pct(total_R, 50),
        "total_R_p95": pct(total_R, 95),
        "max_dd_R_p5": pct(max_dd, 5),
        "max_dd_R_p50": pct(max_dd, 50),
        "max_dd_R_p95": pct(max_dd, 95),
        "max_losing_streak_p50": pct(streak.astype(float), 50),
        "max_losing_streak_p95": pct(streak.astype(float), 95),
        "expectancy_R_mean": float(np.mean(expR)),
        "prob(max_dd_R>=5)": prob_dd_ge_5,
        "prob(max_dd_R>=8)": prob_dd_ge_8,
        "prob(max_dd_R>=12)": prob_dd_ge_12,
    }


def run_mc_suite(df_trades: pd.DataFrame, n_sims: int, seed: int) -> Dict[str, Dict[str, float]]:
    trade_R = df_trades["R"].to_numpy(dtype=float)

    report: Dict[str, Dict[str, float]] = {}

    r_iid = iid_bootstrap(trade_R, n_sims=n_sims, seed=seed)
    report["mc_iid_trades"] = summarize_mc(r_iid)

    for bs in [5, 10, 20]:
        r_blk = block_bootstrap(trade_R, block_size=bs, n_sims=n_sims, seed=seed + bs)
        report[f"mc_block_trades_bs{bs}"] = summarize_mc(r_blk)

    r_day1 = daily_bootstrap(df_trades, n_sims=n_sims, seed=seed + 101, block_days=1)
    report["mc_daily_blocks_1d"] = summarize_mc(r_day1)

    r_day5 = daily_bootstrap(df_trades, n_sims=n_sims, seed=seed + 105, block_days=5)
    report["mc_daily_blocks_5d"] = summarize_mc(r_day5)

    return report


# -----------------------------
# Data loading + gating
# -----------------------------
def _to_utc_naive(ts: pd.Series) -> pd.Series:
    # accept tz-aware or naive; output naive UTC for reliable merge
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(None)


def load_and_merge(trades_path: Path, features_path: Path) -> pd.DataFrame:
    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    feats = pd.read_csv(features_path, parse_dates=["time"])

    trades["entry_time"] = _to_utc_naive(trades["entry_time"])
    feats["time"] = _to_utc_naive(feats["time"])

    df = trades.merge(feats, left_on="entry_time", right_on="time", how="left")
    return df


def apply_gating_v4(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.DataFrame:
    """
    V4 rule:
      ALLOW if (col <= lo) OR (col > hi)
      BLOCK if (lo < col <= hi)

    This matches the toxic bucket you identified around ~0 for spread_ny_open_atr.
    """
    if col not in df.columns:
        raise KeyError(f"Missing column '{col}' in merged dataframe.")
    x = pd.to_numeric(df[col], errors="coerce")
    keep = (x <= lo) | (x > hi)
    return df.loc[keep].copy()


def _write_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="results/runs/<RUN_ID>")
    ap.add_argument(
        "--features_path",
        default="data/anchor_reversion_fx/data/eurusd_m5_features.csv",
        help="Path to features CSV with 'time' column",
    )
    ap.add_argument("--out_dir", default=None, help="Default: <run_dir>/gating_v4")
    ap.add_argument("--mc_sims", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)

    # V4 thresholds (from your qcut bucket approx)
    ap.add_argument("--gate_col", default="spread_ny_open_atr")
    ap.add_argument("--gate_lo", type=float, default=-3.067)
    ap.add_argument("--gate_hi", type=float, default=2.664)

    # Optional: min trades safeguard
    ap.add_argument("--min_trades", type=int, default=50)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    trades_path = run_dir / "trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(trades_path)

    features_path = Path(args.features_path)
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "gating_v4")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(trades_path, features_path)

    # Sanity: merge coverage
    nan_rate_close = float(df["close"].isna().mean()) if "close" in df.columns else None

    # Baseline metrics (ordered by time)
    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    base_r = df_sorted["R"].to_numpy(dtype=float)
    base = summarize_r_series(base_r)

    # Apply gating
    gated = apply_gating_v4(df_sorted, col=args.gate_col, lo=args.gate_lo, hi=args.gate_hi)
    gated = gated.sort_values("entry_time").reset_index(drop=True)
    gated_r = gated["R"].to_numpy(dtype=float)
    gated_m = summarize_r_series(gated_r)

    report = {
        "run_dir": str(run_dir),
        "features_path": str(features_path),
        "merge_nan_rate_close": nan_rate_close,
        "gate": {
            "name": "v4_spread_ny_open_atr_exclude_near_zero",
            "col": args.gate_col,
            "lo": args.gate_lo,
            "hi": args.gate_hi,
            "rule": "ALLOW if x<=lo OR x>hi ; BLOCK if lo<x<=hi",
        },
        "baseline": {
            "n_trades": base.n,
            "expectancy_R": base.expectancy_R,
            "total_R": base.total_R,
            "winrate": base.winrate,
            "max_dd_R": base.max_dd_R,
            "max_losing_streak": base.max_losing_streak,
        },
        "gated": {
            "n_trades": gated_m.n,
            "expectancy_R": gated_m.expectancy_R,
            "total_R": gated_m.total_R,
            "winrate": gated_m.winrate,
            "max_dd_R": gated_m.max_dd_R,
            "max_losing_streak": gated_m.max_losing_streak,
        },
    }

    # Save compare table
    rows = [
        {"variant": "baseline", **report["baseline"]},
        {"variant": "gated_v4", **report["gated"]},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "gating_v4_compare.csv", index=False)
    _write_json(out_dir / "gating_v4_compare.json", report)

    # Save gated trades
    gated_cols = [c for c in df_sorted.columns if c in gated.columns]
    gated[gated_cols].to_csv(out_dir / "trades_gated.csv", index=False)

    # Monte Carlo on baseline + gated (if enough trades)
    mc_dir = out_dir / "mc"
    mc_dir.mkdir(parents=True, exist_ok=True)

    mc_report = {"baseline": {}, "gated_v4": {}}

    if base.n >= args.min_trades:
        mc_report["baseline"] = run_mc_suite(df_sorted[["entry_time", "R"]].copy(), n_sims=args.mc_sims, seed=args.seed)
        _write_json(mc_dir / "mc_baseline.json", mc_report["baseline"])
        pd.DataFrame(
            [{"mc_name": k, **v} for k, v in mc_report["baseline"].items()]
        ).to_csv(mc_dir / "mc_baseline.csv", index=False)

    if gated_m.n >= args.min_trades:
        mc_report["gated_v4"] = run_mc_suite(gated[["entry_time", "R"]].copy(), n_sims=args.mc_sims, seed=args.seed)
        _write_json(mc_dir / "mc_gated_v4.json", mc_report["gated_v4"])
        pd.DataFrame(
            [{"mc_name": k, **v} for k, v in mc_report["gated_v4"].items()]
        ).to_csv(mc_dir / "mc_gated_v4.csv", index=False)

    # Print concise summary
    print(f"[OK] Wrote: {out_dir / 'gating_v4_compare.json'}")
    print(f"[OK] Wrote: {out_dir / 'gating_v4_compare.csv'}")
    print(f"[OK] Wrote: {out_dir / 'trades_gated.csv'}")
    print("\n=== BASELINE ===")
    print(report["baseline"])
    print("\n=== GATED V4 ===")
    print(report["gated"])
    print("\n=== MC OUTPUTS ===")
    print(f"  {mc_dir / 'mc_baseline.csv'}")
    print(f"  {mc_dir / 'mc_gated_v4.csv'}")


if __name__ == "__main__":
    main()
