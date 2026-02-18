# analysis/monte_carlo_robustness.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MCResult:
    total_R: float
    expectancy_R: float
    max_dd_R: float
    max_losing_streak: int


def equity_and_metrics_from_R(r: np.ndarray) -> Tuple[np.ndarray, MCResult]:
    """
    Given a per-trade R sequence, compute equity (cum-sum in R-space) and key metrics.
    NOTE: Contract = r is per-trade (not per-day).
    """
    eq = np.cumsum(r)

    # drawdown in R-space
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(np.max(dd)) if len(dd) else 0.0

    # losing streak (consecutive r < 0)
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


def iid_bootstrap(trade_R: np.ndarray, n_sims: int, seed: int) -> List[MCResult]:
    rng = np.random.default_rng(seed)
    n = len(trade_R)
    out: List[MCResult] = []
    for _ in range(n_sims):
        sample = rng.choice(trade_R, size=n, replace=True)
        _, res = equity_and_metrics_from_R(sample)
        out.append(res)
    return out


def block_bootstrap(trade_R: np.ndarray, block_size: int, n_sims: int, seed: int) -> List[MCResult]:
    rng = np.random.default_rng(seed)
    n = len(trade_R)
    if n == 0:
        return []
    starts = np.arange(0, n - block_size + 1)
    if len(starts) == 0:
        # fallback to IID if too small
        return iid_bootstrap(trade_R, n_sims=n_sims, seed=seed)

    out: List[MCResult] = []
    for _ in range(n_sims):
        chunks = []
        while sum(len(c) for c in chunks) < n:
            s = int(rng.choice(starts))
            chunks.append(trade_R[s : s + block_size])
        sample = np.concatenate(chunks)[:n]
        _, res = equity_and_metrics_from_R(sample)
        out.append(res)
    return out


def daily_bootstrap(df: pd.DataFrame, n_sims: int, seed: int, block_days: int = 1) -> List[MCResult]:
    """
    Resample day-blocks to preserve clustering, BUT return a per-trade R sequence
    so expectancy_R remains comparable across MC variants (unit = R/trade).

    Approach:
      - Group trades by UTC entry day, keep per-day arrays of trade R.
      - Resample blocks of consecutive days (with wrap-around) and concatenate trade R
        until reaching the original number of trades, then truncate.
    """
    rng = np.random.default_rng(seed)

    d0 = df.copy()
    if "entry_time" not in d0.columns:
        raise ValueError("trades.csv must include column 'entry_time' for daily bootstrap")
    if "R" not in d0.columns:
        raise ValueError("trades.csv must include column 'R'")

    d0["entry_time"] = pd.to_datetime(d0["entry_time"], utc=True, errors="coerce")
    d0 = d0.dropna(subset=["entry_time"])
    d0["entry_date"] = d0["entry_time"].dt.floor("D")

    # Keep trades in chronological order within each day for realism
    d0 = d0.sort_values(["entry_date", "entry_time"])

    # Build ordered list of days + per-day trade-R arrays (non-empty only)
    days = d0["entry_date"].dropna().unique()
    days = np.array(sorted(days))
    if len(days) == 0:
        return []

    per_day_R: List[np.ndarray] = []
    for day in days:
        r_day = d0.loc[d0["entry_date"] == day, "R"].to_numpy(dtype=float)
        if len(r_day) > 0:
            per_day_R.append(r_day)

    n_days = len(per_day_R)
    if n_days == 0:
        return []

    n_trades_target = int(len(d0))
    out: List[MCResult] = []

    for _ in range(n_sims):
        chunks: List[np.ndarray] = []
        picked_trades = 0

        # sample day blocks until we have enough trades
        while picked_trades < n_trades_target:
            start = int(rng.integers(0, n_days))
            # consecutive days with wrap-around
            block_arrays = [per_day_R[(start + j) % n_days] for j in range(int(block_days))]
            r_block = np.concatenate(block_arrays) if len(block_arrays) > 1 else block_arrays[0]
            chunks.append(r_block)
            picked_trades += len(r_block)

            # safety guard (should never trigger in normal conditions)
            if len(chunks) > 100000:
                break

        sample_trades = np.concatenate(chunks)[:n_trades_target]
        _, res = equity_and_metrics_from_R(sample_trades)
        out.append(res)

    return out


def summarize(results: List[MCResult]) -> Dict[str, float]:
    if not results:
        return {}
    total_R = np.array([r.total_R for r in results], dtype=float)
    max_dd = np.array([r.max_dd_R for r in results], dtype=float)
    streak = np.array([r.max_losing_streak for r in results], dtype=int)
    expR = np.array([r.expectancy_R for r in results], dtype=float)

    def pct(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p))

    # Probabilities for risk thresholds (tune as needed)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="results/runs/<RUN_ID>")
    ap.add_argument("--n_sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_dir", default=None, help="default: <run_dir>/mc")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    trades_path = run_dir / "trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(trades_path)

    df = pd.read_csv(trades_path)
    if "R" not in df.columns:
        raise ValueError("trades.csv must include column 'R'")

    trade_R = df["R"].to_numpy(dtype=float)

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "mc")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Dict[str, float]] = {}

    # MC-1 IID trades
    r_iid = iid_bootstrap(trade_R, n_sims=args.n_sims, seed=args.seed)
    report["mc_iid_trades"] = summarize(r_iid)

    # MC-2 block trades (multiple block sizes)
    for bs in [5, 10, 20]:
        r_blk = block_bootstrap(trade_R, block_size=bs, n_sims=args.n_sims, seed=args.seed + bs)
        report[f"mc_block_trades_bs{bs}"] = summarize(r_blk)

    # MC-3 daily resample (1-day and 5-day blocks) — now per-trade consistent
    r_day1 = daily_bootstrap(df, n_sims=args.n_sims, seed=args.seed + 101, block_days=1)
    report["mc_daily_blocks_1d"] = summarize(r_day1)

    r_day5 = daily_bootstrap(df, n_sims=args.n_sims, seed=args.seed + 105, block_days=5)
    report["mc_daily_blocks_5d"] = summarize(r_day5)

    # Save report
    (out_dir / "mc_summary.json").write_text(pd.Series(report).to_json(indent=2), encoding="utf-8")

    # Also save a flat CSV summary for quick diffing
    rows = []
    for k, v in report.items():
        row = {"mc_name": k}
        row.update(v)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "mc_summary.csv", index=False)

    print(f"[OK] Wrote: {out_dir / 'mc_summary.json'}")
    print(f"[OK] Wrote: {out_dir / 'mc_summary.csv'}")


if __name__ == "__main__":
    main()
