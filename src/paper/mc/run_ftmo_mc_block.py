from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# FTMO-like challenge parameters
# -----------------------------
DEFAULT_TARGET_PCT = 0.10        # +10%
DEFAULT_MAX_TOTAL_DD_PCT = 0.10  # -10%
DEFAULT_MAX_DAILY_DD_PCT = 0.05  # -5%
DEFAULT_MAX_DAYS = 30

# Monte Carlo parameters
DEFAULT_N_PATHS = 10000
DEFAULT_BLOCK_SIZE_DAYS = 3  # block bootstrap on "days" (not trades)
DEFAULT_SEED = 42

# trades.csv schema candidates
CANDIDATE_R_COLS = ["R", "r", "pnl_R", "r_multiple", "R_multiple"]
CANDIDATE_ENTRY_TIME_COLS = ["entry_time", "open_time", "time_entry", "entry_ts"]
CANDIDATE_EXIT_REASON_COLS = ["exit_reason", "exit_tag", "exit_type", "exit_code"]


@dataclass
class PathResult:
    passed: bool
    fail_reason: str  # "PASS" / "FAIL_TOTAL_DD" / "FAIL_DAILY_DD" / "FAIL_TIMEOUT"
    days_used: int
    trades_used: int
    final_equity: float
    max_total_dd_pct: float
    max_daily_dd_pct: float


@dataclass
class Summary:
    n_paths: int
    risk_pct: float
    target_pct: float
    max_total_dd_pct: float
    max_daily_dd_pct: float
    max_days: int
    block_size_days: int

    pass_rate: float
    fail_total_dd_rate: float
    fail_daily_dd_rate: float
    fail_timeout_rate: float

    median_days_used: float
    median_trades_used: float

    p05_final_equity: float
    p50_final_equity: float
    p95_final_equity: float


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_latest_run_dir(results_root: Path) -> Path:
    run_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {results_root.resolve()}")
    run_dirs.sort(key=lambda p: p.name)
    return run_dirs[-1]


def _load_trades(trades_csv: Path) -> pd.DataFrame:
    if not trades_csv.exists():
        raise FileNotFoundError(f"Missing trades.csv: {trades_csv.resolve()}")
    df = pd.read_csv(trades_csv)
    if df.empty:
        raise ValueError(f"trades.csv is empty: {trades_csv.resolve()}")
    return df


def _prepare_daily_r_sequences(
    trades: pd.DataFrame,
    r_col: str,
    entry_time_col: Optional[str],
    max_days_cap: Optional[int] = None
) -> List[List[float]]:
    """
    Returns: list of days, each day is list of R values in chronological order.

    If entry_time_col exists: group by UTC date.
    Else: treat as 1 trade per "day" in sequence (fallback).
    """
    df = trades.copy()
    df[r_col] = pd.to_numeric(df[r_col], errors="coerce")
    df = df.dropna(subset=[r_col])
    if df.empty:
        raise ValueError("No valid R values after cleaning.")

    if entry_time_col and entry_time_col in df.columns:
        t = pd.to_datetime(df[entry_time_col], utc=True, errors="coerce")
        df = df.assign(_ts=t).dropna(subset=["_ts"]).sort_values("_ts")
        df["_day"] = df["_ts"].dt.date.astype(str)
        grouped = df.groupby("_day")[r_col].apply(list).tolist()
    else:
        # Fallback: assume each trade is one day (conservative for daily DD, optimistic for time)
        grouped = [[float(x)] for x in df[r_col].tolist()]

    if max_days_cap is not None:
        grouped = grouped[:max_days_cap]
    return grouped


def _block_bootstrap_days(
    days: List[List[float]],
    out_len_days: int,
    block_size_days: int,
    rng: random.Random
) -> List[List[float]]:
    """
    Sample blocks of consecutive days (preserve within-day order; preserve short-range clustering).
    """
    n = len(days)
    if n == 0:
        raise ValueError("No days available for bootstrap.")
    if n == 1:
        return days * out_len_days  # degenerate

    block = max(1, int(block_size_days))
    res: List[List[float]] = []
    while len(res) < out_len_days:
        start = rng.randint(0, n - 1)
        # take consecutive days with wrap-around
        for i in range(block):
            if len(res) >= out_len_days:
                break
            res.append(days[(start + i) % n])
    return res


def _simulate_path(
    sampled_days: List[List[float]],
    risk_pct: float,
    target_pct: float,
    max_total_dd_pct: float,
    max_daily_dd_pct: float
) -> Tuple[PathResult, List[float]]:
    """
    Returns PathResult and equity_by_day (length = len(sampled_days)+1 with day0 equity).
    Equity model:
      - Start equity = 1.0 (normalized).
      - Each trade: equity *= (1 + R * risk_pct)
        (approx R-space compounding; acceptable for small risk%)
      - Daily DD: compare day start equity vs day min equity intraday.
      - Total DD: compare peak equity vs trough equity.
    """
    eq = 1.0
    peak = 1.0
    max_total_dd = 0.0
    max_daily_dd = 0.0
    trades_used = 0

    equity_by_day = [eq]

    for day_idx, day_rs in enumerate(sampled_days, start=1):
        day_start = eq
        day_min = eq

        for R in day_rs:
            trades_used += 1
            eq *= (1.0 + float(R) * risk_pct)
            # Track day min, global peak/DD
            day_min = min(day_min, eq)
            peak = max(peak, eq)
            dd_total = (peak - eq) / peak if peak > 0 else 0.0
            max_total_dd = max(max_total_dd, dd_total)

            # Total DD breach
            if dd_total >= max_total_dd_pct - 1e-12:
                return (
                    PathResult(
                        passed=False,
                        fail_reason="FAIL_TOTAL_DD",
                        days_used=day_idx,
                        trades_used=trades_used,
                        final_equity=eq,
                        max_total_dd_pct=max_total_dd,
                        max_daily_dd_pct=max_daily_dd,
                    ),
                    equity_by_day + [eq],
                )

            # Target hit (pass immediately)
            if eq >= 1.0 + target_pct - 1e-12:
                dd_daily = (day_start - day_min) / day_start if day_start > 0 else 0.0
                max_daily_dd = max(max_daily_dd, dd_daily)
                return (
                    PathResult(
                        passed=True,
                        fail_reason="PASS",
                        days_used=day_idx,
                        trades_used=trades_used,
                        final_equity=eq,
                        max_total_dd_pct=max_total_dd,
                        max_daily_dd_pct=max_daily_dd,
                    ),
                    equity_by_day + [eq],
                )

        # End of day: compute daily DD
        dd_daily = (day_start - day_min) / day_start if day_start > 0 else 0.0
        max_daily_dd = max(max_daily_dd, dd_daily)

        # Daily DD breach at day end
        if dd_daily >= max_daily_dd_pct - 1e-12:
            return (
                PathResult(
                    passed=False,
                    fail_reason="FAIL_DAILY_DD",
                    days_used=day_idx,
                    trades_used=trades_used,
                    final_equity=eq,
                    max_total_dd_pct=max_total_dd,
                    max_daily_dd_pct=max_daily_dd,
                ),
                equity_by_day + [eq],
            )

        equity_by_day.append(eq)

    # Timeout
    return (
        PathResult(
            passed=False,
            fail_reason="FAIL_TIMEOUT",
            days_used=len(sampled_days),
            trades_used=trades_used,
            final_equity=eq,
            max_total_dd_pct=max_total_dd,
            max_daily_dd_pct=max_daily_dd,
        ),
        equity_by_day,
    )


def _summarize(results: List[PathResult], equities_final: List[float], cfg: Dict) -> Summary:
    n = len(results)
    passes = sum(1 for r in results if r.fail_reason == "PASS")
    fail_total = sum(1 for r in results if r.fail_reason == "FAIL_TOTAL_DD")
    fail_daily = sum(1 for r in results if r.fail_reason == "FAIL_DAILY_DD")
    fail_timeout = sum(1 for r in results if r.fail_reason == "FAIL_TIMEOUT")

    days_used = [r.days_used for r in results]
    trades_used = [r.trades_used for r in results]

    s = pd.Series(equities_final)
    return Summary(
        n_paths=n,
        risk_pct=cfg["risk_pct"],
        target_pct=cfg["target_pct"],
        max_total_dd_pct=cfg["max_total_dd_pct"],
        max_daily_dd_pct=cfg["max_daily_dd_pct"],
        max_days=cfg["max_days"],
        block_size_days=cfg["block_size_days"],
        pass_rate=passes / n,
        fail_total_dd_rate=fail_total / n,
        fail_daily_dd_rate=fail_daily / n,
        fail_timeout_rate=fail_timeout / n,
        median_days_used=float(pd.Series(days_used).median()),
        median_trades_used=float(pd.Series(trades_used).median()),
        p05_final_equity=float(s.quantile(0.05)),
        p50_final_equity=float(s.quantile(0.50)),
        p95_final_equity=float(s.quantile(0.95)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None, help="Path to results/runs/<run_id>. If omitted, uses latest run.")
    ap.add_argument("--results_root", type=str, default="results/runs", help="Root folder containing run directories.")
    ap.add_argument("--n_paths", type=int, default=DEFAULT_N_PATHS)
    ap.add_argument("--block_days", type=int, default=DEFAULT_BLOCK_SIZE_DAYS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    ap.add_argument("--max_days", type=int, default=DEFAULT_MAX_DAYS)
    ap.add_argument("--target_pct", type=float, default=DEFAULT_TARGET_PCT)
    ap.add_argument("--max_total_dd_pct", type=float, default=DEFAULT_MAX_TOTAL_DD_PCT)
    ap.add_argument("--max_daily_dd_pct", type=float, default=DEFAULT_MAX_DAILY_DD_PCT)

    ap.add_argument("--risk_pcts", type=str, default="0.005,0.0075", help="Comma-separated risk per trade (e.g. 0.005,0.0075)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    results_root = Path(args.results_root)
    run_dir = Path(args.run_dir) if args.run_dir else _load_latest_run_dir(results_root)

    trades_csv = run_dir / "trades.csv"
    trades = _load_trades(trades_csv)

    r_col = _pick_col(trades, CANDIDATE_R_COLS)
    if r_col is None:
        raise ValueError(f"Could not find R column in trades.csv. Tried: {CANDIDATE_R_COLS}")

    entry_time_col = _pick_col(trades, CANDIDATE_ENTRY_TIME_COLS)

    # Build daily sequences from observed OOS trades
    days = _prepare_daily_r_sequences(trades, r_col=r_col, entry_time_col=entry_time_col)

    # Output dir
    out_dir = run_dir / "mc_ftmo"
    out_dir.mkdir(parents=True, exist_ok=True)

    risk_pcts = [float(x.strip()) for x in args.risk_pcts.split(",") if x.strip()]

    # For equity quantiles by day
    # We'll store equity_by_day per path and then compute quantiles day-wise.
    all_outputs = {}

    for risk_pct in risk_pcts:
        cfg = dict(
            risk_pct=risk_pct,
            target_pct=args.target_pct,
            max_total_dd_pct=args.max_total_dd_pct,
            max_daily_dd_pct=args.max_daily_dd_pct,
            max_days=args.max_days,
            block_size_days=args.block_days,
            n_paths=args.n_paths,
            seed=args.seed,
            run_id=run_dir.name,
            r_col=r_col,
            entry_time_col=entry_time_col,
            n_days_in_sample=len(days),
            n_trades_in_file=int(pd.to_numeric(trades[r_col], errors="coerce").dropna().shape[0]),
        )

        path_results: List[PathResult] = []
        finals: List[float] = []
        equity_paths: List[List[float]] = []

        for _ in range(args.n_paths):
            sampled_days = _block_bootstrap_days(days, out_len_days=args.max_days, block_size_days=args.block_days, rng=rng)
            pr, eq_by_day = _simulate_path(
                sampled_days=sampled_days,
                risk_pct=risk_pct,
                target_pct=args.target_pct,
                max_total_dd_pct=args.max_total_dd_pct,
                max_daily_dd_pct=args.max_daily_dd_pct,
            )
            path_results.append(pr)
            finals.append(pr.final_equity)
            # Ensure fixed length max_days+1 for quantiles
            if len(eq_by_day) < args.max_days + 1:
                eq_by_day = eq_by_day + [eq_by_day[-1]] * (args.max_days + 1 - len(eq_by_day))
            else:
                eq_by_day = eq_by_day[: args.max_days + 1]
            equity_paths.append(eq_by_day)

        summary = _summarize(path_results, finals, cfg)

        # Fail reasons table
        fr = pd.Series([r.fail_reason for r in path_results]).value_counts().rename_axis("fail_reason").reset_index(name="n")
        fr["pct"] = fr["n"] / fr["n"].sum() * 100.0

        # Quantiles by day
        eq_df = pd.DataFrame(equity_paths)  # rows=paths, cols=day0..dayN
        quant = pd.DataFrame({
            "day": list(range(args.max_days + 1)),
            "p05": eq_df.quantile(0.05).values,
            "p25": eq_df.quantile(0.25).values,
            "p50": eq_df.quantile(0.50).values,
            "p75": eq_df.quantile(0.75).values,
            "p95": eq_df.quantile(0.95).values,
        })

        tag = f"risk_{risk_pct:.4f}".replace(".", "p")
        # Write files
        (out_dir / f"{tag}__mc_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        fr.to_csv(out_dir / f"{tag}__mc_fail_reasons.csv", index=False)
        quant.to_csv(out_dir / f"{tag}__mc_paths_quantiles.csv", index=False)

        all_outputs[tag] = {
            "summary": asdict(summary),
            "fail_reasons": fr.to_dict(orient="records"),
            "quantiles_path": str((out_dir / f"{tag}__mc_paths_quantiles.csv").as_posix()),
        }

    # Write combined summary
    (out_dir / "mc_summary_all.json").write_text(json.dumps(all_outputs, indent=2), encoding="utf-8")

    print(f"[OK] run_dir: {run_dir}")
    print(f"[OK] wrote outputs to: {out_dir}")
    for tag, payload in all_outputs.items():
        s = payload["summary"]
        print(f"--- {tag} ---")
        print(f"pass_rate={s['pass_rate']:.3f}  fail_total={s['fail_total_dd_rate']:.3f}  fail_daily={s['fail_daily_dd_rate']:.3f}  timeout={s['fail_timeout_rate']:.3f}")
        print(f"median_days={s['median_days_used']:.1f}  median_trades={s['median_trades_used']:.1f}  final_equity p05/p50/p95 = {s['p05_final_equity']:.3f}/{s['p50_final_equity']:.3f}/{s['p95_final_equity']:.3f}")


if __name__ == "__main__":
    main()