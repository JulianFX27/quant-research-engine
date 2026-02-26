from __future__ import annotations

import argparse
import json
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
DEFAULT_N_PATHS = 20000
DEFAULT_BLOCK_SIZE_DAYS = 3
DEFAULT_SEED = 42

# trades.csv schema candidates
CANDIDATE_R_COLS = ["R", "r", "pnl_R", "r_multiple", "R_multiple"]
CANDIDATE_ENTRY_TIME_COLS = ["entry_time", "open_time", "time_entry", "entry_ts"]


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
    policy_name: str
    target_pct: float
    max_total_dd_pct: float
    max_daily_dd_pct: float
    max_days: int
    block_size_days: int
    seed: int

    pass_rate: float
    fail_total_dd_rate: float
    fail_daily_dd_rate: float
    fail_timeout_rate: float

    median_days_used: float
    median_trades_used: float

    p05_final_equity: float
    p50_final_equity: float
    p95_final_equity: float


# -----------------------------
# Helpers
# -----------------------------
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


def _prepare_daily_r_sequences(trades: pd.DataFrame, r_col: str, entry_time_col: Optional[str]) -> List[List[float]]:
    df = trades.copy()
    df[r_col] = pd.to_numeric(df[r_col], errors="coerce")
    df = df.dropna(subset=[r_col])
    if df.empty:
        raise ValueError("No valid R values after cleaning.")

    if entry_time_col and entry_time_col in df.columns:
        t = pd.to_datetime(df[entry_time_col], utc=True, errors="coerce")
        df = df.assign(_ts=t).dropna(subset=["_ts"]).sort_values("_ts")
        df["_day"] = df["_ts"].dt.date.astype(str)
        days = df.groupby("_day")[r_col].apply(lambda s: [float(x) for x in s.tolist()]).tolist()
    else:
        # Fallback: each trade is one "day"
        days = [[float(x)] for x in df[r_col].tolist()]

    if not days:
        raise ValueError("No days constructed from trades.")
    return days


def _block_bootstrap_days(days: List[List[float]], out_len_days: int, block_size_days: int, rng: random.Random) -> List[List[float]]:
    n = len(days)
    if n == 0:
        raise ValueError("No days available for bootstrap.")
    block = max(1, int(block_size_days))
    res: List[List[float]] = []
    while len(res) < out_len_days:
        start = rng.randint(0, n - 1)
        for i in range(block):
            if len(res) >= out_len_days:
                break
            res.append(days[(start + i) % n])
    return res


# -----------------------------
# Risk policies
# -----------------------------
def risk_fixed(risk_pct: float):
    def _f(eq: float) -> float:
        return risk_pct
    return _f


def risk_ramp_equity(eq: float) -> float:
    """
    Recommended schedule:
      eq < 1.03  -> 0.90%
      1.03-1.07  -> 0.75%
      eq > 1.07  -> 0.50%

    NOTE: eq is normalized to 1.0 = start.
    """
    if eq < 1.03:
        return 0.0090
    if eq < 1.07:
        return 0.0075
    return 0.0050


# -----------------------------
# Simulation
# -----------------------------
def _simulate_path(
    sampled_days: List[List[float]],
    risk_fn,
    target_pct: float,
    max_total_dd_pct: float,
    max_daily_dd_pct: float
) -> Tuple[PathResult, List[float]]:
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
            risk_pct = float(risk_fn(eq))
            eq *= (1.0 + float(R) * risk_pct)

            day_min = min(day_min, eq)
            peak = max(peak, eq)

            dd_total = (peak - eq) / peak if peak > 0 else 0.0
            max_total_dd = max(max_total_dd, dd_total)

            if dd_total >= max_total_dd_pct - 1e-12:
                return (
                    PathResult(False, "FAIL_TOTAL_DD", day_idx, trades_used, eq, max_total_dd, max_daily_dd),
                    equity_by_day + [eq],
                )

            if eq >= 1.0 + target_pct - 1e-12:
                dd_daily = (day_start - day_min) / day_start if day_start > 0 else 0.0
                max_daily_dd = max(max_daily_dd, dd_daily)
                return (
                    PathResult(True, "PASS", day_idx, trades_used, eq, max_total_dd, max_daily_dd),
                    equity_by_day + [eq],
                )

        dd_daily = (day_start - day_min) / day_start if day_start > 0 else 0.0
        max_daily_dd = max(max_daily_dd, dd_daily)

        if dd_daily >= max_daily_dd_pct - 1e-12:
            return (
                PathResult(False, "FAIL_DAILY_DD", day_idx, trades_used, eq, max_total_dd, max_daily_dd),
                equity_by_day + [eq],
            )

        equity_by_day.append(eq)

    return (
        PathResult(False, "FAIL_TIMEOUT", len(sampled_days), trades_used, eq, max_total_dd, max_daily_dd),
        equity_by_day,
    )


def _summarize(results: List[PathResult], finals: List[float], cfg: Dict) -> Summary:
    n = len(results)
    passes = sum(1 for r in results if r.fail_reason == "PASS")
    fail_total = sum(1 for r in results if r.fail_reason == "FAIL_TOTAL_DD")
    fail_daily = sum(1 for r in results if r.fail_reason == "FAIL_DAILY_DD")
    fail_timeout = sum(1 for r in results if r.fail_reason == "FAIL_TIMEOUT")

    days_used = pd.Series([r.days_used for r in results])
    trades_used = pd.Series([r.trades_used for r in results])

    s = pd.Series(finals)
    return Summary(
        n_paths=n,
        policy_name=cfg["policy_name"],
        target_pct=cfg["target_pct"],
        max_total_dd_pct=cfg["max_total_dd_pct"],
        max_daily_dd_pct=cfg["max_daily_dd_pct"],
        max_days=cfg["max_days"],
        block_size_days=cfg["block_size_days"],
        seed=cfg["seed"],
        pass_rate=passes / n,
        fail_total_dd_rate=fail_total / n,
        fail_daily_dd_rate=fail_daily / n,
        fail_timeout_rate=fail_timeout / n,
        median_days_used=float(days_used.median()),
        median_trades_used=float(trades_used.median()),
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

    args = ap.parse_args()

    rng = random.Random(args.seed)

    results_root = Path(args.results_root)
    run_dir = Path(args.run_dir) if args.run_dir else _load_latest_run_dir(results_root)

    trades = _load_trades(run_dir / "trades.csv")

    r_col = _pick_col(trades, CANDIDATE_R_COLS)
    if r_col is None:
        raise ValueError(f"Could not find R column in trades.csv. Tried: {CANDIDATE_R_COLS}")

    entry_time_col = _pick_col(trades, CANDIDATE_ENTRY_TIME_COLS)
    days = _prepare_daily_r_sequences(trades, r_col=r_col, entry_time_col=entry_time_col)

    out_dir = run_dir / "mc_ftmo_ramp"
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = {
        "FIXED_0p0050": risk_fixed(0.0050),
        "FIXED_0p0075": risk_fixed(0.0075),
        "RAMP_EQUITY": risk_ramp_equity,
    }

    combined = {}

    for name, risk_fn in policies.items():
        cfg = dict(
            policy_name=name,
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
        )

        path_results: List[PathResult] = []
        finals: List[float] = []
        equity_paths: List[List[float]] = []

        for _ in range(args.n_paths):
            sampled_days = _block_bootstrap_days(days, out_len_days=args.max_days, block_size_days=args.block_days, rng=rng)
            pr, eq_by_day = _simulate_path(
                sampled_days=sampled_days,
                risk_fn=risk_fn,
                target_pct=args.target_pct,
                max_total_dd_pct=args.max_total_dd_pct,
                max_daily_dd_pct=args.max_daily_dd_pct,
            )
            path_results.append(pr)
            finals.append(pr.final_equity)

            if len(eq_by_day) < args.max_days + 1:
                eq_by_day = eq_by_day + [eq_by_day[-1]] * (args.max_days + 1 - len(eq_by_day))
            else:
                eq_by_day = eq_by_day[: args.max_days + 1]
            equity_paths.append(eq_by_day)

        summary = _summarize(path_results, finals, cfg)
        fr = pd.Series([r.fail_reason for r in path_results]).value_counts().rename_axis("fail_reason").reset_index(name="n")
        fr["pct"] = fr["n"] / fr["n"].sum() * 100.0

        eq_df = pd.DataFrame(equity_paths)
        quant = pd.DataFrame({
            "day": list(range(args.max_days + 1)),
            "p05": eq_df.quantile(0.05).values,
            "p25": eq_df.quantile(0.25).values,
            "p50": eq_df.quantile(0.50).values,
            "p75": eq_df.quantile(0.75).values,
            "p95": eq_df.quantile(0.95).values,
        })

        tag = name
        (out_dir / f"{tag}__summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        fr.to_csv(out_dir / f"{tag}__fail_reasons.csv", index=False)
        quant.to_csv(out_dir / f"{tag}__paths_quantiles.csv", index=False)

        combined[tag] = {
            "summary": asdict(summary),
            "fail_reasons": fr.to_dict(orient="records"),
            "quantiles_path": str((out_dir / f"{tag}__paths_quantiles.csv").as_posix()),
        }

    (out_dir / "mc_ramp_all.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print(f"[OK] run_dir: {run_dir}")
    print(f"[OK] wrote outputs to: {out_dir}")
    for tag, payload in combined.items():
        s = payload["summary"]
        print(f"--- {tag} ---")
        print(f"pass_rate={s['pass_rate']:.3f}  fail_total={s['fail_total_dd_rate']:.3f}  fail_daily={s['fail_daily_dd_rate']:.3f}  timeout={s['fail_timeout_rate']:.3f}")
        print(f"median_days={s['median_days_used']:.1f}  median_trades={s['median_trades_used']:.1f}  final_equity p05/p50/p95 = {s['p05_final_equity']:.3f}/{s['p50_final_equity']:.3f}/{s['p95_final_equity']:.3f}")


if __name__ == "__main__":
    main()