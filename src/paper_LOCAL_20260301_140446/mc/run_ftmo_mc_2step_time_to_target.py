from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------- FTMO 2-Step objectives ----------
CHALLENGE_TARGET = 0.10   # +10%
VERIF_TARGET = 0.05       # +5%
MAX_DAILY_DD = 0.05       # -5%
MAX_TOTAL_DD = 0.10       # -10%

MIN_TRADING_DAYS_PER_PHASE = 4  # FTMO Evaluation (Challenge + Verification)

# ---------- Monte Carlo ----------
DEFAULT_N_PATHS = 20000
DEFAULT_BLOCK_DAYS = 3
DEFAULT_SEED = 42

# Technical cap so simulations terminate even if the process is very slow
DEFAULT_CAP_DAYS = 300  # trading days (not calendar days)

# trades.csv schema candidates
CANDIDATE_R_COLS = ["R", "r", "pnl_R", "r_multiple", "R_multiple"]
CANDIDATE_ENTRY_TIME_COLS = ["entry_time", "open_time", "time_entry", "entry_ts"]


@dataclass
class PhaseResult:
    passed: bool
    fail_reason: str  # PASS / FAIL_TOTAL_DD / FAIL_DAILY_DD / CAP_REACHED
    days_used: int
    trades_used: int
    final_equity: float
    max_total_dd_pct: float
    max_daily_dd_pct: float


@dataclass
class TwoStepPath:
    challenge: PhaseResult
    verification: Optional[PhaseResult]
    passed_both: bool
    total_days_used: int
    total_trades_used: int


@dataclass
class Summary:
    n_paths: int
    risk_policy: str
    block_days: int
    cap_days: int
    seed: int

    pass_challenge_rate: float
    pass_both_rate: float

    fail_daily_rate: float
    fail_total_rate: float
    cap_rate: float

    # time-to-target distributions (trading days), conditional on passing
    challenge_days_p50: float
    challenge_days_p90: float
    verif_days_p50: float
    verif_days_p90: float
    total_days_p50: float
    total_days_p90: float


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


def _prepare_days(trades: pd.DataFrame, r_col: str, entry_time_col: Optional[str]) -> List[List[float]]:
    df = trades.copy()
    df[r_col] = pd.to_numeric(df[r_col], errors="coerce")
    df = df.dropna(subset=[r_col])
    if df.empty:
        raise ValueError("No valid R values after cleaning.")

    if entry_time_col and entry_time_col in df.columns:
        t = pd.to_datetime(df[entry_time_col], utc=True, errors="coerce")
        df = df.assign(_ts=t).dropna(subset=["_ts"]).sort_values("_ts")
        df["_day"] = df["_ts"].dt.date.astype(str)
        return df.groupby("_day")[r_col].apply(lambda s: [float(x) for x in s.tolist()]).tolist()

    # Fallback: each trade is a "day"
    return [[float(x)] for x in df[r_col].tolist()]


def _block_bootstrap_days(days: List[List[float]], out_len_days: int, block_days: int, rng: random.Random) -> List[List[float]]:
    n = len(days)
    block = max(1, int(block_days))
    res: List[List[float]] = []
    while len(res) < out_len_days:
        start = rng.randint(0, n - 1)
        for i in range(block):
            if len(res) >= out_len_days:
                break
            res.append(days[(start + i) % n])
    return res


# ---------- Risk policies ----------
def risk_fixed(p: float):
    def _f(eq: float) -> float:
        return p
    return _f


def risk_ramp_equity(eq: float) -> float:
    # Conservative ramp (your earlier ramp caused tiny total-DD failures; keep it mild)
    # <1.03 => 0.80%, 1.03-1.07 => 0.75%, >1.07 => 0.50%
    if eq < 1.03:
        return 0.0080
    if eq < 1.07:
        return 0.0075
    return 0.0050


def _simulate_phase(sampled_days: List[List[float]], risk_fn, target: float, cap_days: int) -> PhaseResult:
    eq = 1.0
    peak = 1.0
    max_total_dd = 0.0
    max_daily_dd = 0.0
    trades_used = 0

    days_used = 0
    for day_rs in sampled_days:
        days_used += 1
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
            if dd_total >= MAX_TOTAL_DD - 1e-12:
                return PhaseResult(False, "FAIL_TOTAL_DD", days_used, trades_used, eq, max_total_dd, max_daily_dd)

            if eq >= 1.0 + target - 1e-12:
                # Still must satisfy minimum trading days to "finish" the phase
                # If hit early, we continue with no trades? In real FTMO you can simply keep trading small.
                # Here we model by continuing to count days until min-days reached, with *1 micro trade/day*.
                # To avoid inventing micro-trade distribution, we just enforce days_used >= MIN_TRADING_DAYS_PER_PHASE externally.
                dd_daily = (day_start - day_min) / day_start if day_start > 0 else 0.0
                max_daily_dd = max(max_daily_dd, dd_daily)
                return PhaseResult(True, "PASS", days_used, trades_used, eq, max_total_dd, max_daily_dd)

        dd_daily = (day_start - day_min) / day_start if day_start > 0 else 0.0
        max_daily_dd = max(max_daily_dd, dd_daily)
        if dd_daily >= MAX_DAILY_DD - 1e-12:
            return PhaseResult(False, "FAIL_DAILY_DD", days_used, trades_used, eq, max_total_dd, max_daily_dd)

        if days_used >= cap_days:
            return PhaseResult(False, "CAP_REACHED", days_used, trades_used, eq, max_total_dd, max_daily_dd)

    return PhaseResult(False, "CAP_REACHED", days_used, trades_used, eq, max_total_dd, max_daily_dd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--results_root", type=str, default="results/runs")
    ap.add_argument("--n_paths", type=int, default=DEFAULT_N_PATHS)
    ap.add_argument("--block_days", type=int, default=DEFAULT_BLOCK_DAYS)
    ap.add_argument("--cap_days", type=int, default=DEFAULT_CAP_DAYS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--policy", type=str, default="fixed_0.0075", choices=["fixed_0.005", "fixed_0.0075", "ramp_equity"])
    args = ap.parse_args()

    rng = random.Random(args.seed)

    run_dir = Path(args.run_dir) if args.run_dir else _load_latest_run_dir(Path(args.results_root))
    trades = _load_trades(run_dir / "trades.csv")

    r_col = _pick_col(trades, CANDIDATE_R_COLS)
    if r_col is None:
        raise ValueError(f"Could not find R column. Tried: {CANDIDATE_R_COLS}")

    entry_time_col = _pick_col(trades, CANDIDATE_ENTRY_TIME_COLS)
    days_sample = _prepare_days(trades, r_col, entry_time_col)

    if args.policy == "fixed_0.005":
        risk_fn = risk_fixed(0.005)
    elif args.policy == "fixed_0.0075":
        risk_fn = risk_fixed(0.0075)
    else:
        risk_fn = risk_ramp_equity

    paths: List[TwoStepPath] = []

    for _ in range(args.n_paths):
        # We sample a long enough day-sequence to cover both phases within cap.
        sampled = _block_bootstrap_days(days_sample, out_len_days=args.cap_days, block_days=args.block_days, rng=rng)

        # Phase 1
        ch = _simulate_phase(sampled, risk_fn, CHALLENGE_TARGET, args.cap_days)

        # Enforce minimum trading days for passing: if PASS but days_used < 4, treat as reaching target early.
        # For time-to-target reporting, we report max(days_used, 4).
        ch_days_effective = max(ch.days_used, MIN_TRADING_DAYS_PER_PHASE) if ch.fail_reason == "PASS" else ch.days_used
        ch_trades_effective = ch.trades_used

        if ch.fail_reason != "PASS":
            paths.append(
                TwoStepPath(
                    challenge=ch,
                    verification=None,
                    passed_both=False,
                    total_days_used=ch.days_used,
                    total_trades_used=ch.trades_used,
                )
            )
            continue

        # Phase 2 starts fresh equity baseline in real life (new account). Model as new independent run on same distribution.
        # We reuse another sampled stream (independent) for verification.
        sampled2 = _block_bootstrap_days(days_sample, out_len_days=args.cap_days, block_days=args.block_days, rng=rng)
        vf = _simulate_phase(sampled2, risk_fn, VERIF_TARGET, args.cap_days)
        vf_days_effective = max(vf.days_used, MIN_TRADING_DAYS_PER_PHASE) if vf.fail_reason == "PASS" else vf.days_used

        passed_both = (vf.fail_reason == "PASS")
        total_days = ch_days_effective + vf_days_effective if vf else ch_days_effective
        total_trades = ch_trades_effective + (vf.trades_used if vf else 0)

        # Store with adjusted days for minimum-days constraint
        ch_adj = PhaseResult(ch.passed, ch.fail_reason, ch_days_effective, ch.trades_used, ch.final_equity, ch.max_total_dd_pct, ch.max_daily_dd_pct)
        vf_adj = None
        if vf:
            vf_adj = PhaseResult(vf.passed, vf.fail_reason, vf_days_effective, vf.trades_used, vf.final_equity, vf.max_total_dd_pct, vf.max_daily_dd_pct)

        paths.append(TwoStepPath(ch_adj, vf_adj, passed_both, total_days, total_trades))

    # Summaries
    n = len(paths)
    pass_ch = sum(1 for p in paths if p.challenge.fail_reason == "PASS")
    pass_both = sum(1 for p in paths if p.passed_both)

    fail_daily = sum(1 for p in paths if p.challenge.fail_reason == "FAIL_DAILY_DD" or (p.verification and p.verification.fail_reason == "FAIL_DAILY_DD"))
    fail_total = sum(1 for p in paths if p.challenge.fail_reason == "FAIL_TOTAL_DD" or (p.verification and p.verification.fail_reason == "FAIL_TOTAL_DD"))
    cap = sum(1 for p in paths if p.challenge.fail_reason == "CAP_REACHED")

    ch_days = [p.challenge.days_used for p in paths if p.challenge.fail_reason == "PASS"]
    vf_days = [p.verification.days_used for p in paths if p.verification and p.verification.fail_reason == "PASS"]
    total_days = [p.total_days_used for p in paths if p.passed_both]

    def q(arr, qq):
        if not arr:
            return float("nan")
        return float(pd.Series(arr).quantile(qq))

    summ = Summary(
        n_paths=n,
        risk_policy=args.policy,
        block_days=args.block_days,
        cap_days=args.cap_days,
        seed=args.seed,
        pass_challenge_rate=pass_ch / n,
        pass_both_rate=pass_both / n,
        fail_daily_rate=fail_daily / n,
        fail_total_rate=fail_total / n,
        cap_rate=cap / n,
        challenge_days_p50=q(ch_days, 0.50),
        challenge_days_p90=q(ch_days, 0.90),
        verif_days_p50=q(vf_days, 0.50),
        verif_days_p90=q(vf_days, 0.90),
        total_days_p50=q(total_days, 0.50),
        total_days_p90=q(total_days, 0.90),
    )

    out_dir = run_dir / "mc_ftmo_2step_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"summary__{args.policy}.json").write_text(json.dumps(asdict(summ), indent=2), encoding="utf-8")

    print(f"[OK] run_dir: {run_dir}")
    print(f"[OK] wrote: {out_dir / f'summary__{args.policy}.json'}")
    print("--- SUMMARY ---")
    print(json.dumps(asdict(summ), indent=2))


if __name__ == "__main__":
    main()