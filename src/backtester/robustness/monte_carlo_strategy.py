from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from backtester.robustness.contracts import RobustnessManifest


CANDIDATE_R_COLS: List[str] = ["R", "r", "pnl_R", "r_multiple", "R_multiple"]
CANDIDATE_ENTRY_TIME_COLS: List[str] = ["entry_time", "open_time", "time_entry", "entry_ts"]


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_trades(trades_csv: Path) -> pd.DataFrame:
    if not trades_csv.exists():
        raise FileNotFoundError(f"Missing trades.csv: {trades_csv.resolve()}")
    df = pd.read_csv(trades_csv)
    if df.empty:
        raise ValueError(f"trades.csv is empty: {trades_csv.resolve()}")
    return df


def _prepare_r_series(trades: pd.DataFrame, r_col: str) -> List[float]:
    s = pd.to_numeric(trades[r_col], errors="coerce").dropna().astype(float)
    vals = s.tolist()
    if len(vals) < 30:
        raise ValueError(f"Too few valid R observations for Monte Carlo: {len(vals)}")
    return vals


def _prepare_daily_r_sequences(
    trades: pd.DataFrame,
    r_col: str,
    entry_time_col: Optional[str],
) -> List[List[float]]:
    df = trades.copy()
    df[r_col] = pd.to_numeric(df[r_col], errors="coerce")
    df = df.dropna(subset=[r_col])

    if df.empty:
        raise ValueError("No valid R values after cleaning.")

    if entry_time_col and entry_time_col in df.columns:
        ts = pd.to_datetime(df[entry_time_col], utc=True, errors="coerce")
        df = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts")
        if df.empty:
            raise ValueError("No valid entry timestamps after parsing.")
        df["_day"] = df["_ts"].dt.date.astype(str)
        grouped = df.groupby("_day")[r_col].apply(lambda s: [float(x) for x in s.tolist()]).tolist()
    else:
        grouped = [[float(x)] for x in df[r_col].tolist()]

    if not grouped:
        raise ValueError("No day sequences constructed from trades.")
    return grouped


def _max_drawdown_abs(path_cum_r: List[float]) -> float:
    if not path_cum_r:
        return 0.0
    peak = path_cum_r[0]
    worst = 0.0
    for x in path_cum_r:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > worst:
            worst = dd
    return float(worst)


def _max_drawdown_pct(path_cum_r: List[float]) -> float:
    """
    Relative drawdown computed on cumulative R path.

    Important:
    - This is NOT portfolio/account equity drawdown.
    - It is a relative drawdown over cumulative R trajectory.
    - If peak <= 0, relative DD is undefined for that segment; those points are skipped.
    """
    if not path_cum_r:
        return 0.0
    peak = path_cum_r[0]
    worst = 0.0
    for x in path_cum_r:
        if x > peak:
            peak = x
        if peak <= 0:
            continue
        dd = (peak - x) / peak
        if dd > worst:
            worst = dd
    return float(worst)


def _max_consecutive_losses(path_r: Sequence[float]) -> int:
    cur = 0
    best = 0
    for x in path_r:
        if x < 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _quantile(series: pd.Series, q: float) -> Optional[float]:
    v = series.quantile(q)
    return float(v) if pd.notna(v) else None


def _iid_bootstrap(r_values: Sequence[float], out_len: int, rng: random.Random) -> List[float]:
    n = len(r_values)
    if n == 0:
        raise ValueError("No R values available for iid bootstrap.")
    return [float(r_values[rng.randrange(n)]) for _ in range(out_len)]


def _block_bootstrap_days(
    days: Sequence[Sequence[float]],
    out_len_days: int,
    block_days: int,
    rng: random.Random,
) -> List[List[float]]:
    n = len(days)
    if n == 0:
        raise ValueError("No day sequences available for block bootstrap.")
    block = max(1, int(block_days))
    res: List[List[float]] = []
    while len(res) < out_len_days:
        start = rng.randint(0, n - 1)
        for i in range(block):
            if len(res) >= out_len_days:
                break
            res.append(list(days[(start + i) % n]))
    return res


def _simulate_path(path_r: Sequence[float], start_r_cum: float = 0.0) -> Tuple[List[float], Dict[str, Any]]:
    """
    Strategy Monte Carlo in additive R-space.

    This does NOT simulate account equity with compounding.
    It simulates cumulative R trajectory, which is the correct canonical
    representation for strategy robustness independent of sizing policy.
    """
    r_cum = float(start_r_cum)
    path_cum_r = [r_cum]

    for r in path_r:
        r_cum += float(r)
        path_cum_r.append(r_cum)

    terminal_r_cum = float(path_cum_r[-1]) if path_cum_r else float(start_r_cum)
    expectancy_r = float(sum(path_r) / len(path_r)) if path_r else 0.0

    stats = {
        "terminal_R_cum": terminal_r_cum,
        "max_drawdown_R_abs": _max_drawdown_abs(path_cum_r),
        "max_drawdown_R_pct": _max_drawdown_pct(path_cum_r),
        "max_consecutive_losses": _max_consecutive_losses(path_r),
        "expectancy_R": expectancy_r,
        "n_trades": int(len(path_r)),
    }
    return path_cum_r, stats


def _build_quantile_paths(path_cum_r_list: List[List[float]]) -> pd.DataFrame:
    max_len = max(len(p) for p in path_cum_r_list)
    padded = []
    for p in path_cum_r_list:
        if len(p) < max_len:
            p = p + [p[-1]] * (max_len - len(p))
        padded.append(p)

    path_df = pd.DataFrame(padded)
    return pd.DataFrame(
        {
            "step": list(range(max_len)),
            "p05_cum_R": path_df.quantile(0.05).values,
            "p25_cum_R": path_df.quantile(0.25).values,
            "p50_cum_R": path_df.quantile(0.50).values,
            "p75_cum_R": path_df.quantile(0.75).values,
            "p95_cum_R": path_df.quantile(0.95).values,
        }
    )


def run_strategy_mc(
    run_dir: str | Path,
    out_root: str | Path = "results/robustness/mc_strategy",
    method: str = "block_days",
    n_paths: int = 20000,
    block_days: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    trades_csv = run_dir / "trades.csv"
    manifest_json = run_dir / "run_manifest.json"
    metrics_json = run_dir / "metrics.json"

    trades = _load_trades(trades_csv)
    run_manifest = _load_json_if_exists(manifest_json)
    run_metrics = _load_json_if_exists(metrics_json)

    r_col = _pick_col(trades, CANDIDATE_R_COLS)
    if r_col is None:
        raise ValueError(f"Could not find R column in trades.csv. Tried: {CANDIDATE_R_COLS}")

    entry_time_col = _pick_col(trades, CANDIDATE_ENTRY_TIME_COLS)

    robustness_id = _utc_now_id()
    out_dir = Path(out_root) / robustness_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    distributions: List[Dict[str, Any]] = []
    path_cum_r_list: List[List[float]] = []

    if method == "iid":
        r_values = _prepare_r_series(trades, r_col)
        out_len = len(r_values)

        for path_idx in range(n_paths):
            path_r = _iid_bootstrap(r_values, out_len=out_len, rng=rng)
            path_cum_r, stats = _simulate_path(path_r)
            stats["path_id"] = path_idx
            distributions.append(stats)
            path_cum_r_list.append(path_cum_r)

    elif method in {"block_days", "day_bootstrap"}:
        day_sequences = _prepare_daily_r_sequences(trades, r_col=r_col, entry_time_col=entry_time_col)
        out_len_days = len(day_sequences)

        for path_idx in range(n_paths):
            sampled_days = _block_bootstrap_days(
                day_sequences,
                out_len_days=out_len_days,
                block_days=(1 if method == "day_bootstrap" else block_days),
                rng=rng,
            )
            path_r = [float(x) for day in sampled_days for x in day]
            path_cum_r, stats = _simulate_path(path_r)
            stats["path_id"] = path_idx
            distributions.append(stats)
            path_cum_r_list.append(path_cum_r)
    else:
        raise ValueError("method must be one of: iid, block_days, day_bootstrap")

    dist_df = pd.DataFrame(distributions)
    quant_df = _build_quantile_paths(path_cum_r_list)

    dist_csv = out_dir / "mc_distribution.csv"
    quant_csv = out_dir / "mc_paths_quantiles.csv"
    summary_json = out_dir / "mc_strategy_summary.json"
    manifest_out = out_dir / "mc_manifest.json"

    dist_df.to_csv(dist_csv, index=False)
    quant_df.to_csv(quant_csv, index=False)

    summary: Dict[str, Any] = {
        "test_type": "monte_carlo_strategy",
        "robustness_id": robustness_id,
        "method": method,
        "n_paths": int(n_paths),
        "n_observations": int(len(dist_df)),
        "p05_terminal_R_cum": _quantile(dist_df["terminal_R_cum"], 0.05),
        "p50_terminal_R_cum": _quantile(dist_df["terminal_R_cum"], 0.50),
        "p95_terminal_R_cum": _quantile(dist_df["terminal_R_cum"], 0.95),
        "p05_max_drawdown_R_abs": _quantile(dist_df["max_drawdown_R_abs"], 0.05),
        "p50_max_drawdown_R_abs": _quantile(dist_df["max_drawdown_R_abs"], 0.50),
        "p95_max_drawdown_R_abs": _quantile(dist_df["max_drawdown_R_abs"], 0.95),
        "p05_max_drawdown_R_pct": _quantile(dist_df["max_drawdown_R_pct"], 0.05),
        "p50_max_drawdown_R_pct": _quantile(dist_df["max_drawdown_R_pct"], 0.50),
        "p95_max_drawdown_R_pct": _quantile(dist_df["max_drawdown_R_pct"], 0.95),
        "p50_max_consecutive_losses": _quantile(dist_df["max_consecutive_losses"], 0.50),
        "p90_max_consecutive_losses": _quantile(dist_df["max_consecutive_losses"], 0.90),
        "p95_max_consecutive_losses": _quantile(dist_df["max_consecutive_losses"], 0.95),
        "p99_max_consecutive_losses": _quantile(dist_df["max_consecutive_losses"], 0.99),
        "p05_expectancy_R": _quantile(dist_df["expectancy_R"], 0.05),
        "p50_expectancy_R": _quantile(dist_df["expectancy_R"], 0.50),
        "p95_expectancy_R": _quantile(dist_df["expectancy_R"], 0.95),
        "notes": {
            "path_space": "additive_R_space",
            "interpretation": "This Monte Carlo estimates cumulative R trajectory, not compounded account equity.",
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = RobustnessManifest(
        test_type="monte_carlo_strategy",
        robustness_id=robustness_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_run_dir=str(run_dir),
        source_trades_csv=str(trades_csv),
        output_dir=str(out_dir),
        dataset_id=run_metrics.get("dataset_id"),
        dataset_fp8=run_metrics.get("dataset_fp8"),
        run_id=run_manifest.get("run_id"),
        method=method,
        n_paths=int(n_paths),
        block_days=(None if method == "iid" else int(1 if method == "day_bootstrap" else block_days)),
        seed=int(seed),
        r_col=str(r_col),
        entry_time_col=entry_time_col,
        notes={
            "methods_supported": ["iid", "block_days", "day_bootstrap"],
            "path_space": "additive_R_space",
            "rationale": "Strategy Monte Carlo measures statistical robustness of the observed trade distribution independent of capital sizing.",
        },
    )
    manifest_out.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    return {
        "robustness_id": robustness_id,
        "output_dir": str(out_dir),
        "summary": summary,
        "artifacts": {
            "distribution_csv": str(dist_csv),
            "quantiles_csv": str(quant_csv),
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_out),
        },
    }