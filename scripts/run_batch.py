from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from backtester.orchestrator.run import run_from_config


@dataclass(frozen=True)
class BatchItem:
    config_path: str


LEADERBOARD_COLUMNS: List[str] = [
    # Batch identity
    "batch_id",
    "dataset_compat",
    # Identity
    "status",
    "config_path",
    "run_id",
    "name",
    "symbol",
    "timeframe",
    "strategy_name",
    # Dataset identity (traceability)
    "dataset_id",
    "dataset_fp8",
    "dataset_rows",
    "dataset_start_time_utc",
    "dataset_end_time_utc",
    # Risk policy (guardrails) - flat metrics
    "risk_max_daily_loss_R",
    "risk_max_trades_per_day",
    "risk_cooldown_bars",
    "risk_blocked_by_daily_stop",
    "risk_blocked_by_max_trades_per_day",
    "risk_blocked_by_cooldown",
    "risk_final_realized_R_today",
    "risk_final_stopped_today",
    # Core metrics (PnL space)
    "n_trades",
    "total_pnl",
    "winrate",
    "avg_pnl",
    "avg_win",
    "avg_loss",
    "profit_factor",
    "profit_factor_is_inf",
    # Risk / dynamics (PnL space)
    "max_drawdown_abs",
    "max_drawdown_pct",
    "max_consecutive_losses",
    "avg_hold_minutes",
    "median_hold_minutes",
    "min_hold_minutes",
    "max_hold_minutes",
    # R-space metrics (research-grade)
    "n_trades_with_risk",
    "expectancy_R",
    "avg_R",
    "median_R",
    "winrate_R",
    "avg_win_R",
    "avg_loss_R",
    "payoff_ratio_R",
    "max_consecutive_losses_R",
    "max_drawdown_R_abs",
    "max_drawdown_R_pct",
    # Outputs
    "run_dir",
    "metrics_path",
    "trades_path",
    "equity_path",
]


def _run_one(config_path: str, out_root: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    res = run_from_config(cfg, out_dir=out_root)
    return {
        "status": "ok",
        "config_path": config_path,
        "run_id": res.get("run_id"),
        "outputs": res.get("outputs", {}),
        "metrics": res.get("metrics", {}),
        "name": cfg.get("name"),
        "symbol": cfg.get("symbol"),
        "timeframe": cfg.get("timeframe"),
        "strategy_name": (cfg.get("strategy") or {}).get("name"),
    }


def _is_header_like_row(row: Dict[str, Any]) -> bool:
    """
    Defensive guard: if a header-like record slips in (values equal column names),
    drop it so the leaderboard CSV does not show a duplicated header line.
    """
    try:
        return (
            str(row.get("status", "")).strip() == "status"
            and str(row.get("config_path", "")).strip() == "config_path"
            and str(row.get("run_id", "")).strip() == "run_id"
        )
    except Exception:
        return False


def _normalize_status(s: Any) -> str:
    s2 = str(s).strip().lower()
    return s2 if s2 in {"ok", "error"} else "error"


def _norm_fp8(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _compute_dataset_compat(rows: List[Dict[str, Any]]) -> str:
    """
    Dataset compatibility for the batch.
    - OK: all non-null dataset_fp8 values are identical and at least one exists
    - MIXED: more than one distinct non-null dataset_fp8 exists
    - UNKNOWN: no dataset_fp8 available at all
    """
    fps = []
    for r in rows:
        fp = _norm_fp8(r.get("dataset_fp8"))
        if fp:
            fps.append(fp)
    uniq = sorted(set(fps))
    if len(uniq) == 0:
        return "UNKNOWN"
    if len(uniq) == 1:
        return "OK"
    return "MIXED"


def _row_from_result(r: Dict[str, Any], batch_id: str, dataset_compat: str) -> Dict[str, Any]:
    m = r.get("metrics", {}) or {}
    outs = r.get("outputs", {}) or {}

    row: Dict[str, Any] = {
        # Batch fields
        "batch_id": batch_id,
        "dataset_compat": dataset_compat,
        # Identity
        "status": _normalize_status(r.get("status", "ok")),
        "config_path": r.get("config_path"),
        "run_id": r.get("run_id"),
        "name": r.get("name"),
        "symbol": r.get("symbol"),
        "timeframe": r.get("timeframe"),
        "strategy_name": r.get("strategy_name"),
        # Dataset (traceability)
        "dataset_id": m.get("dataset_id"),
        "dataset_fp8": m.get("dataset_fp8"),
        "dataset_rows": m.get("dataset_rows"),
        "dataset_start_time_utc": m.get("dataset_start_time_utc"),
        "dataset_end_time_utc": m.get("dataset_end_time_utc"),
        # Risk policy (guardrails)
        "risk_max_daily_loss_R": m.get("risk_max_daily_loss_R"),
        "risk_max_trades_per_day": m.get("risk_max_trades_per_day"),
        "risk_cooldown_bars": m.get("risk_cooldown_bars"),
        "risk_blocked_by_daily_stop": m.get("risk_blocked_by_daily_stop"),
        "risk_blocked_by_max_trades_per_day": m.get("risk_blocked_by_max_trades_per_day"),
        "risk_blocked_by_cooldown": m.get("risk_blocked_by_cooldown"),
        "risk_final_realized_R_today": m.get("risk_final_realized_R_today"),
        "risk_final_stopped_today": m.get("risk_final_stopped_today"),
        # Core
        "n_trades": m.get("n_trades"),
        "total_pnl": m.get("total_pnl"),
        "winrate": m.get("winrate"),
        "avg_pnl": m.get("avg_pnl"),
        "avg_win": m.get("avg_win"),
        "avg_loss": m.get("avg_loss"),
        "profit_factor": m.get("profit_factor"),
        "profit_factor_is_inf": m.get("profit_factor_is_inf"),
        # Risk / dynamics (PnL)
        "max_drawdown_abs": m.get("max_drawdown_abs"),
        "max_drawdown_pct": m.get("max_drawdown_pct"),
        "max_consecutive_losses": m.get("max_consecutive_losses"),
        "avg_hold_minutes": m.get("avg_hold_minutes"),
        "median_hold_minutes": m.get("median_hold_minutes"),
        "min_hold_minutes": m.get("min_hold_minutes"),
        "max_hold_minutes": m.get("max_hold_minutes"),
        # R-space
        "n_trades_with_risk": m.get("n_trades_with_risk"),
        "expectancy_R": m.get("expectancy_R"),
        "avg_R": m.get("avg_R"),
        "median_R": m.get("median_R"),
        "winrate_R": m.get("winrate_R"),
        "avg_win_R": m.get("avg_win_R"),
        "avg_loss_R": m.get("avg_loss_R"),
        "payoff_ratio_R": m.get("payoff_ratio_R"),
        "max_consecutive_losses_R": m.get("max_consecutive_losses_R"),
        "max_drawdown_R_abs": m.get("max_drawdown_R_abs"),
        "max_drawdown_R_pct": m.get("max_drawdown_R_pct"),
        # Outputs
        "run_dir": outs.get("run_dir"),
        "metrics_path": outs.get("metrics"),
        "trades_path": outs.get("trades"),
        "equity_path": outs.get("equity"),
    }

    # Ensure schema stability
    for col in LEADERBOARD_COLUMNS:
        row.setdefault(col, None)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs_dir", required=True, help="Directory containing run YAML configs")
    ap.add_argument("--out", default="results/runs", help="Output root directory for runs")
    ap.add_argument("--jobs", type=int, default=4, help="Number of parallel workers")
    ap.add_argument("--pattern", default="*.yaml", help="Glob pattern for config files (default: *.yaml)")
    args = ap.parse_args()

    configs_dir = Path(args.configs_dir)
    if not configs_dir.exists():
        raise SystemExit(f"configs_dir not found: {configs_dir}")

    config_paths = sorted([str(p) for p in configs_dir.glob(args.pattern)])
    if not config_paths:
        raise SystemExit(f"No config files found in {configs_dir} matching pattern {args.pattern!r}")

    out_root = str(Path(args.out))
    Path(out_root).mkdir(parents=True, exist_ok=True)

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    leaderboard_dir = Path("results/leaderboards")
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = leaderboard_dir / f"leaderboard_{batch_id}.csv"
    manifest_path = leaderboard_dir / f"batch_manifest_{batch_id}.json"

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_run_one, cp, out_root): cp for cp in config_paths}
        for fut in as_completed(futs):
            cp = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append({"config_path": cp, "error": repr(e)})

    # Compute dataset compatibility based on successful runs first
    # (errors don't contribute dataset fingerprints)
    # We'll compute once and stamp into all rows.
    # Build provisional rows with minimal dataset_fp8 to evaluate compat.
    provisional_rows: List[Dict[str, Any]] = []
    for r in results:
        m = r.get("metrics", {}) or {}
        provisional_rows.append({"dataset_fp8": m.get("dataset_fp8")})
    dataset_compat = _compute_dataset_compat(provisional_rows)

    rows: List[Dict[str, Any]] = [_row_from_result(r, batch_id=batch_id, dataset_compat=dataset_compat) for r in results]

    for er in errors:
        rows.append(
            _row_from_result(
                {
                    "status": "error",
                    "config_path": er.get("config_path"),
                    "run_id": None,
                    "outputs": {},
                    "metrics": {},
                    "name": None,
                    "symbol": None,
                    "timeframe": None,
                    "strategy_name": None,
                },
                batch_id=batch_id,
                dataset_compat=dataset_compat,
            )
        )

    # Drop any header-like row that could have slipped in
    rows = [r for r in rows if not _is_header_like_row(r)]

    df = pd.DataFrame(rows).reindex(columns=LEADERBOARD_COLUMNS)

    df = df.sort_values(
        by=["expectancy_R", "total_pnl", "winrate", "n_trades"],
        ascending=[False, False, False, False],
        na_position="last",
    )

    df.to_csv(leaderboard_path, index=False)

    batch_manifest = {
        "batch_id": batch_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configs_dir": str(configs_dir),
        "pattern": args.pattern,
        "jobs": args.jobs,
        "out_root": out_root,
        "leaderboard_csv": str(leaderboard_path),
        "dataset_compat": dataset_compat,
        "n_configs": len(config_paths),
        "n_success": len(results),
        "n_errors": len(errors),
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(batch_manifest, indent=2))

    print("Batch done.")
    print(f"Dataset compat: {dataset_compat}")
    print(f"Leaderboard: {leaderboard_path}")
    if errors:
        print(f"Errors: {len(errors)} (see {manifest_path})")


if __name__ == "__main__":
    main()
