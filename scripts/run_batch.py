# scripts/run_batch.py
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
    policy_id: Optional[str] = None


LEADERBOARD_COLUMNS: List[str] = [
    # Batch identity
    "batch_id",
    "dataset_compat",
    # Identity
    "status",
    "config_path",
    "policy_id",
    "run_id",
    "name",
    "symbol",
    "timeframe",
    "strategy_name",
    # Run validity (research-grade)
    "run_status",
    "invalid_eof",
    "valid_trades",
    "valid_trade_ratio",
    "eof_incomplete_count",
    "eof_forced_count",
    # Execution policy / effective execution
    "execution_policy_id",
    "execution_fill_mode",
    "execution_intrabar_path",
    "execution_intrabar_tie",
    "costs_spread_pips_effective",
    "costs_slippage_pips_effective",
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
    # Exit aggregates (optional but useful)
    "forced_exits_total",
    "forced_eof_total",
    "eof_exits_total",
    "non_forced_exits_total",
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


def _apply_policy_to_cfg(cfg: Dict[str, Any], policy_id: Optional[str]) -> Dict[str, Any]:
    if not policy_id:
        return cfg
    cfg2 = dict(cfg)
    exe = dict((cfg2.get("execution", {}) or {}))
    exe["policy_id"] = str(policy_id).strip()
    cfg2["execution"] = exe
    return cfg2


def _run_one(item: BatchItem, out_root: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(item.config_path).read_text())
    cfg = _apply_policy_to_cfg(cfg, item.policy_id)
    res = run_from_config(cfg, out_dir=out_root)
    return {
        "status": "ok",
        "config_path": item.config_path,
        "policy_id": item.policy_id,
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
    fps: List[str] = []
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


def _as_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


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
        "policy_id": r.get("policy_id"),
        "run_id": r.get("run_id"),
        "name": r.get("name"),
        "symbol": r.get("symbol"),
        "timeframe": r.get("timeframe"),
        "strategy_name": r.get("strategy_name"),
        # Run validity
        "run_status": m.get("run_status"),
        "invalid_eof": _as_bool(m.get("invalid_eof")),
        "valid_trades": m.get("valid_trades"),
        "valid_trade_ratio": m.get("valid_trade_ratio"),
        "eof_incomplete_count": m.get("eof_incomplete_count"),
        "eof_forced_count": m.get("eof_forced_count"),
        # Execution policy (effective)
        "execution_policy_id": m.get("execution_policy_id"),
        "execution_fill_mode": m.get("execution_fill_mode"),
        "execution_intrabar_path": m.get("execution_intrabar_path"),
        "execution_intrabar_tie": m.get("execution_intrabar_tie"),
        "costs_spread_pips_effective": m.get("costs_spread_pips_effective"),
        "costs_slippage_pips_effective": m.get("costs_slippage_pips_effective"),
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
        # Exit aggregates
        "forced_exits_total": m.get("forced_exits_total"),
        "forced_eof_total": m.get("forced_eof_total"),
        "eof_exits_total": m.get("eof_exits_total"),
        "non_forced_exits_total": m.get("non_forced_exits_total"),
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


def _parse_policy_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs_dir", required=True, help="Directory containing run YAML configs")
    ap.add_argument("--out", default="results/runs", help="Output root directory for runs")
    ap.add_argument("--jobs", type=int, default=4, help="Number of parallel workers")
    ap.add_argument("--pattern", default="*.yaml", help="Glob pattern for config files (default: *.yaml)")

    # Execution policy controls
    ap.add_argument("--policy", default=None, help="Force a single execution.policy_id for all configs")
    ap.add_argument(
        "--policy-sweep",
        default=None,
        help="Comma-separated list of policy_ids to run per config (expands runs). Example: baseline,stress_spread_2",
    )

    # Leaderboard filtering (research-grade)
    ap.add_argument(
        "--rank-only-valid",
        action="store_true",
        help="If set: exclude runs marked invalid_eof==True from ranking (still written to CSV).",
    )

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

    # Build batch items (config x policy)
    sweep = _parse_policy_list(args.policy_sweep)
    force_policy = str(args.policy).strip() if args.policy else None

    items: List[BatchItem] = []
    if sweep:
        for cp in config_paths:
            for pid in sweep:
                items.append(BatchItem(config_path=cp, policy_id=pid))
    else:
        for cp in config_paths:
            items.append(BatchItem(config_path=cp, policy_id=force_policy))

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_run_one, it, out_root): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append({"config_path": it.config_path, "policy_id": it.policy_id, "error": repr(e)})

    # Compute dataset compatibility based on successful runs first
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
                    "policy_id": er.get("policy_id"),
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

    rows = [r for r in rows if not _is_header_like_row(r)]
    df = pd.DataFrame(rows).reindex(columns=LEADERBOARD_COLUMNS)

    # Ranking view: optionally exclude invalid EOF runs from sorting priority
    rank_df = df.copy()
    if args.rank_only_valid:
        rank_df = rank_df[~(rank_df["invalid_eof"] == True)]  # noqa: E712

    # Sort ranking with validity first (OK > forced exits present > invalid)
    # We keep a stable CSV containing all; sorting is applied to full df by default,
    # or by rank_df if --rank-only-valid is used.
    sort_df = rank_df if args.rank_only_valid else df

    # Add helper sort keys (not persisted)
    sort_df = sort_df.copy()
    sort_df["_is_ok"] = (sort_df["run_status"] == "OK")
    sort_df["_is_ok_forced"] = (sort_df["run_status"] == "OK_FORCED_EXITS_PRESENT")
    sort_df["_valid_class"] = sort_df["_is_ok"].astype(int) * 2 + sort_df["_is_ok_forced"].astype(int) * 1

    sort_df = sort_df.sort_values(
        by=["_valid_class", "expectancy_R", "total_pnl", "winrate", "n_trades"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).drop(columns=["_is_ok", "_is_ok_forced", "_valid_class"])

    # If we used rank_only_valid, we still want to write ALL rows, but ordered as:
    # 1) ranked rows (valid) first, 2) then the rest.
    if args.rank_only_valid:
        ranked_keys = set(zip(sort_df["config_path"], sort_df["policy_id"], sort_df["run_id"]))
        rest = df.copy()
        rest["_k"] = list(zip(rest["config_path"], rest["policy_id"], rest["run_id"]))
        rest = rest[~rest["_k"].isin(ranked_keys)].drop(columns=["_k"])
        out_df = pd.concat([sort_df, rest], ignore_index=True)
    else:
        out_df = sort_df

    out_df.to_csv(leaderboard_path, index=False)

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
        "n_items": len(items),
        "policy": force_policy,
        "policy_sweep": sweep,
        "n_success": len(results),
        "n_errors": len(errors),
        "errors": errors,
        "rank_only_valid": bool(args.rank_only_valid),
    }
    manifest_path.write_text(json.dumps(batch_manifest, indent=2))

    print("Batch done.")
    print(f"Dataset compat: {dataset_compat}")
    print(f"Leaderboard: {leaderboard_path}")
    if args.rank_only_valid:
        print("Ranking policy: VALID ONLY (invalid EOF excluded from ranking)")
    if errors:
        print(f"Errors: {len(errors)} (see {manifest_path})")


if __name__ == "__main__":
    main()
