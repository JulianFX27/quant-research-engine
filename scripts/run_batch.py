from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from backtester.orchestrator.run import run_from_config


@dataclass(frozen=True)
class BatchItem:
    config_path: str


def _run_one(config_path: str, out_root: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    res = run_from_config(cfg, out_dir=out_root)
    # Persist a small per-config summary (useful if process crashes mid-batch)
    return {
        "config_path": config_path,
        "run_id": res.get("run_id"),
        "outputs": res.get("outputs", {}),
        "metrics": res.get("metrics", {}),
        "name": cfg.get("name"),
        "symbol": cfg.get("symbol"),
        "timeframe": cfg.get("timeframe"),
        "strategy_name": (cfg.get("strategy") or {}).get("name"),
    }


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

    # Build leaderboard table
    rows = []
    for r in results:
        m = r.get("metrics", {}) or {}
        outs = r.get("outputs", {}) or {}
        rows.append(
            {
                "config_path": r.get("config_path"),
                "run_id": r.get("run_id"),
                "name": r.get("name"),
                "symbol": r.get("symbol"),
                "timeframe": r.get("timeframe"),
                "strategy_name": r.get("strategy_name"),
                "n_trades": m.get("n_trades"),
                "total_pnl": m.get("total_pnl"),
                "winrate": m.get("winrate"),
                "avg_pnl": m.get("avg_pnl"),
                "profit_factor": m.get("profit_factor"),
                "profit_factor_is_inf": m.get("profit_factor_is_inf"),
                "run_dir": outs.get("run_dir"),
                "metrics_path": outs.get("metrics"),
                "trades_path": outs.get("trades"),
                "equity_path": outs.get("equity"),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["total_pnl", "winrate", "n_trades"],
        ascending=[False, False, False],
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
        "n_configs": len(config_paths),
        "n_success": len(results),
        "n_errors": len(errors),
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(batch_manifest, indent=2))

    print("Batch done.")
    print(f"Leaderboard: {leaderboard_path}")
    if errors:
        print(f"Errors: {len(errors)} (see {manifest_path})")


if __name__ == "__main__":
    main()
