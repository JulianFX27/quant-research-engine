from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from backtester.orchestrator.run import run_from_config
from backtester.robustness.contracts import RobustnessManifest


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _canonical_cfg_json(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a top-level mapping")
    return cfg


def _ensure_dict(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(d or {})


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _build_stress_cfg(
    base_cfg: Dict[str, Any],
    spread_pips: float,
    slippage_pips: float,
    delay_bars: int,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy via json-safe path

    costs = _ensure_dict(cfg.get("costs"))
    costs["spread_pips"] = float(spread_pips)
    costs["slippage_pips"] = float(slippage_pips)
    cfg["costs"] = costs

    execution = _ensure_dict(cfg.get("execution"))
    execution["entry_delay_bars"] = int(delay_bars)
    cfg["execution"] = execution

    base_name = str(cfg.get("name", "stress_run"))
    cfg["name"] = (
        f"{base_name}"
        f"__spr_{spread_pips:.2f}"
        f"__slip_{slippage_pips:.2f}"
        f"__delay_{delay_bars}"
    )

    return cfg


def run_stress_grid(
    base_config_path: str | Path,
    out_root: str | Path = "results/robustness/stress",
    runs_out_dir: str | Path = "results/runs",
    spread_grid: Optional[List[float]] = None,
    slippage_grid: Optional[List[float]] = None,
    delay_grid: Optional[List[int]] = None,
) -> Dict[str, Any]:
    base_cfg = _load_yaml(base_config_path)

    spread_grid = spread_grid or [1.0, 1.2, 1.5]
    slippage_grid = slippage_grid or [0.0, 0.2, 0.5]
    delay_grid = delay_grid or [0, 1]

    stress_id = _utc_now_id()
    out_dir = Path(out_root) / stress_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for spread_pips, slippage_pips, delay_bars in product(spread_grid, slippage_grid, delay_grid):
        cfg = _build_stress_cfg(
            base_cfg=base_cfg,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            delay_bars=delay_bars,
        )

        out = run_from_config(cfg, out_dir=str(runs_out_dir))
        m = out.get("metrics", {}) or {}

        rows.append(
            {
                "stress_id": stress_id,
                "run_id": out.get("run_id"),
                "config_name": cfg.get("name"),
                "spread_pips": float(spread_pips),
                "slippage_pips": float(slippage_pips),
                "delay_bars": int(delay_bars),

                "dataset_id": m.get("dataset_id"),
                "dataset_fp8": m.get("dataset_fp8"),

                "run_status": m.get("run_status"),
                "invalid_eof": m.get("invalid_eof"),

                "n_trades": m.get("n_trades"),
                "valid_trades": m.get("valid_trades"),
                "expectancy_R": _safe_float(m.get("expectancy_R")),
                "avg_R": _safe_float(m.get("avg_R")),
                "winrate_R": _safe_float(m.get("winrate_R")),
                "profit_factor_R": _safe_float(m.get("profit_factor_R")),
                "max_drawdown_R_abs": _safe_float(m.get("max_drawdown_R_abs")),
                "max_drawdown_R_pct": _safe_float(m.get("max_drawdown_R_pct")),
                "max_consecutive_losses_R": _safe_float(m.get("max_consecutive_losses_R")),

                "total_pnl": _safe_float(m.get("total_pnl")),
                "winrate": _safe_float(m.get("winrate")),
                "profit_factor": _safe_float(m.get("profit_factor")),
                "max_drawdown_abs": _safe_float(m.get("max_drawdown_abs")),
                "max_drawdown_pct": _safe_float(m.get("max_drawdown_pct")),

                "execution_policy_id": m.get("execution_policy_id"),
                "execution_fill_mode": m.get("execution_fill_mode"),
                "execution_intrabar_path": m.get("execution_intrabar_path"),
                "execution_intrabar_tie": m.get("execution_intrabar_tie"),
                "costs_spread_pips_effective": _safe_float(m.get("costs_spread_pips_effective")),
                "costs_slippage_pips_effective": _safe_float(m.get("costs_slippage_pips_effective")),
                "execution_entry_delay_bars": m.get("execution_entry_delay_bars"),

                "run_dir": out.get("outputs", {}).get("run_dir"),
                "metrics_path": out.get("outputs", {}).get("metrics"),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["expectancy_R", "total_pnl", "winrate_R"],
        ascending=[False, False, False],
        na_position="last",
    )

    results_csv = out_dir / "stress_grid_results.csv"
    df.to_csv(results_csv, index=False)

    valid_df = df.copy()
    if "invalid_eof" in valid_df.columns:
        valid_df = valid_df[valid_df["invalid_eof"] != True]  # noqa: E712

    summary = {
        "test_type": "stress_grid",
        "stress_id": stress_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config_path": str(base_config_path),
        "base_cfg_hash_sha256": hashlib.sha256(_canonical_cfg_json(base_cfg).encode("utf-8")).hexdigest(),
        "n_runs": int(len(df)),
        "n_valid_runs": int(len(valid_df)),
        "spread_grid": [float(x) for x in spread_grid],
        "slippage_grid": [float(x) for x in slippage_grid],
        "delay_grid": [int(x) for x in delay_grid],
        "best_expectancy_R": _safe_float(valid_df["expectancy_R"].max()) if len(valid_df) else None,
        "worst_expectancy_R": _safe_float(valid_df["expectancy_R"].min()) if len(valid_df) else None,
        "median_expectancy_R": _safe_float(valid_df["expectancy_R"].median()) if len(valid_df) else None,
        "best_total_pnl": _safe_float(valid_df["total_pnl"].max()) if len(valid_df) else None,
        "worst_total_pnl": _safe_float(valid_df["total_pnl"].min()) if len(valid_df) else None,
        "best_config_by_expectancy": (
            valid_df.sort_values(by="expectancy_R", ascending=False).head(1).to_dict(orient="records")[0]
            if len(valid_df) else None
        ),
        "notes": {
            "purpose": "Stress grid evaluates strategy sensitivity to execution friction.",
            "interpretation": "Look for slow edge decay, not just absolute profitability.",
        },
    }

    summary_json = out_dir / "stress_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = RobustnessManifest(
        test_type="stress_grid",
        robustness_id=stress_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_run_dir="",
        source_trades_csv="",
        output_dir=str(out_dir),
        dataset_id=None,
        dataset_fp8=None,
        run_id=None,
        method="grid_search_execution_friction",
        n_paths=int(len(df)),
        block_days=None,
        seed=0,
        r_col="R",
        entry_time_col="entry_time",
        notes={
            "base_config_path": str(base_config_path),
            "runs_out_dir": str(runs_out_dir),
            "results_csv": str(results_csv),
            "summary_json": str(summary_json),
            "grid": {
                "spread_grid": [float(x) for x in spread_grid],
                "slippage_grid": [float(x) for x in slippage_grid],
                "delay_grid": [int(x) for x in delay_grid],
            },
        },
    )

    manifest_path = out_dir / "stress_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    return {
        "stress_id": stress_id,
        "output_dir": str(out_dir),
        "summary": summary,
        "artifacts": {
            "results_csv": str(results_csv),
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_path),
        },
    }