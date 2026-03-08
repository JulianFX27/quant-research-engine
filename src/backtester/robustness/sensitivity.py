from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from backtester.orchestrator.run import run_from_config
from backtester.robustness.contracts import RobustnessManifest


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a top-level mapping")
    return cfg


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _json_deepcopy(obj: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(obj))


def _set_deep(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in str(dotted_key).split(".") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")

    cur = cfg
    for p in parts[:-1]:
        if p not in cur or cur[p] is None:
            cur[p] = {}
        if not isinstance(cur[p], dict):
            raise ValueError(
                f"Cannot set '{dotted_key}': '{p}' is not a dict "
                f"(found {type(cur[p]).__name__})."
            )
        cur = cur[p]

    cur[parts[-1]] = value


def _get_deep(cfg: Dict[str, Any], dotted_key: str) -> Any:
    parts = [p for p in str(dotted_key).split(".") if p.strip()]
    cur: Any = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _normalize_values(values: List[str]) -> List[Any]:
    out: List[Any] = []
    for x in values:
        s = str(x).strip()
        if s.lower() in {"true", "false"}:
            out.append(s.lower() == "true")
            continue
        try:
            if "." in s:
                out.append(float(s))
            else:
                out.append(int(s))
            continue
        except Exception:
            pass
        out.append(s)
    return out


def run_sensitivity(
    base_config_path: str | Path,
    param_key: str,
    values: List[Any],
    out_root: str | Path = "results/robustness/sensitivity",
    runs_out_dir: str | Path = "results/runs",
) -> Dict[str, Any]:
    """
    One-factor sensitivity runner.

    Example:
        param_key = "strategy.params.entry_threshold_pips"
        values = [6, 7, 8, 9, 10]
    """
    if not values:
        raise ValueError("values must contain at least one element")

    base_cfg = _load_yaml(base_config_path)
    baseline_value = _get_deep(base_cfg, param_key)

    sens_id = _utc_now_id()
    out_dir = Path(out_root) / sens_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for value in values:
        cfg = _json_deepcopy(base_cfg)
        _set_deep(cfg, param_key, value)

        base_name = str(cfg.get("name", "sensitivity_run"))
        cfg["name"] = f"{base_name}__{param_key.replace('.', '_')}__{value}"

        out = run_from_config(cfg, out_dir=str(runs_out_dir))
        m = out.get("metrics", {}) or {}

        rows.append(
            {
                "sensitivity_id": sens_id,
                "param_key": param_key,
                "param_value": value,
                "baseline_value": baseline_value,

                "run_id": out.get("run_id"),
                "config_name": cfg.get("name"),

                "dataset_id": m.get("dataset_id"),
                "dataset_fp8": m.get("dataset_fp8"),

                "run_status": m.get("run_status"),
                "invalid_eof": m.get("invalid_eof"),

                "n_trades": m.get("n_trades"),
                "valid_trades": m.get("valid_trades"),

                "expectancy_R": _safe_float(m.get("expectancy_R")),
                "avg_R": _safe_float(m.get("avg_R")),
                "median_R": _safe_float(m.get("median_R")),
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

                "run_dir": out.get("outputs", {}).get("run_dir"),
                "metrics_path": out.get("outputs", {}).get("metrics"),
            }
        )

    df = pd.DataFrame(rows)

    sort_col = "param_value"
    if sort_col in df.columns:
        try:
            df = df.sort_values(by=[sort_col], ascending=True, na_position="last")
        except Exception:
            pass

    table_csv = out_dir / "sensitivity_table.csv"
    df.to_csv(table_csv, index=False)

    valid_df = df.copy()
    if "invalid_eof" in valid_df.columns:
        valid_df = valid_df[valid_df["invalid_eof"] != True]  # noqa: E712

    best_row = None
    worst_row = None
    if len(valid_df):
        valid_sorted = valid_df.sort_values(by="expectancy_R", ascending=False, na_position="last")
        best_row = valid_sorted.head(1).to_dict(orient="records")[0]
        worst_row = valid_sorted.tail(1).to_dict(orient="records")[0]

    exp_max = _safe_float(valid_df["expectancy_R"].max()) if len(valid_df) else None
    exp_min = _safe_float(valid_df["expectancy_R"].min()) if len(valid_df) else None
    exp_median = _safe_float(valid_df["expectancy_R"].median()) if len(valid_df) else None

    stability_range = None
    if exp_max is not None and exp_min is not None:
        stability_range = float(exp_max - exp_min)

    stability_ratio = None
    if exp_median is not None and exp_max is not None and exp_max not in (0, None):
        stability_ratio = float(exp_median / exp_max)

    summary = {
        "test_type": "sensitivity",
        "sensitivity_id": sens_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config_path": str(base_config_path),
        "param_key": param_key,
        "baseline_value": baseline_value,
        "values_tested": values,
        "n_runs": int(len(df)),
        "n_valid_runs": int(len(valid_df)),
        "best_expectancy_R": exp_max,
        "worst_expectancy_R": exp_min,
        "median_expectancy_R": exp_median,
        "stability_range_expectancy_R": stability_range,
        "stability_ratio_expectancy_R": stability_ratio,
        "best_config_by_expectancy": best_row,
        "worst_config_by_expectancy": worst_row,
        "notes": {
            "purpose": "One-factor sensitivity measures structural robustness around a single parameter.",
            "interpretation": "Look for a stable plateau, not a single narrow optimum.",
        },
    }

    summary_json = out_dir / "sensitivity_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = RobustnessManifest(
        test_type="sensitivity",
        robustness_id=sens_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_run_dir="",
        source_trades_csv="",
        output_dir=str(out_dir),
        dataset_id=None,
        dataset_fp8=None,
        run_id=None,
        method="one_factor_sensitivity",
        n_paths=int(len(df)),
        block_days=None,
        seed=0,
        r_col="R",
        entry_time_col="entry_time",
        notes={
            "base_config_path": str(base_config_path),
            "runs_out_dir": str(runs_out_dir),
            "param_key": param_key,
            "baseline_value": baseline_value,
            "values_tested": values,
            "table_csv": str(table_csv),
            "summary_json": str(summary_json),
        },
    )

    manifest_json = out_dir / "sensitivity_manifest.json"
    manifest_json.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    return {
        "sensitivity_id": sens_id,
        "output_dir": str(out_dir),
        "summary": summary,
        "artifacts": {
            "table_csv": str(table_csv),
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_json),
        },
    }