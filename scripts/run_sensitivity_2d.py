from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


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


def _parse_scalar(s: str) -> Any:
    raw = str(s).strip()

    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"

    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except Exception:
        return raw


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _find_new_run_dir(runs_root: Path, before_names: set[str]) -> Path:
    after = [p for p in runs_root.iterdir() if p.is_dir()]
    new_dirs = [p for p in after if p.name not in before_names]
    if not new_dirs:
        raise RuntimeError("No new run directory detected after backtest execution.")
    new_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return new_dirs[0]


def _run_backtest(config_path: str | Path, runs_root: str | Path = "results/runs") -> str:
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    before_names = {p.name for p in runs_root.iterdir() if p.is_dir()}

    result = subprocess.run(
        [sys.executable, "scripts/run_single.py", "--config", str(config_path), "--out", str(runs_root)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    run_dir = _find_new_run_dir(runs_root, before_names)
    return str(run_dir)


def _load_metrics(run_dir: str | Path) -> Dict[str, Any]:
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in run_dir: {run_dir}")

    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="2D sensitivity runner (2-parameter grid).")
    ap.add_argument("--base_config", required=True)
    ap.add_argument("--param_x", required=True)
    ap.add_argument("--values_x", required=True)
    ap.add_argument("--param_y", required=True)
    ap.add_argument("--values_y", required=True)
    ap.add_argument("--out_root", default="results/robustness/sensitivity_2d")
    ap.add_argument("--runs_out_dir", default="results/runs")
    ap.add_argument("--print_summary", action="store_true")
    args = ap.parse_args()

    values_x: List[Any] = [_parse_scalar(v) for v in args.values_x.split(",") if v.strip()]
    values_y: List[Any] = [_parse_scalar(v) for v in args.values_y.split(",") if v.strip()]

    base_cfg = _load_yaml(args.base_config)

    sensitivity_id = _utc_now_id()
    out_dir = Path(args.out_root) / sensitivity_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for x_val, y_val in itertools.product(values_x, values_y):
        cfg = _json_deepcopy(base_cfg)

        _set_deep(cfg, args.param_x, x_val)
        _set_deep(cfg, args.param_y, y_val)

        base_name = str(cfg.get("name", "sensitivity2d_run"))
        cfg["name"] = (
            f"{base_name}"
            f"__{args.param_x.replace('.', '_')}__{x_val}"
            f"__{args.param_y.replace('.', '_')}__{y_val}"
        )

        cfg_path = out_dir / f"{cfg['name']}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        run_dir = _run_backtest(cfg_path, runs_root=args.runs_out_dir)
        metrics = _load_metrics(run_dir)

        rows.append(
            {
                "entry_threshold_pips": x_val if "entry_threshold_pips" in args.param_x else y_val,
                "tp_pips": y_val if "tp_pips" in args.param_y else x_val,
                "param_x": args.param_x,
                "value_x": x_val,
                "param_y": args.param_y,
                "value_y": y_val,
                "expectancy_R": _safe_float(metrics.get("expectancy_R")),
                "avg_R": _safe_float(metrics.get("avg_R")),
                "winrate_R": _safe_float(metrics.get("winrate_R")),
                "profit_factor_R": _safe_float(metrics.get("profit_factor_R")),
                "max_drawdown_R_abs": _safe_float(metrics.get("max_drawdown_R_abs")),
                "max_drawdown_R_pct": _safe_float(metrics.get("max_drawdown_R_pct")),
                "max_consecutive_losses_R": _safe_float(metrics.get("max_consecutive_losses_R")),
                "total_pnl": _safe_float(metrics.get("total_pnl")),
                "run_status": metrics.get("run_status"),
                "invalid_eof": metrics.get("invalid_eof"),
                "n_trades": metrics.get("n_trades"),
                "valid_trades": metrics.get("valid_trades"),
                "dataset_id": metrics.get("dataset_id"),
                "dataset_fp8": metrics.get("dataset_fp8"),
                "run_dir": run_dir,
                "metrics_path": str(Path(run_dir) / "metrics.json"),
                "config_path": str(cfg_path),
            }
        )

    df = pd.DataFrame(rows)
    table_path = out_dir / "sensitivity_2d_table.csv"
    df.to_csv(table_path, index=False)

    valid_df = df.copy()
    if "invalid_eof" in valid_df.columns:
        valid_df = valid_df[valid_df["invalid_eof"] != True]  # noqa: E712

    best_row = None
    if len(valid_df):
        best_row = (
            valid_df.sort_values(by="expectancy_R", ascending=False, na_position="last")
            .head(1)
            .to_dict(orient="records")[0]
        )

    summary = {
        "test_type": "sensitivity_2d",
        "sensitivity_id": sensitivity_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config_path": args.base_config,
        "param_x": args.param_x,
        "values_x": values_x,
        "param_y": args.param_y,
        "values_y": values_y,
        "n_runs": int(len(df)),
        "n_valid_runs": int(len(valid_df)),
        "best_expectancy_R": _safe_float(valid_df["expectancy_R"].max()) if len(valid_df) else None,
        "worst_expectancy_R": _safe_float(valid_df["expectancy_R"].min()) if len(valid_df) else None,
        "median_expectancy_R": _safe_float(valid_df["expectancy_R"].median()) if len(valid_df) else None,
        "best_config": best_row,
        "table_csv": str(table_path),
    }

    summary_path = out_dir / "sensitivity_2d_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.print_summary:
        print(json.dumps(summary, indent=2))

    print(f"OUT_DIR: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())