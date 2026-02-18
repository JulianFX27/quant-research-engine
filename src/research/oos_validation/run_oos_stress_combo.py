from __future__ import annotations

import json
import shutil
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple


FREEZE_CFG = "configs/frozen/anchor_mr_pure_8p__gating_v6__freeze_v1.yaml"
IS_OVERRIDE = "configs/experiments/split_is_2019_2022.yaml"
OOS_OVERRIDE = "configs/experiments/split_oos_2023_2024__stress_costs_delay1.yaml"

ACCEPT = {
    "expectancy_R_min": 0.0,     # strictly > 0
    "profit_factor_min": 1.0,    # strictly > 1
    "max_drawdown_pct_max": 12.0
}
MIN_TRADES = 50

OUT_ROOT = Path("results/oos_validation")


def run_once(config_path: str) -> Path:
    tr_start, tr_end = load_time_range_from_config(config_path)

    env = os.environ.copy()
    if tr_start:
        env["QRE_TIME_RANGE_START"] = tr_start
    else:
        env.pop("QRE_TIME_RANGE_START", None)

    if tr_end:
        env["QRE_TIME_RANGE_END"] = tr_end
    else:
        env.pop("QRE_TIME_RANGE_END", None)

    cmd = [
        sys.executable,
        "-m",
        "src.backtester.orchestrator.run",
        config_path,  # positional arg (your CLI)
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed:\n"
            f"CMD: {' '.join(str(x) for x in cmd)}\n"
            f"ENV: QRE_TIME_RANGE_START={env.get('QRE_TIME_RANGE_START')} "
            f"QRE_TIME_RANGE_END={env.get('QRE_TIME_RANGE_END')}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    run_dir = infer_latest_run_dir(Path("results/runs"))
    if run_dir is None:
        raise RuntimeError("Could not infer latest run directory under results/runs")

    # STRICT ASSERT: produced trades must lie inside time_range
    assert_run_is_within_timerange(run_dir, tr_start, tr_end)

    return run_dir


def infer_latest_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {run_dir}")
    return load_json(metrics_path)


def summarize_primary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "expectancy_R",
        "profit_factor",
        "max_drawdown_pct",
        "n_trades",
        "max_consecutive_losses",
    ]
    return {k: metrics[k] for k in keys if k in metrics}


def acceptance_check(metrics: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    required = ["expectancy_R", "profit_factor", "max_drawdown_pct", "n_trades"]
    missing = [k for k in required if k not in metrics]

    result: Dict[str, Any] = {"missing_keys": missing, "checks": {}, "pass": False}
    if missing:
        result["checks"]["schema"] = f"FAIL (missing: {missing})"
        return False, result

    exp = float(metrics["expectancy_R"])
    pf = float(metrics["profit_factor"])
    dd = float(metrics["max_drawdown_pct"])
    n = int(metrics["n_trades"])

    result["checks"]["expectancy_R"] = "PASS" if exp > ACCEPT["expectancy_R_min"] else f"FAIL (={exp})"
    result["checks"]["profit_factor"] = "PASS" if pf > ACCEPT["profit_factor_min"] else f"FAIL (={pf})"
    result["checks"]["max_drawdown_pct"] = "PASS" if dd <= ACCEPT["max_drawdown_pct_max"] else f"FAIL (={dd})"
    result["checks"]["n_trades"] = "PASS" if n >= MIN_TRADES else f"FAIL (={n} < {MIN_TRADES})"

    passed = all(v == "PASS" for v in result["checks"].values())
    result["pass"] = passed
    return passed, result


def bundle_run(run_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["metrics.json", "run_manifest.json", "trades.csv", "equity.csv"]:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, dest_dir / fname)


def write_report(bundle_dir: Path, report: Dict[str, Any]) -> None:
    with (bundle_dir / "oos_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def load_time_range_from_config(config_path: str) -> Tuple[str | None, str | None]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML not installed. Install it (pip install pyyaml).") from e

    p = Path(config_path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    tr = (cfg.get("data") or {}).get("time_range") or {}
    start = tr.get("start")
    end = tr.get("end")
    return start, end


def _parse_utc(s: str):
    import pandas as pd
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid time_range timestamp: {s}")
    return ts


def assert_run_is_within_timerange(run_dir: Path, start: str | None, end: str | None) -> None:
    """
    Hard guard: if produced trades are outside requested time_range, the run is NOT a valid IS/OOS slice.
    This protects us from false OOS.
    """
    if not start and not end:
        return  # nothing to assert

    import pandas as pd

    p = run_dir / "trades.csv"
    if not p.exists():
        return  # no trades, allowed (but will fail MIN_TRADES anyway)

    df = pd.read_csv(p)
    if len(df) == 0:
        return

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")

    tmin = min(df["entry_time"].min(), df["exit_time"].min())
    tmax = max(df["entry_time"].max(), df["exit_time"].max())

    s = _parse_utc(start) if start else None
    e = _parse_utc(end) if end else None

    if s is not None and tmin < s:
        raise RuntimeError(
            f"STRICT OOS ASSERT FAILED: trades earlier than start.\n"
            f"start={start}, min_trade_time={tmin}, run_dir={run_dir}"
        )

    if e is not None and tmax > e + pd.Timedelta(days=1) - pd.Timedelta(minutes=1):
        # allow inclusive end date; treat end as end-of-day UTC
        raise RuntimeError(
            f"STRICT OOS ASSERT FAILED: trades later than end.\n"
            f"end={end}, max_trade_time={tmax}, run_dir={run_dir}"
        )


def merge_yaml_files(base: Path, override: Path, out_path: Path) -> None:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML not installed. Install it (pip install pyyaml).") from e

    with base.open("r", encoding="utf-8") as f:
        b = yaml.safe_load(f) or {}
    with override.open("r", encoding="utf-8") as f:
        o = yaml.safe_load(f) or {}

    merged = deep_merge(b, o)

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"anchor_mr_pure_8p_gating_v6_oos_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    is_cfg_path = out_dir / "cfg_is_2019_2022_merged.yaml"
    oos_cfg_path = out_dir / "cfg_oos_2023_2024_merged.yaml"
    merge_yaml_files(Path(FREEZE_CFG), Path(IS_OVERRIDE), is_cfg_path)
    merge_yaml_files(Path(FREEZE_CFG), Path(OOS_OVERRIDE), oos_cfg_path)

    run_dir_is = run_once(str(is_cfg_path))
    metrics_is = get_metrics(run_dir_is)

    run_dir_oos = run_once(str(oos_cfg_path))
    metrics_oos = get_metrics(run_dir_oos)

    passed, checks = acceptance_check(metrics_oos)

    report: Dict[str, Any] = {
        "freeze_config": FREEZE_CFG,
        "is_override": IS_OVERRIDE,
        "oos_override": OOS_OVERRIDE,
        "is_run_dir": str(run_dir_is),
        "oos_run_dir": str(run_dir_oos),
        "is_primary": summarize_primary(metrics_is),
        "oos_primary": summarize_primary(metrics_oos),
        "oos_acceptance": checks,
        "verdict": "PASS" if passed else "FAIL",
        "policy": {
            "oos_strict": True,
            "split": {"is": ["2019-01-01", "2022-12-31"], "oos": ["2023-01-01", "2024-12-31"]},
            "no_recalibration": True,
            "no_selection": True,
            "freeze_untouched": True,
            "min_trades": MIN_TRADES,
            "criteria": {
                "expectancy_R": "> 0",
                "profit_factor": "> 1",
                "max_drawdown_pct": "<= 12",
                "n_trades": f">= {MIN_TRADES}",
            },
            "interpreter": str(sys.executable),
            "env_injection": {
                "QRE_TIME_RANGE_START": "from merged YAML data.time_range.start",
                "QRE_TIME_RANGE_END": "from merged YAML data.time_range.end",
            },
            "strict_assert_timerange": True,
        },
    }

    bundle_run(run_dir_is, out_dir / "IS_2019_2022")
    bundle_run(run_dir_oos, out_dir / "OOS_2023_2024")
    write_report(out_dir, report)

    print(f"[OOS] Python: {sys.executable}")
    print(f"[OOS] Bundle: {out_dir}")
    print(f"[OOS] Verdict: {report['verdict']}")
    if report["oos_acceptance"]["missing_keys"]:
        print(f"[OOS] Missing metric keys: {report['oos_acceptance']['missing_keys']}")
    else:
        for k, v in report["oos_acceptance"]["checks"].items():
            print(f"[OOS] {k}: {v}")


if __name__ == "__main__":
    main()
