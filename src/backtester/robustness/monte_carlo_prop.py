from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from backtester.robustness.contracts import RobustnessManifest


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _copy_if_exists(src: Path, dst: Path) -> Optional[str]:
    if not src.exists():
        return None
    dst.write_bytes(src.read_bytes())
    return str(dst)


def run_prop_mc(
    run_dir: str | Path,
    out_root: str | Path = "results/robustness/mc_prop",
    mode: str = "risk_ramp",
    n_paths: int = 20000,
    block_days: int = 3,
    seed: int = 42,
    policy: str = "fixed_0.0075",
    cap_days: int = 300,
    max_days: int = 30,
    target_pct: float = 0.10,
    max_total_dd_pct: float = 0.10,
    max_daily_dd_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Canonical wrapper for prop Monte Carlo tools.

    Supported modes:
    - risk_ramp
    - block
    - two_step_time_to_target
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    run_manifest = _load_json_if_exists(run_dir / "run_manifest.json")
    run_metrics = _load_json_if_exists(run_dir / "metrics.json")

    robustness_id = _utc_now_id()
    out_dir = Path(out_root) / robustness_id
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]

    if mode == "risk_ramp":
        script_path = repo_root / "src" / "paper" / "mc" / "run_ftmo_mc_risk_ramp.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--run_dir", str(run_dir),
            "--n_paths", str(n_paths),
            "--block_days", str(block_days),
            "--seed", str(seed),
            "--max_days", str(max_days),
            "--target_pct", str(target_pct),
            "--max_total_dd_pct", str(max_total_dd_pct),
            "--max_daily_dd_pct", str(max_daily_dd_pct),
        ]
        produced_dir = run_dir / "mc_ftmo_ramp"

    elif mode == "block":
        script_path = repo_root / "src" / "paper" / "mc" / "run_ftmo_mc_block.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--run_dir", str(run_dir),
            "--n_paths", str(n_paths),
            "--block_days", str(block_days),
            "--seed", str(seed),
            "--max_days", str(max_days),
            "--target_pct", str(target_pct),
            "--max_total_dd_pct", str(max_total_dd_pct),
            "--max_daily_dd_pct", str(max_daily_dd_pct),
            "--risk_pcts", "0.005,0.0075",
        ]
        produced_dir = run_dir / "mc_ftmo"

    elif mode == "two_step_time_to_target":
        script_path = repo_root / "src" / "paper" / "mc" / "run_ftmo_mc_2step_time_to_target.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--run_dir", str(run_dir),
            "--n_paths", str(n_paths),
            "--block_days", str(block_days),
            "--cap_days", str(cap_days),
            "--seed", str(seed),
            "--policy", str(policy),
        ]
        produced_dir = run_dir / "mc_ftmo_2step_time"

    else:
        raise ValueError("mode must be one of: risk_ramp, block, two_step_time_to_target")

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    stdout_path = out_dir / "stdout.txt"
    stderr_path = out_dir / "stderr.txt"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Prop Monte Carlo failed.\n"
            f"mode={mode}\n"
            f"returncode={proc.returncode}\n"
            f"stderr={proc.stderr}"
        )

    copied_artifacts: Dict[str, Optional[str]] = {}

    if mode == "risk_ramp":
        copied_artifacts["mc_ramp_all_json"] = _copy_if_exists(
            produced_dir / "mc_ramp_all.json",
            out_dir / "mc_ramp_all.json",
        )

    elif mode == "block":
        copied_artifacts["mc_summary_all_json"] = _copy_if_exists(
            produced_dir / "mc_summary_all.json",
            out_dir / "mc_summary_all.json",
        )

    elif mode == "two_step_time_to_target":
        summary_file = produced_dir / f"summary__{policy}.json"
        copied_artifacts["summary_json"] = _copy_if_exists(
            summary_file,
            out_dir / summary_file.name,
        )

    manifest = RobustnessManifest(
        test_type="monte_carlo_prop",
        robustness_id=robustness_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_run_dir=str(run_dir),
        source_trades_csv=str(run_dir / "trades.csv"),
        output_dir=str(out_dir),
        dataset_id=run_metrics.get("dataset_id"),
        dataset_fp8=run_metrics.get("dataset_fp8"),
        run_id=run_manifest.get("run_id"),
        method=mode,
        n_paths=int(n_paths),
        block_days=int(block_days),
        seed=int(seed),
        r_col="R",
        entry_time_col="entry_time",
        notes={
            "mode": mode,
            "policy": policy if mode == "two_step_time_to_target" else None,
            "max_days": max_days if mode in {"risk_ramp", "block"} else None,
            "cap_days": cap_days if mode == "two_step_time_to_target" else None,
            "target_pct": target_pct if mode in {"risk_ramp", "block"} else None,
            "max_total_dd_pct": max_total_dd_pct if mode in {"risk_ramp", "block"} else None,
            "max_daily_dd_pct": max_daily_dd_pct if mode in {"risk_ramp", "block"} else None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "produced_dir_original": str(produced_dir),
            "rationale": "Prop Monte Carlo measures operational survivability under prop firm constraints.",
        },
    )

    manifest_path = out_dir / "mc_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    summary_preview: Dict[str, Any] = {}
    for artifact_path in copied_artifacts.values():
        if artifact_path:
            p = Path(artifact_path)
            if p.suffix.lower() == ".json":
                summary_preview[p.name] = _load_json_if_exists(p)

    return {
        "robustness_id": robustness_id,
        "output_dir": str(out_dir),
        "mode": mode,
        "artifacts": {
            **copied_artifacts,
            "manifest_json": str(manifest_path),
            "stdout_txt": str(stdout_path),
            "stderr_txt": str(stderr_path),
        },
        "summary_preview": summary_preview,
    }