# analysis/window_sweep.py
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd


RUN_ID_RE = re.compile(r"^RUN_ID:\s*(\S+)\s*$")
RUN_DIR_RE = re.compile(r"^RUN_DIR:\s*(.+)\s*$")


@dataclass
class Window:
    name: str
    start_utc: str
    end_utc: str


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return p.returncode, p.stdout


def parse_run_info(stdout: str) -> tuple[str, Path]:
    rid = None
    rdir = None
    for line in stdout.splitlines():
        m1 = RUN_ID_RE.match(line.strip())
        if m1:
            rid = m1.group(1)
        m2 = RUN_DIR_RE.match(line.strip())
        if m2:
            rdir = Path(m2.group(1).strip())
    if not rid or not rdir:
        raise RuntimeError(
            "Could not parse RUN_ID / RUN_DIR from orchestrator output.\n"
            "Expected lines like:\n"
            "  RUN_ID: <id>\n"
            "  RUN_DIR: results/runs/<id>\n\n"
            f"--- OUTPUT ---\n{stdout}\n--- END ---"
        )
    return rid, rdir


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def safe_get(d: Dict[str, Any], k: str, default=None):
    return d.get(k, default)


def load_baseline_metrics(run_dir: Path) -> Dict[str, Any]:
    # Prefer metrics.json if present; else return empty.
    p = run_dir / "metrics.json"
    if p.exists():
        return read_json(p)
    return {}


def load_gating_report(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "gating_v4" / "gating_v4_compare.json"
    if not p.exists():
        return {}
    return read_json(p)


def summarize_roll12m(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if len(df) == 0:
        return {"roll12m_n_windows": 0}
    # robust stats (mean/min/p5) for expectancy and dd
    out = {
        "roll12m_n_windows": int(len(df)),
        "roll12m_expectancy_mean": float(df["expectancy_R"].mean()),
        "roll12m_expectancy_min": float(df["expectancy_R"].min()),
        "roll12m_expectancy_p5": float(df["expectancy_R"].quantile(0.05)),
        "roll12m_dd_mean": float(df["max_dd_R"].mean()),
        "roll12m_dd_p95": float(df["max_dd_R"].quantile(0.95)),
        "roll12m_trades_mean": float(df["n_trades"].mean()),
    }
    return out


def make_windows_years(start_year: int, end_year: int) -> List[Window]:
    # inclusive years, each window = calendar year slice
    out: List[Window] = []
    for y in range(start_year, end_year + 1):
        out.append(
            Window(
                name=str(y),
                start_utc=f"{y}-01-01T00:00:00Z",
                end_utc=f"{y}-12-31T23:55:00Z",
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Baseline config YAML path")
    ap.add_argument("--out_dir", default="results/diagnostics/window_sweep", help="Where to write consolidated summaries")
    ap.add_argument("--years", default=None, help="Example: 2019-2024 (makes yearly windows)")
    ap.add_argument("--windows_csv", default=None, help="CSV with columns: name,start_utc,end_utc (alternative to --years)")

    # gating args (passed through to gating_compare_v4)
    ap.add_argument("--mc_sims", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--features_path", default="data/anchor_reversion_fx/data/eurusd_m5_features.csv")
    ap.add_argument("--gate_col", default="spread_ny_open_atr")
    ap.add_argument("--gate_lo", type=float, default=-3.067)
    ap.add_argument("--gate_hi", type=float, default=2.664)
    ap.add_argument("--min_trades", type=int, default=50)

    # roll12m
    ap.add_argument("--roll_min_trades", type=int, default=30)

    ap.add_argument("--python", default="python", help="Python executable to use")
    args = ap.parse_args()

    cfg = Path(args.cfg)
    if not cfg.exists():
        raise FileNotFoundError(cfg)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # build windows
    windows: List[Window] = []
    if args.windows_csv:
        p = Path(args.windows_csv)
        if not p.exists():
            raise FileNotFoundError(p)
        wdf = pd.read_csv(p)
        for _, r in wdf.iterrows():
            windows.append(Window(name=str(r["name"]), start_utc=str(r["start_utc"]), end_utc=str(r["end_utc"])))
    elif args.years:
        m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", args.years)
        if not m:
            raise ValueError("--years must be like 2019-2024")
        y0, y1 = int(m.group(1)), int(m.group(2))
        windows = make_windows_years(y0, y1)
    else:
        raise ValueError("Provide --years 2019-2024 or --windows_csv <path>")

    rows: List[Dict[str, Any]] = []

    for w in windows:
        print(f"\n=== WINDOW {w.name} :: {w.start_utc} -> {w.end_utc} ===")

        # 1) run orchestrator slice
        cmd_run = [
            args.python, "-m", "backtester.orchestrator.run", str(cfg),
            "--set", f'dataset_slice.start_utc="{w.start_utc}"',
            "--set", f'dataset_slice.end_utc="{w.end_utc}"',
            "--print-metrics",
        ]
        rc, out = run_cmd(cmd_run)
        if rc != 0:
            rows.append({
                "window": w.name,
                "start_utc": w.start_utc,
                "end_utc": w.end_utc,
                "run_status": "ERROR",
                "error": out[-4000:],  # tail
            })
            continue

        rid, run_dir = parse_run_info(out)
        run_dir = Path(run_dir)

        # 2) gating compare v4
        cmd_gate = [
            args.python, "analysis/gating_compare_v4.py",
            "--run_dir", str(run_dir),
            "--features_path", str(args.features_path),
            "--mc_sims", str(args.mc_sims),
            "--seed", str(args.seed),
            "--gate_col", str(args.gate_col),
            "--gate_lo", str(args.gate_lo),
            "--gate_hi", str(args.gate_hi),
            "--min_trades", str(args.min_trades),
        ]
        rc2, out2 = run_cmd(cmd_gate)
        if rc2 != 0:
            rows.append({
                "window": w.name,
                "start_utc": w.start_utc,
                "end_utc": w.end_utc,
                "run_id": rid,
                "run_dir": str(run_dir),
                "run_status": "OK",
                "gating_status": "ERROR",
                "gating_error": out2[-4000:],
            })
            continue

        # 3) roll12m baseline
        roll_base_out = run_dir / "roll12m.csv"
        cmd_roll_base = [
            args.python, "analysis/roll12m.py",
            "--trades_csv", str(run_dir / "trades.csv"),
            "--out_path", str(roll_base_out),
            "--min_trades", str(args.roll_min_trades),
        ]
        rc3, out3 = run_cmd(cmd_roll_base)
        # 4) roll12m gated
        gated_trades = run_dir / "gating_v4" / "trades_gated.csv"
        roll_gated_out = run_dir / "gating_v4" / "roll12m_gated.csv"
        cmd_roll_gated = [
            args.python, "analysis/roll12m.py",
            "--trades_csv", str(gated_trades),
            "--out_path", str(roll_gated_out),
            "--min_trades", str(args.roll_min_trades),
        ]
        rc4, out4 = run_cmd(cmd_roll_gated)

        # Load reports
        base_metrics = load_baseline_metrics(run_dir)
        gate_report = load_gating_report(run_dir)

        base = safe_get(gate_report, "baseline", {})
        gated = safe_get(gate_report, "gated", {})

        roll_base = summarize_roll12m(roll_base_out) if rc3 == 0 else {"roll12m_error": out3[-2000:]}
        roll_gated = summarize_roll12m(roll_gated_out) if rc4 == 0 else {"roll12m_gated_error": out4[-2000:]}

        row: Dict[str, Any] = {
            "window": w.name,
            "start_utc": w.start_utc,
            "end_utc": w.end_utc,
            "run_id": rid,
            "run_dir": str(run_dir),
            "run_status": safe_get(base_metrics, "run_status", "OK"),
            "dataset_id": safe_get(base_metrics, "dataset_id"),
            "dataset_slice_fp8": safe_get(base_metrics, "dataset_slice_fp8"),
        }

        # Baseline / gated headline from gating report (R-space)
        for prefix, src in [("baseline", base), ("gated_v4", gated)]:
            row[f"{prefix}_n_trades"] = src.get("n_trades")
            row[f"{prefix}_expectancy_R"] = src.get("expectancy_R")
            row[f"{prefix}_total_R"] = src.get("total_R")
            row[f"{prefix}_winrate"] = src.get("winrate")
            row[f"{prefix}_max_dd_R"] = src.get("max_dd_R")
            row[f"{prefix}_max_losing_streak"] = src.get("max_losing_streak")

        # deltas
        if row.get("baseline_expectancy_R") is not None and row.get("gated_v4_expectancy_R") is not None:
            row["delta_expectancy_R"] = float(row["gated_v4_expectancy_R"] - row["baseline_expectancy_R"])
        if row.get("baseline_max_dd_R") is not None and row.get("gated_v4_max_dd_R") is not None:
            row["delta_max_dd_R"] = float(row["gated_v4_max_dd_R"] - row["baseline_max_dd_R"])
        if row.get("baseline_n_trades") and row.get("gated_v4_n_trades"):
            row["coverage_gated_vs_baseline"] = float(row["gated_v4_n_trades"] / max(1, row["baseline_n_trades"]))

        # roll summaries
        for k, v in roll_base.items():
            row[f"baseline_{k}"] = v
        for k, v in roll_gated.items():
            row[f"gated_v4_{k}"] = v

        rows.append(row)

    # write consolidated outputs
    df_out = pd.DataFrame(rows)
    sweep_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = out_root / f"summary_windows_{sweep_id}.csv"
    out_json = out_root / f"summary_windows_{sweep_id}.json"
    df_out.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[OK] Wrote:", out_csv)
    print("[OK] Wrote:", out_json)


if __name__ == "__main__":
    import pandas as pd  # placed here to keep top imports explicit in some environments
    main()
