from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import json
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    config_paths = sorted(config_dir.glob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No yaml configs found in: {config_dir}")

    rows: list[dict] = []

    for i, cfg_path in enumerate(config_paths, start=1):
        print(f"[{i}/{len(config_paths)}] running {cfg_path.name}")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.run_ny_opening_momentum_v2",
                "--config",
                str(cfg_path),
            ],
            capture_output=True,
            text=True,
        )

        row = {
            "config_name": cfg_path.stem,
            "config_path": str(cfg_path),
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
        }

        if proc.returncode != 0:
            row["stderr"] = proc.stderr[-2000:]
            rows.append(row)
            continue

        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)

            metrics_path = Path(raw["outputs"]["out_dir"]) / "metrics.json"
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            row.update(
                {
                    "threshold_q": raw["signal"]["threshold_q"],
                    "impulse_efficiency_min": raw["signal"]["impulse_efficiency_min"],
                    "entry_delay_min": raw["signal"]["entry_delay_min"],
                    "holding_min": raw["signal"]["holding_min"],
                    **metrics,
                }
            )
        except Exception as e:
            row["status"] = "error"
            row["stderr"] = f"post-run parse error: {e}"

        rows.append(row)

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["status", "sharpe_per_trade", "final_equity_ret", "n_trades"] if c in df.columns]
    ascending = [True, False, False, False][: len(sort_cols)]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)

    df.to_csv(output_csv, index=False)

    print(f"\nSaved batch results: {output_csv}")
    if "status" in df.columns:
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
