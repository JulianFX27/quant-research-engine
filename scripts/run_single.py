from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from backtester.orchestrator.run import run_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to run config YAML")
    ap.add_argument("--out", default="results/run", help="Output directory")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    res = run_from_config(cfg, out_dir=args.out)
    print("Done.")
    print(res["metrics"])


if __name__ == "__main__":
    main()
