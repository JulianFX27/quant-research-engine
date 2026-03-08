from __future__ import annotations

import argparse
import json

from backtester.robustness.monte_carlo_strategy import run_strategy_mc


def main() -> int:
    ap = argparse.ArgumentParser(description="Run canonical Strategy Monte Carlo from a completed run directory.")
    ap.add_argument("--run_dir", required=True, help="Path to results/runs/<run_id>")
    ap.add_argument(
        "--out_root",
        default="results/robustness/mc_strategy",
        help="Root directory for Monte Carlo outputs",
    )
    ap.add_argument(
        "--method",
        default="block_days",
        choices=["iid", "block_days", "day_bootstrap"],
        help="Resampling method",
    )
    ap.add_argument("--n_paths", type=int, default=20000, help="Number of Monte Carlo paths")
    ap.add_argument("--block_days", type=int, default=3, help="Block size in trading days for block bootstrap")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--print_summary", action="store_true", help="Print summary JSON")
    args = ap.parse_args()

    out = run_strategy_mc(
        run_dir=args.run_dir,
        out_root=args.out_root,
        method=args.method,
        n_paths=args.n_paths,
        block_days=args.block_days,
        seed=args.seed,
    )

    print(f"MC_ID: {out['robustness_id']}")
    print(f"OUT_DIR: {out['output_dir']}")

    if args.print_summary:
        print(json.dumps(out["summary"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())