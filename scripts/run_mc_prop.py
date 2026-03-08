from __future__ import annotations

import argparse
import json

from backtester.robustness.monte_carlo_prop import run_prop_mc


def main() -> int:
    ap = argparse.ArgumentParser(description="Run canonical Prop Monte Carlo wrapper from a completed run directory.")
    ap.add_argument("--run_dir", required=True, help="Path to results/runs/<run_id>")
    ap.add_argument(
        "--out_root",
        default="results/robustness/mc_prop",
        help="Root directory for Prop Monte Carlo outputs",
    )
    ap.add_argument(
        "--mode",
        default="risk_ramp",
        choices=["risk_ramp", "block", "two_step_time_to_target"],
        help="Prop Monte Carlo mode",
    )
    ap.add_argument("--n_paths", type=int, default=20000, help="Number of Monte Carlo paths")
    ap.add_argument("--block_days", type=int, default=3, help="Block size in trading days")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--policy", type=str, default="fixed_0.0075", help="Policy for two_step_time_to_target mode")
    ap.add_argument("--cap_days", type=int, default=300, help="Cap days for two_step_time_to_target mode")
    ap.add_argument("--max_days", type=int, default=30, help="Max days for risk_ramp or block mode")
    ap.add_argument("--target_pct", type=float, default=0.10, help="Target pct for risk_ramp or block mode")
    ap.add_argument("--max_total_dd_pct", type=float, default=0.10, help="Max total drawdown pct")
    ap.add_argument("--max_daily_dd_pct", type=float, default=0.05, help="Max daily drawdown pct")
    ap.add_argument("--print_summary", action="store_true", help="Print summary preview JSON")
    args = ap.parse_args()

    out = run_prop_mc(
        run_dir=args.run_dir,
        out_root=args.out_root,
        mode=args.mode,
        n_paths=args.n_paths,
        block_days=args.block_days,
        seed=args.seed,
        policy=args.policy,
        cap_days=args.cap_days,
        max_days=args.max_days,
        target_pct=args.target_pct,
        max_total_dd_pct=args.max_total_dd_pct,
        max_daily_dd_pct=args.max_daily_dd_pct,
    )

    print(f"MC_PROP_ID: {out['robustness_id']}")
    print(f"OUT_DIR: {out['output_dir']}")
    print(f"MODE: {out['mode']}")

    if args.print_summary:
        print(json.dumps(out["summary_preview"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())