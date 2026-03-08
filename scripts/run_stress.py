from __future__ import annotations

import argparse
import json

from backtester.robustness.stress_grid import run_stress_grid


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run canonical stress grid from a base config.")
    ap.add_argument("--base_config", required=True, help="Path to base YAML config")
    ap.add_argument("--out_root", default="results/robustness/stress", help="Output root")
    ap.add_argument("--runs_out_dir", default="results/runs", help="Where underlying runs are stored")
    ap.add_argument("--spread_grid", default="1.0,1.2,1.5", help="Comma-separated spread values")
    ap.add_argument("--slippage_grid", default="0.0,0.2,0.5", help="Comma-separated slippage values")
    ap.add_argument("--delay_grid", default="0,1", help="Comma-separated delay values")
    ap.add_argument("--print_summary", action="store_true", help="Print summary JSON")
    args = ap.parse_args()

    out = run_stress_grid(
        base_config_path=args.base_config,
        out_root=args.out_root,
        runs_out_dir=args.runs_out_dir,
        spread_grid=_parse_float_list(args.spread_grid),
        slippage_grid=_parse_float_list(args.slippage_grid),
        delay_grid=_parse_int_list(args.delay_grid),
    )

    print(f"STRESS_ID: {out['stress_id']}")
    print(f"OUT_DIR: {out['output_dir']}")

    if args.print_summary:
        print(json.dumps(out["summary"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())