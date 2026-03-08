from __future__ import annotations

import argparse
import json

from backtester.robustness.sensitivity import run_sensitivity


def _parse_values(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run canonical one-factor sensitivity from a base config.")
    ap.add_argument("--base_config", required=True, help="Path to base YAML config")
    ap.add_argument("--param_key", required=True, help="Dotted config key to vary")
    ap.add_argument(
        "--values",
        required=True,
        help='Comma-separated values. Example: "6,7,8,9,10"',
    )
    ap.add_argument(
        "--out_root",
        default="results/robustness/sensitivity",
        help="Output root",
    )
    ap.add_argument(
        "--runs_out_dir",
        default="results/runs",
        help="Where underlying runs are stored",
    )
    ap.add_argument("--print_summary", action="store_true", help="Print summary JSON")
    args = ap.parse_args()

    out = run_sensitivity(
        base_config_path=args.base_config,
        param_key=args.param_key,
        values=_parse_values(args.values),
        out_root=args.out_root,
        runs_out_dir=args.runs_out_dir,
    )

    print(f"SENSITIVITY_ID: {out['sensitivity_id']}")
    print(f"OUT_DIR: {out['output_dir']}")

    if args.print_summary:
        print(json.dumps(out["summary"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())