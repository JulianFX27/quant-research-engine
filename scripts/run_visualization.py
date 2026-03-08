from __future__ import annotations

import argparse

from backtester.robustness.visualization import (
    plot_mc_quantile_paths,
    plot_sensitivity_curve,
    plot_stress_heatmap,
    plot_stress_surface_3d,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate robustness visualizations.")
    ap.add_argument(
        "--kind",
        required=True,
        choices=["sensitivity_curve", "stress_heatmap", "mc_quantiles", "stress_surface_3d"],
        help="Visualization kind",
    )
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--title", default=None, help="Optional title")

    # generic / sensitivity
    ap.add_argument("--x_col", default=None, help="X column")
    ap.add_argument("--y_col", default=None, help="Y column")
    ap.add_argument("--value_col", default=None, help="Value column for heatmap")
    ap.add_argument("--z_col", default=None, help="Z column for 3D surface")

    # stress
    ap.add_argument("--fixed_delay", type=int, default=None, help="Optional fixed delay filter for stress plots")

    args = ap.parse_args()

    if args.kind == "sensitivity_curve":
        out = plot_sensitivity_curve(
            csv_path=args.csv,
            out_path=args.out,
            x_col=args.x_col or "param_value",
            y_col=args.y_col or "expectancy_R",
            title=args.title,
        )

    elif args.kind == "stress_heatmap":
        out = plot_stress_heatmap(
            csv_path=args.csv,
            out_path=args.out,
            x_col=args.x_col or "spread_pips",
            y_col=args.y_col or "slippage_pips",
            value_col=args.value_col or "expectancy_R",
            fixed_delay=args.fixed_delay,
            title=args.title,
        )

    elif args.kind == "stress_surface_3d":
        out = plot_stress_surface_3d(
            csv_path=args.csv,
            out_path=args.out,
            x_col=args.x_col or "spread_pips",
            y_col=args.y_col or "slippage_pips",
            z_col=args.z_col or "expectancy_R",
            fixed_delay=args.fixed_delay,
            title=args.title,
        )

    else:
        out = plot_mc_quantile_paths(
            csv_path=args.csv,
            out_path=args.out,
            title=args.title,
        )

    print(f"WROTE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())