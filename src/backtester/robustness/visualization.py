from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def plot_sensitivity_curve(
    csv_path: str | Path,
    out_path: str | Path,
    x_col: str = "param_value",
    y_col: str = "expectancy_R",
    title: Optional[str] = None,
) -> str:
    df = pd.read_csv(csv_path)

    if x_col not in df.columns:
        raise ValueError(f"Column not found in sensitivity table: {x_col}")
    if y_col not in df.columns:
        raise ValueError(f"Column not found in sensitivity table: {y_col}")

    x = _coerce_numeric(df[x_col])
    y = _coerce_numeric(df[y_col])

    plot_df = pd.DataFrame({x_col: x, y_col: y}).dropna().sort_values(by=x_col)
    if plot_df.empty:
        raise ValueError("No valid numeric rows available to plot sensitivity curve.")

    out = _ensure_parent(out_path)

    plt.figure(figsize=(9, 5))
    plt.plot(plot_df[x_col], plot_df[y_col], marker="o")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or f"Sensitivity Curve: {y_col} vs {x_col}")
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()

    return str(out)


def plot_stress_heatmap(
    csv_path: str | Path,
    out_path: str | Path,
    x_col: str = "spread_pips",
    y_col: str = "slippage_pips",
    value_col: str = "expectancy_R",
    fixed_delay: Optional[int] = None,
    title: Optional[str] = None,
) -> str:
    df = pd.read_csv(csv_path)

    required = [x_col, y_col, value_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column not found in stress grid results: {c}")

    if fixed_delay is not None:
        if "delay_bars" not in df.columns:
            raise ValueError("delay_bars column not found in stress grid results.")
        df = df[df["delay_bars"] == fixed_delay].copy()

    if df.empty:
        raise ValueError("No rows available after applying stress heatmap filters.")

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col, value_col])

    if df.empty:
        raise ValueError("No valid numeric rows available to plot stress heatmap.")

    pivot = df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    out = _ensure_parent(out_path)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")

    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(y) for y in pivot.index])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if fixed_delay is None:
        plt.title(title or f"Stress Heatmap: {value_col}")
    else:
        plt.title(title or f"Stress Heatmap: {value_col} (delay_bars={fixed_delay})")

    cbar = plt.colorbar(im)
    cbar.set_label(value_col)

    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()

    return str(out)


def plot_mc_quantile_paths(
    csv_path: str | Path,
    out_path: str | Path,
    step_col: str = "step",
    quantile_cols: Optional[list[str]] = None,
    title: Optional[str] = None,
) -> str:
    df = pd.read_csv(csv_path)

    if step_col not in df.columns:
        raise ValueError(f"Column not found in MC quantiles file: {step_col}")

    if quantile_cols is None:
        if {"p05_cum_R", "p25_cum_R", "p50_cum_R", "p75_cum_R", "p95_cum_R"}.issubset(df.columns):
            quantile_cols = ["p05_cum_R", "p25_cum_R", "p50_cum_R", "p75_cum_R", "p95_cum_R"]
        elif {"p05", "p25", "p50", "p75", "p95"}.issubset(df.columns):
            quantile_cols = ["p05", "p25", "p50", "p75", "p95"]
        else:
            raise ValueError("Could not infer quantile columns from MC quantiles file.")

    for c in quantile_cols:
        if c not in df.columns:
            raise ValueError(f"Quantile column not found in MC quantiles file: {c}")

    plot_df = df[[step_col] + quantile_cols].copy()
    plot_df[step_col] = pd.to_numeric(plot_df[step_col], errors="coerce")
    for c in quantile_cols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    plot_df = plot_df.dropna()

    if plot_df.empty:
        raise ValueError("No valid numeric rows available to plot MC quantile paths.")

    out = _ensure_parent(out_path)

    plt.figure(figsize=(10, 6))
    for c in quantile_cols:
        plt.plot(plot_df[step_col], plot_df[c], label=c)

    plt.xlabel(step_col)
    plt.ylabel("cumulative_R" if "cum_R" in quantile_cols[0] else "value")
    plt.title(title or "Monte Carlo Quantile Paths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()

    return str(out)


def plot_stress_surface_3d(
    csv_path: str | Path,
    out_path: str | Path,
    x_col: str = "spread_pips",
    y_col: str = "slippage_pips",
    z_col: str = "expectancy_R",
    fixed_delay: Optional[int] = None,
    title: Optional[str] = None,
) -> str:
    df = pd.read_csv(csv_path)

    required = [x_col, y_col, z_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column not found in stress grid results: {c}")

    if fixed_delay is not None:
        if "delay_bars" not in df.columns:
            raise ValueError("delay_bars column not found in stress grid results.")
        df = df[df["delay_bars"] == fixed_delay].copy()

    if df.empty:
        raise ValueError("No rows available after applying stress surface filters.")

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[z_col] = pd.to_numeric(df[z_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col, z_col])

    if df.empty:
        raise ValueError("No valid numeric rows available to plot stress surface.")

    pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    x_vals = pivot.columns.to_numpy(dtype=float)
    y_vals = pivot.index.to_numpy(dtype=float)

    if len(x_vals) < 2 or len(y_vals) < 2:
        raise ValueError("Need at least a 2x2 grid to render a surface.")

    X, Y = __import__("numpy").meshgrid(x_vals, y_vals)
    Z = pivot.to_numpy(dtype=float)

    out = _ensure_parent(out_path)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)

    if fixed_delay is None:
        ax.set_title(title or f"Stress Surface 3D: {z_col}")
    else:
        ax.set_title(title or f"Stress Surface 3D: {z_col} (delay_bars={fixed_delay})")

    fig.colorbar(surf, shrink=0.7, aspect=12, pad=0.08)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return str(out)