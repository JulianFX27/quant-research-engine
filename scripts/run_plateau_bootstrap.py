from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _load_grid_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Grid table not found: {p}")
    df = pd.read_csv(p)
    required = {"entry_threshold_pips", "tp_pips", "run_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in grid table: {sorted(missing)}")
    return df


def _load_daily_r_from_run(run_dir: str | Path) -> pd.Series:
    trades_path = Path(run_dir) / "trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades.csv in run_dir: {run_dir}")

    df = pd.read_csv(trades_path)
    if "entry_time" not in df.columns:
        raise ValueError(f"trades.csv missing entry_time: {trades_path}")
    if "R" not in df.columns:
        raise ValueError(f"trades.csv missing R column: {trades_path}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["R"] = pd.to_numeric(df["R"], errors="coerce")
    df = df.dropna(subset=["entry_time", "R"]).copy()
    if df.empty:
        raise ValueError(f"No valid rows in trades.csv: {trades_path}")

    df["day"] = df["entry_time"].dt.date.astype(str)
    daily_r = df.groupby("day")["R"].sum().sort_index()
    return daily_r


def _build_daily_matrix(grid_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    series_by_cfg: Dict[str, pd.Series] = {}
    meta: List[Dict[str, Any]] = []

    for _, row in grid_df.iterrows():
        thr = row["entry_threshold_pips"]
        tp = row["tp_pips"]
        run_dir = row["run_dir"]

        cfg_id = f"thr={thr}|tp={tp}"
        daily_r = _load_daily_r_from_run(run_dir)

        series_by_cfg[cfg_id] = daily_r
        meta.append(
            {
                "cfg_id": cfg_id,
                "entry_threshold_pips": thr,
                "tp_pips": tp,
                "run_dir": run_dir,
                "n_days_with_trades": int(len(daily_r)),
            }
        )

    mat = pd.DataFrame(series_by_cfg).sort_index()
    mat = mat.fillna(0.0)
    return mat, meta


def _sample_block_indices(n: int, out_len: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be > 0")
    block = max(1, int(block_size))
    idxs: List[int] = []
    while len(idxs) < out_len:
        start = int(rng.integers(0, n))
        for i in range(block):
            if len(idxs) >= out_len:
                break
            idxs.append((start + i) % n)
    return np.array(idxs, dtype=int)


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap test for 2D parameter plateau robustness.")
    ap.add_argument("--grid_csv", required=True, help="Path to sensitivity_2d_table.csv")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--block_days", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default="results/robustness/plateau_bootstrap")
    ap.add_argument("--print_summary", action="store_true")
    args = ap.parse_args()

    grid_df = _load_grid_table(args.grid_csv)
    daily_mat, meta = _build_daily_matrix(grid_df)

    boot_id = _utc_now_id()
    out_dir = Path(args.out_root) / boot_id
    out_dir.mkdir(parents=True, exist_ok=True)

    n_days = len(daily_mat)
    cfg_ids = list(daily_mat.columns)
    arr = daily_mat.to_numpy(dtype=float)

    rng = np.random.default_rng(args.seed)

    top1_counter = Counter()
    top3_counter = Counter()
    expectancy_samples: Dict[str, List[float]] = defaultdict(list)

    for _ in range(args.n_boot):
        idx = _sample_block_indices(
            n=n_days,
            out_len=n_days,
            block_size=args.block_days,
            rng=rng,
        )
        sample = arr[idx, :]  # shape: [sample_days, n_cfg]
        exp_r = sample.mean(axis=0)  # daily mean R bootstrap

        order = np.argsort(-exp_r)  # descending
        top1_counter[cfg_ids[order[0]]] += 1
        for j in order[: min(3, len(order))]:
            top3_counter[cfg_ids[j]] += 1

        for i, cfg_id in enumerate(cfg_ids):
            expectancy_samples[cfg_id].append(float(exp_r[i]))

    rows = []
    for m in meta:
        cfg_id = m["cfg_id"]
        vals = np.array(expectancy_samples[cfg_id], dtype=float)

        rows.append(
            {
                "cfg_id": cfg_id,
                "entry_threshold_pips": m["entry_threshold_pips"],
                "tp_pips": m["tp_pips"],
                "n_days_with_trades": m["n_days_with_trades"],
                "boot_mean_daily_R": float(vals.mean()),
                "boot_std_daily_R": float(vals.std(ddof=0)),
                "boot_p05_daily_R": float(np.quantile(vals, 0.05)),
                "boot_p50_daily_R": float(np.quantile(vals, 0.50)),
                "boot_p95_daily_R": float(np.quantile(vals, 0.95)),
                "prob_top1": float(top1_counter[cfg_id] / args.n_boot),
                "prob_top3": float(top3_counter[cfg_id] / args.n_boot),
                "run_dir": m["run_dir"],
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        by=["prob_top1", "prob_top3", "boot_mean_daily_R"],
        ascending=[False, False, False],
    )

    out_csv = out_dir / "plateau_bootstrap_table.csv"
    out_df.to_csv(out_csv, index=False)

    best = out_df.iloc[0].to_dict() if len(out_df) else None

    summary = {
        "test_type": "plateau_bootstrap",
        "bootstrap_id": boot_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grid_csv": str(args.grid_csv),
        "n_boot": int(args.n_boot),
        "block_days": int(args.block_days),
        "seed": int(args.seed),
        "n_days": int(n_days),
        "n_configs": int(len(cfg_ids)),
        "best_cfg_by_prob_top1": best,
        "notes": {
            "method": "block bootstrap on daily aggregated R per config",
            "interpretation": "A real plateau shows clustered high prob_top1 / prob_top3 across neighboring cells, not a single isolated winner.",
        },
    }

    summary_path = out_dir / "plateau_bootstrap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.print_summary:
        print(json.dumps(summary, indent=2))

    print(f"OUT_DIR: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())