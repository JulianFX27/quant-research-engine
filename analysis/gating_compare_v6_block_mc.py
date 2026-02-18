# analysis/gating_compare_v6_block_mc.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _max_drawdown(equity: np.ndarray) -> float:
    # equity: cumulative sum of R
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)) if len(dd) else 0.0


def _run_block_bootstrap(
    r: np.ndarray,
    n_sims: int,
    block_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Block bootstrap over a 1D sequence r (R-multiples in chronological order).
    Produces synthetic sequences of same length by concatenating random contiguous blocks.
    """
    rng = np.random.default_rng(seed)
    n = len(r)
    if n == 0:
        return pd.DataFrame(
            {"sim": [], "total_R": [], "max_dd_R": [], "min_equity_R": [], "max_equity_R": []}
        )

    # number of blocks needed to reach length n
    n_blocks = int(np.ceil(n / block_size))

    out_total = np.empty(n_sims, dtype=float)
    out_dd = np.empty(n_sims, dtype=float)
    out_min_eq = np.empty(n_sims, dtype=float)
    out_max_eq = np.empty(n_sims, dtype=float)

    # valid start indices for blocks
    max_start = max(0, n - block_size)
    starts = np.arange(0, max_start + 1) if max_start >= 0 else np.array([0])

    for i in range(n_sims):
        # choose block starts with replacement
        block_starts = rng.choice(starts, size=n_blocks, replace=True)
        pieces = []
        for s in block_starts:
            pieces.append(r[s : s + block_size])
        sim_r = np.concatenate(pieces)[:n]

        eq = np.cumsum(sim_r)
        out_total[i] = float(eq[-1]) if len(eq) else 0.0
        out_dd[i] = _max_drawdown(eq)
        out_min_eq[i] = float(np.min(eq)) if len(eq) else 0.0
        out_max_eq[i] = float(np.max(eq)) if len(eq) else 0.0

    return pd.DataFrame(
        {
            "sim": np.arange(n_sims),
            "total_R": out_total,
            "max_dd_R": out_dd,
            "min_equity_R": out_min_eq,
            "max_equity_R": out_max_eq,
        }
    )


def _summarize_mc(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "totalR_p5": float("nan"),
            "dd_p50": float("nan"),
            "dd_p95": float("nan"),
            "prob_dd_ge_8": float("nan"),
            "prob_dd_ge_12": float("nan"),
        }

    totalR_p5 = float(np.quantile(df["total_R"], 0.05))
    dd_p50 = float(np.quantile(df["max_dd_R"], 0.50))
    dd_p95 = float(np.quantile(df["max_dd_R"], 0.95))
    prob_dd_ge_8 = float(np.mean(df["max_dd_R"] >= 8.0))
    prob_dd_ge_12 = float(np.mean(df["max_dd_R"] >= 12.0))

    return {
        "totalR_p5": totalR_p5,
        "dd_p50": dd_p50,
        "dd_p95": dd_p95,
        "prob_dd_ge_8": prob_dd_ge_8,
        "prob_dd_ge_12": prob_dd_ge_12,
    }


def _load_base_and_v6(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    baseline trades: results/runs/<rid>/trades.csv
    v6 gated trades: results/runs/<rid>/gating_v6/trades_gated.csv
    """
    base_path = run_dir / "trades.csv"
    v6_path = run_dir / "gating_v6" / "trades_gated.csv"

    if not base_path.exists():
        raise FileNotFoundError(f"Missing baseline trades.csv: {base_path}")
    if not v6_path.exists():
        raise FileNotFoundError(f"Missing V6 gated trades_gated.csv: {v6_path}")

    base = pd.read_csv(base_path)
    v6 = pd.read_csv(v6_path)

    # ensure R column
    if "R" not in base.columns:
        raise ValueError(f"baseline trades.csv missing column 'R': {base_path}")
    if "R" not in v6.columns:
        raise ValueError(f"v6 trades_gated.csv missing column 'R': {v6_path}")

    # preserve chronological order: if entry_time exists, sort; else keep file order
    if "entry_time" in base.columns:
        base["entry_time"] = pd.to_datetime(base["entry_time"], utc=True, errors="coerce")
        base = base.sort_values("entry_time").reset_index(drop=True)
    if "entry_time" in v6.columns:
        v6["entry_time"] = pd.to_datetime(v6["entry_time"], utc=True, errors="coerce")
        v6 = v6.sort_values("entry_time").reset_index(drop=True)

    base["R"] = pd.to_numeric(base["R"], errors="coerce")
    v6["R"] = pd.to_numeric(v6["R"], errors="coerce")

    base = base.dropna(subset=["R"]).reset_index(drop=True)
    v6 = v6.dropna(subset=["R"]).reset_index(drop=True)

    return base, v6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="results/runs/<RID>")
    ap.add_argument("--mc_sims", type=int, default=5000)
    ap.add_argument("--block_size", type=int, default=4, help="typical 3-5")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rid = run_dir.name

    base, v6 = _load_base_and_v6(run_dir)

    base_r = base["R"].to_numpy(dtype=float)
    v6_r = v6["R"].to_numpy(dtype=float)

    mc_dir = run_dir / "gating_v6" / "mc_block"
    mc_dir.mkdir(parents=True, exist_ok=True)

    base_mc = _run_block_bootstrap(base_r, args.mc_sims, args.block_size, args.seed)
    v6_mc = _run_block_bootstrap(v6_r, args.mc_sims, args.block_size, args.seed)

    base_csv = mc_dir / "mc_baseline_block.csv"
    v6_csv = mc_dir / "mc_gated_v6_block.csv"
    base_mc.to_csv(base_csv, index=False)
    v6_mc.to_csv(v6_csv, index=False)

    base_sum = _summarize_mc(base_mc)
    v6_sum = _summarize_mc(v6_mc)

    summary = {
        "rid": rid,
        "mc_sims": int(args.mc_sims),
        "block_size": int(args.block_size),
        "seed": int(args.seed),
        "baseline": base_sum,
        "gated_v6": v6_sum,
        "delta": {
            "d_dd_p95": float(v6_sum["dd_p95"] - base_sum["dd_p95"]),
            "d_prob_dd_ge_12": float(v6_sum["prob_dd_ge_12"] - base_sum["prob_dd_ge_12"]),
            "d_totalR_p5": float(v6_sum["totalR_p5"] - base_sum["totalR_p5"]),
        },
        "n_trades": {"baseline": int(len(base_r)), "gated_v6": int(len(v6_r))},
        "outputs": {"baseline_csv": str(base_csv), "gated_csv": str(v6_csv)},
    }

    out_json = mc_dir / "mc_block_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] RID={rid}")
    print(f"[OK] Wrote: {base_csv}")
    print(f"[OK] Wrote: {v6_csv}")
    print(f"[OK] Wrote: {out_json}")
    print("\n=== BASELINE (BLOCK MC) ===")
    print(base_sum)
    print("\n=== GATED V6 (BLOCK MC) ===")
    print(v6_sum)
    print("\n=== DELTA (GATE - BASE) ===")
    print(summary["delta"])


if __name__ == "__main__":
    main()
