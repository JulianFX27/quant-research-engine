# analysis/gating_compare_v6.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Config (V6)
# ---------------------------
V6_BUCKET_LO = -3.067
V6_BUCKET_HI = 2.664
V6_HOUR_CUTOFF = 12  # hour >= 12
V6_SHOCK_SIGN_POS = True  # shock_z > 0


# ---------------------------
# Metrics
# ---------------------------
def equity_and_dd(r: np.ndarray) -> Tuple[np.ndarray, float]:
    # equity curve in R-space
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(np.max(dd)) if len(dd) else 0.0
    return eq, max_dd


def summarize_r(r: np.ndarray) -> Dict[str, float]:
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return {
            "n_trades": 0,
            "expectancy_R": float("nan"),
            "total_R": 0.0,
            "winrate": float("nan"),
            "max_dd_R": float("nan"),
        }
    _, max_dd = equity_and_dd(r)
    return {
        "n_trades": int(len(r)),
        "expectancy_R": float(np.mean(r)),
        "total_R": float(np.sum(r)),
        "winrate": float(np.mean(r > 0)),
        "max_dd_R": float(max_dd),
    }


def mc_sim(
    r: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    IID bootstrap over trades (with replacement).
    For each sim: resample N trades, keep sampled order, compute totalR and maxDD.
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n == 0:
        return pd.DataFrame(columns=["sim", "total_R", "max_dd_R"])

    out_total = np.empty(n_sims, dtype=float)
    out_dd = np.empty(n_sims, dtype=float)

    for i in range(n_sims):
        idx = rng.integers(0, n, size=n, endpoint=False)
        rs = r[idx]
        out_total[i] = float(np.sum(rs))
        _, max_dd = equity_and_dd(rs)
        out_dd[i] = float(max_dd)

    return pd.DataFrame(
        {"sim": np.arange(n_sims, dtype=int), "total_R": out_total, "max_dd_R": out_dd}
    )


def mc_summary(mc: pd.DataFrame) -> Dict[str, float]:
    if mc.empty:
        return {}
    total = mc["total_R"].to_numpy(float)
    dd = mc["max_dd_R"].to_numpy(float)

    return {
        "totalR_p5": float(np.quantile(total, 0.05)),
        "dd_p50": float(np.quantile(dd, 0.50)),
        "dd_p95": float(np.quantile(dd, 0.95)),
        "prob_dd_ge_8": float(np.mean(dd >= 8.0)),
        "prob_dd_ge_12": float(np.mean(dd >= 12.0)),
    }


# ---------------------------
# Data / gating
# ---------------------------
@dataclass
class Loaded:
    rid: str
    run_dir: Path
    trades: pd.DataFrame
    feats: pd.DataFrame


def load_run(run_dir: Path) -> Loaded:
    run_dir = Path(run_dir)
    rid = run_dir.name

    trades_path = run_dir / "trades.csv"
    feats_path = Path("data/anchor_reversion_fx/data/eurusd_m5_features.csv")

    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades.csv: {trades_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"Missing features file: {feats_path}")

    t = pd.read_csv(trades_path, parse_dates=["entry_time"])
    f = pd.read_csv(feats_path, parse_dates=["time"])

    # normalize datetimes (UTC -> naive)
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce").dt.tz_convert(None)
    f["time"] = pd.to_datetime(f["time"], utc=True, errors="coerce").dt.tz_convert(None)

    # enforce numeric columns used
    for col in ["R"]:
        t[col] = pd.to_numeric(t[col], errors="coerce")
    for col in ["spread_ny_open_atr", "hour", "shock_z"]:
        f[col] = pd.to_numeric(f[col], errors="coerce")

    return Loaded(rid=rid, run_dir=run_dir, trades=t, feats=f)


def apply_v6_gate(loaded: Loaded) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    t = loaded.trades.copy()
    f = loaded.feats.copy()

    m = t.merge(
        f[["time", "spread_ny_open_atr", "hour", "shock_z"]],
        left_on="entry_time",
        right_on="time",
        how="left",
    )

    # NaN rates (merge quality)
    nan_rates = {
        "spread_ny_open_atr": float(m["spread_ny_open_atr"].isna().mean()),
        "hour": float(m["hour"].isna().mean()),
        "shock_z": float(m["shock_z"].isna().mean()),
    }

    x = m["spread_ny_open_atr"].to_numpy(float)
    hour = m["hour"].to_numpy(float)
    shock_z = m["shock_z"].to_numpy(float)

    bucket = (x > V6_BUCKET_LO) & (x <= V6_BUCKET_HI)
    shock_pos = shock_z > 0 if V6_SHOCK_SIGN_POS else shock_z < 0
    late = hour >= V6_HOUR_CUTOFF

    block = bucket & shock_pos & late

    base = m.copy()
    gated = m.loc[~block].copy()

    # Keep columns minimal on export
    keep_cols = [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "R",
        "spread_ny_open_atr",
        "hour",
        "shock_z",
    ]
    keep_cols = [c for c in keep_cols if c in base.columns]

    return base[keep_cols], gated[keep_cols], nan_rates


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. results/runs/<RID>")
    ap.add_argument("--mc_sims", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    loaded = load_run(run_dir)

    out_dir = run_dir / "gating_v6"
    mc_dir = out_dir / "mc"
    out_dir.mkdir(parents=True, exist_ok=True)
    mc_dir.mkdir(parents=True, exist_ok=True)

    base_df, gated_df, nan_rates = apply_v6_gate(loaded)

    base_r = base_df["R"].dropna().to_numpy(float)
    gated_r = gated_df["R"].dropna().to_numpy(float)

    base_sum = summarize_r(base_r)
    gated_sum = summarize_r(gated_r)

    rng = np.random.default_rng(args.seed)
    mc_base = mc_sim(base_r, args.mc_sims, rng)
    mc_gate = mc_sim(gated_r, args.mc_sims, rng)

    mc_base_path = mc_dir / "mc_baseline.csv"
    mc_gate_path = mc_dir / "mc_gated_v6.csv"
    mc_base.to_csv(mc_base_path, index=False)
    mc_gate.to_csv(mc_gate_path, index=False)

    base_mc_sum = mc_summary(mc_base)
    gate_mc_sum = mc_summary(mc_gate)

    compare = {
        "rid": loaded.rid,
        "v6": {
            "bucket_lo": V6_BUCKET_LO,
            "bucket_hi": V6_BUCKET_HI,
            "hour_cutoff": V6_HOUR_CUTOFF,
            "shock_sign": "pos" if V6_SHOCK_SIGN_POS else "neg",
        },
        "baseline": base_sum,
        "gated_v6": gated_sum,
        "merge_nan_rates": nan_rates,
        "mc_baseline": base_mc_sum,
        "mc_gated_v6": gate_mc_sum,
        "deltas": {
            "d_expR": float(gated_sum["expectancy_R"] - base_sum["expectancy_R"]),
            "d_dd": float(gated_sum["max_dd_R"] - base_sum["max_dd_R"]),
            "d_dd_p95": float(gate_mc_sum.get("dd_p95", np.nan) - base_mc_sum.get("dd_p95", np.nan)),
            "d_prob_dd_ge_12": float(
                gate_mc_sum.get("prob_dd_ge_12", np.nan) - base_mc_sum.get("prob_dd_ge_12", np.nan)
            ),
        },
    }

    json_path = out_dir / "gating_v6_compare.json"
    csv_path = out_dir / "gating_v6_compare.csv"
    gated_trades_path = out_dir / "trades_gated.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2)

    # flat csv
    row = {
        "rid": loaded.rid,
        "n_base": base_sum["n_trades"],
        "expR_base": base_sum["expectancy_R"],
        "dd_base": base_sum["max_dd_R"],
        "win_base": base_sum["winrate"],
        "n_gated": gated_sum["n_trades"],
        "expR_gated": gated_sum["expectancy_R"],
        "dd_gated": gated_sum["max_dd_R"],
        "win_gated": gated_sum["winrate"],
        "d_expR": compare["deltas"]["d_expR"],
        "d_dd": compare["deltas"]["d_dd"],
        "base_totalR_p5": base_mc_sum.get("totalR_p5", np.nan),
        "base_dd_p50": base_mc_sum.get("dd_p50", np.nan),
        "base_dd_p95": base_mc_sum.get("dd_p95", np.nan),
        "base_prob_dd_ge_8": base_mc_sum.get("prob_dd_ge_8", np.nan),
        "base_prob_dd_ge_12": base_mc_sum.get("prob_dd_ge_12", np.nan),
        "gate_totalR_p5": gate_mc_sum.get("totalR_p5", np.nan),
        "gate_dd_p50": gate_mc_sum.get("dd_p50", np.nan),
        "gate_dd_p95": gate_mc_sum.get("dd_p95", np.nan),
        "gate_prob_dd_ge_8": gate_mc_sum.get("prob_dd_ge_8", np.nan),
        "gate_prob_dd_ge_12": gate_mc_sum.get("prob_dd_ge_12", np.nan),
        "d_dd_p95": compare["deltas"]["d_dd_p95"],
        "d_prob_dd_ge_12": compare["deltas"]["d_prob_dd_ge_12"],
        "nan_spread_ny_open_atr": nan_rates["spread_ny_open_atr"],
        "nan_hour": nan_rates["hour"],
        "nan_shock_z": nan_rates["shock_z"],
        "status": "OK",
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    gated_df.to_csv(gated_trades_path, index=False)

    print(f"[OK] Wrote: {json_path}")
    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {gated_trades_path}")
    print("")
    print("=== BASELINE ===")
    print(base_sum)
    print("")
    print("=== GATED V6 ===")
    print(gated_sum)
    print("")
    print("=== MC OUTPUTS ===")
    print(f"  {mc_base_path}")
    print(f"  {mc_gate_path}")


if __name__ == "__main__":
    main()
