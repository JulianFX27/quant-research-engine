# analysis/gating_v6_master_summary.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def main():
    runs_root = Path("results/runs")
    rows = []

    for run_dir in sorted(runs_root.glob("*")):
        p = run_dir / "gating_v6" / "gating_v6_compare.csv"
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                rows.append(df.iloc[0].to_dict())

    out_dir = Path("results/research")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "gating_v6_master_summary.csv"
    if not rows:
        pd.DataFrame().to_csv(out_path, index=False)
        print(f"[EMPTY] Wrote: {out_path}")
        return

    m = pd.DataFrame(rows)

    # nice ordering (optional)
    cols = [
        "rid",
        "n_base","expR_base","dd_base","win_base",
        "n_gated","expR_gated","dd_gated","win_gated",
        "d_expR","d_dd",
        "base_totalR_p5","base_dd_p50","base_dd_p95","base_prob_dd_ge_8","base_prob_dd_ge_12",
        "gate_totalR_p5","gate_dd_p50","gate_dd_p95","gate_prob_dd_ge_8","gate_prob_dd_ge_12",
        "d_dd_p95","d_prob_dd_ge_12",
        "nan_spread_ny_open_atr","nan_hour","nan_shock_z",
        "status"
    ]
    cols = [c for c in cols if c in m.columns]
    m = m[cols]

    m.to_csv(out_path, index=False)
    print(f"WROTE: {out_path}")
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
