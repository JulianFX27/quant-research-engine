from __future__ import annotations

from pathlib import Path
import pandas as pd


EVENTS_PATH = Path(
    "results/research/opening_momentum/ny_event_surface_enriched.csv"
)

OUTPUT_PATH = Path(
    "results/research/opening_momentum/ny_range_conditioning_study.csv"
)

THRESHOLD_Q = 0.80
ENTRY_DELAY = 30
HOLDING = 30
PIP_SIZE = 0.0001
COST_PIPS = 1.0


def assign_range_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["range_bucket"] = pd.qcut(
        out["range_30m"],
        q=5,
        labels=[
            "Q1_low_vol",
            "Q2",
            "Q3",
            "Q4",
            "Q5_high_vol",
        ],
    )

    return out


def main() -> None:
    df = pd.read_csv(EVENTS_PATH)

    df = assign_range_bucket(df)

    q_low = df["ret_30m"].quantile(1 - THRESHOLD_Q)
    q_high = df["ret_30m"].quantile(THRESHOLD_Q)

    ret_col = f"ret_fwd_{HOLDING}m_from_{ENTRY_DELAY}m"
    cost_ret = COST_PIPS * PIP_SIZE

    rows: list[dict] = []

    for bucket in [
        "Q1_low_vol",
        "Q2",
        "Q3",
        "Q4",
        "Q5_high_vol",
    ]:

        sub = df[df["range_bucket"] == bucket].copy()

        longs = sub[sub["ret_30m"] > q_high].copy()
        shorts = sub[sub["ret_30m"] < q_low].copy()

        long_pnl = longs[ret_col]
        short_pnl = -shorts[ret_col]

        combined = pd.concat([long_pnl, short_pnl], axis=0).dropna()

        combined = combined - cost_ret

        if len(combined) < 20:
            continue

        rows.append(
            {
                "range_bucket": bucket,
                "n_longs": len(long_pnl.dropna()),
                "n_shorts": len(short_pnl.dropna()),
                "n_trades": len(combined),
                "mean_return_net": combined.mean(),
                "std_return_net": combined.std(),
                "winrate_net": (combined > 0).mean(),
                "sharpe_per_trade_net": (
                    combined.mean() / combined.std()
                    if combined.std() > 0
                    else None
                ),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["sharpe_per_trade_net", "mean_return_net"],
        ascending=[False, False],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(OUTPUT_PATH, index=False)

    print("\n=== NY RANGE CONDITIONING STUDY ===\n")

    print(out.to_string(index=False))

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
