from __future__ import annotations

import pandas as pd
from pathlib import Path


EVENTS_PATH = Path(
    "results/research/opening_momentum/london_events.csv"
)


def main():

    df = pd.read_csv(EVENTS_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    q_low = df["ret_30m"].quantile(0.1)
    q_high = df["ret_30m"].quantile(0.9)

    longs = df[df["ret_30m"] > q_high]
    shorts = df[df["ret_30m"] < q_low]

    longs = longs.assign(pnl=longs["ret_120m"])
    shorts = shorts.assign(pnl=-shorts["ret_120m"])

    trades = pd.concat([longs, shorts])

    stats = (
        trades.groupby("year")
        .agg(
            trades=("pnl", "count"),
            mean_return=("pnl", "mean"),
            std=("pnl", "std")
        )
        .reset_index()
    )

    stats["sharpe"] = stats["mean_return"] / stats["std"]

    print("\n=== YEARLY PERFORMANCE ===\n")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
