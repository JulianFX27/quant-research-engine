from __future__ import annotations

import pandas as pd
from pathlib import Path


EVENTS_PATH = Path(
    "results/research/opening_momentum/london_events.csv"
)


def main():

    df = pd.read_csv(EVENTS_PATH)

    q_low = df["ret_30m"].quantile(0.1)
    q_high = df["ret_30m"].quantile(0.9)

    longs = df[df["ret_30m"] > q_high]
    shorts = df[df["ret_30m"] < q_low]

    long_ret = longs["ret_120m"].mean()
    short_ret = -shorts["ret_120m"].mean()

    combined = pd.concat([
        longs.assign(pnl=longs["ret_120m"]),
        shorts.assign(pnl=-shorts["ret_120m"])
    ])

    print("\n=== STRATEGY RESULTS ===\n")

    print(f"long trades: {len(longs)}")
    print(f"short trades: {len(shorts)}")

    print(f"\nmean long return: {long_ret:.6f}")
    print(f"mean short return: {short_ret:.6f}")

    print(f"\ncombined mean return: {combined['pnl'].mean():.6f}")
    print(f"combined std: {combined['pnl'].std():.6f}")

    sharpe = combined["pnl"].mean() / combined["pnl"].std()

    print(f"\napprox sharpe per trade: {sharpe:.3f}")


if __name__ == "__main__":

    main()
