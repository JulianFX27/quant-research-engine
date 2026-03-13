from __future__ import annotations

import pandas as pd
from pathlib import Path


EVENTS_PATH = Path(
    "results/research/opening_momentum/ny_events_dstaware.csv"
)


def main():

    print("Loading NY events...")

    df = pd.read_csv(EVENTS_PATH)

    print("events:", len(df))

    df["ret30_decile"] = pd.qcut(df["ret_30m"], 10, labels=False)

    table = (
        df.groupby("ret30_decile")
        .agg(
            count=("ret_30m", "count"),
            mean_ret30=("ret_30m", "mean"),
            mean_ret60=("ret_60m", "mean"),
            mean_ret120=("ret_120m", "mean"),
        )
        .reset_index()
    )

    print("\n=== NY EVENT STUDY ===\n")

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
