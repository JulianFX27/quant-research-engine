from __future__ import annotations

import pandas as pd
from pathlib import Path


EVENTS_PATH = Path(
    "results/research/opening_momentum/london_events.csv"
)


def load_events():

    df = pd.read_csv(EVENTS_PATH)

    return df


def compute_deciles(df):

    df["ret30_decile"] = pd.qcut(df["ret_30m"], 10, labels=False)

    return df


def event_study(df):

    stats = (
        df.groupby("ret30_decile")
        .agg(
            count=("ret_30m", "count"),
            mean_ret30=("ret_30m", "mean"),
            mean_ret60=("ret_60m", "mean"),
            mean_ret120=("ret_120m", "mean"),
        )
        .reset_index()
    )

    return stats


def main():

    print("Loading events...")

    df = load_events()

    print(f"events: {len(df)}")

    df = compute_deciles(df)

    stats = event_study(df)

    print("\n=== EVENT STUDY RESULTS ===\n")

    print(stats.to_string(index=False))


if __name__ == "__main__":

    main()
