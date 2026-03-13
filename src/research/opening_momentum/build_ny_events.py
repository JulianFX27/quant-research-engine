from __future__ import annotations

import pandas as pd
from pathlib import Path

from session_times import ny_open_utc


DATASET_PATH = Path(
    "data/canonical/EURUSD_M5_2014-01-01__2024-12-31__dukascopy_v2_clean.csv"
)

OUTPUT_PATH = Path(
    "results/research/opening_momentum/ny_events_dstaware.csv"
)


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.date
    return df


def compute_events(df):

    px = df.set_index("time").sort_index()

    records = []

    for d in sorted(df["date"].unique()):

        t0 = ny_open_utc(d)

        try:

            p0 = float(px.loc[t0]["open"])
            p30 = float(px.loc[t0 + pd.Timedelta(minutes=30)]["close"])
            p60 = float(px.loc[t0 + pd.Timedelta(minutes=60)]["close"])
            p120 = float(px.loc[t0 + pd.Timedelta(minutes=120)]["close"])

        except KeyError:
            continue

        window = px.loc[t0 : t0 + pd.Timedelta(minutes=30)]

        if window.empty:
            continue

        high = float(window["high"].max())
        low = float(window["low"].min())

        records.append(
            {
                "date": d,
                "open_time": t0,
                "open_price": p0,
                "ret_30m": (p30 - p0) / p0,
                "ret_60m": (p60 - p0) / p0,
                "ret_120m": (p120 - p0) / p0,
                "range_30m": (high - low) / p0,
                "direction_30m": 1 if p30 > p0 else -1,
            }
        )

    return pd.DataFrame.from_records(records)


def main():

    print("Loading dataset...")

    df = load_dataset()

    print("Building NY DST-aware events...")

    events = compute_events(df)

    print(f"Events generated: {len(events)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    events.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
