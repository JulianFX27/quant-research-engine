from __future__ import annotations

from pathlib import Path
import pandas as pd

from session_times import london_open_utc


DATASET_PATH = Path(
    "data/canonical/EURUSD_M5_2014-01-01__2024-12-31__dukascopy_v2_clean.csv"
)

OUTPUT_PATH = Path(
    "results/research/opening_momentum/london_event_surface_dstaware.csv"
)

ENTRY_DELAYS_MIN = [30, 35, 40, 45, 50, 60]
HOLDING_PERIODS_MIN = [30, 60, 90, 120, 180, 240]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df = df.sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.date
    return df


def build_surface(df: pd.DataFrame) -> pd.DataFrame:
    px = df.set_index("time").sort_index()
    dates = sorted(df["date"].unique())

    records: list[dict] = []

    for d in dates:
        t0 = london_open_utc(d)

        try:
            open_0 = float(px.loc[t0]["open"])
            close_30 = float(px.loc[t0 + pd.Timedelta(minutes=30)]["close"])
        except KeyError:
            continue

        window_30 = px.loc[t0 : t0 + pd.Timedelta(minutes=30)]
        if window_30.empty:
            continue

        rec: dict = {
            "date": d,
            "open_time": t0,
            "signal_time": t0 + pd.Timedelta(minutes=30),
            "open_price": open_0,
            "ret_30m": (close_30 - open_0) / open_0,
            "range_30m": (window_30["high"].max() - window_30["low"].min()) / open_0,
            "direction_30m": 1 if close_30 > open_0 else -1,
        }

        ok = True

        for delay in ENTRY_DELAYS_MIN:
            entry_time = t0 + pd.Timedelta(minutes=delay)

            try:
                entry_price = float(px.loc[entry_time]["open"])
            except KeyError:
                ok = False
                break

            rec[f"entry_time_{delay}m"] = entry_time
            rec[f"entry_price_{delay}m"] = entry_price

            for hold in HOLDING_PERIODS_MIN:
                exit_time = entry_time + pd.Timedelta(minutes=hold)

                try:
                    exit_price = float(px.loc[exit_time]["close"])
                except KeyError:
                    ok = False
                    break

                rec[f"ret_fwd_{hold}m_from_{delay}m"] = (exit_price - entry_price) / entry_price

            if not ok:
                break

        if ok:
            records.append(rec)

    return pd.DataFrame.from_records(records)


def main() -> None:
    print("Loading dataset...")
    df = load_dataset()

    print("Building London event surface (DST-aware, no-lookahead)...")
    surface = build_surface(df)

    print(f"Surface rows generated: {len(surface)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    surface.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
