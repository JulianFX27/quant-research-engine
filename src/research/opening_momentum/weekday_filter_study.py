from __future__ import annotations

from pathlib import Path
import pandas as pd


EVENTS_PATH = Path(
    "results/research/opening_momentum/london_event_surface.csv"
)

OUTPUT_PATH = Path(
    "results/research/opening_momentum/london_weekday_filter_study.csv"
)

# Mejor configuración actual
THRESHOLD_Q = 0.95
ENTRY_DELAY = 30
HOLDING = 30


def label_range_bucket(df: pd.DataFrame) -> pd.DataFrame:
    q25 = df["range_30m"].quantile(0.25)
    q50 = df["range_30m"].quantile(0.50)
    q75 = df["range_30m"].quantile(0.75)

    def bucket(x: float) -> str:
        if x <= q25:
            return "bottom_25"
        elif x <= q50:
            return "q25_q50"
        elif x <= q75:
            return "q50_q75"
        else:
            return "top_25"

    out = df.copy()
    out["range_bucket"] = out["range_30m"].apply(bucket)
    return out


def main() -> None:
    df = pd.read_csv(EVENTS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.day_name()

    df = label_range_bucket(df)
    df = df[df["range_bucket"] == "top_25"].copy()

    q_low = df["ret_30m"].quantile(1 - THRESHOLD_Q)
    q_high = df["ret_30m"].quantile(THRESHOLD_Q)

    ret_col = f"ret_fwd_{HOLDING}m_from_{ENTRY_DELAY}m"

    longs = df[df["ret_30m"] > q_high].copy()
    shorts = df[df["ret_30m"] < q_low].copy()

    longs["pnl"] = longs[ret_col]
    shorts["pnl"] = -shorts[ret_col]

    longs["side"] = "long"
    shorts["side"] = "short"

    trades = pd.concat([longs, shorts], axis=0).dropna(subset=["pnl"])

    stats = (
        trades.groupby("weekday")
        .agg(
            n_trades=("pnl", "count"),
            mean_return=("pnl", "mean"),
            std_return=("pnl", "std"),
        )
        .reset_index()
    )

    stats["sharpe_per_trade"] = stats["mean_return"] / stats["std_return"]

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    stats["weekday"] = pd.Categorical(stats["weekday"], categories=weekday_order, ordered=True)
    stats = stats.sort_values("weekday")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUTPUT_PATH, index=False)

    print("\n=== WEEKDAY FILTER STUDY ===\n")
    print(stats.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
