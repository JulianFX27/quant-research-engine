from __future__ import annotations

from pathlib import Path
import pandas as pd


EVENTS_PATH = Path(
    "results/research/opening_momentum/london_event_surface.csv"
)

OUTPUT_PATH = Path(
    "results/research/opening_momentum/london_range_filter_study.csv"
)

THRESHOLDS = [0.80, 0.85, 0.90, 0.95]

# Estrategia operable detectada en el alpha surface
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

    df = df.copy()
    df["range_bucket"] = df["range_30m"].apply(bucket)
    return df


def main() -> None:
    df = pd.read_csv(EVENTS_PATH)
    df = label_range_bucket(df)

    ret_col = f"ret_fwd_{HOLDING}m_from_{ENTRY_DELAY}m"

    rows: list[dict] = []

    for q in THRESHOLDS:
        q_low = df["ret_30m"].quantile(1 - q)
        q_high = df["ret_30m"].quantile(q)

        for bucket in ["bottom_25", "q25_q50", "q50_q75", "top_25"]:
            sub = df[df["range_bucket"] == bucket].copy()

            longs = sub[sub["ret_30m"] > q_high].copy()
            shorts = sub[sub["ret_30m"] < q_low].copy()

            long_pnl = longs[ret_col]
            short_pnl = -shorts[ret_col]

            combined = pd.concat([long_pnl, short_pnl], axis=0).dropna()

            if len(combined) < 20:
                continue

            mean_ret = combined.mean()
            std_ret = combined.std()

            rows.append(
                {
                    "threshold_q": q,
                    "range_bucket": bucket,
                    "n_longs": len(long_pnl.dropna()),
                    "n_shorts": len(short_pnl.dropna()),
                    "n_trades": len(combined),
                    "mean_return": mean_ret,
                    "std_return": std_ret,
                    "sharpe_per_trade": mean_ret / std_ret if std_ret > 0 else None,
                }
            )

    out = pd.DataFrame(rows).sort_values(
        ["threshold_q", "sharpe_per_trade", "mean_return"],
        ascending=[True, False, False],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\n=== RANGE FILTER STUDY ===\n")
    print(out.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
