from __future__ import annotations

from pathlib import Path
import pandas as pd


EVENTS_PATH = Path(
    "results/research/opening_momentum/ny_event_surface.csv"
)

OUTPUT_PATH = Path(
    "results/research/opening_momentum/ny_alpha_surface.csv"
)

ENTRY_DELAYS_MIN = [30, 35, 40, 45, 50, 60]
HOLDING_PERIODS_MIN = [30, 60, 90, 120, 180, 240]
THRESHOLDS = [0.80, 0.85, 0.90, 0.95]


def main() -> None:
    df = pd.read_csv(EVENTS_PATH)

    rows: list[dict] = []

    for q in THRESHOLDS:
        q_low = df["ret_30m"].quantile(1 - q)
        q_high = df["ret_30m"].quantile(q)

        longs = df[df["ret_30m"] > q_high].copy()
        shorts = df[df["ret_30m"] < q_low].copy()

        for delay in ENTRY_DELAYS_MIN:
            for hold in HOLDING_PERIODS_MIN:
                col = f"ret_fwd_{hold}m_from_{delay}m"
                if col not in df.columns:
                    continue

                long_pnl = longs[col]
                short_pnl = -shorts[col]

                combined = pd.concat([long_pnl, short_pnl], axis=0).dropna()

                if len(combined) < 20:
                    continue

                mean_ret = combined.mean()
                std_ret = combined.std()

                rows.append(
                    {
                        "threshold_q": q,
                        "entry_delay_min": delay,
                        "holding_min": hold,
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

    print("\n=== TOP 30 NY CONFIGS (NO-LOOKAHEAD) ===\n")
    print(out.head(30).to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
