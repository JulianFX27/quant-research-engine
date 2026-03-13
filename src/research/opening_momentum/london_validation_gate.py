from __future__ import annotations

from pathlib import Path
import pandas as pd


EVENTS_PATH = Path(
    "results/research/opening_momentum/london_event_surface_dstaware.csv"
)

OUTPUT_DIR = Path(
    "results/research/opening_momentum"
)

THRESHOLD_Q = 0.95
ENTRY_DELAY = 30
HOLDING = 30
PIP_SIZE = 0.0001

# Costes totales round-trip en pips para stress
COST_SCENARIOS_PIPS = [0.0, 0.5, 1.0, 1.5, 2.0]


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


def build_trades(df: pd.DataFrame) -> pd.DataFrame:
    df = label_range_bucket(df)
    df = df[df["range_bucket"] == "top_25"].copy()

    q_low = df["ret_30m"].quantile(1 - THRESHOLD_Q)
    q_high = df["ret_30m"].quantile(THRESHOLD_Q)

    ret_col = f"ret_fwd_{HOLDING}m_from_{ENTRY_DELAY}m"

    longs = df[df["ret_30m"] > q_high].copy()
    shorts = df[df["ret_30m"] < q_low].copy()

    longs["side"] = "long"
    longs["gross_ret"] = longs[ret_col]

    shorts["side"] = "short"
    shorts["gross_ret"] = -shorts[ret_col]

    trades = pd.concat([longs, shorts], axis=0).copy()
    trades["date"] = pd.to_datetime(trades["date"])
    trades["year"] = trades["date"].dt.year

    return trades.sort_values(["date", "side"]).reset_index(drop=True)


def summarize_yearly(trades: pd.DataFrame, cost_pips: float) -> pd.DataFrame:
    cost_ret = cost_pips * PIP_SIZE

    df = trades.copy()
    df["net_ret"] = df["gross_ret"] - cost_ret

    stats = (
        df.groupby("year")
        .agg(
            n_trades=("net_ret", "count"),
            mean_return=("net_ret", "mean"),
            std_return=("net_ret", "std"),
            winrate=("net_ret", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    stats["sharpe_per_trade"] = stats["mean_return"] / stats["std_return"]
    stats["cost_pips"] = cost_pips
    return stats


def summarize_global(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cost_pips in COST_SCENARIOS_PIPS:
        cost_ret = cost_pips * PIP_SIZE

        net = trades["gross_ret"] - cost_ret

        rows.append(
            {
                "cost_pips": cost_pips,
                "n_trades": len(net),
                "mean_return": net.mean(),
                "std_return": net.std(),
                "winrate": (net > 0).mean(),
                "sharpe_per_trade": net.mean() / net.std(),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(EVENTS_PATH)

    trades = build_trades(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUTPUT_DIR / "london_validation_trades.csv", index=False)

    global_stats = summarize_global(trades)
    global_stats.to_csv(OUTPUT_DIR / "london_validation_global.csv", index=False)

    yearly_frames = []
    for c in COST_SCENARIOS_PIPS:
        yearly_frames.append(summarize_yearly(trades, c))
    yearly_stats = pd.concat(yearly_frames, axis=0, ignore_index=True)
    yearly_stats.to_csv(OUTPUT_DIR / "london_validation_yearly.csv", index=False)

    print("\n=== LONDON VALIDATION GATE — GLOBAL ===\n")
    print(global_stats.to_string(index=False))

    print("\n=== LONDON VALIDATION GATE — YEARLY (cost=1.0 pip) ===\n")
    print(yearly_stats[yearly_stats["cost_pips"] == 1.0].to_string(index=False))

    print(f"\nSaved trades: {OUTPUT_DIR / 'london_validation_trades.csv'}")
    print(f"Saved global: {OUTPUT_DIR / 'london_validation_global.csv'}")
    print(f"Saved yearly: {OUTPUT_DIR / 'london_validation_yearly.csv'}")


if __name__ == "__main__":
    main()
