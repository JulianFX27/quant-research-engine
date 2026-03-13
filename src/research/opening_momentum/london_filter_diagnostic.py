from __future__ import annotations

from pathlib import Path
import pandas as pd


EVENTS_PATH = Path(
    "results/research/opening_momentum/london_event_surface_dstaware.csv"
)

OUTPUT_PATH = Path(
    "results/research/opening_momentum/london_filter_diagnostic.csv"
)

ENTRY_DELAY = 30
HOLDING = 30
PIP_SIZE = 0.0001
COST_PIPS = 1.0

SCENARIOS = [
    {"name": "base_q90_no_range", "threshold_q": 0.90, "range_top_frac": None},
    {"name": "q90_top40", "threshold_q": 0.90, "range_top_frac": 0.40},
    {"name": "q925_top35", "threshold_q": 0.925, "range_top_frac": 0.35},
]


def apply_range_filter(df: pd.DataFrame, top_frac: float | None) -> pd.DataFrame:
    if top_frac is None:
        out = df.copy()
        out["range_cutoff"] = None
        return out

    cutoff = df["range_30m"].quantile(1 - top_frac)
    out = df[df["range_30m"] >= cutoff].copy()
    out["range_cutoff"] = cutoff
    return out


def build_trades(df: pd.DataFrame, threshold_q: float, range_top_frac: float | None) -> pd.DataFrame:
    sub = apply_range_filter(df, range_top_frac)

    q_low = sub["ret_30m"].quantile(1 - threshold_q)
    q_high = sub["ret_30m"].quantile(threshold_q)

    ret_col = f"ret_fwd_{HOLDING}m_from_{ENTRY_DELAY}m"

    longs = sub[sub["ret_30m"] > q_high].copy()
    shorts = sub[sub["ret_30m"] < q_low].copy()

    longs["side"] = "long"
    longs["gross_ret"] = longs[ret_col]

    shorts["side"] = "short"
    shorts["gross_ret"] = -shorts[ret_col]

    trades = pd.concat([longs, shorts], axis=0).dropna(subset=["gross_ret"]).copy()
    trades["date"] = pd.to_datetime(trades["date"])
    trades["year"] = trades["date"].dt.year
    trades["net_ret"] = trades["gross_ret"] - COST_PIPS * PIP_SIZE

    return trades.sort_values(["date", "side"]).reset_index(drop=True)


def summarize(trades: pd.DataFrame) -> dict:
    yearly = (
        trades.groupby("year")
        .agg(
            n_trades=("net_ret", "count"),
            mean_return=("net_ret", "mean"),
        )
        .reset_index()
    )

    positive_years = int((yearly["mean_return"] > 0).sum())
    total_years = int(len(yearly))
    positive_ratio = positive_years / total_years if total_years > 0 else 0.0

    return {
        "n_trades": len(trades),
        "mean_return_net": trades["net_ret"].mean(),
        "std_return_net": trades["net_ret"].std(),
        "winrate_net": (trades["net_ret"] > 0).mean(),
        "sharpe_per_trade_net": trades["net_ret"].mean() / trades["net_ret"].std(),
        "positive_years": positive_years,
        "total_years": total_years,
        "positive_year_ratio": positive_ratio,
    }


def main() -> None:
    df = pd.read_csv(EVENTS_PATH)

    rows = []

    for sc in SCENARIOS:
        trades = build_trades(
            df=df,
            threshold_q=sc["threshold_q"],
            range_top_frac=sc["range_top_frac"],
        )

        stats = summarize(trades)
        rows.append(
            {
                "scenario": sc["name"],
                "threshold_q": sc["threshold_q"],
                "range_top_frac": sc["range_top_frac"],
                "cost_pips": COST_PIPS,
                **stats,
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["sharpe_per_trade_net", "n_trades"],
        ascending=[False, False],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\n=== LONDON FILTER DIAGNOSTIC ===\n")
    print(out.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
