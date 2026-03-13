from __future__ import annotations

from pathlib import Path
import pandas as pd


EVENTS_PATH = Path(
    "results/research/opening_momentum/london_event_surface_dstaware.csv"
)

OUTPUT_PATH = Path(
    "results/research/opening_momentum/london_expand_n_study.csv"
)

ENTRY_DELAY = 30
HOLDING = 30
PIP_SIZE = 0.0001
COST_PIPS = 1.0

THRESHOLDS = [0.90, 0.925, 0.95]
RANGE_TOPS = [0.25, 0.30, 0.35, 0.40]


def apply_range_filter(df: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    cutoff = df["range_30m"].quantile(1 - top_frac)
    out = df[df["range_30m"] >= cutoff].copy()
    out["range_top_frac"] = top_frac
    out["range_cutoff"] = cutoff
    return out


def summarize_yearly(trades: pd.DataFrame) -> tuple[int, int, float]:
    yearly = (
        trades.groupby("year")
        .agg(mean_return=("net_ret", "mean"), n_trades=("net_ret", "count"))
        .reset_index()
    )
    positive_years = int((yearly["mean_return"] > 0).sum())
    total_years = int(len(yearly))
    positive_ratio = positive_years / total_years if total_years > 0 else 0.0
    return positive_years, total_years, positive_ratio


def main() -> None:
    df = pd.read_csv(EVENTS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    ret_col = f"ret_fwd_{HOLDING}m_from_{ENTRY_DELAY}m"
    cost_ret = COST_PIPS * PIP_SIZE

    rows: list[dict] = []

    for q in THRESHOLDS:
        for top_frac in RANGE_TOPS:
            sub = apply_range_filter(df, top_frac)

            q_low = sub["ret_30m"].quantile(1 - q)
            q_high = sub["ret_30m"].quantile(q)

            longs = sub[sub["ret_30m"] > q_high].copy()
            shorts = sub[sub["ret_30m"] < q_low].copy()

            longs["side"] = "long"
            longs["gross_ret"] = longs[ret_col]

            shorts["side"] = "short"
            shorts["gross_ret"] = -shorts[ret_col]

            trades = pd.concat([longs, shorts], axis=0).dropna(subset=["gross_ret"]).copy()

            if len(trades) < 20:
                continue

            trades["net_ret"] = trades["gross_ret"] - cost_ret

            mean_ret = trades["net_ret"].mean()
            std_ret = trades["net_ret"].std()
            winrate = (trades["net_ret"] > 0).mean()
            sharpe = mean_ret / std_ret if std_ret > 0 else None

            positive_years, total_years, positive_ratio = summarize_yearly(trades)

            rows.append(
                {
                    "threshold_q": q,
                    "range_top_frac": top_frac,
                    "range_cutoff": sub["range_30m"].quantile(1 - top_frac),
                    "cost_pips": COST_PIPS,
                    "n_longs": len(longs),
                    "n_shorts": len(shorts),
                    "n_trades": len(trades),
                    "mean_return_net": mean_ret,
                    "std_return_net": std_ret,
                    "winrate_net": winrate,
                    "sharpe_per_trade_net": sharpe,
                    "positive_years": positive_years,
                    "total_years": total_years,
                    "positive_year_ratio": positive_ratio,
                }
            )

    out = pd.DataFrame(rows).sort_values(
        ["sharpe_per_trade_net", "n_trades", "mean_return_net"],
        ascending=[False, False, False],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\n=== LONDON EXPAND-N STUDY (post-cost) ===\n")
    print(out.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
