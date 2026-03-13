from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


EVENTS_PATH = Path(
    "results/research/opening_momentum/ny_event_surface_enriched.csv"
)

OUTPUT_PATH = Path(
    "results/research/opening_momentum/ny_efficiency_threshold_scan.csv"
)

THRESHOLD_Q = 0.80
ENTRY_DELAY = 30
HOLDING = 30
PIP_SIZE = 0.0001
COST_PIPS = 1.0

EFF_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year

    out["impulse_efficiency"] = np.where(
        out["range_30m"] > 0,
        out["ret_30m"].abs() / out["range_30m"],
        np.nan,
    )

    return out.dropna(subset=["impulse_efficiency"]).copy()


def summarize_yearly(trades: pd.DataFrame) -> tuple[int, int, float]:
    yearly = (
        trades.groupby("year")
        .agg(mean_return=("net_ret", "mean"), n_trades=("net_ret", "count"))
        .reset_index()
    )

    positive_years = int((yearly["mean_return"] > 0).sum())
    total_years = int(len(yearly))
    ratio = positive_years / total_years if total_years > 0 else 0.0

    return positive_years, total_years, ratio


def main() -> None:
    df = pd.read_csv(EVENTS_PATH)
    df = prepare_df(df)

    q_low = df["ret_30m"].quantile(1 - THRESHOLD_Q)
    q_high = df["ret_30m"].quantile(THRESHOLD_Q)

    ret_col = f"ret_fwd_{HOLDING}m_from_{ENTRY_DELAY}m"
    cost_ret = COST_PIPS * PIP_SIZE

    rows: list[dict] = []

    for eff_thr in EFF_THRESHOLDS:
        sub = df[df["impulse_efficiency"] >= eff_thr].copy()

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
                "eff_threshold": eff_thr,
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
                "eff_mean": sub["impulse_efficiency"].mean(),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["sharpe_per_trade_net", "n_trades", "mean_return_net"],
        ascending=[False, False, False],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\n=== NY EFFICIENCY THRESHOLD SCAN ===\n")
    print(out.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
