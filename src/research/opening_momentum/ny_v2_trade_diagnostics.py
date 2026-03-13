from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


EVENTS_PATH = Path(
    "results/research/opening_momentum/ny_event_surface_enriched.csv"
)

OUTPUT_DIR = Path(
    "results/research/opening_momentum/ny_v2_trade_diagnostics"
)

THRESHOLD_Q = 0.80
EFF_THRESHOLD = 0.70
ENTRY_DELAY = 30
HOLDING = 30
PIP_SIZE = 0.0001
COST_PIPS = 1.0


def build_trades(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year

    out["impulse_efficiency"] = np.where(
        out["range_30m"] > 0,
        out["ret_30m"].abs() / out["range_30m"],
        np.nan,
    )
    out = out.dropna(subset=["impulse_efficiency"]).copy()

    # IMPORTANTE:
    # Signal thresholds se calculan sobre TODO el universo NY,
    # no sobre el subconjunto filtrado.
    q_low = out["ret_30m"].quantile(1 - THRESHOLD_Q)
    q_high = out["ret_30m"].quantile(THRESHOLD_Q)

    # Luego aplicamos gating
    gated = out[out["impulse_efficiency"] >= EFF_THRESHOLD].copy()

    ret_col = f"ret_fwd_{HOLDING}m_from_{ENTRY_DELAY}m"
    cost_ret = COST_PIPS * PIP_SIZE

    longs = gated[gated["ret_30m"] > q_high].copy()
    shorts = gated[gated["ret_30m"] < q_low].copy()

    longs["side"] = "long"
    longs["gross_ret"] = longs[ret_col]

    shorts["side"] = "short"
    shorts["gross_ret"] = -shorts[ret_col]

    trades = pd.concat([longs, shorts], axis=0).dropna(subset=["gross_ret"]).copy()
    trades = trades.sort_values(["date", "side"]).reset_index(drop=True)

    trades["net_ret"] = trades["gross_ret"] - cost_ret
    trades["trade_id"] = np.arange(1, len(trades) + 1)

    return trades


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity - running_max
    return float(dd.min())


def max_losing_streak(returns: pd.Series) -> int:
    streak = 0
    max_streak = 0
    for r in returns:
        if r <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def summarize_distribution(trades: pd.DataFrame) -> pd.DataFrame:
    r = trades["net_ret"]

    return pd.DataFrame(
        [
            {
                "n_trades": len(r),
                "mean_return": r.mean(),
                "median_return": r.median(),
                "std_return": r.std(),
                "min_return": r.min(),
                "max_return": r.max(),
                "p05": r.quantile(0.05),
                "p25": r.quantile(0.25),
                "p75": r.quantile(0.75),
                "p95": r.quantile(0.95),
                "winrate": (r > 0).mean(),
                "skew_proxy_mean_minus_median": r.mean() - r.median(),
            }
        ]
    )


def summarize_streaks_and_dd(trades: pd.DataFrame) -> pd.DataFrame:
    eq = trades["net_ret"].cumsum()

    return pd.DataFrame(
        [
            {
                "final_equity_ret": eq.iloc[-1],
                "max_drawdown_ret": max_drawdown(eq),
                "max_losing_streak": max_losing_streak(trades["net_ret"]),
            }
        ]
    )


def summarize_concentration(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = trades.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)

    by_year = (
        df.groupby("year")
        .agg(
            n_trades=("net_ret", "count"),
            mean_return=("net_ret", "mean"),
            total_return=("net_ret", "sum"),
        )
        .reset_index()
    )

    by_month = (
        df.groupby("month")
        .agg(
            n_trades=("net_ret", "count"),
            total_return=("net_ret", "sum"),
        )
        .reset_index()
        .sort_values("total_return", ascending=False)
    )

    return by_year, by_month


def build_equity(trades: pd.DataFrame) -> pd.DataFrame:
    eq = trades[["trade_id", "date", "side", "net_ret"]].copy()
    eq["equity_ret"] = eq["net_ret"].cumsum()
    return eq


def main() -> None:
    df = pd.read_csv(EVENTS_PATH)
    trades = build_trades(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)

    dist_summary = summarize_distribution(trades)
    dist_summary.to_csv(OUTPUT_DIR / "distribution_summary.csv", index=False)

    dd_summary = summarize_streaks_and_dd(trades)
    dd_summary.to_csv(OUTPUT_DIR / "dd_streak_summary.csv", index=False)

    by_year, by_month = summarize_concentration(trades)
    by_year.to_csv(OUTPUT_DIR / "by_year.csv", index=False)
    by_month.to_csv(OUTPUT_DIR / "by_month.csv", index=False)

    equity = build_equity(trades)
    equity.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)

    print("\n=== NY V2 TRADE DISTRIBUTION SUMMARY ===\n")
    print(dist_summary.to_string(index=False))

    print("\n=== NY V2 DRAWDOWN / STREAK SUMMARY ===\n")
    print(dd_summary.to_string(index=False))

    print("\n=== NY V2 BY YEAR ===\n")
    print(by_year.to_string(index=False))

    print("\n=== NY V2 TOP 12 MONTHS BY TOTAL RETURN ===\n")
    print(by_month.head(12).to_string(index=False))

    print(f"\nSaved trades: {OUTPUT_DIR / 'trades.csv'}")
    print(f"Saved equity: {OUTPUT_DIR / 'equity_curve.csv'}")
    print(f"Saved summaries in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
