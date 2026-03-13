from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


EUR_TRADES_PATH = Path(
    "results/research/opening_momentum/ny_v2_validation_gate/base_trades.csv"
)

GBP_TRADES_PATH = Path(
    "results/research/opening_momentum/gbpusd/ny_v2_validation_gate/base_trades.csv"
)

OUTPUT_DIR = Path(
    "results/research/opening_momentum/portfolio_validation_gate"
)

PIP_SIZE = 0.0001
COST_PIPS = 1.0


def load_trades(path: Path, asset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["asset"] = asset
    df["net_ret"] = df["gross_ret"] - COST_PIPS * PIP_SIZE
    return df[["date", "asset", "side", "gross_ret", "net_ret"]].copy()


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


def summarize_returns(df: pd.DataFrame, ret_col: str) -> dict:
    r = df[ret_col]
    eq = r.cumsum()

    return {
        "n_trades": len(r),
        "mean_return": r.mean(),
        "median_return": r.median(),
        "std_return": r.std(),
        "winrate": (r > 0).mean(),
        "sharpe_per_trade": (r.mean() / r.std()) if r.std() > 0 else np.nan,
        "final_equity_ret": eq.iloc[-1] if len(eq) else np.nan,
        "max_drawdown_ret": max_drawdown(eq) if len(eq) else np.nan,
        "max_losing_streak": max_losing_streak(r) if len(r) else np.nan,
    }


def build_individual_table(eur: pd.DataFrame, gbp: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"portfolio": "EURUSD_only", **summarize_returns(eur, "net_ret")},
        {"portfolio": "GBPUSD_only", **summarize_returns(gbp, "net_ret")},
    ]
    return pd.DataFrame(rows)


def build_portfolio_equal_weight(eur: pd.DataFrame, gbp: pd.DataFrame) -> pd.DataFrame:
    eur_daily = eur.groupby("date", as_index=False).agg(ret_eur=("net_ret", "sum"))
    gbp_daily = gbp.groupby("date", as_index=False).agg(ret_gbp=("net_ret", "sum"))

    merged = eur_daily.merge(gbp_daily, on="date", how="outer").fillna(0.0)
    merged = merged.sort_values("date").reset_index(drop=True)

    merged["portfolio_ret"] = 0.5 * merged["ret_eur"] + 0.5 * merged["ret_gbp"]
    merged["equity_ret"] = merged["portfolio_ret"].cumsum()

    return merged


def build_portfolio_all_trades(eur: pd.DataFrame, gbp: pd.DataFrame) -> pd.DataFrame:
    all_trades = pd.concat([eur, gbp], ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    all_trades["equity_ret"] = all_trades["net_ret"].cumsum()
    return all_trades


def build_portfolio_table(equal_weight_daily: pd.DataFrame, all_trades: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "portfolio": "combined_all_trades",
            **summarize_returns(all_trades, "net_ret"),
        },
        {
            "portfolio": "combined_equal_weight_daily",
            **summarize_returns(equal_weight_daily, "portfolio_ret"),
        },
    ]
    return pd.DataFrame(rows)


def build_correlation_table(eur: pd.DataFrame, gbp: pd.DataFrame) -> pd.DataFrame:
    eur_daily = eur.groupby("date", as_index=False).agg(ret_eur=("net_ret", "sum"))
    gbp_daily = gbp.groupby("date", as_index=False).agg(ret_gbp=("net_ret", "sum"))

    merged = eur_daily.merge(gbp_daily, on="date", how="outer").fillna(0.0)
    corr = merged["ret_eur"].corr(merged["ret_gbp"])

    overlap_days = int(((merged["ret_eur"] != 0) & (merged["ret_gbp"] != 0)).sum())

    return pd.DataFrame(
        [
            {
                "daily_return_corr": corr,
                "overlap_days_with_both_trades": overlap_days,
                "n_days_total": len(merged),
            }
        ]
    )


def main() -> None:
    eur = load_trades(EUR_TRADES_PATH, "EURUSD")
    gbp = load_trades(GBP_TRADES_PATH, "GBPUSD")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    individual_table = build_individual_table(eur, gbp)
    equal_weight_daily = build_portfolio_equal_weight(eur, gbp)
    all_trades = build_portfolio_all_trades(eur, gbp)
    portfolio_table = build_portfolio_table(equal_weight_daily, all_trades)
    corr_table = build_correlation_table(eur, gbp)

    individual_table.to_csv(OUTPUT_DIR / "individual_table.csv", index=False)
    equal_weight_daily.to_csv(OUTPUT_DIR / "equal_weight_daily.csv", index=False)
    all_trades.to_csv(OUTPUT_DIR / "all_trades_combined.csv", index=False)
    portfolio_table.to_csv(OUTPUT_DIR / "portfolio_table.csv", index=False)
    corr_table.to_csv(OUTPUT_DIR / "correlation_table.csv", index=False)

    print("\n=== PORTFOLIO VALIDATION GATE — INDIVIDUAL ===\n")
    print(individual_table.to_string(index=False))

    print("\n=== PORTFOLIO VALIDATION GATE — COMBINED ===\n")
    print(portfolio_table.to_string(index=False))

    print("\n=== PORTFOLIO VALIDATION GATE — CORRELATION ===\n")
    print(corr_table.to_string(index=False))

    print(f"\nSaved individual: {OUTPUT_DIR / 'individual_table.csv'}")
    print(f"Saved combined: {OUTPUT_DIR / 'portfolio_table.csv'}")
    print(f"Saved corr: {OUTPUT_DIR / 'correlation_table.csv'}")
    print(f"Saved equal-weight daily: {OUTPUT_DIR / 'equal_weight_daily.csv'}")
    print(f"Saved all trades: {OUTPUT_DIR / 'all_trades_combined.csv'}")


if __name__ == "__main__":
    main()
