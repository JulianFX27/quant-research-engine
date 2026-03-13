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

USDJPY_TRADES_PATH = Path(
    "results/research/opening_momentum/usdjpy/ny_v2_validation_gate/base_trades.csv"
)

OUTPUT_DIR = Path(
    "results/research/opening_momentum/portfolio_validation_gate_3assets"
)

PIP_SIZE_NON_JPY = 0.0001
PIP_SIZE_JPY = 0.01
COST_PIPS = 1.0


def load_trades(path: Path, asset: str, is_jpy: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["asset"] = asset

    if is_jpy:
        if "entry_price" not in df.columns:
            raise ValueError(f"{asset} base_trades.csv must contain entry_price")
        df["cost_ret"] = (COST_PIPS * PIP_SIZE_JPY) / df["entry_price"]
    else:
        df["cost_ret"] = COST_PIPS * PIP_SIZE_NON_JPY

    df["net_ret"] = df["gross_ret"] - df["cost_ret"]

    keep_cols = ["date", "asset", "side", "gross_ret", "net_ret"]
    if "entry_price" in df.columns:
        keep_cols.append("entry_price")
    return df[keep_cols].copy()


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


def build_individual_table(eur: pd.DataFrame, gbp: pd.DataFrame, jpy: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"portfolio": "EURUSD_only", **summarize_returns(eur, "net_ret")},
        {"portfolio": "GBPUSD_only", **summarize_returns(gbp, "net_ret")},
        {"portfolio": "USDJPY_only", **summarize_returns(jpy, "net_ret")},
    ]
    return pd.DataFrame(rows)


def build_equal_weight_daily(eur: pd.DataFrame, gbp: pd.DataFrame, jpy: pd.DataFrame) -> pd.DataFrame:
    eur_daily = eur.groupby("date", as_index=False).agg(ret_eur=("net_ret", "sum"))
    gbp_daily = gbp.groupby("date", as_index=False).agg(ret_gbp=("net_ret", "sum"))
    jpy_daily = jpy.groupby("date", as_index=False).agg(ret_jpy=("net_ret", "sum"))

    merged = eur_daily.merge(gbp_daily, on="date", how="outer")
    merged = merged.merge(jpy_daily, on="date", how="outer")
    merged = merged.fillna(0.0).sort_values("date").reset_index(drop=True)

    merged["portfolio_ret"] = (
        merged["ret_eur"] + merged["ret_gbp"] + merged["ret_jpy"]
    ) / 3.0
    merged["equity_ret"] = merged["portfolio_ret"].cumsum()

    return merged


def build_all_trades(eur: pd.DataFrame, gbp: pd.DataFrame, jpy: pd.DataFrame) -> pd.DataFrame:
    all_trades = pd.concat([eur, gbp, jpy], ignore_index=True)
    all_trades = all_trades.sort_values(["date", "asset"]).reset_index(drop=True)
    all_trades["equity_ret"] = all_trades["net_ret"].cumsum()
    return all_trades


def build_portfolio_table(equal_weight_daily: pd.DataFrame, all_trades: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"portfolio": "combined_all_trades_3assets", **summarize_returns(all_trades, "net_ret")},
        {"portfolio": "combined_equal_weight_daily_3assets", **summarize_returns(equal_weight_daily, "portfolio_ret")},
    ]
    return pd.DataFrame(rows)


def build_correlation_table(eur: pd.DataFrame, gbp: pd.DataFrame, jpy: pd.DataFrame) -> pd.DataFrame:
    eur_daily = eur.groupby("date", as_index=False).agg(ret_eur=("net_ret", "sum"))
    gbp_daily = gbp.groupby("date", as_index=False).agg(ret_gbp=("net_ret", "sum"))
    jpy_daily = jpy.groupby("date", as_index=False).agg(ret_jpy=("net_ret", "sum"))

    merged = eur_daily.merge(gbp_daily, on="date", how="outer")
    merged = merged.merge(jpy_daily, on="date", how="outer")
    merged = merged.fillna(0.0)

    corr = merged[["ret_eur", "ret_gbp", "ret_jpy"]].corr()

    overlap_all_3 = int(((merged["ret_eur"] != 0) & (merged["ret_gbp"] != 0) & (merged["ret_jpy"] != 0)).sum())

    summary = pd.DataFrame(
        [
            {
                "corr_eur_gbp": corr.loc["ret_eur", "ret_gbp"],
                "corr_eur_jpy": corr.loc["ret_eur", "ret_jpy"],
                "corr_gbp_jpy": corr.loc["ret_gbp", "ret_jpy"],
                "overlap_days_all_3": overlap_all_3,
                "n_days_total": len(merged),
            }
        ]
    )

    return summary


def main() -> None:
    eur = load_trades(EUR_TRADES_PATH, "EURUSD", is_jpy=False)
    gbp = load_trades(GBP_TRADES_PATH, "GBPUSD", is_jpy=False)
    jpy = load_trades(USDJPY_TRADES_PATH, "USDJPY", is_jpy=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    individual_table = build_individual_table(eur, gbp, jpy)
    equal_weight_daily = build_equal_weight_daily(eur, gbp, jpy)
    all_trades = build_all_trades(eur, gbp, jpy)
    portfolio_table = build_portfolio_table(equal_weight_daily, all_trades)
    corr_table = build_correlation_table(eur, gbp, jpy)

    individual_table.to_csv(OUTPUT_DIR / "individual_table.csv", index=False)
    equal_weight_daily.to_csv(OUTPUT_DIR / "equal_weight_daily.csv", index=False)
    all_trades.to_csv(OUTPUT_DIR / "all_trades_combined.csv", index=False)
    portfolio_table.to_csv(OUTPUT_DIR / "portfolio_table.csv", index=False)
    corr_table.to_csv(OUTPUT_DIR / "correlation_table.csv", index=False)

    print("\n=== PORTFOLIO VALIDATION GATE 3 ASSETS — INDIVIDUAL ===\n")
    print(individual_table.to_string(index=False))

    print("\n=== PORTFOLIO VALIDATION GATE 3 ASSETS — COMBINED ===\n")
    print(portfolio_table.to_string(index=False))

    print("\n=== PORTFOLIO VALIDATION GATE 3 ASSETS — CORRELATION ===\n")
    print(corr_table.to_string(index=False))

    print(f"\nSaved individual: {OUTPUT_DIR / 'individual_table.csv'}")
    print(f"Saved combined: {OUTPUT_DIR / 'portfolio_table.csv'}")
    print(f"Saved corr: {OUTPUT_DIR / 'correlation_table.csv'}")
    print(f"Saved equal-weight daily: {OUTPUT_DIR / 'equal_weight_daily.csv'}")
    print(f"Saved all trades: {OUTPUT_DIR / 'all_trades_combined.csv'}")


if __name__ == "__main__":
    main()

