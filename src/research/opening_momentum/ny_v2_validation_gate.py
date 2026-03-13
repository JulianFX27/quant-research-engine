from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


EVENTS_PATH = Path(
    "results/research/opening_momentum/ny_event_surface_enriched.csv"
)

OUTPUT_DIR = Path(
    "results/research/opening_momentum/ny_v2_validation_gate"
)

THRESHOLD_Q = 0.80
EFF_THRESHOLD = 0.70
ENTRY_DELAY = 30
HOLDING = 30
PIP_SIZE = 0.0001

COST_SCENARIOS_PIPS = [0.5, 1.0, 1.5, 2.0]


def build_base_trades(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year

    out["impulse_efficiency"] = np.where(
        out["range_30m"] > 0,
        out["ret_30m"].abs() / out["range_30m"],
        np.nan,
    )
    out = out.dropna(subset=["impulse_efficiency"]).copy()

    # Signal thresholds sobre universo NY completo
    q_low = out["ret_30m"].quantile(1 - THRESHOLD_Q)
    q_high = out["ret_30m"].quantile(THRESHOLD_Q)

    gated = out[out["impulse_efficiency"] >= EFF_THRESHOLD].copy()

    ret_col = f"ret_fwd_{HOLDING}m_from_{ENTRY_DELAY}m"

    longs = gated[gated["ret_30m"] > q_high].copy()
    shorts = gated[gated["ret_30m"] < q_low].copy()

    longs["side"] = "long"
    longs["gross_ret"] = longs[ret_col]

    shorts["side"] = "short"
    shorts["gross_ret"] = -shorts[ret_col]

    trades = pd.concat([longs, shorts], axis=0).dropna(subset=["gross_ret"]).copy()
    trades = trades.sort_values(["date", "side"]).reset_index(drop=True)
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


def add_costs(trades: pd.DataFrame, cost_pips: float) -> pd.DataFrame:
    out = trades.copy()
    out["cost_pips"] = cost_pips
    out["net_ret"] = out["gross_ret"] - cost_pips * PIP_SIZE
    return out


def build_global_table(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost in COST_SCENARIOS_PIPS:
        t = add_costs(trades, cost)
        rows.append(
            {
                "cost_pips": cost,
                **summarize_returns(t, "net_ret"),
            }
        )
    return pd.DataFrame(rows)


def build_period_table(trades: pd.DataFrame) -> pd.DataFrame:
    period_map = {
        "2014_2018": (2014, 2018),
        "2019_2024": (2019, 2024),
    }

    rows = []

    for period_name, (y0, y1) in period_map.items():
        sub = trades[(trades["date"].dt.year >= y0) & (trades["date"].dt.year <= y1)].copy()

        for cost in COST_SCENARIOS_PIPS:
            t = add_costs(sub, cost)
            rows.append(
                {
                    "period": period_name,
                    "year_start": y0,
                    "year_end": y1,
                    "cost_pips": cost,
                    **summarize_returns(t, "net_ret"),
                }
            )

    return pd.DataFrame(rows)


def build_yearly_table(trades: pd.DataFrame, cost_pips: float) -> pd.DataFrame:
    t = add_costs(trades, cost_pips)

    rows = []
    for year, sub in t.groupby(t["date"].dt.year):
        rows.append(
            {
                "year": year,
                "cost_pips": cost_pips,
                **summarize_returns(sub, "net_ret"),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(EVENTS_PATH)
    trades = build_base_trades(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades.to_csv(OUTPUT_DIR / "base_trades.csv", index=False)

    global_table = build_global_table(trades)
    period_table = build_period_table(trades)
    yearly_table = build_yearly_table(trades, cost_pips=1.0)

    global_table.to_csv(OUTPUT_DIR / "global_table.csv", index=False)
    period_table.to_csv(OUTPUT_DIR / "period_table.csv", index=False)
    yearly_table.to_csv(OUTPUT_DIR / "yearly_table_cost1.csv", index=False)

    print("\n=== NY V2 VALIDATION GATE — GLOBAL ===\n")
    print(global_table.to_string(index=False))

    print("\n=== NY V2 VALIDATION GATE — BY PERIOD ===\n")
    print(period_table.to_string(index=False))

    print("\n=== NY V2 VALIDATION GATE — YEARLY (cost=1.0 pip) ===\n")
    print(yearly_table.to_string(index=False))

    print(f"\nSaved base trades: {OUTPUT_DIR / 'base_trades.csv'}")
    print(f"Saved global: {OUTPUT_DIR / 'global_table.csv'}")
    print(f"Saved period: {OUTPUT_DIR / 'period_table.csv'}")
    print(f"Saved yearly: {OUTPUT_DIR / 'yearly_table_cost1.csv'}")


if __name__ == "__main__":
    main()
