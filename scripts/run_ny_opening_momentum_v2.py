from __future__ import annotations

import argparse
from pathlib import Path
import json
import yaml
import pandas as pd

from src.strategies.ny_opening_momentum_v2 import (
    NYOpeningMomentumV2Config,
    run_strategy_from_csv,
)


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity - running_max
    return float(dd.min()) if len(dd) else 0.0


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


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "mean_return": None,
            "median_return": None,
            "std_return": None,
            "winrate": None,
            "sharpe_per_trade": None,
            "final_equity_ret": None,
            "max_drawdown_ret": None,
            "max_losing_streak": None,
        }

    r = trades["net_ret"]
    eq = r.cumsum()

    return {
        "n_trades": int(len(r)),
        "mean_return": float(r.mean()),
        "median_return": float(r.median()),
        "std_return": float(r.std()),
        "winrate": float((r > 0).mean()),
        "sharpe_per_trade": float(r.mean() / r.std()) if r.std() > 0 else None,
        "final_equity_ret": float(eq.iloc[-1]),
        "max_drawdown_ret": float(max_drawdown(eq)),
        "max_losing_streak": int(max_losing_streak(r)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = NYOpeningMomentumV2Config(
        symbol=raw["symbol"],
        threshold_q=raw["signal"]["threshold_q"],
        impulse_efficiency_min=raw["signal"]["impulse_efficiency_min"],
        entry_delay_min=raw["signal"]["entry_delay_min"],
        holding_min=raw["signal"]["holding_min"],
        pip_size_price=raw["execution"]["pip_size_price"],
        cost_pips=raw["execution"]["cost_pips"],
    )

    out_dir = Path(raw["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = run_strategy_from_csv(raw["data"]["csv_path"], cfg)
    metrics = summarize(trades)

    trades.to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame(
        {
            "trade_id": trades["trade_id"] if not trades.empty else [],
            "entry_time": trades["entry_time"] if not trades.empty else [],
            "symbol": trades["symbol"] if not trades.empty else [],
            "net_ret": trades["net_ret"] if not trades.empty else [],
            "equity_ret": trades["net_ret"].cumsum() if not trades.empty else [],
        }
    ).to_csv(out_dir / "equity.csv", index=False)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== NY OPENING MOMENTUM V2 ===\n")
    print(f"config: {cfg_path}")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nSaved trades: {out_dir / 'trades.csv'}")
    print(f"Saved equity: {out_dir / 'equity.csv'}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
