from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src" / "research" / "opening_momentum"))

from session_times import ny_open_utc  # noqa: E402


DATA_PATH = ROOT / "data" / "canonical" / "GBPUSD_M5_2014-01-01__2024-12-31__dukascopy_v2_clean.csv"
RESULTS_DIR = ROOT / "results" / "research" / "nyom_v3"

ENTRY_DELAY_MIN = 30
HOLDING_MIN = 30


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True)
    elif "time" in df.columns:
        ts = pd.to_datetime(df["time"], utc=True)
    else:
        raise ValueError(f"No timestamp column found. Columns={list(df.columns)}")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    # Compatibilidad con ny_open_utc(): UTC naive
    df["ts"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    df = df.sort_values("ts").reset_index(drop=True)
    df["date"] = df["ts"].dt.date

    return df


def _scalar_at(px: pd.DataFrame, ts: pd.Timestamp, col: str) -> float:
    value = px.loc[ts, col]
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)


def build_ny_events(df: pd.DataFrame) -> pd.DataFrame:
    px = df.set_index("ts").sort_index()

    events: list[dict] = []

    for d in sorted(df["date"].unique()):
        t0 = pd.Timestamp(ny_open_utc(d))
        t30 = t0 + pd.Timedelta(minutes=30)
        entry_ts = t30
        exit_ts = entry_ts + pd.Timedelta(minutes=HOLDING_MIN)

        try:
            open_price = _scalar_at(px, t0, "open")
            close_30 = _scalar_at(px, t30, "close")
            _ = _scalar_at(px, exit_ts, "close")  # validar que exista salida
        except KeyError:
            continue

        window = px.loc[t0:t30]
        if window.empty:
            continue

        high = float(window["high"].max())
        low = float(window["low"].min())

        range_30m = high - low
        if range_30m <= 0:
            continue

        ret_30m = (close_30 / open_price) - 1.0
        efficiency = abs(ret_30m) / range_30m
        direction = 1 if ret_30m > 0 else -1

        events.append(
            {
                "session_date": d,
                "open_ts": t0,
                "impulse_end_ts": t30,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "ret_30m": ret_30m,
                "range_30m": range_30m,
                "efficiency": efficiency,
                "direction": direction,
            }
        )

    return pd.DataFrame(events)


def assign_buckets(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    events = events.copy()

    ret_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    eff_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    ret_bucket_count = min(5, events["ret_30m"].abs().nunique())
    eff_bucket_count = min(5, events["efficiency"].nunique())

    if ret_bucket_count < 2:
        raise ValueError("Not enough unique ret_30m values to assign quantile buckets.")

    if eff_bucket_count < 2:
        raise ValueError("Not enough unique efficiency values to assign quantile buckets.")

    events["ret30_bucket"] = pd.qcut(
        events["ret_30m"].abs(),
        q=ret_bucket_count,
        labels=ret_labels[:ret_bucket_count],
        duplicates="drop",
    )

    events["eff_bucket"] = pd.qcut(
        events["efficiency"],
        q=eff_bucket_count,
        labels=eff_labels[:eff_bucket_count],
        duplicates="drop",
    )

    return events


def policy_A(events: pd.DataFrame) -> pd.DataFrame:
    return events[
        (events["ret30_bucket"] == "Q4") &
        (events["eff_bucket"] == "Q4")
    ].copy()


def policy_B(events: pd.DataFrame) -> pd.DataFrame:
    return events[
        (events["ret30_bucket"].isin(["Q3", "Q4"])) &
        (events["eff_bucket"] == "Q4")
    ].copy()


def simulate_trades(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "session_date",
                "open_ts",
                "impulse_end_ts",
                "entry_ts",
                "exit_ts",
                "direction",
                "ret_30m",
                "range_30m",
                "efficiency",
                "ret30_bucket",
                "eff_bucket",
                "entry_price",
                "exit_price",
                "pnl",
            ]
        )

    px = df.set_index("ts").sort_index()

    trades: list[dict] = []

    for _, e in events.iterrows():
        try:
            entry_price = _scalar_at(px, e["entry_ts"], "open")
            exit_price = _scalar_at(px, e["exit_ts"], "close")
        except KeyError:
            continue

        pnl = float(e["direction"]) * (exit_price - entry_price)

        trades.append(
            {
                "session_date": e["session_date"],
                "open_ts": e["open_ts"],
                "impulse_end_ts": e["impulse_end_ts"],
                "entry_ts": e["entry_ts"],
                "exit_ts": e["exit_ts"],
                "direction": e["direction"],
                "ret_30m": e["ret_30m"],
                "range_30m": e["range_30m"],
                "efficiency": e["efficiency"],
                "ret30_bucket": e["ret30_bucket"],
                "eff_bucket": e["eff_bucket"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
            }
        )

    return pd.DataFrame(trades)


def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "winrate": None,
            "avg_pnl": None,
            "profit_factor": None,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss_abs = float(abs(losses["pnl"].sum())) if not losses.empty else 0.0

    profit_factor = None
    if gross_loss_abs > 0:
        profit_factor = gross_profit / gross_loss_abs

    return {
        "n_trades": int(len(trades)),
        "winrate": float((trades["pnl"] > 0).mean()),
        "avg_pnl": float(trades["pnl"].mean()),
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": -gross_loss_abs,
    }


def export_results(
    policy_name: str,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict,
) -> None:
    out = RESULTS_DIR / policy_name
    out.mkdir(parents=True, exist_ok=True)

    events.to_csv(out / "events.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def run() -> None:
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("Building DST-aware NY events...")
    events = build_ny_events(df)
    if events.empty:
        raise ValueError("No NY events were generated. Check dataset coverage and session alignment.")

    print(f"NY events generated: {len(events)}")

    print("Assigning quantile buckets...")
    events = assign_buckets(events)

    print("Running Policy A...")
    events_A = policy_A(events)
    trades_A = simulate_trades(df, events_A)
    metrics_A = compute_metrics(trades_A)
    export_results("policy_A", events_A, trades_A, metrics_A)

    print("Running Policy B...")
    events_B = policy_B(events)
    trades_B = simulate_trades(df, events_B)
    metrics_B = compute_metrics(trades_B)
    export_results("policy_B", events_B, trades_B, metrics_B)

    print("NYOM v3 research run complete.")
    print(f"Policy A trades: {metrics_A['n_trades']}")
    print(f"Policy B trades: {metrics_B['n_trades']}")


if __name__ == "__main__":
    run()