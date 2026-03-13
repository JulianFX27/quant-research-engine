from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd

# Reutiliza tu lógica DST-aware ya validada
from src.research.opening_momentum.session_times import ny_open_utc


@dataclass(frozen=True)
class NYOpeningMomentumV2Config:
    symbol: str
    threshold_q: float = 0.80
    impulse_efficiency_min: float = 0.70
    entry_delay_min: int = 30
    holding_min: int = 30
    pip_size_price: float = 0.0001  # EURUSD/GBPUSD; para USDJPY usar 0.01
    cost_pips: float = 1.0


def _load_ohlc(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    df["date"] = df["time"].dt.date
    return df


def build_event_surface_from_ohlc(
    ohlc: pd.DataFrame,
    entry_delays_min: Iterable[int] = (30,),
    holding_periods_min: Iterable[int] = (30,),
) -> pd.DataFrame:
    px = ohlc.set_index("time").sort_index()
    dates = sorted(ohlc["date"].unique())

    records: list[dict] = []

    for d in dates:
        t0 = ny_open_utc(d)
        midnight = pd.Timestamp(f"{d} 00:00:00")

        try:
            midnight_open = float(px.loc[midnight]["open"])
            open_0 = float(px.loc[t0]["open"])
            close_30 = float(px.loc[t0 + pd.Timedelta(minutes=30)]["close"])
        except KeyError:
            continue

        window_30 = px.loc[t0 : t0 + pd.Timedelta(minutes=30)]
        if window_30.empty:
            continue

        rec: dict = {
            "date": pd.Timestamp(d),
            "open_time": t0,
            "signal_time": t0 + pd.Timedelta(minutes=30),
            "open_price": open_0,
            "overnight_ret_to_ny_open": (open_0 - midnight_open) / midnight_open,
            "ret_30m": (close_30 - open_0) / open_0,
            "range_30m": (window_30["high"].max() - window_30["low"].min()) / open_0,
            "direction_30m": 1 if close_30 > open_0 else -1,
        }

        ok = True
        for delay in entry_delays_min:
            entry_time = t0 + pd.Timedelta(minutes=delay)
            try:
                entry_price = float(px.loc[entry_time]["open"])
            except KeyError:
                ok = False
                break

            rec[f"entry_time_{delay}m"] = entry_time
            rec[f"entry_price_{delay}m"] = entry_price

            for hold in holding_periods_min:
                exit_time = entry_time + pd.Timedelta(minutes=hold)
                try:
                    exit_price = float(px.loc[exit_time]["close"])
                except KeyError:
                    ok = False
                    break

                rec[f"exit_time_{hold}m_from_{delay}m"] = exit_time
                rec[f"exit_price_{hold}m_from_{delay}m"] = exit_price
                rec[f"ret_fwd_{hold}m_from_{delay}m"] = (exit_price - entry_price) / entry_price

            if not ok:
                break

        if ok:
            records.append(rec)

    return pd.DataFrame.from_records(records)


def build_trades_from_surface(
    surface: pd.DataFrame,
    cfg: NYOpeningMomentumV2Config,
) -> pd.DataFrame:
    if surface.empty:
        return pd.DataFrame()

    df = surface.copy()
    df["impulse_efficiency"] = np.where(
        df["range_30m"] > 0,
        df["ret_30m"].abs() / df["range_30m"],
        np.nan,
    )
    df = df.dropna(subset=["impulse_efficiency"]).copy()

    q_low = df["ret_30m"].quantile(1 - cfg.threshold_q)
    q_high = df["ret_30m"].quantile(cfg.threshold_q)

    gated = df[df["impulse_efficiency"] >= cfg.impulse_efficiency_min].copy()

    entry_price_col = f"entry_price_{cfg.entry_delay_min}m"
    entry_time_col = f"entry_time_{cfg.entry_delay_min}m"
    exit_price_col = f"exit_price_{cfg.holding_min}m_from_{cfg.entry_delay_min}m"
    exit_time_col = f"exit_time_{cfg.holding_min}m_from_{cfg.entry_delay_min}m"
    ret_col = f"ret_fwd_{cfg.holding_min}m_from_{cfg.entry_delay_min}m"

    longs = gated[gated["ret_30m"] > q_high].copy()
    shorts = gated[gated["ret_30m"] < q_low].copy()

    if longs.empty and shorts.empty:
        return pd.DataFrame()

    longs["side"] = "long"
    longs["gross_ret"] = longs[ret_col]

    shorts["side"] = "short"
    shorts["gross_ret"] = -shorts[ret_col]

    trades = pd.concat([longs, shorts], axis=0, ignore_index=True)
    trades = trades.dropna(subset=[entry_price_col, exit_price_col, "gross_ret"]).copy()

    trades["symbol"] = cfg.symbol
    trades["entry_time"] = pd.to_datetime(trades[entry_time_col])
    trades["exit_time"] = pd.to_datetime(trades[exit_time_col])
    trades["entry_price"] = trades[entry_price_col].astype(float)
    trades["exit_price"] = trades[exit_price_col].astype(float)

    trades["cost_ret"] = (cfg.cost_pips * cfg.pip_size_price) / trades["entry_price"]
    trades["net_ret"] = trades["gross_ret"] - trades["cost_ret"]

    trades = trades.sort_values(["entry_time", "side"]).reset_index(drop=True)
    trades["trade_id"] = np.arange(1, len(trades) + 1)

    keep_cols = [
        "trade_id",
        "symbol",
        "date",
        "open_time",
        "signal_time",
        "entry_time",
        "exit_time",
        "side",
        "open_price",
        "entry_price",
        "exit_price",
        "ret_30m",
        "range_30m",
        "impulse_efficiency",
        "overnight_ret_to_ny_open",
        "gross_ret",
        "cost_ret",
        "net_ret",
    ]
    return trades[keep_cols].copy()


def run_strategy_from_csv(
    csv_path: str,
    cfg: NYOpeningMomentumV2Config,
) -> pd.DataFrame:
    ohlc = _load_ohlc(csv_path)
    surface = build_event_surface_from_ohlc(
        ohlc=ohlc,
        entry_delays_min=(cfg.entry_delay_min,),
        holding_periods_min=(cfg.holding_min,),
    )
    return build_trades_from_surface(surface=surface, cfg=cfg)
