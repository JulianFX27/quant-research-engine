from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from zoneinfo import ZoneInfo


@dataclass
class LiveFeatureBuilderConfig:
    # Input OHLCV from MT5 EA
    mt5_csv_path: str
    mt5_time_col: str = "time"
    tz_in: str = "Europe/Prague"   # MT5 timestamps timezone (naive -> localized)
    tf_minutes: int = 5

    # Output enriched CSV
    out_csv_path: str = ""

    # Anchor/session definitions
    ny_tz: str = "America/New_York"
    ny_open_hhmm: str = "08:30"    # adjust if your project defines differently

    # ATR / shock configs (baseline, consistent + deterministic)
    atr_period: int = 14

    # shock_log_ret and shock_z are computed on close-to-close returns:
    # shock_log_ret := log_ret
    # shock_z := zscore(log_ret) over rolling window
    shock_z_window: int = 288      # ~1 day of M5 bars
    shock_min_std: float = 1e-9

    # runtime
    poll_seconds: int = 2


# -------------------- parsing/time --------------------
def _parse_mt5_time(s: str) -> datetime:
    """
    MT5 format: 'YYYY.MM.DD HH:MM' (your EA writes TIME_DATE|TIME_MINUTES)
    returns naive datetime
    """
    s = str(s).strip()
    for fmt in ("%Y.%m.%d %H:%M", "%Y.%m.%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    # fallback: allow ISO-like
    return datetime.fromisoformat(s)


def _to_utc(dt_naive: datetime, tz_in: str) -> datetime:
    tz = ZoneInfo(tz_in)
    return dt_naive.replace(tzinfo=tz).astimezone(timezone.utc)


def _ny_day_id(ts_utc: datetime, ny_tz: str) -> str:
    ny = ZoneInfo(ny_tz)
    return ts_utc.astimezone(ny).strftime("%Y-%m-%d")


def _ny_open_ts_for_day(ts_utc: datetime, ny_tz: str, hhmm: str) -> datetime:
    ny = ZoneInfo(ny_tz)
    local = ts_utc.astimezone(ny)
    hh, mm = hhmm.split(":")
    open_local = datetime(local.year, local.month, local.day, int(hh), int(mm), tzinfo=ny)
    return open_local.astimezone(timezone.utc)


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -------------------- features --------------------
def _compute_log_ret(close: pd.Series) -> pd.Series:
    close = close.astype("float64")
    prev = close.shift(1)
    lr = (close / prev).apply(lambda x: math.log(x) if pd.notna(x) and x > 0 else float("nan"))
    return lr.astype("float64")


def _compute_atr_14(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Classic ATR on True Range (Wilder-style smoothing approximated with EMA here for simplicity).
    Deterministic and stable for live.
    """
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder ATR is RMA; EMA(alpha=1/period) is a common equivalent
    atr = tr.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
    return atr.astype("float64")


def _compute_ny_open(df: pd.DataFrame, cfg: LiveFeatureBuilderConfig) -> pd.Series:
    """
    ny_open = close of first bar at/after NY open time for each NY day, forward-filled for that NY day.
    """
    ts = df["ts_utc"]
    day = ts.apply(lambda x: _ny_day_id(x, cfg.ny_tz))

    tmp = df.copy()
    tmp["_ny_day"] = day
    tmp["_ny_open_ts_utc"] = ts.apply(lambda x: _ny_open_ts_for_day(x, cfg.ny_tz, cfg.ny_open_hhmm))

    ny_open_val = {}
    for dkey, g in tmp.groupby("_ny_day", sort=True):
        g2 = g[g["ts_utc"] >= g["_ny_open_ts_utc"]]
        ny_open_val[dkey] = float("nan") if len(g2) == 0 else float(g2.iloc[0]["close"])

    return tmp["_ny_day"].map(ny_open_val).astype("float64")


def _compute_shock_z(log_ret: pd.Series, window: int, min_std: float) -> pd.Series:
    mu = log_ret.rolling(window, min_periods=max(10, window // 10)).mean()
    sd = log_ret.rolling(window, min_periods=max(10, window // 10)).std(ddof=0).clip(lower=min_std)
    return ((log_ret - mu) / sd).astype("float64")


# -------------------- build/write loop --------------------
def build_once(cfg: LiveFeatureBuilderConfig) -> pd.DataFrame:
    raw = pd.read_csv(cfg.mt5_csv_path)

    need = [cfg.mt5_time_col, "open", "high", "low", "close"]
    for c in need:
        if c not in raw.columns:
            raise ValueError(f"Missing required column {c!r}. Found={list(raw.columns)}")

    # parse time -> ts_utc
    dt_naive = raw[cfg.mt5_time_col].apply(_parse_mt5_time)
    ts_utc = dt_naive.apply(lambda d: _to_utc(d, cfg.tz_in))

    df = pd.DataFrame({
        "time": raw[cfg.mt5_time_col].astype(str),  # keep MT5 format as the runner time_col
        "ts_utc": ts_utc,
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
    })

    if "tick_volume" in raw.columns:
        df["tick_volume"] = raw["tick_volume"]
    elif "volume" in raw.columns:
        df["tick_volume"] = raw["volume"]
    else:
        df["tick_volume"] = ""

    # core features (matching gate expectations)
    df["log_ret"] = _compute_log_ret(df["close"])
    df["atr_14"] = _compute_atr_14(df, cfg.atr_period)

    # anchors
    df["ny_open"] = _compute_ny_open(df, cfg)

    # shock features (per gate contract)
    df["shock_log_ret"] = df["log_ret"]
    df["shock_z"] = _compute_shock_z(df["shock_log_ret"], cfg.shock_z_window, cfg.shock_min_std)

    # output columns: keep a minimal set + ts_utc for debugging
    out_cols = [
        "time", "open", "high", "low", "close", "tick_volume",
        "log_ret", "atr_14",
        "ny_open",
        "shock_log_ret", "shock_z",
        "ts_utc",
    ]
    return df[out_cols]


def run_loop(cfg: LiveFeatureBuilderConfig) -> None:
    _ensure_parent_dir(cfg.out_csv_path)
    print(f"[live_features] in ={cfg.mt5_csv_path}")
    print(f"[live_features] out={cfg.out_csv_path}")

    last_mtime: Optional[float] = None

    while True:
        try:
            st = os.stat(cfg.mt5_csv_path)
            if last_mtime is None or st.st_mtime > last_mtime:
                df = build_once(cfg)

                # write UTF-8 no BOM (avoid your earlier utf-8 decode issues)
                df.to_csv(cfg.out_csv_path, index=False, encoding="utf-8", lineterminator="\n")
                last_mtime = st.st_mtime
                print(f"[live_features] wrote rows={len(df)} mtime={last_mtime}")
        except Exception as e:
            print(f"[live_features] ERROR: {e}")

        time.sleep(max(1, cfg.poll_seconds))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mt5_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tz_in", default="Europe/Prague")
    ap.add_argument("--ny_open_hhmm", default="08:30")
    ap.add_argument("--shock_z_window", type=int, default=288)
    ap.add_argument("--poll", type=int, default=2)
    args = ap.parse_args()

    cfg = LiveFeatureBuilderConfig(
        mt5_csv_path=args.mt5_csv,
        out_csv_path=args.out_csv,
        tz_in=args.tz_in,
        ny_open_hhmm=args.ny_open_hhmm,
        shock_z_window=args.shock_z_window,
        poll_seconds=args.poll,
    )
    run_loop(cfg)