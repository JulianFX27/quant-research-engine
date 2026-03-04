from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo


@dataclass
class LiveFeatureBuilderConfig:
    # Input OHLCV from MT5 EA
    mt5_csv_path: str
    mt5_time_col: str = "time"

    # IMPORTANT:
    # Many MT5 brokers use "server time" around UTC+2 (common in FX).
    # If tz_in is wrong, ts_utc can drift into the future/past and break live gating.
    # Default set to UTC+2 as a safer baseline than Prague for most brokers.
    tz_in: str = "Etc/GMT-2"  # UTC+2 (Etc/GMT-2 == UTC+2 by convention)

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

    # Safety: do not emit features if ts_utc appears in the future beyond tolerance
    future_tolerance_seconds: int = 90  # allow small skew; > this implies tz_in mismatch


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
    # treat MT5 timestamp as local in tz_in
    return dt_naive.replace(tzinfo=tz).astimezone(timezone.utc)


def _suggest_tz_in_for_last_bar(dt_naive_last: datetime, now_utc: datetime) -> Tuple[str, float]:
    """
    Suggest a tz_in (whole-hour offset) that makes the last bar close time plausible.
    We brute-force offsets [-12..+14] hours and pick the one that makes ts_utc
    closest to now_utc WITHOUT being too far in the future.
    Returns (tz_name, age_minutes).
    """
    best = None  # (score, tz_name, age_min)
    for offset in range(-12, 15):
        # Using Etc/GMT semantics: Etc/GMT-2 == UTC+2, Etc/GMT+3 == UTC-3
        # So to represent UTC+offset, we need "Etc/GMT-<offset>" when offset>0,
        # and "Etc/GMT+<abs(offset)>" when offset<0.
        if offset >= 0:
            tz_name = f"Etc/GMT-{offset}"
        else:
            tz_name = f"Etc/GMT+{abs(offset)}"

        try:
            ts_utc = _to_utc(dt_naive_last, tz_name)
        except Exception:
            continue

        age_min = (now_utc - ts_utc).total_seconds() / 60.0

        # We prefer small positive age (recent past).
        # Penalize future timestamps heavily.
        if age_min < -2.0:
            score = 1e9 + abs(age_min)
        else:
            score = abs(age_min)

        cand = (score, tz_name, age_min)
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        return ("Etc/GMT-0", float("nan"))
    _, tz_name, age_min = best
    return (tz_name, age_min)


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

    # parse MT5 time -> naive datetime
    dt_naive = raw[cfg.mt5_time_col].apply(_parse_mt5_time)

    # convert -> ts_utc using tz_in
    ts_utc = dt_naive.apply(lambda d: _to_utc(d, cfg.tz_in))

    # Sanity: last ts_utc must NOT be in the future beyond tolerance
    now_utc = datetime.now(timezone.utc)
    last_ts = ts_utc.iloc[-1]
    if isinstance(last_ts, pd.Timestamp):
        last_ts = last_ts.to_pydatetime()

    drift_sec = (last_ts - now_utc).total_seconds()
    if drift_sec > cfg.future_tolerance_seconds:
        # Suggest tz_in fix
        last_naive = dt_naive.iloc[-1]
        sug_tz, sug_age_min = _suggest_tz_in_for_last_bar(last_naive, now_utc)
        msg = (
            f"FUTURE ts_utc detected. last_ts_utc={last_ts.isoformat()} now_utc={now_utc.isoformat()} "
            f"drift_sec=+{drift_sec:.0f}. This usually means tz_in mismatch.\n"
            f"Current tz_in={cfg.tz_in!r}. Suggested tz_in ~ {sug_tz!r} (age_min={sug_age_min:.2f})."
        )
        raise ValueError(msg)

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
    print(f"[live_features] tz_in={cfg.tz_in} future_tol_s={cfg.future_tolerance_seconds}")

    last_mtime: Optional[float] = None

    while True:
        try:
            st = os.stat(cfg.mt5_csv_path)
            if last_mtime is None or st.st_mtime > last_mtime:
                df = build_once(cfg)

                # write UTF-8 no BOM
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
    ap.add_argument("--tz_in", default="Etc/GMT-2")
    ap.add_argument("--ny_open_hhmm", default="08:30")
    ap.add_argument("--shock_z_window", type=int, default=288)
    ap.add_argument("--poll", type=int, default=2)
    ap.add_argument("--future_tol_s", type=int, default=90)
    args = ap.parse_args()

    cfg = LiveFeatureBuilderConfig(
        mt5_csv_path=args.mt5_csv,
        out_csv_path=args.out_csv,
        tz_in=args.tz_in,
        ny_open_hhmm=args.ny_open_hhmm,
        shock_z_window=args.shock_z_window,
        poll_seconds=args.poll,
        future_tolerance_seconds=args.future_tol_s,
    )
    run_loop(cfg)