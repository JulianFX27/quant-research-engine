from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# =========================
# CONFIG
# =========================

RAW_FEED = Path(
    r"C:\Users\julia\AppData\Roaming\MetaQuotes\Terminal\Common\Files\anchor_reversion_fx\prod\v8_policy_r1\live_feed\eurusd_m5_latest.csv"
)

FEATURES = Path(
    r"C:\Users\julia\AppData\Roaming\MetaQuotes\Terminal\Common\Files\anchor_reversion_fx\prod\v8_policy_r1\live_feed\eurusd_m5_features_latest.csv"
)

RUNS_DIR = Path("results/runs")
SHADOW_DIR = Path("results/shadow/runs")

# MT5 raw feed timezone contract
RAW_TZ = "Etc/GMT-3"

# Health thresholds
RAW_MAX_LAG_S = 900
FEATURES_MAX_LAG_S = 900
PAPER_MAX_LAG_S = 1800
SHADOW_MAX_LAG_S = 1800


# =========================
# HELPERS
# =========================

def clear_console() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def colorize(text: str, color: str, enabled: bool = True) -> str:
    if not enabled:
        return text

    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def get_latest_run(path: Path) -> Optional[Path]:
    if not path.exists():
        return None

    runs = [p for p in path.iterdir() if p.is_dir()]
    if not runs:
        return None

    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def parse_ts_to_utc(value, source: str) -> Optional[pd.Timestamp]:
    if pd.isna(value):
        return None

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None

    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert("UTC")

    if source == "raw":
        return ts.tz_localize(RAW_TZ).tz_convert("UTC")

    return ts.tz_localize("UTC")


def last_timestamp(csv_path: Path, source: str = "generic") -> Optional[pd.Timestamp]:
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df.empty:
        return None

    candidate_cols = ["ts_utc", "timestamp", "time", "ts", "datetime"]

    for col in candidate_cols:
        if col in df.columns:
            return parse_ts_to_utc(df[col].iloc[-1], source=source)

    return None


def count_rows(csv_path: Path) -> Optional[int]:
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    return len(df)


def lag_seconds(ts: Optional[pd.Timestamp]) -> Optional[float]:
    if ts is None:
        return None

    now = datetime.now(timezone.utc)

    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return (now - ts).total_seconds()


def fmt_ts(ts: Optional[pd.Timestamp]) -> str:
    return str(ts) if ts is not None else "N/A"


def fmt_lag(lag: Optional[float]) -> str:
    return f"{lag:.1f}s" if lag is not None else "N/A"


def fmt_rows(rows: Optional[int]) -> str:
    return str(rows) if rows is not None else "N/A"


def file_exists_label(path: Path) -> str:
    return "YES" if path.exists() else "NO"


def read_trades_info(run: Optional[Path]) -> tuple[Optional[Path], Optional[int]]:
    if run is None:
        return None, None

    trades = run / "trades.csv"
    if not trades.exists():
        return trades, None

    rows = count_rows(trades)
    return trades, rows


def evaluate_component(name: str, lag: Optional[float], max_lag_s: int) -> str:
    if lag is None:
        return "DOWN"
    if lag < -60:
        return "DEGRADED"
    if lag <= max_lag_s:
        return "HEALTHY"
    return "DEGRADED"


def overall_status(raw_lag, feat_lag, paper_lag, shadow_lag) -> str:
    if raw_lag is None and feat_lag is None and paper_lag is None and shadow_lag is None:
        return "DOWN"

    if raw_lag is not None and raw_lag < -60:
        return "DEGRADED (raw timezone mismatch suspected)"

    if raw_lag is not None and raw_lag <= RAW_MAX_LAG_S:
        if feat_lag is None or feat_lag > FEATURES_MAX_LAG_S:
            return "DEGRADED (builder/features stale)"

    if feat_lag is not None and feat_lag <= FEATURES_MAX_LAG_S:
        if paper_lag is not None and paper_lag > PAPER_MAX_LAG_S:
            return "DEGRADED (paper stale)"
        if shadow_lag is not None and shadow_lag > SHADOW_MAX_LAG_S:
            return "DEGRADED (shadow stale)"

    if (
        raw_lag is not None and raw_lag <= RAW_MAX_LAG_S and
        feat_lag is not None and feat_lag <= FEATURES_MAX_LAG_S and
        paper_lag is not None and paper_lag <= PAPER_MAX_LAG_S and
        shadow_lag is not None and shadow_lag <= SHADOW_MAX_LAG_S
    ):
        return "HEALTHY"

    return "DEGRADED"


def status_color(status: str) -> str:
    s = status.upper()
    if "HEALTHY" in s:
        return "green"
    if "DOWN" in s:
        return "red"
    return "yellow"


# =========================
# CHECKS
# =========================

def check_raw(use_color: bool = True) -> Optional[float]:
    ts = last_timestamp(RAW_FEED, source="raw")
    lag = lag_seconds(ts)
    rows = count_rows(RAW_FEED)
    comp = evaluate_component("raw", lag, RAW_MAX_LAG_S)

    print(colorize("\nRAW FEED", "cyan", use_color))
    print("path:", RAW_FEED)
    print("exists:", file_exists_label(RAW_FEED))
    print("last bar:", fmt_ts(ts))
    print("lag:", fmt_lag(lag))
    print("rows:", fmt_rows(rows))
    print("status:", colorize(comp, status_color(comp), use_color))

    return lag


def check_features(use_color: bool = True) -> Optional[float]:
    ts = last_timestamp(FEATURES, source="features")
    lag = lag_seconds(ts)
    rows = count_rows(FEATURES)
    comp = evaluate_component("features", lag, FEATURES_MAX_LAG_S)

    print(colorize("\nFEATURES", "cyan", use_color))
    print("path:", FEATURES)
    print("exists:", file_exists_label(FEATURES))
    print("last bar:", fmt_ts(ts))
    print("lag:", fmt_lag(lag))
    print("rows:", fmt_rows(rows))
    print("status:", colorize(comp, status_color(comp), use_color))

    return lag


def check_paper(use_color: bool = True) -> Optional[float]:
    run = get_latest_run(RUNS_DIR)

    print(colorize("\nPAPER", "cyan", use_color))
    if run is None:
        print("run: N/A")
        print("equity heartbeat: N/A")
        print("status:", colorize("DOWN", "red", use_color))
        return None

    equity = run / "equity.csv"
    ts = last_timestamp(equity, source="paper")
    lag = lag_seconds(ts)
    comp = evaluate_component("paper", lag, PAPER_MAX_LAG_S)

    trades_path, trade_rows = read_trades_info(run)

    print("run:", run.name)
    print("equity path:", equity)
    print("equity exists:", file_exists_label(equity))
    print("equity heartbeat:", fmt_ts(ts))
    print("lag:", fmt_lag(lag))
    print("trades path:", trades_path if trades_path is not None else "N/A")
    print("trades exists:", file_exists_label(trades_path) if trades_path is not None else "NO")
    print("trades rows:", fmt_rows(trade_rows))
    print("status:", colorize(comp, status_color(comp), use_color))

    return lag


def check_shadow(use_color: bool = True) -> Optional[float]:
    run = get_latest_run(SHADOW_DIR)

    print(colorize("\nSHADOW", "cyan", use_color))
    if run is None:
        print("run: N/A")
        print("last decision: N/A")
        print("status:", colorize("DOWN", "red", use_color))
        return None

    decisions = run / "shadow_decisions.csv"
    ts = last_timestamp(decisions, source="shadow")
    lag = lag_seconds(ts)
    rows = count_rows(decisions)
    comp = evaluate_component("shadow", lag, SHADOW_MAX_LAG_S)

    print("run:", run.name)
    print("decisions path:", decisions)
    print("exists:", file_exists_label(decisions))
    print("last decision:", fmt_ts(ts))
    print("lag:", fmt_lag(lag))
    print("rows:", fmt_rows(rows))
    print("status:", colorize(comp, status_color(comp), use_color))

    return lag


def run_once(use_color: bool = True) -> int:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(colorize("\nINVARIANT LIVE HEALTH CHECK", "bold", use_color))
    print("----------------------------")
    print("checked_at:", now_utc)

    raw_lag = check_raw(use_color=use_color)
    feat_lag = check_features(use_color=use_color)
    paper_lag = check_paper(use_color=use_color)
    shadow_lag = check_shadow(use_color=use_color)

    status = overall_status(raw_lag, feat_lag, paper_lag, shadow_lag)

    print(colorize("\nSYSTEM STATUS", "bold", use_color))
    print(colorize(status, status_color(status), use_color))
    print("\nSTATUS CHECK COMPLETE")

    if "DOWN" in status.upper():
        return 2
    if "DEGRADED" in status.upper():
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invariant live stack health monitor")
    parser.add_argument("--watch", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=15, help="Refresh interval in seconds")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear console in watch mode")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_color = not args.no_color

    if not args.watch:
        code = run_once(use_color=use_color)
        sys.exit(code)

    try:
        while True:
            if not args.no_clear:
                clear_console()
            code = run_once(use_color=use_color)
            print(f"\nNext refresh in {args.interval}s | exit_code={code} | Ctrl+C to stop")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nLive health monitor stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()