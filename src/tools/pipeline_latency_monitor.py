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

RAW_TZ = "Etc/GMT-3"
BAR_MINUTES = 5


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


def load_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def last_bar_ts(csv_path: Path, source: str) -> Optional[pd.Timestamp]:
    df = load_csv(csv_path)
    if df is None:
        return None

    col = first_existing_column(df, ["ts_utc", "timestamp", "time", "ts", "datetime"])
    if col is None:
        return None

    return parse_ts_to_utc(df[col].iloc[-1], source=source)


def shadow_decision_ts(csv_path: Path) -> Optional[pd.Timestamp]:
    df = load_csv(csv_path)
    if df is None:
        return None

    # Prefer explicit decision/write timestamp columns if they exist.
    col = first_existing_column(
        df,
        [
            "decision_ts_utc",
            "written_at_utc",
            "processed_at_utc",
            "created_at_utc",
            "ts_utc",
            "timestamp",
            "time",
        ],
    )
    if col is None:
        return None

    return parse_ts_to_utc(df[col].iloc[-1], source="shadow")


def file_mtime_utc(path: Path) -> Optional[datetime]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def count_rows(csv_path: Path) -> Optional[int]:
    df = load_csv(csv_path)
    if df is None:
        return None
    return len(df)


def diff_seconds(newer, older) -> Optional[float]:
    if newer is None or older is None:
        return None

    if hasattr(newer, "to_pydatetime"):
        newer = newer.to_pydatetime()
    if hasattr(older, "to_pydatetime"):
        older = older.to_pydatetime()

    if newer.tzinfo is None:
        newer = newer.replace(tzinfo=timezone.utc)
    if older.tzinfo is None:
        older = older.replace(tzinfo=timezone.utc)

    return (newer - older).total_seconds()


def fmt_ts(ts) -> str:
    return str(ts) if ts is not None else "N/A"


def fmt_s(x: Optional[float]) -> str:
    return f"{x:.1f}s" if x is not None else "N/A"


def current_utc() -> datetime:
    return datetime.now(timezone.utc)


def expected_last_closed_bar_utc(now: Optional[datetime] = None, bar_minutes: int = 5) -> datetime:
    if now is None:
        now = current_utc()

    floored_minute = (now.minute // bar_minutes) * bar_minutes
    return now.replace(minute=floored_minute, second=0, microsecond=0)


def status_for_alignment(delta_s: Optional[float], warn_s: float = 1.0, crit_s: float = 60.0) -> str:
    if delta_s is None:
        return "N/A"
    x = abs(delta_s)
    if x <= warn_s:
        return "OK"
    if x <= crit_s:
        return "WARN"
    return "CRIT"


def status_color(status: str) -> str:
    s = status.upper()
    if s == "OK":
        return "green"
    if s == "WARN":
        return "yellow"
    if s == "CRIT":
        return "red"
    return "yellow"


def estimate_bar_close_to_visibility_s(expected_bar_ts: Optional[datetime], observed_bar_ts: Optional[pd.Timestamp]) -> Optional[float]:
    # If observed bar == expected bar, latency ≈ 0 relative to bar availability.
    # If observed bar lags by one full bar, result ≈ 300s on M5.
    return diff_seconds(expected_bar_ts, observed_bar_ts)


# =========================
# CORE REPORT
# =========================

def run_once(use_color: bool = True) -> int:
    now_utc = current_utc()
    expected_bar = expected_last_closed_bar_utc(now_utc, bar_minutes=BAR_MINUTES)

    raw_bar = last_bar_ts(RAW_FEED, source="raw")
    feat_bar = last_bar_ts(FEATURES, source="features")

    latest_paper_run = get_latest_run(RUNS_DIR)
    latest_shadow_run = get_latest_run(SHADOW_DIR)

    paper_equity = latest_paper_run / "equity.csv" if latest_paper_run else None
    shadow_decisions = latest_shadow_run / "shadow_decisions.csv" if latest_shadow_run else None

    paper_bar = last_bar_ts(paper_equity, source="paper") if paper_equity else None
    shadow_bar = shadow_decision_ts(shadow_decisions) if shadow_decisions else None

    raw_mtime = file_mtime_utc(RAW_FEED)
    feat_mtime = file_mtime_utc(FEATURES)
    paper_mtime = file_mtime_utc(paper_equity) if paper_equity else None
    shadow_mtime = file_mtime_utc(shadow_decisions) if shadow_decisions else None

    raw_rows = count_rows(RAW_FEED)
    feat_rows = count_rows(FEATURES)
    paper_rows = count_rows(paper_equity) if paper_equity else None
    shadow_rows = count_rows(shadow_decisions) if shadow_decisions else None

    raw_vs_expected_s = estimate_bar_close_to_visibility_s(expected_bar, raw_bar)
    feat_vs_raw_s = diff_seconds(feat_bar, raw_bar)
    paper_vs_features_s = diff_seconds(paper_bar, feat_bar)
    shadow_vs_features_s = diff_seconds(shadow_bar, feat_bar)

    raw_mtime_vs_bar_s = diff_seconds(raw_mtime, raw_bar)
    feat_mtime_vs_bar_s = diff_seconds(feat_mtime, feat_bar)
    paper_mtime_vs_bar_s = diff_seconds(paper_mtime, paper_bar)
    shadow_mtime_vs_shadow_ts_s = diff_seconds(shadow_mtime, shadow_bar)

    feat_align = status_for_alignment(feat_vs_raw_s)
    paper_align = status_for_alignment(paper_vs_features_s)
    shadow_align = status_for_alignment(shadow_vs_features_s, warn_s=10.0, crit_s=300.0)

    print(colorize("\nINVARIANT PIPELINE LATENCY MONITOR", "bold", use_color))
    print("-----------------------------------")
    print("checked_at_utc:", now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"))
    print("expected_last_closed_bar_utc:", expected_bar)

    print(colorize("\nBAR STATE", "cyan", use_color))
    print("raw_bar_ts        :", fmt_ts(raw_bar))
    print("features_bar_ts   :", fmt_ts(feat_bar))
    print("paper_bar_ts      :", fmt_ts(paper_bar))
    print("shadow_ts         :", fmt_ts(shadow_bar))

    print(colorize("\nROW COUNTS", "cyan", use_color))
    print("raw_rows          :", raw_rows if raw_rows is not None else "N/A")
    print("features_rows     :", feat_rows if feat_rows is not None else "N/A")
    print("paper_equity_rows :", paper_rows if paper_rows is not None else "N/A")
    print("shadow_rows       :", shadow_rows if shadow_rows is not None else "N/A")

    print(colorize("\nPIPELINE ALIGNMENT", "cyan", use_color))
    print("raw_vs_expected_s     :", fmt_s(raw_vs_expected_s))
    print(
        "features_vs_raw_s     :",
        fmt_s(feat_vs_raw_s),
        colorize(f"[{feat_align}]", status_color(feat_align), use_color),
    )
    print(
        "paper_vs_features_s   :",
        fmt_s(paper_vs_features_s),
        colorize(f"[{paper_align}]", status_color(paper_align), use_color),
    )
    print(
        "shadow_vs_features_s  :",
        fmt_s(shadow_vs_features_s),
        colorize(f"[{shadow_align}]", status_color(shadow_align), use_color),
    )

    print(colorize("\nFILE MTIME PROXIES", "cyan", use_color))
    print("raw_file_mtime_utc    :", fmt_ts(raw_mtime))
    print("features_file_mtime_utc:", fmt_ts(feat_mtime))
    print("paper_file_mtime_utc  :", fmt_ts(paper_mtime))
    print("shadow_file_mtime_utc :", fmt_ts(shadow_mtime))

    print("raw_mtime_vs_bar_s    :", fmt_s(raw_mtime_vs_bar_s))
    print("features_mtime_vs_bar_s:", fmt_s(feat_mtime_vs_bar_s))
    print("paper_mtime_vs_bar_s  :", fmt_s(paper_mtime_vs_bar_s))
    print("shadow_mtime_vs_ts_s  :", fmt_s(shadow_mtime_vs_shadow_ts_s))

    print(colorize("\nINTERPRETATION", "cyan", use_color))
    if raw_vs_expected_s is not None and raw_vs_expected_s >= BAR_MINUTES * 60:
        print("raw feed appears at least one full bar behind expected close.")
        print("This can still be normal in MT5 if export is tick-driven and next-bar tick has not arrived.")
    else:
        print("raw feed is consistent with expected bar availability.")

    if feat_align == "OK" and paper_align == "OK":
        print("No evidence of internal pipeline delay between raw → features → paper.")
    else:
        print("Potential internal pipeline misalignment detected. Review alignment metrics above.")

    print("\nSTATUS CHECK COMPLETE")

    if paper_align == "CRIT" or feat_align == "CRIT":
        return 2
    if paper_align == "WARN" or feat_align == "WARN" or shadow_align == "WARN":
        return 1
    return 0


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invariant pipeline latency monitor")
    parser.add_argument("--watch", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds")
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
        print("\nPipeline latency monitor stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()