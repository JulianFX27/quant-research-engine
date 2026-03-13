from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo


NY_TZ = ZoneInfo("America/New_York")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return int(x)
    except Exception:
        return default


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "run_manifest.json"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _load_trades(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "trades.csv"
    if not p.exists():
        raise FileNotFoundError(f"trades.csv not found in run_dir: {run_dir}")

    df = pd.read_csv(p)
    if df.empty:
        return df

    required = ["entry_time", "exit_time", "exit_reason", "pnl"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trades.csv missing required columns: {missing}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")

    df = df.dropna(subset=["entry_time", "exit_time"]).copy()
    if df.empty:
        return df

    if "R" not in df.columns:
        df["R"] = None
    if "hold_minutes" not in df.columns:
        df["hold_minutes"] = None

    df["R"] = pd.to_numeric(df["R"], errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["hold_minutes"] = pd.to_numeric(df["hold_minutes"], errors="coerce")

    df["entry_time_ny"] = df["entry_time"].dt.tz_convert(NY_TZ)
    df["exit_time_ny"] = df["exit_time"].dt.tz_convert(NY_TZ)

    df["entry_hour_utc"] = df["entry_time"].dt.hour.astype("Int64")
    df["entry_hour_ny"] = df["entry_time_ny"].dt.hour.astype("Int64")
    df["entry_weekday_ny"] = df["entry_time_ny"].dt.weekday.astype("Int64")

    # ---------- FIX CRÍTICO ----------
    # Bucket 30m debe ser único por media hora, no por minuto exacto del trade.
    # Antes podían aparecer múltiples labels ("10:05", "10:10", etc.) para el mismo bucket.
    df["entry_bucket_30m_ny"] = (
        df["entry_time_ny"].dt.hour * 2 + (df["entry_time_ny"].dt.minute // 30)
    ).astype("Int64")

    bucket_hour = (df["entry_bucket_30m_ny"] // 2).astype(int)
    bucket_min = ((df["entry_bucket_30m_ny"] % 2) * 30).astype(int)
    df["entry_bucket_label_30m_ny"] = [
        f"{h:02d}:{m:02d}" for h, m in zip(bucket_hour, bucket_min)
    ]
    # ---------- FIN FIX ----------

    return df


def _max_consecutive_losses_from_r(values: list[float]) -> int:
    max_streak = 0
    cur = 0
    for x in values:
        if pd.notna(x) and float(x) < 0:
            cur += 1
            if cur > max_streak:
                max_streak = cur
        else:
            cur = 0
    return int(max_streak)


def _profit_factor_from_pnl(values: pd.Series) -> float | None:
    if values.empty:
        return None
    gross_profit = float(values[values > 0].sum())
    gross_loss_abs = float(abs(values[values < 0].sum()))
    if gross_loss_abs <= 0:
        return float("inf") if gross_profit > 0 else None
    return gross_profit / gross_loss_abs


def _summarize_group(g: pd.DataFrame) -> dict[str, Any]:
    g = g.sort_values("entry_time", kind="stable").copy()

    n = int(len(g))
    r = pd.to_numeric(g["R"], errors="coerce")
    pnl = pd.to_numeric(g["pnl"], errors="coerce")
    hold = pd.to_numeric(g["hold_minutes"], errors="coerce")

    valid_r = r.dropna()
    wins_r = valid_r[valid_r > 0]
    losses_r = valid_r[valid_r < 0]

    forced = g["exit_reason"].astype(str).isin(["FORCE_MAX_HOLD", "FORCE_EOF"])
    force_max_hold = g["exit_reason"].astype(str).eq("FORCE_MAX_HOLD")

    out: dict[str, Any] = {
        "n_trades": n,
        "expectancy_R": float(valid_r.mean()) if len(valid_r) > 0 else None,
        "winrate_R": float((valid_r > 0).mean()) if len(valid_r) > 0 else None,
        "avg_win_R": float(wins_r.mean()) if len(wins_r) > 0 else None,
        "avg_loss_R": float(losses_r.mean()) if len(losses_r) > 0 else None,
        "profit_factor_pnl": _profit_factor_from_pnl(pnl.dropna()),
        "total_pnl": float(pnl.sum()) if len(pnl.dropna()) > 0 else None,
        "avg_hold_minutes": float(hold.mean()) if len(hold.dropna()) > 0 else None,
        "median_hold_minutes": float(hold.median()) if len(hold.dropna()) > 0 else None,
        "forced_exit_ratio": float(forced.mean()) if n > 0 else None,
        "force_max_hold_ratio": float(force_max_hold.mean()) if n > 0 else None,
        "max_consecutive_losses_R": _max_consecutive_losses_from_r(valid_r.tolist()),
    }
    return out


def _group_table(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    gb = df.groupby(by_cols, dropna=False, sort=True)

    for key, g in gb:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(by_cols, key)}
        row.update(_summarize_group(g))
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(by_cols, kind="stable").reset_index(drop=True)


def _weekday_label(n: int) -> str:
    mapping = {
        0: "Mon",
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri",
        5: "Sat",
        6: "Sun",
    }
    return mapping.get(int(n), str(n))


def build_reports(run_dir: Path) -> dict[str, Any]:
    manifest = _load_manifest(run_dir)
    trades = _load_trades(run_dir)

    if trades.empty:
        outputs = {
            "by_hour_utc": run_dir / "session_adversity_by_hour_utc.csv",
            "by_hour_ny": run_dir / "session_adversity_by_hour_ny.csv",
            "by_bucket_30m_ny": run_dir / "session_adversity_by_bucket_30m_ny.csv",
            "by_weekday_ny": run_dir / "session_adversity_by_weekday_ny.csv",
        }
        for p in outputs.values():
            pd.DataFrame().to_csv(p, index=False)

        return {
            "run_id": run_dir.name,
            "name": manifest.get("name"),
            "symbol": manifest.get("symbol"),
            "timeframe": manifest.get("timeframe"),
            "n_trades_loaded": 0,
            "outputs": {k: Path(v).name for k, v in outputs.items()},
        }

    by_hour_utc = _group_table(trades, ["entry_hour_utc"])

    by_hour_ny = _group_table(trades, ["entry_hour_ny"])

    by_bucket_30m_ny = _group_table(
        trades,
        ["entry_bucket_30m_ny", "entry_bucket_label_30m_ny"],
    )

    by_weekday_ny = _group_table(trades, ["entry_weekday_ny"])
    if not by_weekday_ny.empty:
        by_weekday_ny["entry_weekday_label_ny"] = by_weekday_ny["entry_weekday_ny"].map(
            lambda x: _weekday_label(_safe_int(x))
        )
        cols = ["entry_weekday_ny", "entry_weekday_label_ny"] + [
            c for c in by_weekday_ny.columns
            if c not in {"entry_weekday_ny", "entry_weekday_label_ny"}
        ]
        by_weekday_ny = by_weekday_ny[cols]

    out_hour_utc = run_dir / "session_adversity_by_hour_utc.csv"
    out_hour_ny = run_dir / "session_adversity_by_hour_ny.csv"
    out_bucket = run_dir / "session_adversity_by_bucket_30m_ny.csv"
    out_weekday = run_dir / "session_adversity_by_weekday_ny.csv"

    by_hour_utc.to_csv(out_hour_utc, index=False)
    by_hour_ny.to_csv(out_hour_ny, index=False)
    by_bucket_30m_ny.to_csv(out_bucket, index=False)
    by_weekday_ny.to_csv(out_weekday, index=False)

    return {
        "run_id": run_dir.name,
        "name": manifest.get("name"),
        "symbol": manifest.get("symbol"),
        "timeframe": manifest.get("timeframe"),
        "n_trades_loaded": int(len(trades)),
        "outputs": {
            "by_hour_utc": out_hour_utc.name,
            "by_hour_ny": out_hour_ny.name,
            "by_bucket_30m_ny": out_bucket.name,
            "by_weekday_ny": out_weekday.name,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build session adversity maps from a backtest run directory.")
    ap.add_argument("--run-dir", required=True, help="Path to results/runs/<run_id>")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found or not a directory: {run_dir}")

    result = build_reports(run_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())