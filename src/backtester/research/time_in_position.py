# src/backtester/research/time_in_position.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


# -------------------------
# Utils
# -------------------------
def _to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _safe_div(a: float, b: float) -> Optional[float]:
    if b == 0 or not np.isfinite(b):
        return None
    return float(a / b)


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "run_manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"run_manifest.json not found in {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_trades(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "trades.csv"
    if not p.exists():
        raise FileNotFoundError(f"trades.csv not found in {run_dir}")
    df = pd.read_csv(p)

    # Parse timestamps if present
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    return df


def _infer_tf_minutes_from_timeframe(tf: str) -> int:
    # Expected: "M5", "M15", "H1", etc.
    s = str(tf).strip().upper()
    if s.startswith("M"):
        return int(s[1:])
    if s.startswith("H"):
        return int(s[1:]) * 60
    raise ValueError(f"Unsupported timeframe format: {tf!r}")


def _find_index_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Try common index column names used in trades exports.
    """
    candidates = [
        ("entry_idx", "exit_idx"),
        ("entry_i", "exit_i"),
        ("entry_bar_idx", "exit_bar_idx"),
        ("entry_bar_index", "exit_bar_index"),
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    return None, None


def _compute_r_multiple(df: pd.DataFrame) -> pd.Series:
    """
    R = pnl / (risk_price * abs(qty))
    Fallback risk_price = abs(entry_price - sl_price) when available.
    """
    pnl = df.get("pnl")
    qty = df.get("qty")
    risk_price = df.get("risk_price")

    if pnl is None or qty is None:
        return pd.Series([np.nan] * len(df), index=df.index)

    pnl_f = pd.to_numeric(pnl, errors="coerce").astype(float)
    qty_f = pd.to_numeric(qty, errors="coerce").astype(float)

    rp = None
    if risk_price is not None:
        rp = pd.to_numeric(risk_price, errors="coerce").astype(float)

    # fallback if needed
    if rp is None or rp.isna().all():
        entry = pd.to_numeric(df.get("entry_price"), errors="coerce").astype(float) if "entry_price" in df.columns else None
        sl = pd.to_numeric(df.get("sl_price"), errors="coerce").astype(float) if "sl_price" in df.columns else None
        if entry is not None and sl is not None:
            rp = (entry - sl).abs()
        else:
            rp = pd.Series([np.nan] * len(df), index=df.index)

    denom = rp.abs() * qty_f.abs()
    r = pnl_f / denom
    r = r.replace([np.inf, -np.inf], np.nan)
    return r


def _forced_flags(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    reason = df.get("exit_reason")
    if reason is None:
        return (
            pd.Series([False] * len(df), index=df.index),
            pd.Series([False] * len(df), index=df.index),
        )
    r = reason.astype(str).fillna("UNKNOWN")
    forced = r.str.startswith("FORCE_")
    force_max_hold = r.eq("FORCE_MAX_HOLD")
    return forced, force_max_hold


def _bucket_table(
    values: pd.Series,
    r_mult: pd.Series,
    forced: pd.Series,
    force_max_hold: pd.Series,
    exit_reason: pd.Series,
    bins: List[float],
    unit: str,
) -> pd.DataFrame:
    """
    values: duration measure (bars or minutes)
    """
    x = pd.to_numeric(values, errors="coerce").astype(float)
    R = pd.to_numeric(r_mult, errors="coerce").astype(float)

    # build buckets
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"[{bins[i]},{bins[i+1]})")
    cat = pd.cut(x, bins=bins, right=False, labels=labels, include_lowest=True)

    out_rows = []
    for b in labels:
        m = (cat == b)
        n = int(m.sum())
        if n <= 0:
            out_rows.append(
                dict(
                    bucket=b,
                    unit=unit,
                    n=0,
                    expectancy_R=np.nan,
                    winrate_R=np.nan,
                    avg_win_R=np.nan,
                    avg_loss_R=np.nan,
                    forced_ratio=np.nan,
                    force_max_hold_ratio=np.nan,
                    exit_reason_top="",
                )
            )
            continue

        Rb = R[m].dropna()
        forced_ratio = float(forced[m].mean()) if n else np.nan
        fmh_ratio = float(force_max_hold[m].mean()) if n else np.nan

        # exit reason top
        er = exit_reason[m].astype(str).fillna("UNKNOWN")
        top = er.value_counts().index[0] if len(er) else "UNKNOWN"

        if len(Rb) == 0:
            out_rows.append(
                dict(
                    bucket=b,
                    unit=unit,
                    n=n,
                    expectancy_R=np.nan,
                    winrate_R=np.nan,
                    avg_win_R=np.nan,
                    avg_loss_R=np.nan,
                    forced_ratio=forced_ratio,
                    force_max_hold_ratio=fmh_ratio,
                    exit_reason_top=str(top),
                )
            )
            continue

        wins = Rb[Rb > 0]
        losses = Rb[Rb < 0]

        out_rows.append(
            dict(
                bucket=b,
                unit=unit,
                n=n,
                expectancy_R=float(Rb.mean()),
                winrate_R=float((Rb > 0).mean()),
                avg_win_R=float(wins.mean()) if len(wins) else 0.0,
                avg_loss_R=float(losses.mean()) if len(losses) else 0.0,
                forced_ratio=forced_ratio,
                force_max_hold_ratio=fmh_ratio,
                exit_reason_top=str(top),
            )
        )

    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=str)
    ap.add_argument("--out-dir", default=None, type=str)
    ap.add_argument("--bars-bins", default=None, type=str, help="Comma-separated bins for bars, e.g. 0,5,10,15,20,30,40,60,80,100,120,140,200,300,500,1000,inf")
    ap.add_argument("--minutes-bins", default=None, type=str, help="Comma-separated bins for minutes")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(run_dir)
    df = _load_trades(run_dir)

    timeframe = str(manifest.get("timeframe", "M5"))
    tf_minutes = _infer_tf_minutes_from_timeframe(timeframe)

    # Duration minutes (calendar)
    if "entry_time" in df.columns and "exit_time" in df.columns:
        dur_minutes = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60.0
        dur_minutes = dur_minutes.clip(lower=0)
    else:
        dur_minutes = pd.Series([np.nan] * len(df), index=df.index)

    # Duration bars (prefer index-diff)
    entry_idx_col, exit_idx_col = _find_index_cols(df)
    duration_bars_mode = "time_proxy"
    if entry_idx_col and exit_idx_col:
        e = pd.to_numeric(df[entry_idx_col], errors="coerce").astype(float)
        x = pd.to_numeric(df[exit_idx_col], errors="coerce").astype(float)
        dur_bars = (x - e).clip(lower=0)
        duration_bars_mode = f"index_diff:{entry_idx_col}/{exit_idx_col}"
    else:
        dur_bars = (dur_minutes / float(tf_minutes)).round().clip(lower=0)

    # R and forced flags
    r_mult = _compute_r_multiple(df)
    forced, force_max_hold = _forced_flags(df)
    exit_reason = df["exit_reason"].astype(str).fillna("UNKNOWN") if "exit_reason" in df.columns else pd.Series(["UNKNOWN"] * len(df))

    # Bins
    def _parse_bins(s: Optional[str]) -> List[float]:
        if not s:
            return []
        parts = [p.strip().lower() for p in s.split(",") if p.strip()]
        out: List[float] = []
        for p in parts:
            if p == "inf":
                out.append(float("inf"))
            else:
                out.append(float(p))
        return out

    bars_bins = _parse_bins(args.bars_bins) if args.bars_bins else [0,5,10,15,20,30,40,60,80,100,120,140,200,300,500,1000,float("inf")]
    minutes_bins = _parse_bins(args.minutes_bins) if args.minutes_bins else [0,15,30,45,60,90,120,180,240,300,360,480,600,700,1000,2500,5000,float("inf")]

    # Tables
    by_bars = _bucket_table(dur_bars, r_mult, forced, force_max_hold, exit_reason, bars_bins, unit="bars")
    by_minutes = _bucket_table(dur_minutes, r_mult, forced, force_max_hold, exit_reason, minutes_bins, unit="minutes")

    # Summary stats
    overall_R = pd.to_numeric(r_mult, errors="coerce").dropna()
    overall = {
        "expectancy_R": float(overall_R.mean()) if len(overall_R) else None,
        "forced_ratio_overall": float(forced.mean()) if len(df) else None,
        "force_max_hold_ratio_overall": float(force_max_hold.mean()) if len(df) else None,
    }

    def _quantiles(x: pd.Series) -> Dict[str, Any]:
        x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
        if len(x) == 0:
            return {"n": 0}
        return {
            "n": int(len(x)),
            "min": float(x.min()),
            "p50": float(x.quantile(0.50)),
            "p90": float(x.quantile(0.90)),
            "p95": float(x.quantile(0.95)),
            "p99": float(x.quantile(0.99)),
            "max": float(x.max()),
            "mean": float(x.mean()),
        }

    report = {
        "run_id": str(manifest.get("run_id", "")),
        "name": str(manifest.get("name", "")),
        "symbol": str(manifest.get("symbol", "")),
        "timeframe": timeframe,
        "tf_minutes": tf_minutes,
        "dataset_id": str((manifest.get("dataset") or {}).get("dataset_id") or manifest.get("dataset_id") or ""),
        "max_holding_bars_cfg": (manifest.get("risk") or {}).get("max_holding_bars"),
        "n_trades_loaded": int(len(df)),
        "duration_bars_mode": duration_bars_mode,
        "duration_bars": _quantiles(dur_bars),
        "duration_minutes": _quantiles(dur_minutes),
        "overall": overall,
    }

    # Write outputs
    (out_dir / "time_in_position_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    by_bars.to_csv(out_dir / "time_in_position_by_bars.csv", index=False)
    by_minutes.to_csv(out_dir / "time_in_position_by_minutes.csv", index=False)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
