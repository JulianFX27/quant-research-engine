#!/usr/bin/env python3
"""
QA report generator for a completed run.

Reads:
  results/runs/<run_id>/trades.csv
  results/runs/<run_id>/equity.csv
  results/runs/<run_id>/run_manifest.json (optional)

Writes:
  results/runs/<run_id>/qa_report.json

Notes:
- Tooling-only: does not touch engine/strategy code paths.
- Designed to be robust to missing columns and locale issues.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return None
            # Handle commas (some exports/locales)
            s = s.replace(",", ".")
            return float(s)
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    f = safe_float(x)
    if f is None or math.isnan(f):
        return None
    try:
        return int(f)
    except Exception:
        return None


def parse_dt_utc(x: Any) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    try:
        ts = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    p = min(max(p, 0.0), 1.0)
    idx = int(math.floor((len(sorted_vals) - 1) * p))
    return float(sorted_vals[idx])


def stdev_pop(vals: List[float]) -> Optional[float]:
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return math.sqrt(var)


def max_streak(signs: List[int], target: int) -> int:
    # target: -1 for losses, +1 for wins
    cur = 0
    mx = 0
    for s in signs:
        if s == target:
            cur += 1
        else:
            cur = 0
        mx = max(mx, cur)
    return mx


def equity_dd_R(rs: List[float]) -> float:
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rs:
        eq += r
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


# -----------------------------
# Core QA
# -----------------------------
@dataclass
class QAReport:
    run_dir: str
    created_at_utc: str
    inputs: Dict[str, Any]
    counts: Dict[str, Any]
    r_stats: Dict[str, Any]
    drawdown: Dict[str, Any]
    holds: Dict[str, Any]
    exits: Dict[str, Any]
    gate: Dict[str, Any]
    notes: List[str]


def compute_report(run_dir: Path) -> QAReport:
    trades_path = run_dir / "trades.csv"
    equity_path = run_dir / "equity.csv"
    manifest_path = run_dir / "run_manifest.json"

    notes: List[str] = []

    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades.csv at: {trades_path}")
    if not equity_path.exists():
        raise FileNotFoundError(f"Missing equity.csv at: {equity_path}")

    tr = pd.read_csv(trades_path)
    eq = pd.read_csv(equity_path)

    # ---- Basic columns
    # Trades: expect entry_time_utc, exit_time_utc, R, exit_reason (best effort)
    missing_cols = []
    for c in ["entry_time_utc", "exit_time_utc", "R"]:
        if c not in tr.columns:
            missing_cols.append(c)
    if missing_cols:
        notes.append(f"trades.csv missing expected columns: {missing_cols}")

    # Equity: expect ts_utc, dd_pct
    if "ts_utc" not in eq.columns:
        notes.append("equity.csv missing ts_utc (time column).")
    if "dd_pct" not in eq.columns:
        notes.append("equity.csv missing dd_pct (drawdown percent).")

    # ---- Same-bar exits (hold_min <= 0)
    entry_ts = tr["entry_time_utc"].apply(parse_dt_utc) if "entry_time_utc" in tr.columns else None
    exit_ts = tr["exit_time_utc"].apply(parse_dt_utc) if "exit_time_utc" in tr.columns else None

    hold_min: List[Optional[float]] = []
    same_bar_rows: List[int] = []
    if entry_ts is not None and exit_ts is not None:
        for i, (et, xt) in enumerate(zip(entry_ts, exit_ts)):
            if et is None or xt is None:
                hold_min.append(None)
                continue
            hm = float((xt - et) / pd.Timedelta(minutes=1))
            hold_min.append(hm)
            if hm <= 0:
                same_bar_rows.append(i)
    else:
        notes.append("Could not compute hold_min (missing entry/exit times).")

    same_bar_count = len(same_bar_rows)
    n_trades = int(len(tr))

    same_bar_by_reason: Dict[str, int] = {}
    if same_bar_rows and "exit_reason" in tr.columns:
        reasons = tr.loc[same_bar_rows, "exit_reason"].astype(str).fillna("")
        for r in reasons.tolist():
            same_bar_by_reason[r] = same_bar_by_reason.get(r, 0) + 1

    # ---- R stats
    R_vals: List[float] = []
    if "R" in tr.columns:
        for x in tr["R"].tolist():
            v = safe_float(x)
            if v is not None and not math.isnan(v):
                R_vals.append(float(v))
    else:
        notes.append("No R column found; cannot compute R-space stats.")

    r_mean = (sum(R_vals) / len(R_vals)) if R_vals else None
    r_min = min(R_vals) if R_vals else None
    r_max = max(R_vals) if R_vals else None
    r_stdev = stdev_pop(R_vals) if R_vals else None

    signs: List[int] = []
    for r in R_vals:
        if r > 0:
            signs.append(1)
        elif r < 0:
            signs.append(-1)
        else:
            signs.append(0)

    max_consec_losses = max_streak(signs, -1) if signs else None
    max_consec_wins = max_streak(signs, 1) if signs else None
    dd_R = equity_dd_R(R_vals) if R_vals else None

    # ---- Equity DD%
    max_dd_pct: Optional[float] = None
    worst_dd_row: Optional[Dict[str, Any]] = None
    if "dd_pct" in eq.columns:
        dd_list: List[Tuple[int, float]] = []
        for i, x in enumerate(eq["dd_pct"].tolist()):
            v = safe_float(x)
            if v is None or math.isnan(v):
                continue
            dd_list.append((i, float(v)))
        if dd_list:
            i_max, v_max = max(dd_list, key=lambda t: t[1])
            max_dd_pct = v_max
            row = eq.iloc[i_max].to_dict()
            worst_dd_row = {
                "ts_utc": row.get("ts_utc"),
                "equity": safe_float(row.get("equity")),
                "equity_peak": safe_float(row.get("equity_peak")),
                "dd_pct": safe_float(row.get("dd_pct")),
            }
    else:
        notes.append("No dd_pct in equity.csv; cannot compute max_drawdown_pct.")

    # ---- Holds stats
    holds_clean = [hm for hm in hold_min if hm is not None]
    holds_sorted = sorted(holds_clean) if holds_clean else []
    hold_avg = (sum(holds_clean) / len(holds_clean)) if holds_clean else None
    hold_min_v = min(holds_clean) if holds_clean else None
    hold_max_v = max(holds_clean) if holds_clean else None
    hold_p50 = percentile(holds_sorted, 0.50) if holds_sorted else None
    hold_p95 = percentile(holds_sorted, 0.95) if holds_sorted else None

    # ---- Exits distribution
    exit_counts: Dict[str, int] = {}
    if "exit_reason" in tr.columns:
        for r in tr["exit_reason"].astype(str).fillna("").tolist():
            exit_counts[r] = exit_counts.get(r, 0) + 1
    else:
        notes.append("No exit_reason in trades.csv; cannot compute exit distribution.")

    # ---- Gate stats (from manifest if present)
    gate: Dict[str, Any] = {}
    inputs: Dict[str, Any] = {"trades_csv": str(trades_path), "equity_csv": str(equity_path)}
    if manifest_path.exists():
        try:
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            inputs["run_manifest_json"] = str(manifest_path)
            # best-effort gate fields
            g = man.get("gate", {}) if isinstance(man, dict) else {}
            if isinstance(g, dict) and g.get("enabled") is True:
                allow = safe_float(g.get("gate_allow"))
                block = safe_float(g.get("gate_block"))
                gate.update(
                    {
                        "enabled": True,
                        "policy_id": g.get("policy_id"),
                        "gate_allow": allow,
                        "gate_block": block,
                        "allow_rate": (allow / (allow + block)) if (allow is not None and block is not None and (allow + block) > 0) else None,
                    }
                )
            else:
                gate.update({"enabled": False})
        except Exception as e:
            notes.append(f"Failed reading run_manifest.json: {e}")
    else:
        gate.update({"enabled": None})
        notes.append("run_manifest.json not found (optional).")

    report = QAReport(
        run_dir=str(run_dir),
        created_at_utc=utc_now_iso(),
        inputs=inputs,
        counts={
            "n_trades": n_trades,
            "same_bar_exits_n": same_bar_count,
            "same_bar_exits_rate": (same_bar_count / n_trades) if n_trades > 0 else None,
        },
        r_stats={
            "mean_R": r_mean,
            "min_R": r_min,
            "max_R": r_max,
            "stdev_R_pop": r_stdev,
            "max_consecutive_losses": max_consec_losses,
            "max_consecutive_wins": max_consec_wins,
        },
        drawdown={
            "max_drawdown_R": dd_R,
            "max_drawdown_pct": max_dd_pct,
            "worst_dd_row": worst_dd_row,
        },
        holds={
            "n_with_times": len(holds_clean),
            "avg_min": hold_avg,
            "min_min": hold_min_v,
            "max_min": hold_max_v,
            "p50_min": hold_p50,
            "p95_min": hold_p95,
        },
        exits={
            "by_exit_reason": exit_counts,
            "same_bar_by_exit_reason": same_bar_by_reason,
        },
        gate=gate,
        notes=notes,
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help=r'Path like "results\runs\20260225_044113_3caf7ebd"')
    ap.add_argument("--out", default=None, help="Optional output path. Default: <run_dir>/qa_report.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    rep = compute_report(run_dir)

    out_path = Path(args.out) if args.out else (run_dir / "qa_report.json")
    out_path.write_text(json.dumps(rep.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")

    # Minimal console summary (no spam)
    print("Wrote:", str(out_path))
    print("n_trades:", rep.counts.get("n_trades"))
    print("same_bar_exits_rate:", rep.counts.get("same_bar_exits_rate"))
    print("max_drawdown_R:", rep.drawdown.get("max_drawdown_R"))
    print("max_drawdown_pct:", rep.drawdown.get("max_drawdown_pct"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())