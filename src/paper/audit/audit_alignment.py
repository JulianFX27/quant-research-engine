# src/paper/audit/audit_alignment.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


# ---------- Config ----------
RESULTS_DIR = Path("results/runs")

# Adjust these names if your schema differs.
CANDIDATE_R_COLS = ["R", "r", "pnl_R", "r_multiple", "R_multiple"]
CANDIDATE_EXIT_COLS = ["exit_reason", "exit_tag", "exit_type", "exit_code"]
CANDIDATE_HOLD_COLS = ["hold_bars", "bars_held", "holding_bars", "duration_bars"]

TP_LABELS = {"TP", "TAKE_PROFIT", "PROFIT"}
SL_LABELS = {"SL", "STOP_LOSS", "LOSS"}

# If you have canonical names for these non-TP/SL exits, list them here.
NON_TPSL_HINTS = {"ANCHOR_TOUCH", "TIME_STOP", "FORCE_WEEKEND", "FORCE_EOF", "FORCE_MAX_HOLD"}


@dataclass
class SliceMetrics:
    n_trades: int
    expectancy_R: float
    winrate: float
    profit_factor_R: float
    avg_win_R: float
    avg_loss_R: float
    median_R: float


def _pick_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_metrics(df: pd.DataFrame, r_col: str) -> SliceMetrics:
    r = df[r_col].astype(float)
    n = int(r.notna().sum())
    if n == 0:
        return SliceMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    wins = r[r > 0]
    losses = r[r < 0]

    expectancy = r.mean()
    winrate = (r > 0).mean()

    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0

    return SliceMetrics(
        n_trades=n,
        expectancy_R=float(expectancy),
        winrate=float(winrate),
        profit_factor_R=float(pf),
        avg_win_R=float(avg_win),
        avg_loss_R=float(avg_loss),
        median_R=float(r.median()),
    )


def load_latest_run(results_dir: Path) -> Tuple[str, Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results dir: {results_dir.resolve()}")

    run_dirs = [p for p in results_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {results_dir.resolve()}")

    # Sort by folder name; if your run_id starts with timestamp, this works.
    run_dirs.sort(key=lambda p: p.name)
    latest = run_dirs[-1]
    return latest.name, latest


def main():
    run_id, run_path = load_latest_run(RESULTS_DIR)

    trades_path = run_path / "trades.csv"
    metrics_path = run_path / "metrics.json"

    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades.csv: {trades_path.resolve()}")

    trades = pd.read_csv(trades_path)

    r_col = _pick_col(trades, CANDIDATE_R_COLS)
    exit_col = _pick_col(trades, CANDIDATE_EXIT_COLS)
    hold_col = _pick_col(trades, CANDIDATE_HOLD_COLS)

    if r_col is None:
        raise ValueError(f"Could not find R column. Tried: {CANDIDATE_R_COLS}. Found cols: {list(trades.columns)[:30]}...")

    if exit_col is None:
        raise ValueError(f"Could not find exit_reason column. Tried: {CANDIDATE_EXIT_COLS}.")

    trades[exit_col] = trades[exit_col].astype(str)

    # Identify TP/SL rows
    exit_upper = trades[exit_col].str.upper()
    is_tp = exit_upper.isin(TP_LABELS)
    is_sl = exit_upper.isin(SL_LABELS)
    is_tpsl = is_tp | is_sl

    full = trades.copy()
    tpsl = trades[is_tpsl].copy()

    m_full = compute_metrics(full, r_col)
    m_tpsl = compute_metrics(tpsl, r_col)

    # Exit mix
    exit_mix = (
        full.groupby(exit_col)
        .size()
        .rename("n")
        .to_frame()
        .assign(pct=lambda x: (x["n"] / x["n"].sum()) * 100.0)
        .sort_values("n", ascending=False)
        .reset_index()
    )

    # R by exit_reason
    r_by_exit = (
        full.groupby(exit_col)[r_col]
        .agg(n="count", mean="mean", median="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75))
        .sort_values("n", ascending=False)
        .reset_index()
    )

    # Hold time diagnostics (optional)
    hold_diag = None
    if hold_col is not None:
        hold_diag = (
            full.groupby(exit_col)[hold_col]
            .agg(n="count", mean="mean", median="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75))
            .sort_values("n", ascending=False)
            .reset_index()
        )

    # Monthly stability (if entry_time exists)
    monthly = None
    time_cols = [c for c in trades.columns if "time" in c.lower() or "date" in c.lower()]
    entry_time_col = None
    for c in ["entry_time", "open_time", "time_entry", "entry_ts"]:
        if c in trades.columns:
            entry_time_col = c
            break

    if entry_time_col:
        tt = pd.to_datetime(full[entry_time_col], utc=True, errors="coerce")
        full["_month"] = tt.dt.to_period("M").astype(str)
        monthly = full.groupby("_month")[r_col].agg(n="count", expectancy_R="mean").reset_index().sort_values("_month")

    # Compare deltas
    deltas = {
        "delta_expectancy_R": _safe_float(m_full.expectancy_R) - _safe_float(m_tpsl.expectancy_R),
        "delta_profit_factor_R": _safe_float(m_full.profit_factor_R) - _safe_float(m_tpsl.profit_factor_R),
        "delta_winrate": _safe_float(m_full.winrate) - _safe_float(m_tpsl.winrate),
        "tpsl_coverage_pct": (len(tpsl) / len(full) * 100.0) if len(full) else float("nan"),
    }

    # Read metrics.json if present (for cross-check)
    metrics_json = None
    if metrics_path.exists():
        metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir = run_path / "audit_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tables export
    tables = []
    exit_mix2 = exit_mix.copy()
    exit_mix2["table"] = "exit_mix"
    tables.append(exit_mix2)

    r_by_exit2 = r_by_exit.copy()
    r_by_exit2["table"] = "R_by_exit"
    tables.append(r_by_exit2)

    if hold_diag is not None:
        hold_diag2 = hold_diag.copy()
        hold_diag2["table"] = "hold_by_exit"
        tables.append(hold_diag2)

    if monthly is not None:
        monthly2 = monthly.copy()
        monthly2["table"] = "monthly_expectancy"
        tables.append(monthly2)

    all_tables = pd.concat(tables, axis=0, ignore_index=True)
    all_tables.to_csv(out_dir / "alignment_tables.csv", index=False)

    # Markdown report
    def fmt(x, nd=4):
        if x is None:
            return "NA"
        try:
            if pd.isna(x):
                return "NA"
        except Exception:
            pass
        return f"{float(x):.{nd}f}"

    report = []
    report.append(f"# Alignment Audit — run_id: {run_id}")
    report.append("")
    report.append("## Slice Metrics (R-space)")
    report.append("")
    report.append("| Slice | n_trades | expectancy_R | winrate | PF_R | avg_win_R | avg_loss_R | median_R |")
    report.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    report.append(
        f"| FULL | {m_full.n_trades} | {fmt(m_full.expectancy_R)} | {fmt(m_full.winrate)} | {fmt(m_full.profit_factor_R)} | "
        f"{fmt(m_full.avg_win_R)} | {fmt(m_full.avg_loss_R)} | {fmt(m_full.median_R)} |"
    )
    report.append(
        f"| TP/SL-only | {m_tpsl.n_trades} | {fmt(m_tpsl.expectancy_R)} | {fmt(m_tpsl.winrate)} | {fmt(m_tpsl.profit_factor_R)} | "
        f"{fmt(m_tpsl.avg_win_R)} | {fmt(m_tpsl.avg_loss_R)} | {fmt(m_tpsl.median_R)} |"
    )
    report.append("")
    report.append("## Deltas (FULL − TP/SL)")
    report.append("")
    for k, v in deltas.items():
        report.append(f"- {k}: {fmt(v)}")

    report.append("")
    report.append("## Exit mix (top)")
    report.append("")
    report.append(exit_mix.head(15).to_markdown(index=False))

    report.append("")
    report.append("## R by exit_reason (top)")
    report.append("")
    report.append(r_by_exit.head(15).to_markdown(index=False))

    if hold_diag is not None:
        report.append("")
        report.append("## Hold time by exit_reason (top)")
        report.append("")
        report.append(hold_diag.head(15).to_markdown(index=False))

    if monthly is not None:
        report.append("")
        report.append("## Monthly expectancy_R")
        report.append("")
        report.append(monthly.to_markdown(index=False))

    if metrics_json is not None:
        report.append("")
        report.append("## metrics.json (cross-check excerpt)")
        report.append("")
        # Keep it short
        keys = ["n_trades", "expectancy_R", "profit_factor_R", "max_drawdown_pct", "forced_exits_total", "non_forced_exits_total"]
        excerpt = {k: metrics_json.get(k) for k in keys if k in metrics_json}
        report.append("```json")
        report.append(json.dumps(excerpt, indent=2))
        report.append("```")

    (out_dir / "alignment_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[OK] Wrote: {out_dir / 'alignment_report.md'}")
    print(f"[OK] Wrote: {out_dir / 'alignment_tables.csv'}")


if __name__ == "__main__":
    main()