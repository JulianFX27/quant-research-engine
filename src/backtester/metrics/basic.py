# src/backtester/metrics/basic.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np

from backtester.execution.engine import Trade


def _to_float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _calc_max_consecutive_losses(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    max_streak = 0
    cur = 0
    for v in values:
        if v < 0:
            cur += 1
            if cur > max_streak:
                max_streak = cur
        else:
            cur = 0
    return int(max_streak)


def _calc_drawdown(equity: np.ndarray) -> Tuple[float, float]:
    """
    Returns (max_dd_abs, max_dd_pct) where pct is relative to peak.

    IMPORTANT:
      - Equity is expected to be strictly interpretable (e.g., includes an initial equity > 0).
      - If peak <= 0, pct is not updated (remains 0.0).
    """
    if equity.size == 0:
        return 0.0, 0.0

    peak = float(equity[0])
    max_dd_abs = 0.0
    max_dd_pct = 0.0

    for x in equity:
        x = float(x)
        if x > peak:
            peak = x
        dd = peak - x
        if dd > max_dd_abs:
            max_dd_abs = dd
        if peak > 0:
            dd_pct = dd / peak
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

    return float(max_dd_abs), float(max_dd_pct)


def summarize_trades(trades: List[Trade]) -> Dict[str, Any]:
    """
    Research-grade summary.
    - PnL-space metrics
    - Hold-time metrics
    - Exit reason counts + forced/EOF aggregates
    - R-space metrics (requires risk_price and qty; fallback to abs(entry-sl) if needed)

    Option A policy:
      - PnL-space equity uses an explicit initial equity of 1.0 to make DD% stable.
      - R-space drawdown reported in absolute R only (DD% in R-space is set to None).
    """
    if not trades:
        return {"n_trades": 0}

    # -------------------------
    # PnL-space core stats
    # -------------------------
    pnls = np.array([_to_float_or_none(getattr(t, "pnl", None)) or 0.0 for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    out: Dict[str, Any] = {
        "n_trades": int(len(trades)),
        "total_pnl": float(pnls.sum()),
        "winrate": float((pnls > 0).mean()),
        "avg_pnl": float(pnls.mean()) if pnls.size else 0.0,
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 1e-12 else float("inf"),
    }

    # Option A: stable equity base for meaningful drawdown%
    initial_equity = 1.0
    equity = initial_equity + np.cumsum(pnls)
    max_dd_abs, max_dd_pct = _calc_drawdown(equity)
    out["max_drawdown_abs"] = max_dd_abs
    out["max_drawdown_pct"] = max_dd_pct
    out["max_consecutive_losses"] = _calc_max_consecutive_losses(pnls)

    # -------------------------
    # Hold time
    # -------------------------
    hold_mins: list[float] = []
    for t in trades:
        et = getattr(t, "entry_time", None)
        xt = getattr(t, "exit_time", None)
        if et is None or xt is None:
            continue
        try:
            dt = (xt - et).total_seconds() / 60.0
        except Exception:
            continue
        if np.isfinite(dt) and dt >= 0:
            hold_mins.append(float(dt))

    if hold_mins:
        hm = np.array(hold_mins, dtype=float)
        out["avg_hold_minutes"] = float(hm.mean())
        out["median_hold_minutes"] = float(np.median(hm))
        out["min_hold_minutes"] = float(hm.min())
        out["max_hold_minutes"] = float(hm.max())
    else:
        out["avg_hold_minutes"] = None
        out["median_hold_minutes"] = None
        out["min_hold_minutes"] = None
        out["max_hold_minutes"] = None

    # -------------------------
    # Exit reason counts + aggregates
    # -------------------------
    reason_counts: Dict[str, int] = {}
    for t in trades:
        r = str(getattr(t, "exit_reason", "") or "UNKNOWN").strip()
        reason_counts[r] = int(reason_counts.get(r, 0)) + 1

    for r, n in sorted(reason_counts.items()):
        out[f"exit_reason_count__{r}"] = int(n)

    forced_exits_total = 0
    forced_eof_total = 0
    eof_exits_total = 0

    for rk, n in reason_counts.items():
        rk = str(rk)
        n = int(n)
        if rk.startswith("FORCE_"):
            forced_exits_total += n
            if rk == "FORCE_EOF":
                forced_eof_total += n
                eof_exits_total += n  # count EOF even if expressed as FORCE_EOF
        if rk == "EOF":
            eof_exits_total += n

    out["forced_exits_total"] = int(forced_exits_total)
    out["forced_eof_total"] = int(forced_eof_total)
    out["eof_exits_total"] = int(eof_exits_total)
    out["non_forced_exits_total"] = int(len(trades) - forced_exits_total)

    # -------------------------
    # R-space metrics (correct contract uses qty)
    #   R = pnl / (risk_price * abs(qty))
    # -------------------------
    r_mults: list[float] = []
    for t in trades:
        pnl = _to_float_or_none(getattr(t, "pnl", None))
        if pnl is None:
            continue

        qty = _to_float_or_none(getattr(t, "qty", None))
        if qty is None or abs(qty) <= 0:
            continue

        rp = _to_float_or_none(getattr(t, "risk_price", None))
        if rp is None or rp <= 0:
            entry = _to_float_or_none(getattr(t, "entry_price", None))
            slp = _to_float_or_none(getattr(t, "sl_price", None))
            if entry is not None and slp is not None:
                rp = abs(entry - slp)

        if rp is None or rp <= 0:
            continue

        denom = float(rp) * abs(float(qty))
        if denom <= 0:
            continue

        r_mults.append(float(pnl / denom))

    if r_mults:
        R = np.array(r_mults, dtype=float)

        out["n_trades_with_risk"] = int(R.size)
        out["expectancy_R"] = float(R.mean())
        out["avg_R"] = float(R.mean())
        out["median_R"] = float(np.median(R))

        wins_R = R[R > 0]
        losses_R = R[R < 0]

        out["winrate_R"] = float((R > 0).mean())
        out["avg_win_R"] = float(wins_R.mean()) if wins_R.size else 0.0
        out["avg_loss_R"] = float(losses_R.mean()) if losses_R.size else 0.0
        out["payoff_ratio_R"] = (
            float(out["avg_win_R"] / abs(out["avg_loss_R"])) if abs(out["avg_loss_R"]) > 1e-12 else None
        )

        out["max_consecutive_losses_R"] = _calc_max_consecutive_losses(R)

        # Option A: R-space drawdown in absolute R only (DD% in R-space is not meaningful/stable)
        eq_R = np.cumsum(R)
        dd_R_abs, _dd_R_pct_unused = _calc_drawdown(eq_R)
        out["max_drawdown_R_abs"] = float(dd_R_abs)
        out["max_drawdown_R_pct"] = None
    else:
        out["n_trades_with_risk"] = 0
        out["expectancy_R"] = None
        out["avg_R"] = None
        out["median_R"] = None
        out["winrate_R"] = None
        out["avg_win_R"] = None
        out["avg_loss_R"] = None
        out["payoff_ratio_R"] = None
        out["max_consecutive_losses_R"] = None
        out["max_drawdown_R_abs"] = None
        out["max_drawdown_R_pct"] = None

    return out


def trades_to_dicts(trades: List[Trade]) -> List[Dict[str, Any]]:
    """
    Export trades as dicts for CSV output.

    Contract:
      - risk_price is persisted in Trade when possible.
      - If missing (older runs), compute as abs(entry_price - sl_price) when available.
    """
    out: List[Dict[str, Any]] = []
    for t in trades:
        d = asdict(t)

        rp = _to_float_or_none(d.get("risk_price"))
        if rp is None or rp <= 0:
            entry = _to_float_or_none(d.get("entry_price"))
            sl = _to_float_or_none(d.get("sl_price"))
            d["risk_price"] = abs(entry - sl) if (entry is not None and sl is not None) else None

        out.append(d)

    return out
