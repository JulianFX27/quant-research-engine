from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

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
    return v


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


def _calc_drawdown(equity: np.ndarray) -> tuple[float, float]:
    if equity.size == 0:
        return 0.0, 0.0

    peak = equity[0]
    max_dd_abs = 0.0
    max_dd_pct = 0.0

    for x in equity:
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
    if not trades:
        return {"n_trades": 0}

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

    equity = np.cumsum(pnls)
    max_dd_abs, max_dd_pct = _calc_drawdown(equity)
    out["max_drawdown_abs"] = max_dd_abs
    out["max_drawdown_pct"] = max_dd_pct
    out["max_consecutive_losses"] = _calc_max_consecutive_losses(pnls)

    # Hold time
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

    # R-multiples (primary: risk_price field; fallback: abs(entry - sl))
    r_mults: list[float] = []
    for t in trades:
        pnl = _to_float_or_none(getattr(t, "pnl", None))
        if pnl is None:
            continue

        rp = _to_float_or_none(getattr(t, "risk_price", None))
        if rp is None or rp <= 0:
            entry = _to_float_or_none(getattr(t, "entry_price", None))
            slp = _to_float_or_none(getattr(t, "sl_price", None))
            if entry is not None and slp is not None:
                rp = abs(entry - slp)

        if rp is None or rp <= 0:
            continue

        r_mults.append(float(pnl / rp))

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

        eq_R = np.cumsum(R)
        dd_R_abs, dd_R_pct = _calc_drawdown(eq_R)
        out["max_drawdown_R_abs"] = dd_R_abs
        out["max_drawdown_R_pct"] = dd_R_pct
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
