# src/backtester/metrics/basic.py
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List


# --- Safety constants (research-grade hygiene) ---
# Any trade with risk_price <= RISK_EPS is considered invalid for R metrics.
RISK_EPS: float = 1e-12

# Cap extreme R values to avoid a single bad trade destroying aggregates.
# This does NOT "hide" issues because we also report extreme counts + max raw |R|.
R_CAP_ABS: float = 1000.0


def _is_finite(x: Any) -> bool:
    try:
        return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _quantile(sorted_vals: List[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w)


def _max_drawdown_abs(equity: List[float]) -> float | None:
    if not equity:
        return None
    peak = equity[0]
    max_dd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _max_drawdown_pct(equity: List[float]) -> float | None:
    # pct based on peak value; if peak == 0 then pct undefined
    if not equity:
        return None
    peak = equity[0]
    max_pct = 0.0
    for x in equity:
        if x > peak:
            peak = x
        if peak == 0:
            continue
        pct = (peak - x) / abs(peak)
        if pct > max_pct:
            max_pct = pct
    return float(max_pct) if max_pct > 0 else 0.0


def _get_hold_minutes(t: Any) -> float | None:
    """
    Robust hold time:
      - Prefer Trade.hold_minutes if present
      - Else compute from exit_time - entry_time if both exist
    """
    hm = _safe_float(getattr(t, "hold_minutes", None), default=None)
    if hm is not None:
        return hm

    et = getattr(t, "entry_time", None)
    xt = getattr(t, "exit_time", None)
    if et is None or xt is None:
        return None

    # entry_time/exit_time in engine are usually pd.Timestamp
    try:
        delta = (xt - et)
        secs = delta.total_seconds()
        if not math.isfinite(secs):
            return None
        return float(secs) / 60.0
    except Exception:
        return None


def _is_time_stop_normal_exit(t: Any) -> bool:
    """
    Time-stop semantics:
      If exit_reason == FORCE_MAX_HOLD AND strategy did not use SL/TP in price,
      then this is the *normal* exit (not a "forced/invalid" artifact).
    """
    er = getattr(t, "exit_reason", None)
    if er != "FORCE_MAX_HOLD":
        return False
    slp = getattr(t, "sl_price", None)
    tpp = getattr(t, "tp_price", None)
    return (slp is None) and (tpp is None)


def summarize_trades(trades: Iterable[Any]) -> Dict[str, Any]:
    """
    Research-grade summary.

    Assumptions about Trade object:
      - pnl (float)
      - entry_time / exit_time (pd.Timestamp or parseable)
      - hold_minutes (optional float)
      - exit_reason (optional str)
      - risk_price (optional float): risk in price units (proxy allowed)
      - sl_price/tp_price optional (used only for time-stop reclassification)

    NOTE:
      - R metrics here intentionally use pnl/risk_price (legacy behavior).
      - If you want R to be *strictly* pnl/(risk_price*abs(qty)), you can extend
        summarize_trades similarly to trades_to_dicts. For now we keep your metrics
        output stable; the CSV will have correct R.
    """
    pnl_list: List[float] = []
    win_list: List[float] = []
    loss_list: List[float] = []
    hold_minutes: List[float] = []
    exit_reason_counts: Dict[str, int] = {}

    # R-space lists
    R_list: List[float] = []
    R_win: List[float] = []
    R_loss: List[float] = []

    # R diagnostics
    n_trades_with_risk = 0
    n_trades_bad_risk = 0
    n_trades_tiny_risk = 0
    n_trades_extreme_R = 0
    R_max_abs_seen_raw: float | None = None

    # forced exit accounting
    # NOTE: FORCE_MAX_HOLD is conditionally reclassified as normal exit for time-stop strategies.
    forced_eof_reasons = {"FORCE_EOF", "EOF"}
    forced_exits_total = 0
    forced_eof_total = 0
    eof_exits_total = 0
    non_forced_exits_total = 0

    # Build equity curve in pnl-space for DD
    equity: List[float] = []
    eq = 0.0

    for t in trades:
        pnl = _safe_float(getattr(t, "pnl", None), default=None)
        if pnl is None:
            continue

        pnl_list.append(pnl)
        eq += pnl
        equity.append(eq)

        if pnl >= 0:
            win_list.append(pnl)
        else:
            loss_list.append(pnl)

        hm = _get_hold_minutes(t)
        if hm is not None:
            hold_minutes.append(hm)

        er = getattr(t, "exit_reason", None)
        if isinstance(er, str) and er:
            exit_reason_counts[er] = exit_reason_counts.get(er, 0) + 1

            # Time-stop: treat FORCE_MAX_HOLD as normal exit when no SL/TP in price
            if _is_time_stop_normal_exit(t):
                non_forced_exits_total += 1
            else:
                if er in forced_eof_reasons or er.startswith("FORCE_"):
                    forced_exits_total += 1
                    if er == "FORCE_EOF":
                        forced_eof_total += 1
                    if er in {"EOF", "FORCE_EOF"}:
                        eof_exits_total += 1
                else:
                    non_forced_exits_total += 1
        else:
            non_forced_exits_total += 1

        # --- R-space (legacy) ---
        rp = _safe_float(getattr(t, "risk_price", None), default=None)
        if rp is None:
            continue

        n_trades_with_risk += 1

        if (not math.isfinite(rp)) or rp <= 0:
            n_trades_bad_risk += 1
            continue

        if rp <= RISK_EPS:
            n_trades_tiny_risk += 1
            continue

        R_raw = pnl / rp
        if not math.isfinite(R_raw):
            n_trades_bad_risk += 1
            continue

        ar = abs(R_raw)
        if R_max_abs_seen_raw is None or ar > R_max_abs_seen_raw:
            R_max_abs_seen_raw = float(ar)

        if ar > R_CAP_ABS:
            n_trades_extreme_R += 1
            R = float(max(-R_CAP_ABS, min(R_CAP_ABS, R_raw)))
        else:
            R = float(R_raw)

        R_list.append(R)
        if R >= 0:
            R_win.append(R)
        else:
            R_loss.append(R)

    n_trades = len(pnl_list)
    total_pnl = float(sum(pnl_list)) if pnl_list else 0.0
    avg_pnl = (total_pnl / n_trades) if n_trades > 0 else None

    winrate = (len(win_list) / n_trades) if n_trades > 0 else None
    avg_win = (sum(win_list) / len(win_list)) if win_list else None
    avg_loss = (sum(loss_list) / len(loss_list)) if loss_list else None

    gross_win = float(sum(win_list)) if win_list else 0.0
    gross_loss_abs = float(abs(sum(loss_list))) if loss_list else 0.0
    profit_factor = (gross_win / gross_loss_abs) if gross_loss_abs > 0 else (float("inf") if gross_win > 0 else None)

    max_dd_abs = _max_drawdown_abs(equity)
    max_dd_pct = _max_drawdown_pct(equity)

    # Consecutive losses in pnl-space
    max_consec_losses = 0
    cur = 0
    for p in pnl_list:
        if p < 0:
            cur += 1
            if cur > max_consec_losses:
                max_consec_losses = cur
        else:
            cur = 0

    # Hold stats
    hold_sorted = sorted(hold_minutes) if hold_minutes else []
    avg_hold = (sum(hold_sorted) / len(hold_sorted)) if hold_sorted else None
    med_hold = _quantile(hold_sorted, 0.5)
    min_hold = float(hold_sorted[0]) if hold_sorted else None
    max_hold = float(hold_sorted[-1]) if hold_sorted else None

    # R metrics
    R_sorted = sorted(R_list) if R_list else []
    expectancy_R = (sum(R_list) / len(R_list)) if R_list else None
    avg_R = expectancy_R
    median_R = _quantile(R_sorted, 0.5)

    winrate_R = (len(R_win) / len(R_list)) if R_list else None
    avg_win_R = (sum(R_win) / len(R_win)) if R_win else None
    avg_loss_R = (sum(R_loss) / len(R_loss)) if R_loss else None

    payoff_ratio_R = None
    if avg_win_R is not None and avg_loss_R is not None and avg_loss_R != 0:
        payoff_ratio_R = float(abs(avg_win_R / avg_loss_R))

    gross_win_R = float(sum([r for r in R_list if r > 0])) if R_list else 0.0
    gross_loss_R_abs = float(abs(sum([r for r in R_list if r < 0]))) if R_list else 0.0
    profit_factor_R = (gross_win_R / gross_loss_R_abs) if gross_loss_R_abs > 0 else (float("inf") if gross_win_R > 0 else None)

    # Consecutive losses in R-space (capped)
    max_consec_losses_R = 0
    curR = 0
    for r in R_list:
        if r < 0:
            curR += 1
            if curR > max_consec_losses_R:
                max_consec_losses_R = curR
        else:
            curR = 0

    out: Dict[str, Any] = {
        "n_trades": n_trades,
        "total_pnl": total_pnl,
        "winrate": winrate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown_abs": max_dd_abs,
        "max_drawdown_pct": max_dd_pct,
        "max_consecutive_losses": int(max_consec_losses),

        "avg_hold_minutes": avg_hold,
        "median_hold_minutes": med_hold,
        "min_hold_minutes": min_hold,
        "max_hold_minutes": max_hold,

        "forced_exits_total": int(forced_exits_total),
        "forced_eof_total": int(forced_eof_total),
        "eof_exits_total": int(eof_exits_total),
        "non_forced_exits_total": int(non_forced_exits_total),

        "n_trades_with_risk": int(n_trades_with_risk),
        "n_trades_bad_risk": int(n_trades_bad_risk),
        "n_trades_tiny_risk": int(n_trades_tiny_risk),
        "n_trades_extreme_R": int(n_trades_extreme_R),
        "R_cap_abs": float(R_CAP_ABS),
        "R_max_abs_seen_raw": R_max_abs_seen_raw,

        "expectancy_R": expectancy_R,
        "avg_R": avg_R,
        "median_R": median_R,
        "winrate_R": winrate_R,
        "avg_win_R": avg_win_R,
        "avg_loss_R": avg_loss_R,
        "payoff_ratio_R": payoff_ratio_R,
        "profit_factor_R": profit_factor_R,
        "max_consecutive_losses_R": int(max_consec_losses_R),
    }

    for k, v in sorted(exit_reason_counts.items()):
        out[f"exit_reason_count__{k}"] = int(v)

    return out


def trades_to_dicts(trades: Iterable[Any]) -> List[Dict[str, Any]]:
    """
    Convert trades to dicts for CSV persistence.

    Persist risk_price and R using the formal contract:
      R = pnl / (risk_price * abs(qty))

    Also persist audit columns: side, qty, tag.
    """
    rows: List[Dict[str, Any]] = []
    for t in trades:
        d: Dict[str, Any] = {}

        d["entry_time"] = getattr(t, "entry_time", None)
        d["exit_time"] = getattr(t, "exit_time", None)

        d["entry_idx"] = getattr(t, "entry_idx", None)
        d["exit_idx"] = getattr(t, "exit_idx", None)

        # ---- NEW: audit columns ----
        d["side"] = getattr(t, "side", None)
        qty = _safe_float(getattr(t, "qty", None), default=None)
        d["qty"] = qty
        d["tag"] = getattr(t, "tag", None)

        d["entry_price"] = getattr(t, "entry_price", None)
        d["exit_price"] = getattr(t, "exit_price", None)
        d["sl_price"] = getattr(t, "sl_price", None)
        d["tp_price"] = getattr(t, "tp_price", None)

        pnl = _safe_float(getattr(t, "pnl", None), default=None)
        d["pnl"] = pnl

        hm = _get_hold_minutes(t)
        d["hold_minutes"] = hm

        d["exit_reason"] = getattr(t, "exit_reason", None)

        rp = _safe_float(getattr(t, "risk_price", None), default=None)
        d["risk_price"] = rp

        # Formal R contract
        if pnl is not None and rp is not None and math.isfinite(rp) and rp > RISK_EPS and qty is not None and qty != 0:
            denom = float(rp) * abs(float(qty))
            if denom > 0:
                R_raw = float(pnl) / denom
                d["R"] = float(R_raw) if math.isfinite(R_raw) else None
            else:
                d["R"] = None
        else:
            d["R"] = None

        rows.append(d)

    return rows
