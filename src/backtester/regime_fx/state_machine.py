from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .thresholds import Thresholds


@dataclass(frozen=True)
class StateMachineParams:
    confirm_bars: int = 2
    min_state_bars: int = 9
    cooldown_bars: int = 9
    hysteresis_enabled: bool = True
    hysteresis_margin_frac: float = 0.10


def _compute_exit_thresholds(thr: Thresholds, margin_frac: float) -> Thresholds:
    """
    Hysteresis: construye umbrales de salida (exit) más estrictos.
    margin = margin_frac * (trend - range)
    Para TREND, para "seguir siendo TREND" exigimos er >= er_trend - margin, dp >= dp_trend - margin_dp.
    Para RANGE, para "seguir siendo RANGE" exigimos er <= er_range + margin, dp <= dp_range + margin_dp.

    Nota: implementamos esto como checks internos al detectar cambio desde el estado actual.
    """
    er_margin = margin_frac * (thr.er_trend - thr.er_range)
    dp_margin = margin_frac * (thr.dp_trend - thr.dp_range)

    return Thresholds(
        er_trend=thr.er_trend - er_margin,
        er_range=thr.er_range + er_margin,
        dp_trend=thr.dp_trend - dp_margin,
        dp_range=thr.dp_range + dp_margin,
    )


def apply_state_machine(
    raw_regime: pd.Series,
    er: pd.Series,
    dp: pd.Series,
    thr: Thresholds,
    params: StateMachineParams,
) -> pd.Series:
    """
    Deterministic finite state machine:
      - confirm candidate for confirm_bars consecutive bars
      - enforce min_state_bars
      - enforce cooldown_bars after change
      - hysteresis uses "exit thresholds" to reduce flip-flop
    """
    idx = raw_regime.index
    state = []
    if len(idx) == 0:
        return pd.Series([], dtype="object", index=idx)

    current = "NO_TRADE"
    bars_in_state = 0
    cooldown_left = 0

    candidate: Optional[str] = None
    candidate_count = 0

    exit_thr = _compute_exit_thresholds(thr, params.hysteresis_margin_frac) if params.hysteresis_enabled else thr

    def can_leave_current(t: int) -> bool:
        # If current is TREND, keep TREND unless ER/DP fall below exit thresholds.
        if current == "TREND":
            e = er.iat[t]
            d = dp.iat[t]
            if np.isnan(e) or np.isnan(d):
                return True
            return not (e >= exit_thr.er_trend and d >= exit_thr.dp_trend)
        if current == "RANGE":
            e = er.iat[t]
            d = dp.iat[t]
            if np.isnan(e) or np.isnan(d):
                return True
            return not (e <= exit_thr.er_range and d <= exit_thr.dp_range)
        return True  # NO_TRADE can always switch

    for t in range(len(idx)):
        r = raw_regime.iat[t]

        # default push current
        if cooldown_left > 0:
            cooldown_left -= 1
            bars_in_state += 1
            state.append(current)
            continue

        # If raw suggests staying, reset candidate
        if r == current:
            candidate = None
            candidate_count = 0
            bars_in_state += 1
            state.append(current)
            continue

        # Hysteresis: if not allowed to leave current yet, stay.
        if params.hysteresis_enabled and not can_leave_current(t):
            candidate = None
            candidate_count = 0
            bars_in_state += 1
            state.append(current)
            continue

        # Enforce minimum time in state before allowing change
        if bars_in_state < params.min_state_bars:
            candidate = None
            candidate_count = 0
            bars_in_state += 1
            state.append(current)
            continue

        # Update candidate confirmation
        if candidate is None or candidate != r:
            candidate = r
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count >= params.confirm_bars:
            # Commit change
            current = candidate
            bars_in_state = 0
            cooldown_left = params.cooldown_bars
            candidate = None
            candidate_count = 0
            state.append(current)
            continue

        # Not confirmed yet -> hold current
        bars_in_state += 1
        state.append(current)

    return pd.Series(state, index=idx, dtype="object")
