# src/backtester/metrics/test_drawdown_option_a.py
import numpy as np

from backtester.metrics.basic import _calc_drawdown


def test_drawdown_pct_stable_with_initial_equity_base():
    """
    Regression test for the pathological case where equity peaks near 0.

    Under Option A, equity is expected to include an explicit positive initial base (e.g., 1.0),
    making drawdown percentage interpretable and not exploding.
    """
    initial_equity = 1.0

    # Mimics: peak near 0 in raw cumulative pnl, but stable after adding initial_equity.
    pnls = np.array(
        [-0.00111, -0.00108, -0.00113, 0.00394, -1.13815],
        dtype=float,
    )
    equity = initial_equity + np.cumsum(pnls)

    dd_abs, dd_pct = _calc_drawdown(equity)

    assert dd_abs > 0.5
    # With initial equity base, dd_pct should be O(1), not thousands.
    assert 0.5 < dd_pct < 3.0
