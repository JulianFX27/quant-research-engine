from __future__ import annotations

import pandas as pd

from backtester.risk.guardrails import Guardrails


def test_time_window_blocks_outside_window() -> None:
    gr = Guardrails(
        {
            "max_concurrent_positions": 1,
            "time_window_enabled": True,
            "window_start_utc": "13:30",
            "window_end_utc": "16:00",
            "max_holding_bars": 0,
        }
    )

    # 12:00 UTC is outside 13:30-16:00
    ts = pd.Timestamp("2026-01-01T12:00:00Z")
    ok, reason = gr.allow_entry(ts, active_positions=0)
    assert ok is False
    assert reason == "by_time_window"

    rep = gr.report()
    assert rep["blocked"]["by_time_window"] == 1


def test_max_concurrent_positions_blocks() -> None:
    gr = Guardrails(
        {
            "max_concurrent_positions": 1,
            "time_window_enabled": False,
            "max_holding_bars": 0,
        }
    )
    ts = pd.Timestamp("2026-01-01T14:00:00Z")
    ok, reason = gr.allow_entry(ts, active_positions=1)
    assert ok is False
    assert reason == "by_max_concurrent_positions"

    rep = gr.report()
    assert rep["blocked"]["by_max_concurrent_positions"] == 1


def test_max_holding_bars_forces_exit() -> None:
    gr = Guardrails(
        {
            "max_concurrent_positions": 1,
            "time_window_enabled": False,
            "max_holding_bars": 3,
        }
    )
    ok, reason = gr.should_force_exit(holding_bars=2)
    assert ok is False and reason is None

    ok, reason = gr.should_force_exit(holding_bars=3)
    assert ok is True and reason == "by_max_holding_bars"

    rep = gr.report()
    assert rep["forced_exits"]["by_max_holding_bars"] == 1
