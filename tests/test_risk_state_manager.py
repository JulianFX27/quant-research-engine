from datetime import datetime, timezone

import pytest

from src.risk.risk_state_manager import RiskStateManager, RiskConfig
from tests._helpers import write_json, base_state


def dt_utc(y, m, d, hh=0, mm=0, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


@pytest.fixture
def tmp_paths(tmp_path):
    state_path = str(tmp_path / "state.json")
    override_path = str(tmp_path / "override.json")
    return state_path, override_path


def test_rollover_initializes_day(tmp_paths):
    state_path, override_path = tmp_paths
    s = base_state()
    write_json(state_path, s)

    rsm = RiskStateManager(
        state_path=state_path,
        override_path=override_path,
        config=RiskConfig(tz_name="Europe/Prague")
    )

    state = rsm.load_state()
    state = rsm.rollover_if_needed(state, run_id="run1", now_utc=dt_utc(2026, 2, 19, 10, 0, 0))

    assert state["current_day_id"] is not None
    assert state["equity_start_day"] == pytest.approx(1.0)
    assert state["trades_taken_today"] == 0
    assert state["daily_stop_triggered"] is False
    assert state["trading_enabled"] is True
    assert len(state["audit_tail"]) >= 1


def test_daily_stop_triggers(tmp_paths):
    state_path, override_path = tmp_paths
    s = base_state()
    s["current_day_id"] = "2026-02-19"
    s["equity_start_day"] = 1.0
    write_json(state_path, s)

    rsm = RiskStateManager(
        state_path=state_path,
        override_path=override_path,
        config=RiskConfig(daily_stop_pct=-1.5)
    )

    state = rsm.load_state()
    # equity drops -1.6% intraday
    state = rsm.on_trade_closed(state, equity_after=0.984, run_id="run2", now_utc=dt_utc(2026, 2, 19, 12, 0, 0))

    assert state["daily_stop_triggered"] is True
    assert state["trading_enabled"] is False
    assert state["daily_pnl_pct"] <= -1.5


def test_hard_stop_dd_triggers(tmp_paths):
    state_path, override_path = tmp_paths
    s = base_state()
    s["current_day_id"] = "2026-02-19"
    s["equity_start_day"] = 1.0
    s["equity_peak"] = 1.0
    write_json(state_path, s)

    rsm = RiskStateManager(
        state_path=state_path,
        override_path=override_path,
        config=RiskConfig(hard_stop_dd_pct=8.0)
    )

    state = rsm.load_state()
    # equity drops -8.1% from peak -> hard stop
    state = rsm.on_trade_closed(state, equity_after=0.919, run_id="run3", now_utc=dt_utc(2026, 2, 19, 12, 0, 0))

    assert state["hard_stop_dd_triggered"] is True
    assert state["trading_enabled"] is False
    assert state["dd_current_pct"] >= 8.0


def test_max_trades_per_day_blocks_reentry(tmp_paths):
    state_path, override_path = tmp_paths
    s = base_state()
    s["current_day_id"] = "2026-02-19"
    s["equity_start_day"] = 1.0
    s["max_trades_per_day"] = 1
    s["trades_taken_today"] = 0
    write_json(state_path, s)

    rsm = RiskStateManager(state_path=state_path, override_path=override_path)

    state = rsm.load_state()
    state = rsm.on_trade_closed(state, equity_after=1.005, run_id="run4", now_utc=dt_utc(2026, 2, 19, 12, 0, 0))

    assert state["trades_taken_today"] == 1
    # can_trade should now be false
    assert rsm.can_trade(state, run_id="run4", now_utc=dt_utc(2026, 2, 19, 12, 5, 0)) is False


def test_rollover_does_not_reenable_after_hard_stop(tmp_paths):
    state_path, override_path = tmp_paths
    s = base_state()
    s["current_day_id"] = "2026-02-19"
    s["equity_start_day"] = 1.0
    s["hard_stop_dd_triggered"] = True
    s["trading_enabled"] = False
    write_json(state_path, s)

    rsm = RiskStateManager(state_path=state_path, override_path=override_path)

    # Next day rollover
    state = rsm.load_state()
    state = rsm.rollover_if_needed(state, run_id="run5", now_utc=dt_utc(2026, 2, 20, 10, 0, 0))

    assert state["current_day_id"] == "2026-02-20"
    assert state["trading_enabled"] is False  # still blocked


def test_manual_reset_requires_override(tmp_paths):
    state_path, override_path = tmp_paths
    s = base_state()
    s["current_day_id"] = "2026-02-19"
    s["equity_start_day"] = 1.0
    s["hard_stop_dd_triggered"] = True
    s["trading_enabled"] = False
    write_json(state_path, s)

    rsm = RiskStateManager(state_path=state_path, override_path=override_path)

    # Attempt reset without override file: no change
    state = rsm.load_state()
    state2 = rsm.try_manual_reset_hard_stop(state, run_id="run6", now_utc=dt_utc(2026, 2, 19, 13, 0, 0))
    assert state2["hard_stop_dd_triggered"] is True

    # Now enable override and retry
    write_json(override_path, {
        "allow_reset_hard_stop": True,
        "reason": "post-mortem completed",
        "approved_by": "Julian",
        "ts_utc": "2026-02-19T13:01:00Z"
    })

    state3 = rsm.try_manual_reset_hard_stop(state2, run_id="run6", now_utc=dt_utc(2026, 2, 19, 13, 2, 0))
    assert state3["hard_stop_dd_triggered"] is False
