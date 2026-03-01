from __future__ import annotations

from pathlib import Path

from src.live.io.orders_log import OrdersLog
from src.live.io.incidents_log import IncidentsLog


def main():
    base = Path("results/live_runs/_smoke_logs")
    orders = OrdersLog(base / "orders.csv")
    incidents = IncidentsLog(base / "incidents.csv")

    # Minimal fake event
    orders.append_event(
        run_id="RUN_SMOKE_001",
        run_mode="DEMO_LIVE",
        portfolio_mode="A",
        ftmo_day_id="2026-02-27",
        intent_id="INTENT_TEST_001",
        strategy_id="ANCHOR_MR_PURE_8P",
        strategy_version="freeze_v1_2",
        instrument="EURUSD",
        timeframe="M5",
        side="LONG",
        order_type="MARKET",
        time_in_force="IOC",
        risk_mode="FIXED_PCT_EQUITY",
        risk_value_pct=0.5,
        requested_sl_pips=8.0,
        requested_tp_pips=8.0,
        max_hold_min=60,
        valid_to_utc="2026-02-27T18:00:00Z",
        idempotency_key="TEST_KEY_001",
        client_order_id="COID-TEST-001",
        event_type="ORDER_SUBMIT",
        event_state="SUBMITTED",
        event_reason="OK",
        requested_qty=0.10,
        request_price=1.18235,
        attempt_no=1,
    )

    incidents.log(
        run_id="RUN_SMOKE_001",
        run_mode="DEMO_LIVE",
        portfolio_mode="A",
        severity="WARN",
        type="SMOKE_TEST",
        action_taken="IGNORE",
        ftmo_day_id="2026-02-27",
        instrument="EURUSD",
        details="Log smoke test OK",
        intent_id="INTENT_TEST_001",
        strategy_id="ANCHOR_MR_PURE_8P",
    )

    print("SMOKE LOGS OK:", base)


if __name__ == "__main__":
    main()