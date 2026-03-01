from __future__ import annotations

from pathlib import Path

from src.live.mt5.adapter import MT5Adapter
from src.live.mt5.position_sync import PositionSync
from src.live.state.idempotency_store import IdempotencyStore
from src.live.io.orders_log import OrdersLog
from src.live.io.incidents_log import IncidentsLog


def main():
    mt = MT5Adapter()
    mt.connect()
    try:
        store = IdempotencyStore("results/live_state/idempotency.sqlite")
        orders_log = OrdersLog("results/live_runs/_smoke_sync/orders.csv")
        incidents_log = IncidentsLog("results/live_runs/_smoke_sync/incidents.csv")

        sync = PositionSync(
            mt5=mt,
            store=store,
            orders_log=orders_log,
            incidents_log=incidents_log,
            run_id="RUN_SMOKE_SYNC_001",
            run_mode="DEMO_LIVE",
            portfolio_mode="A",
            instrument="EURUSD",
            strategy_id="ANCHOR_MR_PURE_8P",
            strategy_version="freeze_v1_2",
        )

        state = sync.reconcile(ftmo_day_id="2026-02-27")
        print("STATE:", state.symbol, state.position_count, state.pending_order_count, state.assoc, "safe_halt=", state.safe_halt)

        print("SMOKE POSITION_SYNC OK")
    finally:
        mt.shutdown()


if __name__ == "__main__":
    main()