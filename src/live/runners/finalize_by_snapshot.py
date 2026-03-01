from __future__ import annotations

from datetime import datetime, timezone

from src.live.mt5.adapter import MT5Adapter
from src.live.mt5.close_watcher import CloseWatcher
from src.live.state.idempotency_store import IdempotencyStore
from src.live.io.orders_log import OrdersLog
from src.live.io.incidents_log import IncidentsLog


def main():
    run_id = "RUN_MANUAL_DEMO_001"
    run_mode = "DEMO_LIVE"
    portfolio_mode = "A"
    instrument = "EURUSD"
    strategy_id = "ANCHOR_MR_PURE_8P"
    strategy_version = "freeze_v1_2"

    ftmo_day_id = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    mt = MT5Adapter()
    mt.connect()
    try:
        store = IdempotencyStore("results/live_state/idempotency.sqlite")
        orders_log = OrdersLog(f"results/live_runs/{run_id}/orders.csv")
        incidents_log = IncidentsLog(f"results/live_runs/{run_id}/incidents.csv")

        w = CloseWatcher(
            mt5=mt,
            store=store,
            orders_log=orders_log,
            incidents_log=incidents_log,
            run_id=run_id,
            run_mode=run_mode,
            portfolio_mode=portfolio_mode,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            instrument=instrument,
        )

        n = w.finalize_if_closed(ftmo_day_id=ftmo_day_id)
        print("FINALIZED_BY_SNAPSHOT:", n)

    finally:
        mt.shutdown()


if __name__ == "__main__":
    main()