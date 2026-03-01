from __future__ import annotations

from datetime import datetime, timezone, timedelta

import MetaTrader5 as mt5

from src.live.mt5.adapter import MT5Adapter
from src.live.mt5.position_sync import PositionSync
from src.live.mt5.order_manager import OrderManager, ManualIntent
from src.live.state.idempotency_store import IdempotencyStore
from src.live.io.orders_log import OrdersLog
from src.live.io.incidents_log import IncidentsLog


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ftmo_day_id() -> str:
    return _utc_now().strftime("%Y-%m-%d")


def _preflight_or_raise():
    """
    Fails fast with actionable errors if terminal is not ready.
    """
    term = mt5.terminal_info()
    if term is None:
        raise RuntimeError(f"terminal_info() is None: {mt5.last_error()}")

    # In MT5, algo trading toggle / permissions can block order_send.
    # terminal_info usually exposes: trade_allowed, dlls_allowed, connected, etc.
    trade_allowed = getattr(term, "trade_allowed", None)
    connected = getattr(term, "connected", None)

    if connected is not None and not bool(connected):
        raise RuntimeError("MT5 terminal is not connected to broker/server (connected=False).")

    if trade_allowed is not None and not bool(trade_allowed):
        # This is the exact issue you hit (10027)
        raise RuntimeError(
            "AutoTrading / trade not allowed in terminal (trade_allowed=False).\n"
            "Fix:\n"
            "  1) In MT5, enable 'Algo Trading' (button must be GREEN).\n"
            "  2) Tools -> Options -> Expert Advisors -> enable 'Allow algorithmic trading'.\n"
            "  3) If your broker/FTMO has extra restrictions, ensure trading is allowed on this account."
        )

    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("No account detected. Make sure MT5 is OPEN and LOGGED IN.")

    # Print diagnostics (useful in logs)
    print("TERMINAL:", term)
    print("ACCOUNT :", f"login={acc.login} server={acc.server} currency={acc.currency} equity={acc.equity} balance={acc.balance}")


def main():
    # --- config for a safe demo trade ---
    run_id = "RUN_MANUAL_DEMO_001"
    run_mode = "DEMO_LIVE"
    portfolio_mode = "A"
    instrument = "EURUSD"

    strategy_id = "ANCHOR_MR_PURE_8P"
    strategy_version = "freeze_v1_2"

    ftmo_day_id = _ftmo_day_id()
    now = _utc_now()

    # Manual intent (one-shot)
    intent = ManualIntent(
        intent_id=f"MANUAL_{now.strftime('%Y%m%d_%H%M%S')}",
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        instrument=instrument,
        timeframe="M5",
        side="LONG",  # change to SHORT if needed
        order_type="MARKET",
        tif="IOC",
        sl_pips=8.0,
        tp_pips=8.0,
        max_hold_min=60,
        risk_mode="FIXED_PCT_EQUITY",
        risk_value_pct=0.5,
        valid_to_utc=(now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        decision_ts_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        bar_ts_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    requested_qty = 0.01  # ultra pequeño para demo

    mt = MT5Adapter()
    mt.connect()
    try:
        # Preflight checks (fail fast before any submission)
        _preflight_or_raise()

        store = IdempotencyStore("results/live_state/idempotency.sqlite")
        orders_log = OrdersLog(f"results/live_runs/{run_id}/orders.csv")
        incidents_log = IncidentsLog(f"results/live_runs/{run_id}/incidents.csv")

        sync = PositionSync(
            mt5=mt,
            store=store,
            orders_log=orders_log,
            incidents_log=incidents_log,
            run_id=run_id,
            run_mode=run_mode,
            portfolio_mode=portfolio_mode,
            instrument=instrument,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
        )

        mgr = OrderManager(
            mt5=mt,
            sync=sync,
            store=store,
            orders_log=orders_log,
            incidents_log=incidents_log,
            run_id=run_id,
            run_mode=run_mode,
            portfolio_mode=portfolio_mode,
        )

        # Broker state before
        before = sync.reconcile(ftmo_day_id=ftmo_day_id)
        print(
            "BROKER BEFORE:",
            f"pos={before.position_count} ord={before.pending_order_count} assoc={before.assoc.status} safe_halt={before.safe_halt}",
        )
        if before.position_count > 0:
            p = before.positions[0]
            print("POS:", p.ticket, p.symbol, p.type, p.volume, "open=", p.price_open, "SL=", p.sl, "TP=", p.tp, "comment=", p.comment)

        print("About to execute manual demo intent:", intent.intent_id)

        # Execute (OrderManager logs everything)
        mgr.execute_manual_intent(
            ftmo_day_id=ftmo_day_id,
            intent=intent,
            requested_qty=requested_qty,
            comment=None,  # Later: i=<token>|s=...|v=...|r=...
            magic=None,
        )

        # Broker state after
        after = sync.reconcile(ftmo_day_id=ftmo_day_id)
        print(
            "BROKER AFTER :",
            f"pos={after.position_count} ord={after.pending_order_count} assoc={after.assoc.status} safe_halt={after.safe_halt}",
        )
        if after.position_count > 0:
            p = after.positions[0]
            print("POS:", p.ticket, p.symbol, p.type, p.volume, "open=", p.price_open, "SL=", p.sl, "TP=", p.tp, "comment=", p.comment)

        print("Check logs:", f"results/live_runs/{run_id}/")

    finally:
        mt.shutdown()


if __name__ == "__main__":
    main()