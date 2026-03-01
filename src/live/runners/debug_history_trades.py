from __future__ import annotations

from datetime import datetime, timedelta
import MetaTrader5 as mt5


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _p(obj, name: str, default=None):
    return getattr(obj, name, default)


def _print_deal(d) -> None:
    print(
        "DEAL",
        "ticket=", _p(d, "ticket"),
        "time=", _p(d, "time"),
        "time_msc=", _p(d, "time_msc"),
        "symbol=", repr(_p(d, "symbol")),
        "position_id=", _p(d, "position_id"),
        "order=", _p(d, "order"),
        "type=", _p(d, "type"),
        "entry=", _p(d, "entry"),
        "volume=", _p(d, "volume"),
        "price=", _p(d, "price"),
        "profit=", _p(d, "profit"),
        "comment=", _p(d, "comment"),
    )


def _print_order(o) -> None:
    print(
        "ORDER",
        "ticket=", _p(o, "ticket"),
        "time_setup=", _p(o, "time_setup"),
        "time_done=", _p(o, "time_done"),
        "time_done_msc=", _p(o, "time_done_msc"),
        "symbol=", repr(_p(o, "symbol")),
        "position_id=", _p(o, "position_id"),
        "type=", _p(o, "type"),
        "state=", _p(o, "state"),
        "volume_initial=", _p(o, "volume_initial"),
        "volume_current=", _p(o, "volume_current"),
        "price_open=", _p(o, "price_open"),
        "price_current=", _p(o, "price_current"),
        "sl=", _p(o, "sl"),
        "tp=", _p(o, "tp"),
        "comment=", _p(o, "comment"),
    )


def main():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        print("TERMINAL:", mt5.terminal_info())
        print("ACCOUNT :", mt5.account_info())

        now = datetime.now()                # LOCAL naive (lo que MT5 espera mejor)
        frm = now - timedelta(days=30)

        print("\nWINDOW (local naive):", _fmt(frm), "->", _fmt(now))

        deals = mt5.history_deals_get(frm, now)
        if deals is None:
            print("history_deals_get=None:", mt5.last_error())
            deals_list = []
        else:
            deals_list = list(deals)

        orders = mt5.history_orders_get(frm, now)
        if orders is None:
            print("history_orders_get=None:", mt5.last_error())
            orders_list = []
        else:
            orders_list = list(orders)

        print("\nDEALS COUNT :", len(deals_list))
        print("ORDERS COUNT:", len(orders_list))

        print("\nLAST 20 DEALS:")
        for d in deals_list[-20:]:
            _print_deal(d)

        print("\nLAST 20 ORDERS:")
        for o in orders_list[-20:]:
            _print_order(o)

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()