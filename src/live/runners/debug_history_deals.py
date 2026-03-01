from __future__ import annotations

from datetime import datetime, timedelta
import MetaTrader5 as mt5


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _print_deal(d) -> None:
    # Deal fields vary by build; we print defensively
    print(
        "DEAL",
        "ticket=", getattr(d, "ticket", None),
        "time=", getattr(d, "time", None),
        "time_msc=", getattr(d, "time_msc", None),
        "symbol=", getattr(d, "symbol", None),
        "position_id=", getattr(d, "position_id", None),
        "order=", getattr(d, "order", None),
        "type=", getattr(d, "type", None),
        "entry=", getattr(d, "entry", None),
        "volume=", getattr(d, "volume", None),
        "price=", getattr(d, "price", None),
        "profit=", getattr(d, "profit", None),
        "comment=", getattr(d, "comment", None),
    )


def main():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        term = mt5.terminal_info()
        acc = mt5.account_info()
        print("TERMINAL:", term)
        print("ACCOUNT :", acc)

        # Use local naive datetimes (this is often what MT5 expects)
        now_local = datetime.now()
        frm_local = now_local - timedelta(hours=48)

        print("QUERY window (local naive):", _fmt_dt(frm_local), "->", _fmt_dt(now_local))
        deals = mt5.history_deals_get(frm_local, now_local)
        if deals is None:
            print("history_deals_get returned None:", mt5.last_error())
            return

        deals = list(deals)
        print("DEALS COUNT:", len(deals))

        # Print last 20 deals
        for d in deals[-20:]:
            _print_deal(d)

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()