from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.live.mt5.adapter import MT5Adapter


def main():
    symbol = "EURUSD"
    mt = MT5Adapter()
    mt.connect()
    try:
        mt.ensure_symbol(symbol)

        acc = mt.get_account()
        print("ACCOUNT:", acc)

        tick = mt.get_tick(symbol)
        print("TICK:", tick)

        pos = mt.positions(symbol)
        ords = mt.orders(symbol)
        print(f"POSITIONS[{symbol}]:", len(pos))
        print(f"ORDERS[{symbol}]:", len(ords))

        now = datetime.now(timezone.utc)
        deals = mt.history_deals(now - timedelta(days=1), now, symbol=symbol)
        print(f"DEALS last 24h [{symbol}]:", len(deals))

        # Print a few deals if any
        for d in deals[:5]:
            print("DEAL:", d)

        print("SMOKE TEST OK")
    finally:
        mt.shutdown()


if __name__ == "__main__":
    main()