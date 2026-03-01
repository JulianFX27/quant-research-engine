from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional, List

import MetaTrader5 as mt5

from .mt5_types import SendResult, Tick, AccountSnapshot, Position, Order, Deal


def _utc_from_ts(ts: int) -> datetime:
    # MT5 timestamps are usually seconds since epoch
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class MT5Adapter:
    """
    Thin wrapper over MetaTrader5 package.
    No business logic here. Only I/O + normalization.
    """

    def __init__(self, *, path: Optional[str] = None, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None):
        self._path = path
        self._login = login
        self._password = password
        self._server = server
        self._connected = False

    # ---------- lifecycle ----------

    def connect(self) -> None:
        """
        Connect to already running MT5 terminal.
        We DO NOT login programmatically.
        MT5 must be open and logged in manually.
        """
    
        if self._connected:
            return
    
        # Attach to running terminal
        ok = mt5.initialize()
    
        if not ok:
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    
        term = mt5.terminal_info()
        if term is None:
            raise RuntimeError(f"terminal_info() failed: {mt5.last_error()}")
    
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError(
                "MT5 connected but no account detected.\n"
                "Make sure MT5 is OPEN and LOGGED IN."
            )
    
        self._connected = True

    def shutdown(self) -> None:
        if self._connected:
            mt5.shutdown()
        self._connected = False

    # ---------- symbols / market data ----------

    def ensure_symbol(self, symbol: str) -> None:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info({symbol}) is None: {mt5.last_error()}")

        if not info.visible:
            ok = mt5.symbol_select(symbol, True)
            if not ok:
                raise RuntimeError(f"symbol_select({symbol}, True) failed: {mt5.last_error()}")

    def get_tick(self, symbol: str) -> Tick:
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise RuntimeError(f"symbol_info_tick({symbol}) returned None: {mt5.last_error()}")
        # Some terminals provide .time as seconds
        return Tick(
            symbol=symbol,
            time_utc=_utc_from_ts(int(t.time)),
            bid=float(t.bid),
            ask=float(t.ask),
            last=float(getattr(t, "last", 0.0)) if hasattr(t, "last") else None,
        )

    def get_account(self) -> AccountSnapshot:
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError(f"account_info() returned None: {mt5.last_error()}")
        return AccountSnapshot(
            login=int(acc.login),
            server=str(acc.server),
            currency=str(acc.currency),
            equity=float(acc.equity),
            balance=float(acc.balance),
            margin=float(acc.margin),
            margin_free=float(acc.margin_free),
            leverage=int(acc.leverage),
        )

    # ---------- trading ----------

    def send_market_order(
        self,
        *,
        symbol: str,
        side: str,  # "LONG" or "SHORT"
        volume: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        comment: Optional[str],
        magic: Optional[int],
        deviation_points: int = 20,
    ) -> SendResult:
        """
        Sends a MARKET order.
        Note: MetaTrader5 order_send request shape differs by broker and terminal.
        This implementation follows the typical MT5 python package format.
        """
        if side not in ("LONG", "SHORT"):
            raise ValueError("side must be LONG or SHORT")

        self.ensure_symbol(symbol)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return SendResult(ok=False, retcode=-1, comment=f"NO_TICK {mt5.last_error()}")

        # Determine order type & price
        if side == "LONG":
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "deviation": int(deviation_points),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Best-effort metadata for STRONG_ID
        if comment:
            request["comment"] = str(comment)
        if magic is not None:
            request["magic"] = int(magic)

        # Best-effort SL/TP in request (some brokers allow, some require modify)
        if sl_price is not None:
            request["sl"] = float(sl_price)
        if tp_price is not None:
            request["tp"] = float(tp_price)

        res = mt5.order_send(request)
        if res is None:
            err = mt5.last_error()
            return SendResult(ok=False, retcode=-1, comment=f"order_send returned None: {err}")

        ok = (res.retcode == mt5.TRADE_RETCODE_DONE) or (res.retcode == mt5.TRADE_RETCODE_PLACED)
        # order_send result fields vary; use getattr defensively
        broker_order_id = int(getattr(res, "order", 0)) if getattr(res, "order", 0) else None
        broker_deal_id = int(getattr(res, "deal", 0)) if getattr(res, "deal", 0) else None

        return SendResult(
            ok=bool(ok),
            retcode=int(res.retcode),
            comment=str(getattr(res, "comment", "")),
            request_id=int(getattr(res, "request_id", 0)) if getattr(res, "request_id", 0) else None,
            broker_order_id=broker_order_id,
            broker_deal_id=broker_deal_id,
            raw=res,
        )

    def positions(self, symbol: Optional[str] = None) -> List[Position]:
        ps = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if ps is None:
            # None can also mean "no positions" or an error; last_error clarifies
            err = mt5.last_error()
            # If it's truly none positions, brokers usually return empty tuple, but handle anyway:
            if err and err[0] != 0:
                raise RuntimeError(f"positions_get failed: {err}")
            return []

        out: List[Position] = []
        for p in ps:
            out.append(
                Position(
                    ticket=int(p.ticket),
                    symbol=str(p.symbol),
                    type=int(p.type),
                    volume=float(p.volume),
                    price_open=float(p.price_open),
                    sl=float(p.sl),
                    tp=float(p.tp),
                    time_utc=_utc_from_ts(int(p.time)),
                    magic=int(getattr(p, "magic", 0)) if hasattr(p, "magic") else None,
                    comment=str(getattr(p, "comment", "")) if hasattr(p, "comment") else None,
                )
            )
        return out

    def orders(self, symbol: Optional[str] = None) -> List[Order]:
        os = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        if os is None:
            err = mt5.last_error()
            if err and err[0] != 0:
                raise RuntimeError(f"orders_get failed: {err}")
            return []

        out: List[Order] = []
        for o in os:
            out.append(
                Order(
                    ticket=int(o.ticket),
                    symbol=str(o.symbol),
                    type=int(o.type),
                    volume_current=float(o.volume_current),
                    price_open=float(o.price_open),
                    sl=float(o.sl),
                    tp=float(o.tp),
                    time_setup_utc=_utc_from_ts(int(o.time_setup)),
                    magic=int(getattr(o, "magic", 0)) if hasattr(o, "magic") else None,
                    comment=str(getattr(o, "comment", "")) if hasattr(o, "comment") else None,
                )
            )
        return out

    def history_deals(self, time_from_utc: datetime, time_to_utc: datetime, symbol: Optional[str] = None) -> List[Deal]:
        if time_from_utc.tzinfo is None or time_to_utc.tzinfo is None:
            raise ValueError("history_deals expects timezone-aware UTC datetimes")

        deals = mt5.history_deals_get(time_from_utc, time_to_utc, group=symbol) if symbol else mt5.history_deals_get(time_from_utc, time_to_utc)
        if deals is None:
            err = mt5.last_error()
            if err and err[0] != 0:
                raise RuntimeError(f"history_deals_get failed: {err}")
            return []

        out: List[Deal] = []
        for d in deals:
            out.append(
                Deal(
                    ticket=int(d.ticket),
                    order=int(d.order),
                    position_id=int(d.position_id),
                    symbol=str(d.symbol),
                    type=int(d.type),
                    volume=float(d.volume),
                    price=float(d.price),
                    profit=float(d.profit),
                    time_utc=_utc_from_ts(int(d.time)),
                    comment=str(getattr(d, "comment", "")) if hasattr(d, "comment") else None,
                    magic=int(getattr(d, "magic", 0)) if hasattr(d, "magic") else None,
                )
            )
        return out