from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class SendResult:
    ok: bool
    retcode: int
    comment: str
    request_id: Optional[int] = None
    broker_order_id: Optional[int] = None
    broker_deal_id: Optional[int] = None
    raw: Optional[object] = None


@dataclass(frozen=True)
class Tick:
    symbol: str
    time_utc: datetime
    bid: float
    ask: float
    last: float | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    login: int
    server: str
    currency: str
    equity: float
    balance: float
    margin: float
    margin_free: float
    leverage: int


@dataclass(frozen=True)
class Position:
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    sl: float
    tp: float
    time_utc: datetime
    magic: int | None = None
    comment: str | None = None


@dataclass(frozen=True)
class Order:
    ticket: int
    symbol: str
    type: int
    volume_current: float
    price_open: float
    sl: float
    tp: float
    time_setup_utc: datetime
    magic: int | None = None
    comment: str | None = None


@dataclass(frozen=True)
class Deal:
    ticket: int
    order: int
    position_id: int
    symbol: str
    type: int
    volume: float
    price: float
    profit: float
    time_utc: datetime
    comment: str | None = None
    magic: int | None = None