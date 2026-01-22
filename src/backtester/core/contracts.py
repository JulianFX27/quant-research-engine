from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    asset_class: Literal["FX", "CRYPTO", "FUTURES", "EQUITY"]
    timezone: str = "UTC"
    pip_size: Optional[float] = None
    tick_size: Optional[float] = None
    multiplier: float = 1.0


@dataclass(frozen=True)
class Bar:
    ts_utc: "object"  # pandas.Timestamp, kept generic to avoid hard dep in typing
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    spread: float | None = None  # optional, in price units


@dataclass(frozen=True)
class OrderIntent:
    """A minimal intent produced by a strategy.

    Expand later: limit/stop, tif, oco, tags, etc.
    """

    side: Literal["BUY", "SELL"]
    qty: float
    sl_price: float | None = None
    tp_price: float | None = None
    tag: str = ""
