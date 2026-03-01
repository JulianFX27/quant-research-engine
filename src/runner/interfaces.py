from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal
from datetime import datetime


Direction = Literal["LONG", "SHORT"]
Action = Literal["ENTER", "EXIT"]


@dataclass(frozen=True)
class Bar:
    ts_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)  # <-- para features: anchor, shock_z, etc.


@dataclass(frozen=True)
class StrategyContext:
    day_id_ftmo: str
    equity_current: float
    dd_current_pct: float
    trades_taken_today: int
    trading_enabled: bool
    account_mode: Literal["challenge", "funded"]
    instrument: str = "EURUSD"
    pip_size: float = 0.0001


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    ts_utc: datetime
    action: Action                       # ENTER o EXIT
    direction: Optional[Direction] = None # requerido si action=ENTER
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    exit_reason: Optional[str] = None     # usado si action=EXIT
    meta: Dict[str, Any] = field(default_factory=dict)
