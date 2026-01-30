# src/backtester/core/contracts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OrderIntent:
    """
    Strategy -> Engine intent.

    Contract:
      - side: "BUY" or "SELL"
      - qty: position size in units
      - sl_price / tp_price: absolute price levels (NOT pips)
      - tag: free-form label for attribution/audit
    """
    side: str
    qty: float
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    tag: str = ""

    def __post_init__(self) -> None:
        s = str(self.side).upper().strip()
        if s not in ("BUY", "SELL"):
            raise ValueError(f"Invalid OrderIntent.side: {self.side!r} (expected 'BUY' or 'SELL')")
        object.__setattr__(self, "side", s)

        q = float(self.qty)
        if q <= 0:
            raise ValueError(f"Invalid OrderIntent.qty: {self.qty!r} (must be > 0)")

        # Validate numeric-ness if provided
        if self.sl_price is not None:
            float(self.sl_price)
        if self.tp_price is not None:
            float(self.tp_price)
