# src/backtester/core/contracts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


ActionType = Literal["ENTER", "EXIT", "UPDATE"]


@dataclass(frozen=True)
class OrderIntent:
    """
    Strategy -> Engine intent.

    Enterprise-grade contract (backwards compatible):

    - action:
        - "ENTER": open a new position (BUY/SELL required)
        - "EXIT": close current position (side can be omitted; engine will close whatever is open)
        - "UPDATE": update protective levels (e.g., trailing SL/TP) for current position

    - side:
        - For ENTER: "BUY" or "SELL"
        - For EXIT/UPDATE: optional; if provided must be BUY/SELL

    - qty:
        - For ENTER: required (>0)
        - For EXIT: if None -> close full position; if set -> partial close (engine may or may not support)
        - For UPDATE: ignored (can be None or current qty)

    - sl_price / tp_price:
        Absolute price levels (NOT pips). Optional for all actions.

    - risk_price:
        Risk unit in price terms used for R-metrics and guardrails.
        If sl_price is present, engine can derive risk_price = abs(entry - sl_price).
        If sl_price is absent (non-bracket strategies), strategy MUST provide risk_price (or engine marks INVALID_R_CONTRACT).

    - tag:
        Free-form label for attribution/audit (recommend structured "k=v|k=v" style).

    - exit_reason:
        Optional standardized reason for EXIT actions (e.g., "SIGNAL", "TIME", "REGIME", "EOF").
    """
    action: ActionType = "ENTER"
    side: Optional[str] = None
    qty: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    risk_price: Optional[float] = None
    tag: str = ""
    exit_reason: str = ""

    def __post_init__(self) -> None:
        a = str(self.action).upper().strip()
        if a not in ("ENTER", "EXIT", "UPDATE"):
            raise ValueError(f"Invalid OrderIntent.action: {self.action!r} (expected ENTER/EXIT/UPDATE)")
        object.__setattr__(self, "action", a)

        # side validation (optional except ENTER requires it)
        if self.side is not None:
            s = str(self.side).upper().strip()
            if s not in ("BUY", "SELL"):
                raise ValueError(f"Invalid OrderIntent.side: {self.side!r} (expected 'BUY' or 'SELL')")
            object.__setattr__(self, "side", s)
        else:
            if a == "ENTER":
                raise ValueError("OrderIntent.side is required for action=ENTER (BUY/SELL)")

        # qty validation depends on action
        if a == "ENTER":
            if self.qty is None:
                raise ValueError("OrderIntent.qty is required for action=ENTER")
            q = float(self.qty)
            if q <= 0:
                raise ValueError(f"Invalid OrderIntent.qty: {self.qty!r} (must be > 0)")
            object.__setattr__(self, "qty", q)
        else:
            # EXIT/UPDATE: qty optional; if provided must be >0
            if self.qty is not None:
                q = float(self.qty)
                if q <= 0:
                    raise ValueError(f"Invalid OrderIntent.qty: {self.qty!r} (must be > 0 if provided)")
                object.__setattr__(self, "qty", q)

        # Validate numeric-ness if provided
        if self.sl_price is not None:
            float(self.sl_price)
        if self.tp_price is not None:
            float(self.tp_price)
        if self.risk_price is not None:
            rp = float(self.risk_price)
            if rp <= 0:
                raise ValueError(f"Invalid OrderIntent.risk_price: {self.risk_price!r} (must be > 0)")
            object.__setattr__(self, "risk_price", rp)

        # Normalize tag/reason
        if self.tag is None:
            object.__setattr__(self, "tag", "")
        if self.exit_reason is None:
            object.__setattr__(self, "exit_reason", "")
