from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
from datetime import datetime

from src.runner.interfaces import Bar, OrderIntent, Direction


TieBreak = Literal["tp_first", "sl_first"]
IntrabarPath = Literal["OHLC", "OLHC"]


@dataclass
class Fill:
    filled_entry_price: float
    filled_time_utc: datetime


@dataclass
class Trade:
    trade_id: str
    intent_id: str
    direction: Direction

    entry_time_utc: datetime
    entry_price: float

    sl_price: float
    tp_price: float

    exit_time_utc: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "TP" | "SL" | "FORCE_EOF" | etc.

    risk_base_pct: float = 0.0
    risk_multiplier: float = 1.0
    risk_effective_pct: float = 0.0

    risk_price: float = 0.0  # abs(entry - sl)
    pnl: float = 0.0         # equity-space pnl in % terms: e.g., +0.0023 = +0.23%
    R: float = 0.0           # pnl / (risk_effective_pct)

    meta: Dict[str, Any] = field(default_factory=dict)


class PaperExecutionError(Exception):
    pass


class PaperEngine:
    """
    Single-position paper execution engine.

    Design:
    - Entry is filled at NEXT bar open (next_open).
    - Intrabar exit is evaluated on each bar while position open:
        path OHLC or OLHC
        tie-break: tp_first or sl_first when both hit within same bar.
    - Risk is expressed as percentage of equity (r_eff).
      PnL in equity space: pnl = r_eff * R_multiple
    """

    def __init__(
        self,
        *,
        intrabar_path: IntrabarPath = "OHLC",
        tie_break: TieBreak = "tp_first",
        allow_same_bar_exit: bool = True,
    ):
        self.intrabar_path = intrabar_path
        self.tie_break = tie_break
        self.allow_same_bar_exit = allow_same_bar_exit

        self._pending_intent: Optional[OrderIntent] = None
        self._pending_risk: Optional[Dict[str, float]] = None  # base, mult, eff
        self._open_trade: Optional[Trade] = None

        self._trade_seq = 0

    # ----------------------------
    # State
    # ----------------------------
    def has_open_position(self) -> bool:
        return self._open_trade is not None

    def has_pending_intent(self) -> bool:
        return self._pending_intent is not None

    def get_open_trade(self) -> Optional[Trade]:
        return self._open_trade

    # ----------------------------
    # Intent submission
    # ----------------------------
    def submit_intent(
        self,
        intent: OrderIntent,
        *,
        risk_base_pct: float,
        risk_multiplier: float,
        risk_effective_pct: float,
    ) -> None:
        """
        Stage an intent to be filled at the next bar open.

        Caller must ensure:
        - no open position
        - no pending intent
        - risk_effective_pct > 0
        """
        if self._open_trade is not None:
            raise PaperExecutionError("Cannot submit intent: position already open.")
        if self._pending_intent is not None:
            raise PaperExecutionError("Cannot submit intent: another intent is already pending.")
        if risk_effective_pct <= 0:
            raise PaperExecutionError("risk_effective_pct must be > 0 to submit intent.")

        self._pending_intent = intent
        self._pending_risk = {
            "risk_base_pct": float(risk_base_pct),
            "risk_multiplier": float(risk_multiplier),
            "risk_effective_pct": float(risk_effective_pct),
        }

    # ----------------------------
    # Engine step (bar-by-bar)
    # ----------------------------
    def on_bar(self, bar: Bar) -> Optional[Trade]:
        """
        Advance engine by one bar.

        Returns:
        - Trade if a trade CLOSED on this bar
        - None otherwise

        Behavior order:
        1) If pending intent and no open position -> fill at bar open.
        2) If open position -> check intrabar for SL/TP.
        """
        closed_trade: Optional[Trade] = None

        # 1) Fill pending intent at bar open
        if self._pending_intent is not None and self._open_trade is None:
            self._open_trade = self._fill_pending_intent(bar)

        # 2) Evaluate exit for open position
        if self._open_trade is not None:
            # Optionally disallow exiting on the same bar as entry
            if not self.allow_same_bar_exit and self._open_trade.entry_time_utc == bar.ts_utc:
                return None

            hit = self._check_exit_intrabar(self._open_trade, bar)
            if hit is not None:
                price, reason = hit
                closed_trade = self._close_trade(self._open_trade, bar.ts_utc, price, reason)
                self._open_trade = None  # clear after close

        return closed_trade

    # ----------------------------
    # Force close (EOF, etc.)
    # ----------------------------
    def force_close(self, ts_utc: datetime, price: float, reason: str = "FORCE_EOF") -> Optional[Trade]:
        if self._open_trade is None:
            return None
        tr = self._close_trade(self._open_trade, ts_utc, price, reason)
        self._open_trade = None
        return tr

    # ----------------------------
    # Internals
    # ----------------------------
    def _fill_pending_intent(self, bar: Bar) -> Trade:
        intent = self._pending_intent
        risk = self._pending_risk
        assert intent is not None and risk is not None

        self._trade_seq += 1
        trade_id = f"T{self._trade_seq:06d}"

        entry_price = float(bar.open)  # next_open fill
        entry_time = bar.ts_utc

        # Risk price contract
        risk_price = abs(entry_price - float(intent.sl_price))
        if risk_price <= 0:
            raise PaperExecutionError("Invalid risk_price (entry == SL).")

        tr = Trade(
            trade_id=trade_id,
            intent_id=intent.intent_id,
            direction=intent.direction,
            entry_time_utc=entry_time,
            entry_price=entry_price,
            sl_price=float(intent.sl_price),
            tp_price=float(intent.tp_price),
            risk_base_pct=float(risk["risk_base_pct"]),
            risk_multiplier=float(risk["risk_multiplier"]),
            risk_effective_pct=float(risk["risk_effective_pct"]),
            risk_price=float(risk_price),
            meta=dict(intent.meta),
        )

        # clear pending
        self._pending_intent = None
        self._pending_risk = None

        return tr

    def _check_exit_intrabar(self, tr: Trade, bar: Bar) -> Optional[tuple[float, str]]:
        """
        Return (exit_price, reason) if TP or SL is hit within this bar.
        """
        o, h, l, c = float(bar.open), float(bar.high), float(bar.low), float(bar.close)
        tp, sl = float(tr.tp_price), float(tr.sl_price)

        # Determine which levels are "inside" the bar range
        tp_hit = (l <= tp <= h)
        sl_hit = (l <= sl <= h)

        if not tp_hit and not sl_hit:
            return None

        # If only one is hit -> easy
        if tp_hit and not sl_hit:
            return tp, "TP"
        if sl_hit and not tp_hit:
            return sl, "SL"

        # Both hit within the same bar -> need intrabar model
        return self._resolve_both_hit(tr, bar)

    def _resolve_both_hit(self, tr: Trade, bar: Bar) -> tuple[float, str]:
        """
        Both TP and SL are within the bar range.
        Use intrabar_path model:
          OHLC: price visits Open -> High -> Low -> Close
          OLHC: price visits Open -> Low -> High -> Close
        For LONG/SHORT, determine which is reached first.
        If ambiguous (e.g., both in same leg), use tie_break.
        """
        o, h, l, c = float(bar.open), float(bar.high), float(bar.low), float(bar.close)
        tp, sl = float(tr.tp_price), float(tr.sl_price)

        if self.intrabar_path == "OHLC":
            path = [o, h, l, c]
        elif self.intrabar_path == "OLHC":
            path = [o, l, h, c]
        else:
            raise PaperExecutionError(f"Unknown intrabar_path: {self.intrabar_path}")

        # Evaluate each segment in order
        for a, b in zip(path[:-1], path[1:]):
            seg_low = min(a, b)
            seg_high = max(a, b)

            tp_in = (seg_low <= tp <= seg_high)
            sl_in = (seg_low <= sl <= seg_high)

            if not tp_in and not sl_in:
                continue

            if tp_in and not sl_in:
                return tp, "TP"
            if sl_in and not tp_in:
                return sl, "SL"

            # both within same segment => ambiguous, apply tie-break
            if self.tie_break == "tp_first":
                return tp, "TP"
            return sl, "SL"

        # Fallback: if somehow not resolved, apply tie-break
        if self.tie_break == "tp_first":
            return tp, "TP"
        return sl, "SL"

    def _close_trade(self, tr: Trade, ts_utc: datetime, exit_price: float, reason: str) -> Trade:
        tr.exit_time_utc = ts_utc
        tr.exit_price = float(exit_price)
        tr.exit_reason = reason

        # Compute R multiple based on direction
        # LONG: profit = exit - entry
        # SHORT: profit = entry - exit
        if tr.direction == "LONG":
            raw = tr.exit_price - tr.entry_price
            risk_unit = tr.entry_price - tr.sl_price
        else:
            raw = tr.entry_price - tr.exit_price
            risk_unit = tr.sl_price - tr.entry_price

        if risk_unit <= 0:
            # should not happen if SL is on correct side, but guard
            raise PaperExecutionError("Invalid SL placement relative to entry (risk_unit <= 0).")

        R_multiple = raw / risk_unit

        # Equity PnL in pct terms = risk_effective_pct * R_multiple
        pnl = tr.risk_effective_pct * R_multiple

        tr.R = float(R_multiple)
        tr.pnl = float(pnl)

        return tr
