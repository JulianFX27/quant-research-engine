from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from backtester.core.contracts import OrderIntent


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    tag: str
    exit_reason: str


class SimpleBarEngine:
    """Minimal bar-by-bar engine (v1 hardened intrabar).

    Features:
      - One position at a time
      - Entry fill_mode:
          - "close": fill at close of signal bar
          - "next_open": fill at open of next bar
      - Exit intrabar:
          - If TP and SL both reachable in bar, decide via intrabar_path:
              - "OHLC": open->high->low->close
              - "OLHC": open->low->high->close
          - Fallback tie-break via intrabar_tie
      - Costs:
          - commission (abs per-unit per trade)
          - spread_price (full spread in price units)
          - slippage_price (adverse per fill)
    """

    def __init__(self, *, costs: Dict[str, Any], exec_cfg: Dict[str, Any]):
        self.costs = costs
        self.exec_cfg = exec_cfg
        self.tie_break = exec_cfg.get("intrabar_tie", "sl_first")  # sl_first | tp_first
        self.fill_mode = exec_cfg.get("fill_mode", "close")        # close | next_open

        raw_path = str(exec_cfg.get("intrabar_path", "OHLC")).replace(" ", "").upper()
        self.intrabar_path = raw_path if raw_path in ("OHLC", "OLHC") else "OHLC"

    def _apply_costs(self, pnl: float, qty: float) -> float:
        commission = float(self.costs.get("commission", 0.0))
        return pnl - commission * abs(qty)

    def _spread_half(self) -> float:
        spread = float(self.costs.get("spread_price", 0.0))
        return 0.5 * spread

    def _slip(self) -> float:
        return float(self.costs.get("slippage_price", 0.0))

    def _fill_entry(self, mid: float, side: str) -> float:
        half = self._spread_half()
        slip = self._slip()
        if side == "BUY":
            return mid + half + slip
        return mid - half - slip

    def _fill_exit(self, level_price: float, pos_side: str) -> float:
        half = self._spread_half()
        slip = self._slip()
        if pos_side == "BUY":
            return level_price - half - slip
        return level_price + half + slip

    def _decide_exit_reason(self, o: float, h: float, l: float, c: float, pos_side: str, sl: float, tp: float) -> Optional[str]:
        """
        Decide whether SL or TP happens first within the bar using intrabar_path.

        Assumption: if a level is between low/high, it is reachable.
        Path models which extreme is visited first.
        """
        # Reachability
        hit_sl = l <= sl <= h
        hit_tp = l <= tp <= h
        if not hit_sl and not hit_tp:
            return None
        if hit_sl and not hit_tp:
            return "SL"
        if hit_tp and not hit_sl:
            return "TP"

        # Both reachable: use path + position direction
        # For a long:
        #  - TP is above, SL is below typically.
        #  - If path goes to High first (OHLC), TP occurs before SL.
        #  - If path goes to Low first (OLHC), SL occurs before TP.
        #
        # For a short, TP usually below and SL above; the reasoning mirrors:
        #  - OHLC (high first) tends to hit SL first (if SL above), then later TP (low)
        #  - OLHC (low first) tends to hit TP first, then later SL
        if self.intrabar_path == "OHLC":
            if pos_side == "BUY":
                return "TP"  # high first
            return "SL"      # high first hits short SL (above) before low hits TP (below)
        else:  # OLHC
            if pos_side == "BUY":
                return "SL"  # low first
            return "TP"      # low first hits short TP (below) before high hits SL (above)

    def run(self, df: pd.DataFrame, intents_by_bar: List[List[OrderIntent]]) -> List[Trade]:
        trades: List[Trade] = []

        pos_side: Optional[str] = None
        pos_qty: float = 0.0
        entry_price: float = 0.0
        entry_time: Optional[pd.Timestamp] = None
        sl: Optional[float] = None
        tp: Optional[float] = None
        tag: str = ""

        pending_intent: Optional[OrderIntent] = None
        pending_tag: str = ""

        for i, (ts, row) in enumerate(df.iterrows()):
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            # Fill pending entry at this bar open
            if pos_side is None and pending_intent is not None:
                pos_side = pending_intent.side
                pos_qty = float(pending_intent.qty)
                entry_price = self._fill_entry(o, pos_side)
                entry_time = ts
                sl = float(pending_intent.sl_price) if pending_intent.sl_price is not None else None
                tp = float(pending_intent.tp_price) if pending_intent.tp_price is not None else None
                tag = pending_tag
                pending_intent = None
                pending_tag = ""

            # Manage existing position
            if pos_side is not None and entry_time is not None:
                exit_reason: Optional[str] = None
                exit_level: Optional[float] = None

                if sl is not None or tp is not None:
                    # Determine exit reason
                    if sl is not None and tp is not None:
                        reason = self._decide_exit_reason(o, h, l, c, pos_side, float(sl), float(tp))
                        if reason == "SL":
                            exit_reason, exit_level = "SL", float(sl)
                        elif reason == "TP":
                            exit_reason, exit_level = "TP", float(tp)
                    elif sl is not None:
                        if l <= float(sl) <= h:
                            exit_reason, exit_level = "SL", float(sl)
                    elif tp is not None:
                        if l <= float(tp) <= h:
                            exit_reason, exit_level = "TP", float(tp)

                # Fallback tie-break (only if both hit and path logic couldn't decide, rare)
                if exit_reason is None and sl is not None and tp is not None:
                    hit_sl = l <= float(sl) <= h
                    hit_tp = l <= float(tp) <= h
                    if hit_sl and hit_tp:
                        if self.tie_break == "tp_first":
                            exit_reason, exit_level = "TP", float(tp)
                        else:
                            exit_reason, exit_level = "SL", float(sl)

                if exit_reason is not None and exit_level is not None:
                    exit_fill = self._fill_exit(float(exit_level), pos_side)

                    if pos_side == "BUY":
                        raw_pnl = (exit_fill - entry_price) * pos_qty
                    else:
                        raw_pnl = (entry_price - exit_fill) * pos_qty

                    pnl = self._apply_costs(raw_pnl, pos_qty)

                    trades.append(
                        Trade(
                            entry_time=entry_time,
                            exit_time=ts,
                            side=pos_side,
                            qty=pos_qty,
                            entry_price=float(entry_price),
                            exit_price=float(exit_fill),
                            pnl=float(pnl),
                            tag=tag,
                            exit_reason=exit_reason,
                        )
                    )

                    pos_side = None
                    pos_qty = 0.0
                    entry_price = 0.0
                    entry_time = None
                    sl = None
                    tp = None
                    tag = ""

            # Schedule / fill entries
            if pos_side is None and pending_intent is None:
                intents = intents_by_bar[i] if i < len(intents_by_bar) else []
                if intents:
                    intent = intents[0]

                    if self.fill_mode == "next_open":
                        if i < len(df) - 1:
                            pending_intent = intent
                            pending_tag = intent.tag
                    else:
                        pos_side = intent.side
                        pos_qty = float(intent.qty)
                        entry_price = self._fill_entry(c, pos_side)
                        entry_time = ts
                        sl = float(intent.sl_price) if intent.sl_price is not None else None
                        tp = float(intent.tp_price) if intent.tp_price is not None else None
                        tag = intent.tag

        return trades
