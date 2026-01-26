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


    # Persist SL/TP used for this trade (if available) so research metrics can compute R
    sl_price: float | None = None
    tp_price: float | None = None
    # Formal R contract: risk measured in price units
    risk_price: float | None = None


class SimpleBarEngine:
    """Minimal bar-by-bar engine (v1 hardened intrabar) + guardrails.

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

    Guardrails (risk_cfg):
      - max_daily_loss_R: float | None
          Stop taking new entries for the rest of the UTC day once realized_R <= -max_daily_loss_R.
      - max_trades_per_day: int | None
          Limit number of entries per UTC day.
      - cooldown_bars: int
          After a trade exits, block new entries until i >= cooldown_until_index.
    """

    def __init__(self, *, costs: Dict[str, Any], exec_cfg: Dict[str, Any], risk_cfg: Optional[Dict[str, Any]] = None):
        self.costs = costs
        self.exec_cfg = exec_cfg
        self.tie_break = exec_cfg.get("intrabar_tie", "sl_first")  # sl_first | tp_first
        self.fill_mode = exec_cfg.get("fill_mode", "close")        # close | next_open

        raw_path = str(exec_cfg.get("intrabar_path", "OHLC")).replace(" ", "").upper()
        self.intrabar_path = raw_path if raw_path in ("OHLC", "OLHC") else "OHLC"

        # Risk / guardrails config
        risk_cfg = risk_cfg or {}
        self.max_daily_loss_R: Optional[float] = (
            float(risk_cfg["max_daily_loss_R"]) if risk_cfg.get("max_daily_loss_R") is not None else None
        )
        self.max_trades_per_day: Optional[int] = (
            int(risk_cfg["max_trades_per_day"]) if risk_cfg.get("max_trades_per_day") is not None else None
        )
        self.cooldown_bars: int = int(risk_cfg.get("cooldown_bars", 0) or 0)
        if self.cooldown_bars < 0:
            self.cooldown_bars = 0

        # Exposed after run() for persistence/audit (orchestrator can copy into metrics/manifest)
        self.last_risk_report: Dict[str, Any] = {}

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

    def _decide_exit_reason(
        self, o: float, h: float, l: float, c: float, pos_side: str, sl: float, tp: float
    ) -> Optional[str]:
        """
        Decide whether SL or TP happens first within the bar using intrabar_path.

        Assumption: if a level is between low/high, it is reachable.
        Path models which extreme is visited first.
        """
        hit_sl = l <= sl <= h
        hit_tp = l <= tp <= h
        if not hit_sl and not hit_tp:
            return None
        if hit_sl and not hit_tp:
            return "SL"
        if hit_tp and not hit_sl:
            return "TP"

        if self.intrabar_path == "OHLC":
            if pos_side == "BUY":
                return "TP"
            return "SL"
        else:  # OLHC
            if pos_side == "BUY":
                return "SL"
            return "TP"

    @staticmethod
    def _utc_day_key(ts: pd.Timestamp) -> str:
        # ts is already UTC in your loader contract; normalize to midnight UTC
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            # contract says UTC; keep as-is
            return t.normalize().isoformat()
        return t.tz_convert("UTC").normalize().isoformat()

    @staticmethod
    def _safe_r_multiple(pnl: float, risk_price: Optional[float], qty: float) -> Optional[float]:
        """
        Compute realized R multiple using formal risk contract:
          R = pnl / (risk_price * abs(qty))
        where:
          pnl is in (price * qty) units after costs,
          risk_price is abs(entry_fill - sl_level) in price units,
          qty is position size in "units".
        """
        if risk_price is None:
            return None
        denom = float(risk_price) * abs(float(qty))
        if denom <= 0:
            return None
        return float(pnl) / denom

    def run(self, df: pd.DataFrame, intents_by_bar: List[List[OrderIntent]]) -> List[Trade]:
        trades: List[Trade] = []

        pos_side: Optional[str] = None
        pos_qty: float = 0.0
        entry_price: float = 0.0
        entry_time: Optional[pd.Timestamp] = None
        sl: Optional[float] = None
        tp: Optional[float] = None
        tag: str = ""

        pos_sl_price: Optional[float] = None
        pos_tp_price: Optional[float] = None
        pos_risk_price: Optional[float] = None

        pending_intent: Optional[OrderIntent] = None
        pending_tag: str = ""

        # Guardrails state (UTC-day scoped)
        current_day: Optional[str] = None
        entries_today: int = 0
        realized_R_today: float = 0.0
        stopped_today: bool = False

        # Cooldown state (bar-index scoped)
        cooldown_until_index: int = 0

        # Counters for audit
        n_blocked_by_daily_stop: int = 0
        n_blocked_by_max_trades: int = 0
        n_blocked_by_cooldown: int = 0

        for i, (ts, row) in enumerate(df.iterrows()):
            day_key = self._utc_day_key(ts)

            # Day rollover resets daily guardrails
            if current_day is None:
                current_day = day_key
            elif day_key != current_day:
                current_day = day_key
                entries_today = 0
                realized_R_today = 0.0
                stopped_today = False

            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            # Fill pending entry at this bar open
            if pos_side is None and pending_intent is not None:
                pos_side = pending_intent.side
                pos_qty = float(pending_intent.qty)
                entry_price = self._fill_entry(o, pos_side)
                entry_time = ts

                sl = float(pending_intent.sl_price) if pending_intent.sl_price is not None else None
                tp = float(pending_intent.tp_price) if pending_intent.tp_price is not None else None
                pos_sl_price = sl
                pos_tp_price = tp

                # Formal risk contract: abs(filled_entry_price - sl_level_price)
                pos_risk_price = abs(float(entry_price) - float(sl)) if (sl is not None) else None

                tag = pending_tag
                pending_intent = None
                pending_tag = ""

                # Count entry (for max trades/day)
                entries_today += 1

            # Manage existing position
            if pos_side is not None and entry_time is not None:
                exit_reason: Optional[str] = None
                exit_level: Optional[float] = None

                if sl is not None or tp is not None:
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

                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=ts,
                        side=pos_side,
                        qty=pos_qty,
                        entry_price=float(entry_price),
                        exit_price=float(exit_fill),
                        pnl=float(pnl),
                        tag=tag,
                        exit_reason=exit_reason,
                        sl_price=pos_sl_price,
                        tp_price=pos_tp_price,
                        risk_price=pos_risk_price,
                    )
                    trades.append(trade)

                    # Update realized R today (daily stop uses realized R of exits)
                    r_mult = self._safe_r_multiple(trade.pnl, trade.risk_price, trade.qty)
                    if r_mult is not None:
                        realized_R_today += float(r_mult)

                    # Apply daily stop if configured
                    if (self.max_daily_loss_R is not None) and (realized_R_today <= -abs(float(self.max_daily_loss_R))):
                        stopped_today = True

                    # Start cooldown window after exit
                    if self.cooldown_bars > 0:
                        cooldown_until_index = max(cooldown_until_index, i + self.cooldown_bars + 1)

                    # Reset position
                    pos_side = None
                    pos_qty = 0.0
                    entry_price = 0.0
                    entry_time = None
                    sl = None
                    tp = None
                    tag = ""
                    pos_sl_price = None
                    pos_tp_price = None
                    pos_risk_price = None

            # Schedule / fill entries (only if flat and no pending)
            if pos_side is None and pending_intent is None:
                intents = intents_by_bar[i] if i < len(intents_by_bar) else []
                if intents:
                    intent = intents[0]

                    # ---- Guardrails gate (entries only) ----
                    blocked = False

                    # Daily stop gate
                    if stopped_today:
                        n_blocked_by_daily_stop += 1
                        blocked = True

                    # Max trades/day gate (count entries; note pending fill also increments when filled)
                    if (not blocked) and (self.max_trades_per_day is not None) and (entries_today >= int(self.max_trades_per_day)):
                        n_blocked_by_max_trades += 1
                        blocked = True

                    # Cooldown gate
                    if (not blocked) and (i < cooldown_until_index):
                        n_blocked_by_cooldown += 1
                        blocked = True

                    if blocked:
                        continue

                    # ---- Proceed with entry scheduling/fill ----
                    if self.fill_mode == "next_open":
                        if i < len(df) - 1:
                            pending_intent = intent
                            pending_tag = intent.tag
                        # If last bar, ignore (cannot fill next open)
                    else:
                        pos_side = intent.side
                        pos_qty = float(intent.qty)
                        entry_price = self._fill_entry(c, pos_side)
                        entry_time = ts

                        sl = float(intent.sl_price) if intent.sl_price is not None else None
                        tp = float(intent.tp_price) if intent.tp_price is not None else None
                        pos_sl_price = sl
                        pos_tp_price = tp

                        pos_risk_price = abs(float(entry_price) - float(sl)) if (sl is not None) else None

                        tag = intent.tag

                        # Count entry immediately (close fill)
                        entries_today += 1

        # Expose risk/guardrails report for orchestrator persistence
        self.last_risk_report = {
            "risk_cfg": {
                "max_daily_loss_R": self.max_daily_loss_R,
                "max_trades_per_day": self.max_trades_per_day,
                "cooldown_bars": self.cooldown_bars,
            },
            "final_day_key_utc": current_day,
            "final_entries_today": entries_today,
            "final_realized_R_today": realized_R_today,
            "final_stopped_today": stopped_today,
            "blocked": {
                "by_daily_stop": n_blocked_by_daily_stop,
                "by_max_trades_per_day": n_blocked_by_max_trades,
                "by_cooldown": n_blocked_by_cooldown,
            },
        }

        return trades
