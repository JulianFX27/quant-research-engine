# src/backtester/execution/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from backtester.core.contracts import OrderIntent
from backtester.risk.guardrails import Guardrails


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
      - max_trades_per_day: int | None
      - cooldown_bars: int

    Guardrails v2 (via backtester.risk.guardrails.Guardrails):
      - max_concurrent_positions (engine supports only 1 today; >1 is refused)
      - time_window_enabled + window_start_utc/window_end_utc (UTC)
      - max_holding_bars (force exit at close)

    Research-grade additions:
      - FORCE_EOF: close open position at end of dataset
      - dropped entry audits (pending intent at EOF / no next open)
      - time-in-position metrics
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

        # Guardrails v2
        self.guardrails = Guardrails(risk_cfg)

        # Engine is single-position today. Refuse configs that claim otherwise.
        mcp = int((risk_cfg or {}).get("max_concurrent_positions", 1) or 1)
        if mcp != 1:
            raise ValueError(
                "ENGINE_SINGLE_POSITION_ONLY: max_concurrent_positions must be 1 for SimpleBarEngine.\n"
                f"got max_concurrent_positions={mcp}\n"
                "Fix: set risk.max_concurrent_positions: 1 (or implement multi-position engine).\n"
            )

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
        """Decide whether SL or TP happens first within the bar using intrabar_path."""
        hit_sl = l <= sl <= h
        hit_tp = l <= tp <= h
        if not hit_sl and not hit_tp:
            return None
        if hit_sl and not hit_tp:
            return "SL"
        if hit_tp and not hit_sl:
            return "TP"

        # both hit
        if self.intrabar_path == "OHLC":
            return "TP" if pos_side == "BUY" else "SL"
        else:  # OLHC
            return "SL" if pos_side == "BUY" else "TP"

    @staticmethod
    def _utc_day_key(ts: pd.Timestamp) -> str:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            return t.normalize().isoformat()
        return t.tz_convert("UTC").normalize().isoformat()

    @staticmethod
    def _safe_r_multiple(pnl: float, risk_price: Optional[float], qty: float) -> Optional[float]:
        if risk_price is None:
            return None
        denom = float(risk_price) * abs(float(qty))
        if denom <= 0:
            return None
        return float(pnl) / denom

    @staticmethod
    def _bump(d: Dict[str, int], key: str, n: int = 1) -> None:
        d[key] = int(d.get(key, 0)) + int(n)

    def _close_position(
        self,
        *,
        ts: pd.Timestamp,
        exit_level: float,
        exit_reason: str,
        pos_side: str,
        pos_qty: float,
        entry_price: float,
        entry_time: pd.Timestamp,
        tag: str,
        sl_price: Optional[float],
        tp_price: Optional[float],
        risk_price: Optional[float],
    ) -> Trade:
        exit_fill = self._fill_exit(float(exit_level), pos_side)
        if pos_side == "BUY":
            raw_pnl = (exit_fill - entry_price) * pos_qty
        else:
            raw_pnl = (entry_price - exit_fill) * pos_qty

        pnl = self._apply_costs(raw_pnl, pos_qty)

        return Trade(
            entry_time=entry_time,
            exit_time=ts,
            side=pos_side,
            qty=pos_qty,
            entry_price=float(entry_price),
            exit_price=float(exit_fill),
            pnl=float(pnl),
            tag=tag,
            exit_reason=exit_reason,
            sl_price=sl_price,
            tp_price=tp_price,
            risk_price=risk_price,
        )

    def run(self, df: pd.DataFrame, intents_by_bar: List[List[OrderIntent]]) -> List[Trade]:
        trades: List[Trade] = []

        pos_side: Optional[str] = None
        pos_qty: float = 0.0
        entry_price: float = 0.0
        entry_time: Optional[pd.Timestamp] = None
        entry_index: Optional[int] = None
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

        # Legacy counters (keep for backward compatibility with your orchestrator extraction)
        n_blocked_by_daily_stop: int = 0
        n_blocked_by_max_trades: int = 0
        n_blocked_by_cooldown: int = 0

        # RiskReport v1 counters
        attempted_entries: int = 0
        blocked_total: int = 0
        blocked_by_reason: Dict[str, int] = {}
        blocked_unique_bars: Set[int] = set()

        # Guardrails v2 counters (entry gating reasons)
        blocked_v2_by_reason: Dict[str, int] = {}

        # Forced exits counters
        forced_exits: Dict[str, int] = {}

        # Research-grade: dropped entries (not blocked by gates, but could not be filled)
        dropped_entries_total: int = 0
        dropped_entries_by_reason: Dict[str, int] = {}

        # Research-grade: time in position
        time_in_position_bars: int = 0

        # For EOF close metadata
        last_bar_ts: Optional[pd.Timestamp] = None
        last_bar_close: Optional[float] = None

        for i, (ts, row) in enumerate(df.iterrows()):
            last_bar_ts = ts
            last_bar_close = float(row["close"])

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
                entry_index = i

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

            # Track time in position
            if pos_side is not None:
                time_in_position_bars += 1

            # Manage existing position
            if pos_side is not None and entry_time is not None and entry_index is not None:
                exit_reason: Optional[str] = None
                exit_level: Optional[float] = None

                # SL/TP intrabar
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

                # Tie-break fallback
                if exit_reason is None and sl is not None and tp is not None:
                    hit_sl = l <= float(sl) <= h
                    hit_tp = l <= float(tp) <= h
                    if hit_sl and hit_tp:
                        if self.tie_break == "tp_first":
                            exit_reason, exit_level = "TP", float(tp)
                        else:
                            exit_reason, exit_level = "SL", float(sl)

                # Force-exit by max holding bars (close)
                if exit_reason is None:
                    holding_bars = i - int(entry_index)
                    force, force_reason = self.guardrails.should_force_exit(holding_bars=holding_bars)
                    if force:
                        exit_reason = "FORCE_MAX_HOLD"
                        exit_level = float(c)
                        if force_reason:
                            self._bump(forced_exits, str(force_reason))
                        else:
                            self._bump(forced_exits, "MAX_HOLDING_BARS")

                if exit_reason is not None and exit_level is not None:
                    trade = self._close_position(
                        ts=ts,
                        exit_level=float(exit_level),
                        exit_reason=str(exit_reason),
                        pos_side=pos_side,
                        pos_qty=pos_qty,
                        entry_price=float(entry_price),
                        entry_time=entry_time,
                        tag=tag,
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
                    entry_index = None
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
                    attempted_entries += 1

                    blocked = False
                    blocked_reason: Optional[str] = None

                    # Daily stop gate
                    if stopped_today:
                        n_blocked_by_daily_stop += 1
                        blocked = True
                        blocked_reason = "DAILY_STOP"

                    # Max trades/day gate
                    if (not blocked) and (self.max_trades_per_day is not None) and (entries_today >= int(self.max_trades_per_day)):
                        n_blocked_by_max_trades += 1
                        blocked = True
                        blocked_reason = "MAX_TRADES_PER_DAY"

                    # Cooldown gate
                    if (not blocked) and (i < cooldown_until_index):
                        n_blocked_by_cooldown += 1
                        blocked = True
                        blocked_reason = "COOLDOWN"

                    # Guardrails v2 gate (UTC window + max concurrent positions)
                    if not blocked:
                        active_positions = 0  # engine is flat here by construction
                        if self.fill_mode == "next_open":
                            if i < len(df) - 1:
                                entry_ts = df.index[i + 1]
                                ok, reason = self.guardrails.allow_entry(
                                    bar_time_utc=pd.Timestamp(entry_ts),
                                    active_positions=active_positions,
                                )
                                if not ok:
                                    blocked = True
                                    blocked_reason = f"V2_{str(reason) if reason else 'GUARDRAILS'}"
                                    self._bump(blocked_v2_by_reason, str(reason) if reason else "UNKNOWN")
                            else:
                                # Cannot schedule next_open on last bar => treat as dropped (not a "risk gate")
                                dropped_entries_total += 1
                                self._bump(dropped_entries_by_reason, "NO_NEXT_OPEN")
                                continue
                        else:
                            ok, reason = self.guardrails.allow_entry(
                                bar_time_utc=pd.Timestamp(ts),
                                active_positions=active_positions,
                            )
                            if not ok:
                                blocked = True
                                blocked_reason = f"V2_{str(reason) if reason else 'GUARDRAILS'}"
                                self._bump(blocked_v2_by_reason, str(reason) if reason else "UNKNOWN")

                    if blocked:
                        blocked_total += 1
                        blocked_unique_bars.add(i)
                        if blocked_reason:
                            self._bump(blocked_by_reason, blocked_reason)
                        else:
                            self._bump(blocked_by_reason, "UNKNOWN")
                        continue

                    # ---- Proceed with entry scheduling/fill ----
                    if self.fill_mode == "next_open":
                        if i < len(df) - 1:
                            pending_intent = intent
                            pending_tag = intent.tag
                        else:
                            # Same as above: last bar cannot fill next open => dropped
                            dropped_entries_total += 1
                            self._bump(dropped_entries_by_reason, "NO_NEXT_OPEN")
                    else:
                        pos_side = intent.side
                        pos_qty = float(intent.qty)
                        entry_price = self._fill_entry(c, pos_side)
                        entry_time = ts
                        entry_index = i

                        sl = float(intent.sl_price) if intent.sl_price is not None else None
                        tp = float(intent.tp_price) if intent.tp_price is not None else None
                        pos_sl_price = sl
                        pos_tp_price = tp
                        pos_risk_price = abs(float(entry_price) - float(sl)) if (sl is not None) else None

                        tag = intent.tag
                        entries_today += 1

        # If we end with a pending intent, it was never filled
        if pending_intent is not None:
            dropped_entries_total += 1
            self._bump(dropped_entries_by_reason, "PENDING_INTENT_AT_EOF")

        # If we end with an open position, force-close at last close (research-grade default)
        if pos_side is not None and entry_time is not None and (last_bar_ts is not None) and (last_bar_close is not None):
            trade = self._close_position(
                ts=last_bar_ts,
                exit_level=float(last_bar_close),
                exit_reason="FORCE_EOF",
                pos_side=pos_side,
                pos_qty=pos_qty,
                entry_price=float(entry_price),
                entry_time=entry_time,
                tag=tag,
                sl_price=pos_sl_price,
                tp_price=pos_tp_price,
                risk_price=pos_risk_price,
            )
            trades.append(trade)
            self._bump(forced_exits, "EOF")

            # Update realized R today too (if applicable)
            r_mult = self._safe_r_multiple(trade.pnl, trade.risk_price, trade.qty)
            if r_mult is not None:
                realized_R_today += float(r_mult)

            # Reset position to keep internal consistency
            pos_side = None
            pending_intent = None

        # Guardrails report from component (existing contract)
        gr_rep = self.guardrails.report()
        if not isinstance(gr_rep, dict):
            gr_rep = {"raw": gr_rep}

        # Augment guardrails report (non-breaking)
        gr_rep.setdefault("entry_gate", {})
        gr_rep["entry_gate"] = {
            "attempted_entries": attempted_entries,
            "blocked_total": blocked_total,
            "blocked_unique_bars": len(blocked_unique_bars),
            "blocked_v2_by_reason": dict(blocked_v2_by_reason),
        }

        bars_total = int(len(df) or 0)
        time_in_position_rate = (float(time_in_position_bars) / float(bars_total)) if bars_total > 0 else 0.0

        # Expose risk/guardrails report for orchestrator persistence
        self.last_risk_report = {
            "risk_report_version": "v1",
            "risk_cfg": {
                "max_daily_loss_R": self.max_daily_loss_R,
                "max_trades_per_day": self.max_trades_per_day,
                "cooldown_bars": self.cooldown_bars,
            },
            "final_day_key_utc": current_day,
            "final_entries_today": entries_today,
            "final_realized_R_today": realized_R_today,
            "final_stopped_today": stopped_today,

            # Backwards compatible (legacy fields your orchestrator already expects)
            "blocked": {
                "by_daily_stop": n_blocked_by_daily_stop,
                "by_max_trades_per_day": n_blocked_by_max_trades,
                "by_cooldown": n_blocked_by_cooldown,
            },

            # New v1 schema counters (research-grade)
            "entry_gate": {
                "attempted_entries": attempted_entries,
                "blocked_total": blocked_total,
                "blocked_unique_bars": len(blocked_unique_bars),
                "blocked_by_reason": dict(blocked_by_reason),
            },

            # Guardrails v2 full report (plus entry_gate augment)
            "guardrails": gr_rep,

            # Forced exits audit (engine-level)
            "forced_exits": dict(forced_exits),

            # NEW: dropped entries (not gated, but not fillable)
            "entry_fill_dropped": {
                "dropped_total": int(dropped_entries_total),
                "dropped_by_reason": dict(dropped_entries_by_reason),
            },

            # NEW: engine utilization metrics
            "engine": {
                "bars_total": int(bars_total),
                "time_in_position_bars": int(time_in_position_bars),
                "time_in_position_rate": float(time_in_position_rate),
            },
        }

        return trades
