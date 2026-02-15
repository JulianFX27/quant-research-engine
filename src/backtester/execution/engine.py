# src/backtester/execution/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from backtester.core.contracts import OrderIntent
from backtester.risk.guardrails import Guardrails


def _extract_max_holding_bars(risk_cfg: dict) -> int:
    """
    Read max holding bars from the supported config paths.

    Priority:
      1) risk_cfg['guardrails_cfg']['max_holding_bars']
      2) risk_cfg['guardrails']['max_holding_bars']
      3) risk_cfg['max_holding_bars']  (legacy / convenience)
    """
    if not isinstance(risk_cfg, dict):
        return 0

    for path in (
        ("guardrails_cfg", "max_holding_bars"),
        ("guardrails", "max_holding_bars"),
        ("max_holding_bars",),
    ):
        cur: Any = risk_cfg
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and cur is not None:
            try:
                v = int(cur)
                return max(0, v)
            except Exception:
                return 0

    return 0


def _infer_bar_minutes_from_index(index: pd.Index, max_samples: int = 2000) -> Optional[float]:
    """
    Robust bar-size inference from datetime-like index.
    Uses median of positive deltas (in minutes) to ignore gaps.
    """
    try:
        if index is None or len(index) < 2:
            return None
        # Use only first N deltas for speed
        idx = pd.DatetimeIndex(index[: min(len(index), max_samples)])
        deltas = idx.to_series().diff().dropna()
        if deltas.empty:
            return None
        mins = (deltas.dt.total_seconds() / 60.0)
        mins = mins[mins > 0]
        if mins.empty:
            return None
        m = float(mins.median())
        if not pd.notna(m) or m <= 0:
            return None
        return m
    except Exception:
        return None


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

    # Research-grade duration field (derived from bars; avoids gaps artifacts)
    hold_minutes: float | None = None

    # Bar indices (critical for research: duration in TRUE bars, not time gaps)
    entry_idx: int | None = None
    exit_idx: int | None = None

    # Persist SL/TP used for this trade (if available) so research metrics can compute R
    sl_price: float | None = None
    tp_price: float | None = None

    # Formal R contract: risk measured in price units (can be proxy)
    risk_price: float | None = None


class SimpleBarEngine:
    """Minimal bar-by-bar engine (v1 hardened intrabar) + guardrails.

    Key semantics:
      - Single position only
      - fill_mode: close | next_open
      - intrabar_path: OHLC | OLHC
      - intrabar_tie: sl_first | tp_first

    ARCH A/B:
      - force_exit_on_eof:
          True  => close at EOF (FORCE_EOF trade)
          False => do NOT close; report open_position_at_eof (A_PURE_SLTP)

    max_holding_bars semantics:
      - 0 / None => DISABLED
      - >0       => ENABLED (force exit at close when holding_bars >= max_holding_bars)

    eof_buffer_bars semantics:
      - 0 => DISABLED
      - >0 => block new entries if remaining_bars_to_eof <= eof_buffer_bars

    Time-stop strategies (no SL/TP in price):
      - Set risk.allow_missing_sl: true
      - Provide risk.risk_proxy_price > 0 (in price units) to enable R metrics

    NEW (regime-aware exits / gating):
      - risk.regime_exit_enabled: bool
      - risk.regime_exit_col: str (default "state_regime")
      - risk.regime_exit_allowed: list[str] (e.g. ["RANGE"])
      - risk.regime_gate_entries_enabled: bool
      - risk.regime_gate_entries_col: str (default "state_regime")
      - risk.regime_gate_entries_allowed: list[str]

    NEW (strategy-driven exits):
      - Strategy can emit OrderIntent(action="EXIT") while a position is open.
        - fill_mode=close     => exit at current close
        - fill_mode=next_open => exit at next bar open (no lookahead; same pattern as regime exit)
    """

    def __init__(self, *, costs: Dict[str, Any], exec_cfg: Dict[str, Any], risk_cfg: Optional[Dict[str, Any]] = None):
        self.costs = costs
        self.exec_cfg = exec_cfg
        self.tie_break = exec_cfg.get("intrabar_tie", "sl_first")
        self.fill_mode = exec_cfg.get("fill_mode", "close")

        raw_path = str(exec_cfg.get("intrabar_path", "OHLC")).replace(" ", "").upper()
        self.intrabar_path = raw_path if raw_path in ("OHLC", "OLHC") else "OHLC"

        risk_cfg_in = risk_cfg or {}
        risk_cfg_norm: Dict[str, Any] = dict(risk_cfg_in)

        self.force_exit_on_eof: bool = bool(risk_cfg_norm.get("force_exit_on_eof", True))
        risk_cfg_norm["force_exit_on_eof"] = self.force_exit_on_eof

        # max_holding_bars normalization (supports nested guardrails_cfg paths)
        _mhb_int = 0
        try:
            _mhb_int = int(_extract_max_holding_bars(risk_cfg_norm) or 0)
        except Exception:
            _mhb_int = 0

        self.max_holding_bars: Optional[int] = _mhb_int if _mhb_int > 0 else None
        risk_cfg_norm["max_holding_bars"] = int(self.max_holding_bars) if self.max_holding_bars is not None else 0
        if isinstance(risk_cfg_norm.get("guardrails_cfg"), dict) and self.max_holding_bars is not None:
            risk_cfg_norm["guardrails_cfg"]["max_holding_bars"] = int(self.max_holding_bars)

        # eof_buffer_bars normalization: 0/None => disabled
        _ebb = risk_cfg_norm.get("eof_buffer_bars", 0)
        try:
            _ebb_int = int(_ebb) if _ebb is not None else 0
        except Exception:
            _ebb_int = 0
        if _ebb_int < 0:
            _ebb_int = 0
        self.eof_buffer_bars: int = _ebb_int
        risk_cfg_norm["eof_buffer_bars"] = int(self.eof_buffer_bars)

        self.no_entry_on_last_bar: bool = bool(risk_cfg_norm.get("no_entry_on_last_bar", False))
        risk_cfg_norm["no_entry_on_last_bar"] = bool(self.no_entry_on_last_bar)

        # min_risk_price (only applies when SL exists)
        _mrp = risk_cfg_norm.get("min_risk_price", 0.0)
        try:
            _mrp_f = float(_mrp) if _mrp is not None else 0.0
        except Exception:
            _mrp_f = 0.0
        if _mrp_f < 0:
            _mrp_f = 0.0
        self.min_risk_price: float = _mrp_f
        risk_cfg_norm["min_risk_price"] = float(self.min_risk_price)

        # --- allow time-stop strategies with no SL ---
        self.allow_missing_sl: bool = bool(risk_cfg_norm.get("allow_missing_sl", False))
        risk_cfg_norm["allow_missing_sl"] = bool(self.allow_missing_sl)

        _rpp = risk_cfg_norm.get("risk_proxy_price", None)
        try:
            self.risk_proxy_price: float | None = float(_rpp) if _rpp is not None else None
        except Exception:
            self.risk_proxy_price = None
        if self.risk_proxy_price is not None and (not pd.notna(self.risk_proxy_price) or self.risk_proxy_price <= 0):
            self.risk_proxy_price = None
        risk_cfg_norm["risk_proxy_price"] = float(self.risk_proxy_price) if self.risk_proxy_price is not None else None

        # v1 daily risk gates
        self.max_daily_loss_R: Optional[float] = (
            float(risk_cfg_norm["max_daily_loss_R"]) if risk_cfg_norm.get("max_daily_loss_R") is not None else None
        )
        self.max_trades_per_day: Optional[int] = (
            int(risk_cfg_norm["max_trades_per_day"]) if risk_cfg_norm.get("max_trades_per_day") is not None else None
        )
        self.cooldown_bars: int = int(risk_cfg_norm.get("cooldown_bars", 0) or 0)
        if self.cooldown_bars < 0:
            self.cooldown_bars = 0
        risk_cfg_norm["cooldown_bars"] = self.cooldown_bars

        # -------- REGIME EXIT / ENTRY GATING (engine-level) --------
        self.regime_exit_enabled: bool = bool(risk_cfg_norm.get("regime_exit_enabled", False))
        self.regime_exit_col: str = str(risk_cfg_norm.get("regime_exit_col", "state_regime"))
        _rea = risk_cfg_norm.get("regime_exit_allowed", None)
        if isinstance(_rea, (list, tuple, set)):
            self.regime_exit_allowed: Set[str] = set([str(x) for x in _rea])
        elif _rea is None:
            self.regime_exit_allowed = set()
        else:
            self.regime_exit_allowed = set([str(_rea)])

        self.regime_gate_entries_enabled: bool = bool(risk_cfg_norm.get("regime_gate_entries_enabled", False))
        self.regime_gate_entries_col: str = str(risk_cfg_norm.get("regime_gate_entries_col", "state_regime"))
        _rgea = risk_cfg_norm.get("regime_gate_entries_allowed", None)
        if isinstance(_rgea, (list, tuple, set)):
            self.regime_gate_entries_allowed: Set[str] = set([str(x) for x in _rgea])
        elif _rgea is None:
            self.regime_gate_entries_allowed = set()
        else:
            self.regime_gate_entries_allowed = set([str(_rgea)])

        # Normalize into cfg for manifest/debug visibility
        risk_cfg_norm["regime_exit_enabled"] = bool(self.regime_exit_enabled)
        risk_cfg_norm["regime_exit_col"] = str(self.regime_exit_col)
        risk_cfg_norm["regime_exit_allowed"] = sorted(list(self.regime_exit_allowed))
        risk_cfg_norm["regime_gate_entries_enabled"] = bool(self.regime_gate_entries_enabled)
        risk_cfg_norm["regime_gate_entries_col"] = str(self.regime_gate_entries_col)
        risk_cfg_norm["regime_gate_entries_allowed"] = sorted(list(self.regime_gate_entries_allowed))
        # ----------------------------------------------------------

        mcp = int(risk_cfg_norm.get("max_concurrent_positions", 1) or 1)
        if mcp != 1:
            raise ValueError(
                "ENGINE_SINGLE_POSITION_ONLY: max_concurrent_positions must be 1 for SimpleBarEngine.\n"
                f"got max_concurrent_positions={mcp}\n"
            )

        self.guardrails = Guardrails(risk_cfg_norm)
        self._risk_cfg_norm = risk_cfg_norm
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
        hit_sl = l <= sl <= h
        hit_tp = l <= tp <= h
        if not hit_sl and not hit_tp:
            return None
        if hit_sl and not hit_tp:
            return "SL"
        if hit_tp and not hit_sl:
            return "TP"

        if self.intrabar_path == "OHLC":
            return "TP" if pos_side == "BUY" else "SL"
        else:
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

    def _validate_sl_relation(self, *, side: str, entry_price: float, sl_price: Optional[float]) -> Tuple[bool, str]:
        if sl_price is None:
            if self.allow_missing_sl:
                return True, "OK_NO_SL"
            return False, "MISSING_SL"

        sl = float(sl_price)
        ep = float(entry_price)
        if side == "BUY":
            if not (sl < ep):
                return False, "INVALID_SL_RELATION"
        else:  # SELL
            if not (sl > ep):
                return False, "INVALID_SL_RELATION"
        return True, "OK"

    def _validate_min_risk(self, *, entry_price: float, sl_price: Optional[float]) -> Tuple[bool, str, float]:
        if sl_price is None:
            if not self.allow_missing_sl:
                return False, "MISSING_SL", 0.0
            if self.risk_proxy_price is None or self.risk_proxy_price <= 0:
                return False, "MISSING_RISK_PROXY", 0.0
            return True, "OK_PROXY", float(self.risk_proxy_price)

        rp = abs(float(entry_price) - float(sl_price))
        if rp <= 0:
            return False, "ZERO_RISK", float(rp)
        if self.min_risk_price > 0 and rp < float(self.min_risk_price):
            return False, "TINY_RISK", float(rp)
        return True, "OK", float(rp)

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
        entry_idx: Optional[int],
        exit_idx: Optional[int],
        bar_minutes: Optional[float],
    ) -> Trade:
        exit_fill = self._fill_exit(float(exit_level), pos_side)
        if pos_side == "BUY":
            raw_pnl = (exit_fill - entry_price) * pos_qty
        else:
            raw_pnl = (entry_price - exit_fill) * pos_qty

        pnl = self._apply_costs(raw_pnl, pos_qty)

        # Research-grade duration:
        # Prefer bar-based duration to avoid gaps artifacts.
        hold_m: float | None = None
        if (entry_idx is not None) and (exit_idx is not None) and (bar_minutes is not None) and (bar_minutes > 0):
            try:
                hold_m = float((int(exit_idx) - int(entry_idx)) * float(bar_minutes))
            except Exception:
                hold_m = None
        else:
            # Fallback to timestamp delta
            try:
                hold_m = float((pd.Timestamp(ts) - pd.Timestamp(entry_time)).total_seconds() / 60.0)
            except Exception:
                hold_m = None

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
            hold_minutes=hold_m,
            entry_idx=int(entry_idx) if entry_idx is not None else None,
            exit_idx=int(exit_idx) if exit_idx is not None else None,
            sl_price=sl_price,
            tp_price=tp_price,
            risk_price=risk_price,
        )

    def run(self, df: pd.DataFrame, intents_by_bar: List[List[OrderIntent]]) -> List[Trade]:
        trades: List[Trade] = []

        # Infer bar size once (robust to gaps)
        bar_minutes = _infer_bar_minutes_from_index(df.index)

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

        pending_exit_reason: Optional[str] = None

        current_day: Optional[str] = None
        entries_today: int = 0
        realized_R_today: float = 0.0
        stopped_today: bool = False

        cooldown_until_index: int = 0

        n_blocked_by_daily_stop: int = 0
        n_blocked_by_max_trades: int = 0
        n_blocked_by_cooldown: int = 0

        attempted_entries: int = 0
        blocked_total: int = 0
        blocked_by_reason: Dict[str, int] = {}
        blocked_unique_bars: Set[int] = set()

        blocked_v2_by_reason: Dict[str, int] = {}
        forced_exits: Dict[str, int] = {}

        dropped_entries_total: int = 0
        dropped_entries_by_reason: Dict[str, int] = {}

        entry_skipped_invalid_sl_relation: int = 0
        entry_skipped_missing_sl: int = 0
        entry_skipped_tiny_risk: int = 0

        time_in_position_bars: int = 0

        last_bar_ts: Optional[pd.Timestamp] = None
        last_bar_close: Optional[float] = None

        open_position_at_eof: bool = False

        n_bars = int(len(df) or 0)

        for i, (ts, row) in enumerate(df.iterrows()):
            last_bar_ts = ts
            last_bar_close = float(row["close"])

            day_key = self._utc_day_key(ts)

            if current_day is None:
                current_day = day_key
            elif day_key != current_day:
                current_day = day_key
                entries_today = 0
                realized_R_today = 0.0
                stopped_today = False

            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            # Apply pending exit at this bar OPEN (next_open semantics)
            if (
                (pos_side is not None)
                and (entry_time is not None)
                and (entry_index is not None)
                and (pending_exit_reason is not None)
            ):
                trade = self._close_position(
                    ts=ts,
                    exit_level=float(o),
                    exit_reason=str(pending_exit_reason),
                    pos_side=pos_side,
                    pos_qty=pos_qty,
                    entry_price=float(entry_price),
                    entry_time=entry_time,
                    tag=tag,
                    sl_price=pos_sl_price,
                    tp_price=pos_tp_price,
                    risk_price=pos_risk_price,
                    entry_idx=int(entry_index),
                    exit_idx=int(i),
                    bar_minutes=bar_minutes,
                )
                trades.append(trade)

                self._bump(forced_exits, str(pending_exit_reason))

                r_mult = self._safe_r_multiple(trade.pnl, trade.risk_price, trade.qty)
                if r_mult is not None:
                    realized_R_today += float(r_mult)

                if (self.max_daily_loss_R is not None) and (realized_R_today <= -abs(float(self.max_daily_loss_R))):
                    stopped_today = True

                if self.cooldown_bars > 0:
                    cooldown_until_index = max(cooldown_until_index, i + self.cooldown_bars + 1)

                # Reset position + pending exit
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
                pending_exit_reason = None

            # Fill pending entry at this bar open
            if pos_side is None and pending_intent is not None:
                side = pending_intent.side
                qty = float(pending_intent.qty)

                filled_entry = self._fill_entry(o, side)

                _sl = float(pending_intent.sl_price) if pending_intent.sl_price is not None else None
                _tp = float(pending_intent.tp_price) if pending_intent.tp_price is not None else None

                ok_sl, reason_sl = self._validate_sl_relation(side=side, entry_price=filled_entry, sl_price=_sl)
                if not ok_sl:
                    if reason_sl == "MISSING_SL":
                        entry_skipped_missing_sl += 1
                        self._bump(dropped_entries_by_reason, "ENTRY_SKIPPED_MISSING_SL")
                    else:
                        entry_skipped_invalid_sl_relation += 1
                        self._bump(dropped_entries_by_reason, "ENTRY_SKIPPED_INVALID_SL_RELATION")
                    dropped_entries_total += 1
                    pending_intent = None
                    pending_tag = ""
                else:
                    ok_risk, reason_risk, rp = self._validate_min_risk(entry_price=filled_entry, sl_price=_sl)
                    if not ok_risk:
                        if reason_risk in ("MISSING_SL",):
                            entry_skipped_missing_sl += 1
                            self._bump(dropped_entries_by_reason, "ENTRY_SKIPPED_MISSING_SL")
                        elif reason_risk == "MISSING_RISK_PROXY":
                            self._bump(dropped_entries_by_reason, "ENTRY_SKIPPED_MISSING_RISK_PROXY")
                        else:
                            entry_skipped_tiny_risk += 1
                            self._bump(dropped_entries_by_reason, f"ENTRY_SKIPPED_{reason_risk}")
                        dropped_entries_total += 1
                        pending_intent = None
                        pending_tag = ""
                    else:
                        pos_side = side
                        pos_qty = qty
                        entry_price = float(filled_entry)
                        entry_time = ts
                        entry_index = i

                        sl = _sl
                        tp = _tp
                        pos_sl_price = sl
                        pos_tp_price = tp
                        pos_risk_price = float(rp)

                        tag = pending_tag
                        pending_intent = None
                        pending_tag = ""
                        entries_today += 1

            if pos_side is not None:
                time_in_position_bars += 1

            # ---- Strategy-driven EXIT intents ----
            if pos_side is not None and pending_exit_reason is None:
                intents_here = intents_by_bar[i] if i < len(intents_by_bar) else []
                if intents_here:
                    for it in intents_here:
                        if str(getattr(it, "action", "ENTER")).upper() == "EXIT":
                            reason = str(getattr(it, "exit_reason", "") or "").strip() or "SIGNAL_EXIT"
                            if self.fill_mode == "next_open":
                                if i < n_bars - 1:
                                    pending_exit_reason = reason
                                else:
                                    trade = self._close_position(
                                        ts=ts,
                                        exit_level=float(c),
                                        exit_reason=reason,
                                        pos_side=pos_side,
                                        pos_qty=pos_qty,
                                        entry_price=float(entry_price),
                                        entry_time=entry_time,
                                        tag=tag,
                                        sl_price=pos_sl_price,
                                        tp_price=pos_tp_price,
                                        risk_price=pos_risk_price,
                                        entry_idx=int(entry_index) if entry_index is not None else None,
                                        exit_idx=int(i),
                                        bar_minutes=bar_minutes,
                                    )
                                    trades.append(trade)
                                    self._bump(forced_exits, f"{reason}_FALLBACK_CLOSE")

                                    r_mult = self._safe_r_multiple(trade.pnl, trade.risk_price, trade.qty)
                                    if r_mult is not None:
                                        realized_R_today += float(r_mult)

                                    if (self.max_daily_loss_R is not None) and (
                                        realized_R_today <= -abs(float(self.max_daily_loss_R))
                                    ):
                                        stopped_today = True

                                    if self.cooldown_bars > 0:
                                        cooldown_until_index = max(cooldown_until_index, i + self.cooldown_bars + 1)

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
                                    pending_exit_reason = None
                            else:
                                trade = self._close_position(
                                    ts=ts,
                                    exit_level=float(c),
                                    exit_reason=reason,
                                    pos_side=pos_side,
                                    pos_qty=pos_qty,
                                    entry_price=float(entry_price),
                                    entry_time=entry_time,
                                    tag=tag,
                                    sl_price=pos_sl_price,
                                    tp_price=pos_tp_price,
                                    risk_price=pos_risk_price,
                                    entry_idx=int(entry_index) if entry_index is not None else None,
                                    exit_idx=int(i),
                                    bar_minutes=bar_minutes,
                                )
                                trades.append(trade)
                                self._bump(forced_exits, reason)

                                r_mult = self._safe_r_multiple(trade.pnl, trade.risk_price, trade.qty)
                                if r_mult is not None:
                                    realized_R_today += float(r_mult)

                                if (self.max_daily_loss_R is not None) and (
                                    realized_R_today <= -abs(float(self.max_daily_loss_R))
                                ):
                                    stopped_today = True

                                if self.cooldown_bars > 0:
                                    cooldown_until_index = max(cooldown_until_index, i + self.cooldown_bars + 1)

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
                                pending_exit_reason = None

                            break
            # ---------------------------------------------------------------------------

            # Manage existing position
            if pos_side is not None and entry_time is not None and entry_index is not None:
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

                if exit_reason is None and self.regime_exit_enabled and len(self.regime_exit_allowed) > 0:
                    if self.regime_exit_col in df.columns:
                        cur_reg = row.get(self.regime_exit_col, None)
                        cur_reg_s = str(cur_reg) if cur_reg is not None and pd.notna(cur_reg) else None
                        if cur_reg_s is not None and (cur_reg_s not in self.regime_exit_allowed):
                            if self.fill_mode == "next_open":
                                if i < n_bars - 1:
                                    pending_exit_reason = "FORCE_REGIME_EXIT"
                                else:
                                    exit_reason, exit_level = "FORCE_REGIME_EXIT", float(c)
                                    self._bump(forced_exits, "FORCE_REGIME_EXIT_FALLBACK_CLOSE")
                            else:
                                exit_reason, exit_level = "FORCE_REGIME_EXIT", float(c)
                    else:
                        self._bump(forced_exits, "FORCE_REGIME_EXIT_MISSING_COL")

                if exit_reason is None and pending_exit_reason is None and (self.max_holding_bars is not None):
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
                        entry_idx=int(entry_index),
                        exit_idx=int(i),
                        bar_minutes=bar_minutes,
                    )
                    trades.append(trade)

                    r_mult = self._safe_r_multiple(trade.pnl, trade.risk_price, trade.qty)
                    if r_mult is not None:
                        realized_R_today += float(r_mult)

                    if (self.max_daily_loss_R is not None) and (realized_R_today <= -abs(float(self.max_daily_loss_R))):
                        stopped_today = True

                    if self.cooldown_bars > 0:
                        cooldown_until_index = max(cooldown_until_index, i + self.cooldown_bars + 1)

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
                    pending_exit_reason = None

            # Schedule entries (only if flat and no pending)
            if pos_side is None and pending_intent is None:
                intents = intents_by_bar[i] if i < len(intents_by_bar) else []
                if intents:
                    intent = None
                    for cand in intents:
                        if str(getattr(cand, "action", "ENTER")).upper() == "ENTER":
                            intent = cand
                            break
                    if intent is None:
                        continue

                    attempted_entries += 1

                    blocked = False
                    blocked_reason: Optional[str] = None

                    if stopped_today:
                        n_blocked_by_daily_stop += 1
                        blocked = True
                        blocked_reason = "DAILY_STOP"

                    if (not blocked) and (self.max_trades_per_day is not None) and (entries_today >= int(self.max_trades_per_day)):
                        n_blocked_by_max_trades += 1
                        blocked = True
                        blocked_reason = "MAX_TRADES_PER_DAY"

                    if (not blocked) and (i < cooldown_until_index):
                        n_blocked_by_cooldown += 1
                        blocked = True
                        blocked_reason = "COOLDOWN"

                    if not blocked:
                        active_positions = 0

                        fill_index = i + 1 if self.fill_mode == "next_open" else i
                        remaining_bars = (n_bars - 1) - int(fill_index)

                        if self.no_entry_on_last_bar and fill_index >= (n_bars - 1):
                            blocked = True
                            blocked_reason = "V2_by_eof_buffer"
                            self._bump(blocked_v2_by_reason, "by_eof_buffer")

                        if (not blocked) and (self.eof_buffer_bars > 0) and (remaining_bars <= self.eof_buffer_bars):
                            blocked = True
                            blocked_reason = "V2_by_eof_buffer"
                            self._bump(blocked_v2_by_reason, "by_eof_buffer")

                        if (not blocked) and self.regime_gate_entries_enabled and len(self.regime_gate_entries_allowed) > 0:
                            col = self.regime_gate_entries_col
                            if col in df.columns and 0 <= fill_index < n_bars:
                                reg_v = df.iloc[int(fill_index)][col]
                                reg_s = str(reg_v) if reg_v is not None and pd.notna(reg_v) else None
                                if reg_s is None or (reg_s not in self.regime_gate_entries_allowed):
                                    blocked = True
                                    blocked_reason = "REGIME_GATE"
                                    self._bump(blocked_v2_by_reason, "by_regime_gate")
                            else:
                                self._bump(blocked_v2_by_reason, "by_regime_gate_missing_col")

                        if not blocked:
                            if self.fill_mode == "next_open":
                                if i < n_bars - 1:
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
                        self._bump(blocked_by_reason, blocked_reason or "UNKNOWN")
                        continue

                    if self.fill_mode == "next_open":
                        if i < n_bars - 1:
                            pending_intent = intent
                            pending_tag = intent.tag
                        else:
                            dropped_entries_total += 1
                            self._bump(dropped_entries_by_reason, "NO_NEXT_OPEN")
                    else:
                        side = intent.side
                        qty = float(intent.qty)
                        filled_entry = self._fill_entry(c, side)

                        _sl = float(intent.sl_price) if intent.sl_price is not None else None
                        _tp = float(intent.tp_price) if intent.tp_price is not None else None

                        ok_sl, reason_sl = self._validate_sl_relation(side=side, entry_price=filled_entry, sl_price=_sl)
                        if not ok_sl:
                            dropped_entries_total += 1
                            if reason_sl == "MISSING_SL":
                                entry_skipped_missing_sl += 1
                                self._bump(dropped_entries_by_reason, "ENTRY_SKIPPED_MISSING_SL")
                            else:
                                entry_skipped_invalid_sl_relation += 1
                                self._bump(dropped_entries_by_reason, "ENTRY_SKIPPED_INVALID_SL_RELATION")
                            continue

                        ok_risk, reason_risk, rp = self._validate_min_risk(entry_price=filled_entry, sl_price=_sl)
                        if not ok_risk:
                            dropped_entries_total += 1
                            if reason_risk == "MISSING_SL":
                                entry_skipped_missing_sl += 1
                                self._bump(dropped_entries_by_reason, "ENTRY_SKIPPED_MISSING_SL")
                            elif reason_risk == "MISSING_RISK_PROXY":
                                self._bump(dropped_entries_by_reason, "ENTRY_SKIPPED_MISSING_RISK_PROXY")
                            else:
                                entry_skipped_tiny_risk += 1
                                self._bump(dropped_entries_by_reason, f"ENTRY_SKIPPED_{reason_risk}")
                            continue

                        pos_side = side
                        pos_qty = qty
                        entry_price = float(filled_entry)
                        entry_time = ts
                        entry_index = i

                        sl = _sl
                        tp = _tp
                        pos_sl_price = sl
                        pos_tp_price = tp
                        pos_risk_price = float(rp)

                        tag = intent.tag
                        entries_today += 1

        if pending_intent is not None:
            dropped_entries_total += 1
            self._bump(dropped_entries_by_reason, "PENDING_INTENT_AT_EOF")

        if (pos_side is not None) and (entry_time is not None) and (last_bar_ts is not None) and (last_bar_close is not None):
            eof_idx = n_bars - 1 if n_bars > 0 else None
            if self.force_exit_on_eof:
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
                    entry_idx=int(entry_index) if entry_index is not None else None,
                    exit_idx=int(eof_idx) if eof_idx is not None else None,
                    bar_minutes=bar_minutes,
                )
                trades.append(trade)
                self._bump(forced_exits, "EOF")

                r_mult = self._safe_r_multiple(trade.pnl, trade.risk_price, trade.qty)
                if r_mult is not None:
                    realized_R_today += float(r_mult)
            else:
                open_position_at_eof = True
                self._bump(forced_exits, "EOF_SKIPPED_BY_POLICY")

            pos_side = None
            pending_intent = None

        gr_rep = self.guardrails.report()
        if not isinstance(gr_rep, dict):
            gr_rep = {"raw": gr_rep}

        gr_rep.setdefault("entry_gate", {})
        gr_rep["entry_gate"] = {
            "attempted_entries": attempted_entries,
            "blocked_total": blocked_total,
            "blocked_unique_bars": len(blocked_unique_bars),
            "blocked_v2_by_reason": dict(blocked_v2_by_reason),
        }

        bars_total = int(len(df) or 0)
        time_in_position_rate = (float(time_in_position_bars) / float(bars_total)) if bars_total > 0 else 0.0

        self.last_risk_report = {
            "risk_report_version": "v1",
            "risk_cfg": {
                "max_daily_loss_R": self.max_daily_loss_R,
                "max_trades_per_day": self.max_trades_per_day,
                "cooldown_bars": self.cooldown_bars,
                "force_exit_on_eof": bool(self.force_exit_on_eof),
                "max_holding_bars": int(self.max_holding_bars) if self.max_holding_bars is not None else 0,
                "eof_buffer_bars": int(self.eof_buffer_bars),
                "no_entry_on_last_bar": bool(self.no_entry_on_last_bar),
                "min_risk_price": float(self.min_risk_price),
                "allow_missing_sl": bool(self.allow_missing_sl),
                "risk_proxy_price": float(self.risk_proxy_price) if self.risk_proxy_price is not None else None,
                "regime_exit_enabled": bool(self.regime_exit_enabled),
                "regime_exit_col": str(self.regime_exit_col),
                "regime_exit_allowed": sorted(list(self.regime_exit_allowed)),
                "regime_gate_entries_enabled": bool(self.regime_gate_entries_enabled),
                "regime_gate_entries_col": str(self.regime_gate_entries_col),
                "regime_gate_entries_allowed": sorted(list(self.regime_gate_entries_allowed)),
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
            "entry_gate": {
                "attempted_entries": attempted_entries,
                "blocked_total": blocked_total,
                "blocked_unique_bars": len(blocked_unique_bars),
                "blocked_by_reason": dict(blocked_by_reason),
                "entry_skipped_invalid_sl_relation": int(entry_skipped_invalid_sl_relation),
                "entry_skipped_missing_sl": int(entry_skipped_missing_sl),
                "entry_skipped_tiny_risk": int(entry_skipped_tiny_risk),
            },
            "guardrails": gr_rep,
            "forced_exits": dict(forced_exits),
            "entry_fill_dropped": {
                "dropped_total": int(dropped_entries_total),
                "dropped_by_reason": dict(dropped_entries_by_reason),
            },
            "engine": {
                "bars_total": int(bars_total),
                "time_in_position_bars": int(time_in_position_bars),
                "time_in_position_rate": float(time_in_position_rate),
                "open_position_at_eof": bool(open_position_at_eof),
            },
        }

        return trades
