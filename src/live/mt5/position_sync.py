from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from src.live.mt5.adapter import MT5Adapter
from src.live.mt5.mt5_types import Position, Order
from src.live.state.idempotency_store import IdempotencyStore
from src.live.io.orders_log import OrdersLog
from src.live.io.incidents_log import IncidentsLog


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_INTENT_RE = re.compile(r"(?:^|\|)i=([A-Z0-9]{6,16})(?:\||$)")  # e.g. i=K9Q2M7P1XA


@dataclass(frozen=True)
class AssocResult:
    status: str      # OK|ASSOCIATED|AMBIGUOUS|UNASSOCIATED
    method: str      # STRONG_ID|STORE_ID|NONE
    confidence: str  # HIGH|MED|LOW|NONE
    intent_id: str
    details: str


@dataclass(frozen=True)
class BrokerState:
    ts_utc: str
    symbol: str
    positions: List[Position]
    orders: List[Order]
    position_count: int
    pending_order_count: int
    assoc: AssocResult
    safe_halt: bool


class PositionSync:
    """
    Reconciles broker truth and applies conservative safety flags.

    Mode A:
      - if any open position or pending order exists -> Coordinator should block new entries
      - safe_halt triggers only on truly unexplained exposure.
      - IMPORTANT: safe_halt is allowed to CLEAR if association becomes valid (Mode A only).
    """

    def __init__(
        self,
        *,
        mt5: MT5Adapter,
        store: IdempotencyStore,
        orders_log: OrdersLog,
        incidents_log: IncidentsLog,
        run_id: str,
        run_mode: str,
        portfolio_mode: str,
        instrument: str,
        strategy_id: str,
        strategy_version: str,
    ):
        self.mt5 = mt5
        self.store = store
        self.orders_log = orders_log
        self.incidents_log = incidents_log

        self.run_id = run_id
        self.run_mode = run_mode
        self.portfolio_mode = portfolio_mode
        self.instrument = instrument
        self.strategy_id = strategy_id
        self.strategy_version = strategy_version

        self.safe_halt = False  # latched, but can clear in Mode A when assoc becomes valid

    def reconcile(self, *, ftmo_day_id: str, intent_candidates: Optional[List[Dict[str, Any]]] = None) -> BrokerState:
        symbol = self.instrument
        ts = _now_utc_iso()

        positions = self.mt5.positions(symbol)
        orders = self.mt5.orders(symbol)

        pos_count = len(positions)
        ord_count = len(orders)

        assoc = self._associate(positions=positions, orders=orders)

        # --- SAFE_HALT POLICY (Mode A) ---
        # If we can associate, we allow safe_halt to clear.
        if assoc.status == "ASSOCIATED" or assoc.status == "OK":
            self.safe_halt = False
        else:
            # Latch only if there is broker exposure and we truly cannot associate
            if (pos_count > 0 or ord_count > 0) and assoc.status in ("AMBIGUOUS", "UNASSOCIATED"):
                self.safe_halt = True

        # Always log reconcile (audit trail)
        self.orders_log.append_event(
            run_id=self.run_id,
            run_mode=self.run_mode,
            portfolio_mode=self.portfolio_mode,
            ftmo_day_id=ftmo_day_id,
            intent_id=assoc.intent_id or "",
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            instrument=self.instrument,
            timeframe="M5",
            side="",
            order_type="",
            time_in_force="",
            risk_mode="",
            risk_value_pct=0.0,
            requested_sl_pips=0.0,
            requested_tp_pips=0.0,
            max_hold_min=0,
            valid_to_utc="",
            idempotency_key="",
            client_order_id="",
            event_type="SYNC_RECONCILE",
            event_state=assoc.status,
            event_reason=f"{assoc.method}:{assoc.confidence}:{assoc.details}",
            broker_order_id=orders[0].ticket if ord_count > 0 else None,
            broker_position_id=positions[0].ticket if pos_count > 0 else None,
        )

        # Log incidents only when status indicates a real problem
        if assoc.status == "AMBIGUOUS":
            self.incidents_log.log(
                run_id=self.run_id,
                run_mode=self.run_mode,
                portfolio_mode=self.portfolio_mode,
                severity="ERROR",
                type="ASSOC_AMBIGUOUS",
                action_taken="HALT",
                ftmo_day_id=ftmo_day_id,
                instrument=self.instrument,
                details=assoc.details,
                intent_id=assoc.intent_id or "",
                strategy_id=self.strategy_id,
                broker_order_id=orders[0].ticket if ord_count > 0 else None,
                broker_position_id=positions[0].ticket if pos_count > 0 else None,
            )

        if assoc.status == "UNASSOCIATED":
            self.incidents_log.log(
                run_id=self.run_id,
                run_mode=self.run_mode,
                portfolio_mode=self.portfolio_mode,
                severity="CRITICAL",
                type="UNASSOCIATED_BROKER_STATE",
                action_taken="HALT",
                ftmo_day_id=ftmo_day_id,
                instrument=self.instrument,
                details=assoc.details,
                intent_id="",
                strategy_id=self.strategy_id,
                broker_order_id=orders[0].ticket if ord_count > 0 else None,
                broker_position_id=positions[0].ticket if pos_count > 0 else None,
            )

        return BrokerState(
            ts_utc=ts,
            symbol=symbol,
            positions=positions,
            orders=orders,
            position_count=pos_count,
            pending_order_count=ord_count,
            assoc=assoc,
            safe_halt=self.safe_halt,
        )

    # ---------- association ----------

    def _associate(self, *, positions: List[Position], orders: List[Order]) -> AssocResult:
        if not positions and not orders:
            return AssocResult(status="OK", method="NONE", confidence="NONE", intent_id="", details="NO_BROKER_EXPOSURE")

        # 0) Ambiguity guard (Mode A expects single exposure)
        if len(positions) > 1 or len(orders) > 1:
            return AssocResult(
                status="AMBIGUOUS",
                method="NONE",
                confidence="LOW",
                intent_id="",
                details=f"MULTIPLE_EXPOSURES pos={len(positions)} ord={len(orders)}",
            )

        # 1) STRONG_ID via comment parsing
        intent_tok = self._extract_intent_from_broker(positions, orders)
        if intent_tok:
            return AssocResult(status="ASSOCIATED", method="STRONG_ID", confidence="HIGH", intent_id=intent_tok, details="FOUND_INTENT_IN_COMMENT")

        # 2) STORE_ID by broker tickets
        store_match = self._store_match_by_ticket(positions, orders)
        if store_match:
            return AssocResult(status="ASSOCIATED", method="STORE_ID", confidence="HIGH", intent_id=store_match, details="MATCHED_BROKER_TICKET_IN_STORE")

        # 3) Unassociated
        return AssocResult(
            status="UNASSOCIATED",
            method="NONE",
            confidence="NONE",
            intent_id="",
            details="BROKER_HAS_STATE_BUT_NO_STRONG_ID_AND_NO_STORE_TICKET_MATCH",
        )

    def _extract_intent_from_broker(self, positions: List[Position], orders: List[Order]) -> Optional[str]:
        for p in positions:
            c = (p.comment or "").strip()
            if not c:
                continue
            m = _INTENT_RE.search(c)
            if m:
                return m.group(1)

        for o in orders:
            c = (o.comment or "").strip()
            if not c:
                continue
            m = _INTENT_RE.search(c)
            if m:
                return m.group(1)

        return None

    def _store_match_by_ticket(self, positions: List[Position], orders: List[Order]) -> Optional[str]:
        pos_ids = {p.ticket for p in positions}
        ord_ids = {o.ticket for o in orders}

        recent = self.store.list_recent(limit=500)
        for r in recent:
            bo = r.get("broker_order_id")
            bp = r.get("broker_position_id")
            if bo is not None and int(bo) in ord_ids:
                return str(r.get("intent_id", ""))
            if bp is not None and int(bp) in pos_ids:
                return str(r.get("intent_id", ""))
        return None