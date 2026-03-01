from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

import MetaTrader5 as mt5

from src.live.mt5.adapter import MT5Adapter
from src.live.state.idempotency_store import IdempotencyStore
from src.live.io.orders_log import OrdersLog
from src.live.io.incidents_log import IncidentsLog


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return _utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class CloseEvent:
    intent_id: str
    instrument: str
    broker_position_id: int
    close_deal_ticket: int
    close_time_utc: datetime
    close_price: float
    profit: float
    volume: float
    reason: str  # UNKNOWN|SL|TP (best-effort)


class TradeFinalizer:
    """
    Finalizes closed trades by querying MT5 deal history.
    Robustness:
      - queries history using LOCAL naive datetimes (most reliable for mt5.history_deals_get)
      - does NOT depend on symbol filter at API level
      - filters by symbol + position_id locally
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
        strategy_id: str,
        strategy_version: str,
        instrument: str,
    ):
        self.mt5 = mt5
        self.store = store
        self.orders_log = orders_log
        self.incidents_log = incidents_log

        self.run_id = run_id
        self.run_mode = run_mode
        self.portfolio_mode = portfolio_mode
        self.strategy_id = strategy_id
        self.strategy_version = strategy_version
        self.instrument = instrument

    def finalize_window(self, *, ftmo_day_id: str, time_from_utc: datetime, time_to_utc: datetime) -> int:
        # Pull store rows that are "open-ish"
        rows = self.store.list_recent(limit=3000)
        openish = [r for r in rows if r.get("state") in ("FILLED", "ATTACHED", "OPEN_VERIFIED", "ACCEPTED")]

        if not openish:
            return 0

        # MT5 history_deals_get is most reliable with LOCAL naive datetimes
        # Convert UTC-aware window into local naive window of the same absolute instants.
        # We do: utc -> local -> naive
        from_utc = _utc(time_from_utc)
        to_utc = _utc(time_to_utc)

        from_local_naive = from_utc.astimezone().replace(tzinfo=None)
        to_local_naive = to_utc.astimezone().replace(tzinfo=None)

        deals = mt5.history_deals_get(from_local_naive, to_local_naive)
        if deals is None:
            self.incidents_log.log(
                run_id=self.run_id,
                run_mode=self.run_mode,
                portfolio_mode=self.portfolio_mode,
                severity="ERROR",
                type="HISTORY_DEALS_GET_NONE",
                action_taken="IGNORE",
                ftmo_day_id=ftmo_day_id,
                instrument=self.instrument,
                details=f"history_deals_get returned None: {mt5.last_error()}",
            )
            return 0

        deals = list(deals)

        # Index deals by position_id
        deals_by_pos: Dict[int, List[Any]] = {}
        for d in deals:
            try:
                pos_id = int(getattr(d, "position_id", 0) or 0)
            except Exception:
                pos_id = 0
            if pos_id <= 0:
                continue
            deals_by_pos.setdefault(pos_id, []).append(d)

        finalized = 0

        for r in openish:
            intent_id = str(r.get("intent_id", ""))
            idem_key = str(r.get("idempotency_key", ""))
            pos_id = r.get("broker_position_id")
            if not pos_id:
                continue
            pos_id = int(pos_id)

            # If still open live, skip
            live_positions = self.mt5.positions(self.instrument)
            if any(int(p.ticket) == pos_id for p in live_positions):
                continue

            # Get deals for that position
            ds = deals_by_pos.get(pos_id, [])
            if not ds:
                continue

            # Filter to instrument locally (defensive)
            ds = [d for d in ds if (getattr(d, "symbol", "") == self.instrument)]
            if not ds:
                continue

            ds_sorted = sorted(ds, key=lambda x: getattr(x, "time_msc", 0) or 0)
            close_deal = ds_sorted[-1]

            close_time = getattr(close_deal, "time", None)
            if close_time is None:
                # Fallback: if time missing, use now
                close_time_utc = datetime.now(timezone.utc)
            else:
                # MT5 deal.time is local timestamp in seconds; convert via datetime.fromtimestamp then local->utc
                close_time_local = datetime.fromtimestamp(int(close_time))
                close_time_utc = close_time_local.astimezone(timezone.utc)

            ce = CloseEvent(
                intent_id=intent_id,
                instrument=self.instrument,
                broker_position_id=pos_id,
                close_deal_ticket=int(getattr(close_deal, "ticket", 0) or 0),
                close_time_utc=close_time_utc,
                close_price=float(getattr(close_deal, "price", 0.0) or 0.0),
                profit=float(getattr(close_deal, "profit", 0.0) or 0.0),
                volume=float(getattr(close_deal, "volume", 0.0) or 0.0),
                reason=self._infer_close_reason(close_deal),
            )

            # Update store -> CLOSED
            try:
                self.store.mark_state(idem_key, state="CLOSED")
            except Exception as e:
                self.incidents_log.log(
                    run_id=self.run_id,
                    run_mode=self.run_mode,
                    portfolio_mode=self.portfolio_mode,
                    severity="ERROR",
                    type="STORE_UPDATE_FAILED",
                    action_taken="IGNORE",
                    ftmo_day_id=ftmo_day_id,
                    instrument=self.instrument,
                    details=f"Failed to mark CLOSED intent={intent_id} pos_id={pos_id}: {e}",
                    intent_id=intent_id,
                    strategy_id=self.strategy_id,
                    broker_position_id=pos_id,
                )
                continue

            # Log close event (orders.csv)
            self.orders_log.append_event(
                run_id=self.run_id,
                run_mode=self.run_mode,
                portfolio_mode=self.portfolio_mode,
                ftmo_day_id=ftmo_day_id,
                intent_id=intent_id,
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
                idempotency_key=idem_key,
                client_order_id=str(r.get("client_order_id", "")),
                broker_position_id=pos_id,
                broker_order_id=r.get("broker_order_id"),
                event_type="POSITION_CLOSED",
                event_state="CLOSED",
                event_reason=f"{ce.reason}|deal={ce.close_deal_ticket}|profit={ce.profit}",
                filled_qty=ce.volume,
                fill_price=ce.close_price,
                fill_ts_utc=_iso(ce.close_time_utc),
            )

            finalized += 1

        return finalized

    @staticmethod
    def _infer_close_reason(deal: Any) -> str:
        c = (getattr(deal, "comment", "") or "").lower()
        if "tp" in c:
            return "TP"
        if "sl" in c:
            return "SL"
        return "UNKNOWN"