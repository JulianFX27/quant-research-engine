from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.live.mt5.adapter import MT5Adapter
from src.live.mt5.mt5_types import Position
from src.live.state.idempotency_store import IdempotencyStore, IdempotencyRecord
from src.live.io.orders_log import OrdersLog
from src.live.io.incidents_log import IncidentsLog
from src.live.mt5.position_sync import PositionSync


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ManualIntent:
    # Minimal subset to execute one demo trade
    intent_id: str
    strategy_id: str
    strategy_version: str
    instrument: str
    timeframe: str  # "M5"
    side: str  # LONG/SHORT
    order_type: str  # MARKET
    tif: str  # IOC
    sl_pips: float
    tp_pips: float
    max_hold_min: int
    risk_mode: str  # FIXED_PCT_EQUITY (not used here)
    risk_value_pct: float  # not used here
    valid_to_utc: str
    decision_ts_utc: str
    bar_ts_utc: str


class OrderManager:
    """
    State machine for a single trade execution (Mode A).
    - Idempotent submit (store)
    - SYNC reconcile after submit
    - Ensure SL/TP present; if cannot attach => close position (failsafe)
    """

    def __init__(
        self,
        *,
        mt5: MT5Adapter,
        sync: PositionSync,
        store: IdempotencyStore,
        orders_log: OrdersLog,
        incidents_log: IncidentsLog,
        run_id: str,
        run_mode: str,
        portfolio_mode: str,
        deviation_points: int = 20,
        fill_timeout_ms: int = 4000,
        attach_attempts: int = 3,
    ):
        self.mt5 = mt5
        self.sync = sync
        self.store = store
        self.orders_log = orders_log
        self.incidents_log = incidents_log

        self.run_id = run_id
        self.run_mode = run_mode
        self.portfolio_mode = portfolio_mode

        self.deviation_points = deviation_points
        self.fill_timeout_ms = fill_timeout_ms
        self.attach_attempts = attach_attempts

    # ---------- main ----------

    def execute_manual_intent(
        self,
        *,
        ftmo_day_id: str,
        intent: ManualIntent,
        requested_qty: float,
        comment: Optional[str] = None,
        magic: Optional[int] = None,
    ) -> None:
        """
        Executes one manual intent end-to-end.
        Safety gates:
          - broker must have no positions/orders (Mode A)
          - sync must not be in safe_halt
        """
        # Reconcile first
        bs = self.sync.reconcile(ftmo_day_id=ftmo_day_id)
        if bs.safe_halt:
            raise RuntimeError("SAFE_HALT is active; refusing to trade.")
        if bs.position_count > 0 or bs.pending_order_count > 0:
            raise RuntimeError("Broker has exposure; refusing to trade in Mode A.")

        # Build deterministic idempotency key (manual intent)
        idempotency_key = _sha1(
            f"{intent.intent_id}|{intent.instrument}|{intent.side}|{intent.bar_ts_utc}|{intent.strategy_version}"
        )[:16]
        client_order_id = f"COID-{intent.intent_id[:8]}-{intent.bar_ts_utc.replace(':','').replace('-','')[:13]}"

        # Upsert record if not exists
        self.store.upsert_new(
            IdempotencyRecord(
                idempotency_key=idempotency_key,
                intent_id=intent.intent_id,
                client_order_id=client_order_id,
                state="READY",
                attempt_no=1,
            )
        )

        existing = self.store.get(idempotency_key)
        if existing and existing.get("state") in ("ACCEPTED", "FILLED", "ATTACHED", "OPEN_VERIFIED"):
            # Do not submit; just reconcile
            self._log_sync(ftmo_day_id, intent, idempotency_key, client_order_id, "RECOVER_EXISTING_STATE")
            return

        # Submit order (best-effort SL/TP included)
        t0 = time.time()

        tick = self.mt5.get_tick(intent.instrument)
        entry_price_ref = tick.ask if intent.side == "LONG" else tick.bid
        sl_price_req, tp_price_req = self._prices_from_pips(entry_price_ref, intent.side, intent.sl_pips, intent.tp_pips)

        self.orders_log.append_event(
            run_id=self.run_id,
            run_mode=self.run_mode,
            portfolio_mode=self.portfolio_mode,
            ftmo_day_id=ftmo_day_id,
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            strategy_version=intent.strategy_version,
            instrument=intent.instrument,
            timeframe=intent.timeframe,
            side=intent.side,
            order_type=intent.order_type,
            time_in_force=intent.tif,
            risk_mode=intent.risk_mode,
            risk_value_pct=float(intent.risk_value_pct),
            requested_sl_pips=float(intent.sl_pips),
            requested_tp_pips=float(intent.tp_pips),
            max_hold_min=int(intent.max_hold_min),
            valid_to_utc=intent.valid_to_utc,
            idempotency_key=idempotency_key,
            client_order_id=client_order_id,
            event_type="ORDER_SUBMIT",
            event_state="SUBMITTED",
            event_reason="OK",
            requested_qty=requested_qty,
            request_price=float(entry_price_ref),
            sl_price=float(sl_price_req),
            tp_price=float(tp_price_req),
            attempt_no=int(existing.get("attempt_no", 1)) if existing else 1,
        )

        res = self.mt5.send_market_order(
            symbol=intent.instrument,
            side=intent.side,
            volume=float(requested_qty),
            sl_price=sl_price_req,
            tp_price=tp_price_req,
            comment=comment,
            magic=magic,
            deviation_points=self.deviation_points,
        )
        latency_ms = int((time.time() - t0) * 1000)

        if not res.ok:
            reason = f"RET={res.retcode} {res.comment}"
            self.orders_log.append_event(
                run_id=self.run_id,
                run_mode=self.run_mode,
                portfolio_mode=self.portfolio_mode,
                ftmo_day_id=ftmo_day_id,
                intent_id=intent.intent_id,
                strategy_id=intent.strategy_id,
                strategy_version=intent.strategy_version,
                instrument=intent.instrument,
                timeframe=intent.timeframe,
                side=intent.side,
                order_type=intent.order_type,
                time_in_force=intent.tif,
                risk_mode=intent.risk_mode,
                risk_value_pct=float(intent.risk_value_pct),
                requested_sl_pips=float(intent.sl_pips),
                requested_tp_pips=float(intent.tp_pips),
                max_hold_min=int(intent.max_hold_min),
                valid_to_utc=intent.valid_to_utc,
                idempotency_key=idempotency_key,
                client_order_id=client_order_id,
                event_type="ORDER_REJECT",
                event_state="REJECTED",
                event_reason=reason,
                last_error_code=res.retcode,
                last_error_msg=res.comment,
                latency_ms=latency_ms,
                attempt_no=int(existing.get("attempt_no", 1)) if existing else 1,
            )
            self.store.mark_state(idempotency_key, state="FAILED_FINAL")
            return

        # Accepted/Done
        self.store.mark_state(
            idempotency_key,
            state="ACCEPTED",
            broker_order_id=res.broker_order_id,
            broker_deal_id=res.broker_deal_id,
        )

        self.orders_log.append_event(
            run_id=self.run_id,
            run_mode=self.run_mode,
            portfolio_mode=self.portfolio_mode,
            ftmo_day_id=ftmo_day_id,
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            strategy_version=intent.strategy_version,
            instrument=intent.instrument,
            timeframe=intent.timeframe,
            side=intent.side,
            order_type=intent.order_type,
            time_in_force=intent.tif,
            risk_mode=intent.risk_mode,
            risk_value_pct=float(intent.risk_value_pct),
            requested_sl_pips=float(intent.sl_pips),
            requested_tp_pips=float(intent.tp_pips),
            max_hold_min=int(intent.max_hold_min),
            valid_to_utc=intent.valid_to_utc,
            idempotency_key=idempotency_key,
            client_order_id=client_order_id,
            broker_order_id=res.broker_order_id,
            event_type="ORDER_ACCEPT",
            event_state="ACCEPTED",
            event_reason=f"RET={res.retcode}",
            latency_ms=latency_ms,
        )

        # Wait for fill (via PositionSync polling)
        pos = self._wait_for_position(ftmo_day_id, intent.instrument, timeout_ms=self.fill_timeout_ms)
        if pos is None:
            self.incidents_log.log(
                run_id=self.run_id,
                run_mode=self.run_mode,
                portfolio_mode=self.portfolio_mode,
                severity="ERROR",
                type="NO_FILL_TIMEOUT",
                action_taken="HALT",
                ftmo_day_id=ftmo_day_id,
                instrument=intent.instrument,
                details="Order accepted but no position detected within timeout",
                intent_id=intent.intent_id,
                strategy_id=intent.strategy_id,
                broker_order_id=res.broker_order_id,
            )
            self.store.mark_state(idempotency_key, state="FAILED_FINAL")
            return

        # Record fill
        self.store.mark_state(idempotency_key, state="FILLED", broker_position_id=pos.ticket)

        # NEW: snapshot entry equity/time for close-watcher PnL (Mode A)
        try:
            acc = self.mt5.get_account()
            entry_equity = float(acc.equity)
            entry_ts_utc = _iso(_now_utc())

            if hasattr(self.store, "update_fields"):
                # Preferred (generic) API
                self.store.update_fields(idempotency_key, entry_equity=entry_equity, entry_ts_utc=entry_ts_utc)
            else:
                # Fallback: try mark_state with extra fields if your store supports it
                try:
                    self.store.mark_state(idempotency_key, state="FILLED", entry_equity=entry_equity, entry_ts_utc=entry_ts_utc)
                except TypeError:
                    # Store doesn't support extra fields; log and continue
                    self.incidents_log.log(
                        run_id=self.run_id,
                        run_mode=self.run_mode,
                        portfolio_mode=self.portfolio_mode,
                        severity="WARN",
                        type="ENTRY_EQUITY_NOT_PERSISTED",
                        action_taken="IGNORE",
                        ftmo_day_id=ftmo_day_id,
                        instrument=intent.instrument,
                        details="IdempotencyStore has no update_fields() and mark_state() doesn't accept extra fields.",
                        intent_id=intent.intent_id,
                        strategy_id=intent.strategy_id,
                        broker_position_id=pos.ticket,
                    )
        except Exception as e:
            self.incidents_log.log(
                run_id=self.run_id,
                run_mode=self.run_mode,
                portfolio_mode=self.portfolio_mode,
                severity="WARN",
                type="ENTRY_EQUITY_STORE_FAILED",
                action_taken="IGNORE",
                ftmo_day_id=ftmo_day_id,
                instrument=intent.instrument,
                details=str(e),
                intent_id=intent.intent_id,
                strategy_id=intent.strategy_id,
                broker_position_id=pos.ticket,
            )

        self.orders_log.append_event(
            run_id=self.run_id,
            run_mode=self.run_mode,
            portfolio_mode=self.portfolio_mode,
            ftmo_day_id=ftmo_day_id,
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            strategy_version=intent.strategy_version,
            instrument=intent.instrument,
            timeframe=intent.timeframe,
            side=intent.side,
            order_type=intent.order_type,
            time_in_force=intent.tif,
            risk_mode=intent.risk_mode,
            risk_value_pct=float(intent.risk_value_pct),
            requested_sl_pips=float(intent.sl_pips),
            requested_tp_pips=float(intent.tp_pips),
            max_hold_min=int(intent.max_hold_min),
            valid_to_utc=intent.valid_to_utc,
            idempotency_key=idempotency_key,
            client_order_id=client_order_id,
            broker_order_id=res.broker_order_id,
            broker_position_id=pos.ticket,
            event_type="ORDER_FILL",
            event_state="FILLED",
            event_reason="POSITION_DETECTED",
            filled_qty=pos.volume,
            fill_price=pos.price_open,
            fill_ts_utc=_iso(pos.time_utc),
            sl_price=pos.sl,
            tp_price=pos.tp,
        )

        # Verify SL/TP; if missing, attempt attach (v1: failsafe close)
        if not self._has_sltp(pos):
            self._failsafe_close_position(ftmo_day_id, intent, pos, reason="MISSING_SLTP_AFTER_FILL")
            self.store.mark_state(idempotency_key, state="FAILED_FINAL")
            return

        self.store.mark_state(idempotency_key, state="OPEN_VERIFIED")

        # Done for manual test; do not manage exit here
        self.orders_log.append_event(
            run_id=self.run_id,
            run_mode=self.run_mode,
            portfolio_mode=self.portfolio_mode,
            ftmo_day_id=ftmo_day_id,
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            strategy_version=intent.strategy_version,
            instrument=intent.instrument,
            timeframe=intent.timeframe,
            side=intent.side,
            order_type=intent.order_type,
            time_in_force=intent.tif,
            risk_mode=intent.risk_mode,
            risk_value_pct=float(intent.risk_value_pct),
            requested_sl_pips=float(intent.sl_pips),
            requested_tp_pips=float(intent.tp_pips),
            max_hold_min=int(intent.max_hold_min),
            valid_to_utc=intent.valid_to_utc,
            idempotency_key=idempotency_key,
            client_order_id=client_order_id,
            broker_order_id=res.broker_order_id,
            broker_position_id=pos.ticket,
            event_type="OPEN_VERIFIED",
            event_state="OK",
            event_reason="SLTP_PRESENT",
            sl_price=pos.sl,
            tp_price=pos.tp,
        )

    # ---------- helpers ----------

    def _wait_for_position(self, ftmo_day_id: str, symbol: str, timeout_ms: int) -> Optional[Position]:
        deadline = time.time() + (timeout_ms / 1000.0)
        while time.time() < deadline:
            bs = self.sync.reconcile(ftmo_day_id=ftmo_day_id)
            if bs.position_count > 0:
                return bs.positions[0]
            time.sleep(0.25)
        return None

    @staticmethod
    def _prices_from_pips(entry_price: float, side: str, sl_pips: float, tp_pips: float) -> tuple[float, float]:
        pip = 0.0001  # EURUSD
        if side == "LONG":
            sl = entry_price - sl_pips * pip
            tp = entry_price + tp_pips * pip
        else:
            sl = entry_price + sl_pips * pip
            tp = entry_price - tp_pips * pip
        return (round(sl, 5), round(tp, 5))

    @staticmethod
    def _has_sltp(pos: Position) -> bool:
        return (pos.sl is not None and pos.sl > 0) and (pos.tp is not None and pos.tp > 0)

    def _failsafe_close_position(self, ftmo_day_id: str, intent: ManualIntent, pos: Position, reason: str) -> bool:
        side = "SHORT" if intent.side == "LONG" else "LONG"
        res = self.mt5.send_market_order(
            symbol=pos.symbol,
            side=side,
            volume=float(pos.volume),
            sl_price=None,
            tp_price=None,
            comment=None,
            magic=None,
            deviation_points=self.deviation_points,
        )
        self.orders_log.append_event(
            run_id=self.run_id,
            run_mode=self.run_mode,
            portfolio_mode=self.portfolio_mode,
            ftmo_day_id=ftmo_day_id,
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            strategy_version=intent.strategy_version,
            instrument=pos.symbol,
            timeframe=intent.timeframe,
            side=side,
            order_type="MARKET",
            time_in_force=intent.tif,
            risk_mode=intent.risk_mode,
            risk_value_pct=float(intent.risk_value_pct),
            requested_sl_pips=float(intent.sl_pips),
            requested_tp_pips=float(intent.tp_pips),
            max_hold_min=int(intent.max_hold_min),
            valid_to_utc=intent.valid_to_utc,
            idempotency_key="",
            client_order_id="",
            broker_position_id=pos.ticket,
            event_type="POSITION_CLOSE",
            event_state="CLOSED" if res.ok else "FAILED",
            event_reason=reason,
            filled_qty=pos.volume,
            last_error_code=res.retcode if not res.ok else None,
            last_error_msg=res.comment if not res.ok else None,
        )
        self.incidents_log.log(
            run_id=self.run_id,
            run_mode=self.run_mode,
            portfolio_mode=self.portfolio_mode,
            severity="CRITICAL",
            type="FAILSAFE_CLOSE",
            action_taken="FORCE_CLOSE",
            ftmo_day_id=ftmo_day_id,
            instrument=pos.symbol,
            details=reason,
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            broker_position_id=pos.ticket,
        )
        return res.ok

    def _log_sync(self, ftmo_day_id: str, intent: ManualIntent, idempotency_key: str, client_order_id: str, reason: str) -> None:
        self.orders_log.append_event(
            run_id=self.run_id,
            run_mode=self.run_mode,
            portfolio_mode=self.portfolio_mode,
            ftmo_day_id=ftmo_day_id,
            intent_id=intent.intent_id,
            strategy_id=intent.strategy_id,
            strategy_version=intent.strategy_version,
            instrument=intent.instrument,
            timeframe=intent.timeframe,
            side=intent.side,
            order_type=intent.order_type,
            time_in_force=intent.tif,
            risk_mode=intent.risk_mode,
            risk_value_pct=float(intent.risk_value_pct),
            requested_sl_pips=float(intent.sl_pips),
            requested_tp_pips=float(intent.tp_pips),
            max_hold_min=int(intent.max_hold_min),
            valid_to_utc=intent.valid_to_utc,
            idempotency_key=idempotency_key,
            client_order_id=client_order_id,
            event_type="SYNC_RECONCILE",
            event_state="RECOVER",
            event_reason=reason,
        )