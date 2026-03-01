from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from src.live.mt5.adapter import MT5Adapter
from src.live.state.idempotency_store import IdempotencyStore
from src.live.io.orders_log import OrdersLog
from src.live.io.incidents_log import IncidentsLog


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class CloseDetected:
    idempotency_key: str
    intent_id: str
    instrument: str
    broker_position_id: int
    close_ts_utc: str
    approx_profit_usd: float
    reason: str  # UNKNOWN (for now)


class CloseWatcher:
    """
    Finalizes trades without relying on MT5 history API.
    In Mode A (single position), we can approximate profit by equity delta.

    Logic:
      - Find store rows with broker_position_id and state OPEN_VERIFIED (or FILLED/ATTACHED)
      - If position_id not present in live positions -> consider it CLOSED
      - Mark store CLOSED and append POSITION_CLOSED event
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

    def finalize_if_closed(self, *, ftmo_day_id: str) -> int:
        rows = self.store.list_recent(limit=3000)
        openish = [r for r in rows if r.get("state") in ("FILLED", "ATTACHED", "OPEN_VERIFIED")]

        if not openish:
            return 0

        live_positions = self.mt5.positions(self.instrument)
        live_pos_ids = {int(p.ticket) for p in live_positions}

        acc = self.mt5.get_account()
        equity_now = float(acc.equity)

        finalized = 0

        for r in openish:
            idem_key = str(r.get("idempotency_key", ""))
            intent_id = str(r.get("intent_id", ""))
            pos_id = r.get("broker_position_id")
            if not pos_id:
                continue
            pos_id = int(pos_id)

            if pos_id in live_pos_ids:
                continue  # still open

            # Closed detected
            entry_equity = float(r.get("entry_equity", equity_now))  # fallback
            approx_profit = equity_now - entry_equity

            cd = CloseDetected(
                idempotency_key=idem_key,
                intent_id=intent_id,
                instrument=self.instrument,
                broker_position_id=pos_id,
                close_ts_utc=_now_utc_iso(),
                approx_profit_usd=float(approx_profit),
                reason="UNKNOWN",
            )

            # mark CLOSED
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

            # log close
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
                event_reason=f"{cd.reason}|approx_profit_usd={cd.approx_profit_usd}",
                fill_ts_utc=cd.close_ts_utc,
            )

            finalized += 1

        return finalized