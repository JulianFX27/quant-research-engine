from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from src.data.providers import CSVBarProvider, CSVProviderConfig, DataValidationError
from src.execution.paper_engine import PaperEngine
from src.risk.risk_state_manager import RiskStateManager, RiskConfig
from src.runner.interfaces import StrategyContext, OrderIntent
from src.runner.config_loader import load_config
from src.runner.run_id import file_sha256, make_run_id
from src.runner.artifacts import (
    ensure_dir,
    write_json,
    append_log,
    append_trade_csv,
    append_equity_csv,
    upsert_daily_summary,
)

from src.strategies.anchor_adapter import AnchorReversionAdapter, AnchorAdapterConfig


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def throttle_multiplier_from_schedule(dd_pct: float, schedule: list[dict]) -> float:
    """
    schedule items: {dd_min_pct, dd_max_pct, risk_multiplier}
    dd_pct in percent units (e.g., 3.2)
    """
    for row in schedule:
        if dd_pct >= float(row["dd_min_pct"]) and dd_pct < float(row["dd_max_pct"]):
            return float(row["risk_multiplier"])
    return 0.0


def _intent_with_meta(intent: OrderIntent, extra_meta: Dict[str, Any]) -> OrderIntent:
    """
    OrderIntent is frozen. Create a new instance with merged meta.
    Uses dataclasses.replace for safety.
    """
    merged = dict(intent.meta or {})
    merged.update(extra_meta)
    return replace(intent, meta=merged)


def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--csv", required=True, help="Path to EURUSD M5 CSV (with OHLC + time + extras)")
    ap.add_argument("--time_col", default="time")
    ap.add_argument("--tz_in", default="auto", help="auto OR IANA tz like America/Bogota")
    ap.add_argument("--assume_utc_if_naive", action="store_true")

    # Policy + state
    ap.add_argument("--execution_policy", required=True, help="YAML execution policy path")
    ap.add_argument("--state_path", required=True, help="state/daily_risk_state.json")
    ap.add_argument("--override_path", required=True, help="state/manual_override.json")
    ap.add_argument("--account_mode", choices=["challenge", "funded"], required=True)

    # Results
    ap.add_argument("--results_dir", default="results/runs")
    ap.add_argument("--run_tag", default=None)

    # Anchor adapter params
    ap.add_argument("--anchor_col", default="ny_open")  # IMPORTANT: your CSV uses ny_open
    ap.add_argument("--entry_threshold_pips", type=float, default=8.0)
    ap.add_argument("--exit_threshold_pips", type=float, default=0.0)
    ap.add_argument("--sl_pips", type=float, default=20.0)
    ap.add_argument("--tp_pips", type=float, default=12.0)
    ap.add_argument("--warmup_bars", type=int, default=0)
    ap.add_argument("--max_hold_bars", type=int, default=24)
    ap.add_argument("--pip_size", type=float, default=0.0001)

    args = ap.parse_args()

    # Load execution policy
    exec_cfg = load_config(args.execution_policy)
    ep = exec_cfg["execution_policy"]

    policy_name = ep["name"]
    policy_hash = file_sha256(args.execution_policy)
    run_id = make_run_id(args.run_tag, policy_hash)

    run_dir = Path(args.results_dir) / run_id
    ensure_dir(str(run_dir))

    paths = {
        "manifest": str(run_dir / "run_manifest.json"),
        "trades": str(run_dir / "trades.csv"),
        "equity": str(run_dir / "equity.csv"),
        "daily": str(run_dir / "daily_summary.csv"),
        "logs": str(run_dir / "logs.txt"),
    }

    append_log(paths["logs"], f"[{utc_now_iso()}] run_id={run_id} policy={policy_name} account_mode={args.account_mode}")
    append_log(paths["logs"], f"[{utc_now_iso()}] execution_policy_sha256={policy_hash}")
    append_log(paths["logs"], f"[{utc_now_iso()}] csv={args.csv} time_col={args.time_col} tz_in={args.tz_in} assume_utc_if_naive={args.assume_utc_if_naive}")
    append_log(paths["logs"], f"[{utc_now_iso()}] anchor_col={args.anchor_col} entry_thr_pips={args.entry_threshold_pips} exit_thr_pips={args.exit_threshold_pips} sl_pips={args.sl_pips} tp_pips={args.tp_pips}")

    # Risk manager config
    tz_name = ep.get("timezone_day_rollover", "Europe/Prague")

    # YAML stores internal_limits as decimals (0.015 = 1.5%, 0.08 = 8%)
    daily_stop_abs_pct = float(ep["internal_limits"]["daily_stop_pct"]) * 100.0          # 1.5
    hard_stop_dd_pct = float(ep["internal_limits"]["hard_stop_total_dd_pct"]) * 100.0    # 8.0

    rsm = RiskStateManager(
        state_path=args.state_path,
        override_path=args.override_path,
        config=RiskConfig(
            tz_name=tz_name,
            daily_stop_pct=-abs(daily_stop_abs_pct),  # -1.5
            hard_stop_dd_pct=hard_stop_dd_pct,        # 8.0
            max_audit=200,
        ),
    )

    # Ensure state is loadable and consistent
    state = rsm.load_state()
    state["account_mode"] = args.account_mode
    rsm.save_state(state)

    # Provider (extras enabled in CSVBarProvider)
    prov = CSVBarProvider(CSVProviderConfig(
        csv_path=args.csv,
        time_col=args.time_col,
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        volume_col="tick_volume" if "tick_volume" else None,  # if missing, provider will ignore
        tz_in=args.tz_in,
        assume_utc_if_naive=args.assume_utc_if_naive,
    ))

    # Engine
    engine = PaperEngine(
        intrabar_path=ep["order_execution"]["intrabar"]["path_mode"],
        tie_break=ep["order_execution"]["intrabar"]["tie_break"],
        allow_same_bar_exit=True,
    )

    # Strategy (Anchor adapter using ny_open)
    strat = AnchorReversionAdapter(
        AnchorAdapterConfig(
            anchor_col=args.anchor_col,
            entry_threshold_pips=args.entry_threshold_pips,
            exit_threshold_pips=args.exit_threshold_pips,
            sl_pips=args.sl_pips,
            tp_pips=args.tp_pips,
            warmup_bars=args.warmup_bars,
            max_hold_bars=args.max_hold_bars,
            tag="ANCHOR_ADAPTER_V1",
        )
    )

    instrument = ep.get("instrument", "EURUSD")

    # Policy params
    r_base = float(ep["risk_mode"][args.account_mode]["base_risk_pct_per_trade"]) / 100.0  # 0.50 -> 0.005
    throttle_schedule = ep["throttle"]["schedule"]

    # Track max intraday dd by FTMO day
    max_intraday_dd_by_day: Dict[str, float] = {}

    # Manifest skeleton
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": utc_now_iso(),
        "ended_at_utc": None,
        "mode": "paper",
        "account_mode": args.account_mode,
        "instrument": instrument,
        "strategy_name": getattr(strat, "name", strat.__class__.__name__),
        "strategy_version": getattr(strat, "version", "unknown"),
        "policy_name": policy_name,
        "execution_policy_path": args.execution_policy,
        "execution_policy_sha256": policy_hash,
        "state_path": args.state_path,
        "override_path": args.override_path,
        "engine": {
            "fill_mode": ep["order_execution"]["fill_mode"],
            "intrabar_path": ep["order_execution"]["intrabar"]["path_mode"],
            "tie_break": ep["order_execution"]["intrabar"]["tie_break"],
        },
        "strategy_params": {
            "anchor_col": args.anchor_col,
            "entry_threshold_pips": args.entry_threshold_pips,
            "exit_threshold_pips": args.exit_threshold_pips,
            "sl_pips": args.sl_pips,
            "tp_pips": args.tp_pips,
            "warmup_bars": args.warmup_bars,
            "max_hold_bars": args.max_hold_bars,
            "pip_size": args.pip_size,
        },
        "events": {
            "bars": 0,
            "intents": 0,
            "entries_submitted": 0,
            "fills": 0,
            "forced_exits": 0,
            "closes": 0,
            "day_rollovers": 0,
        },
    }
    write_json(paths["manifest"], manifest)

    def close_and_persist(trade, bar_ts_utc: datetime, reason_hint: str):
        nonlocal state

        # Reset strategy after close (TP/SL/forced)
        if hasattr(strat, "on_trade_closed_reset"):
            strat.on_trade_closed_reset()

        manifest["events"]["closes"] += 1

        state = rsm.load_state()
        equity_after = float(state["equity_current"]) * (1.0 + float(trade.pnl))
        state = rsm.on_trade_closed(state, equity_after=equity_after, run_id=run_id, now_utc=bar_ts_utc)
        rsm.save_state(state)

        day_id2 = state["current_day_id"] or rsm.current_day_id(bar_ts_utc)

        append_trade_csv(
            paths["trades"],
            trade,
            day_id_ftmo=day_id2,
            dd_at_entry_pct=float(trade.meta.get("_dd_at_entry_pct", 0.0)),
            dd_at_exit_pct=float(state.get("dd_current_pct", 0.0)),
            daily_pnl_pct_at_entry=float(trade.meta.get("_daily_pnl_pct_at_entry", 0.0)),
        )

        equity_start_day = float(state.get("equity_start_day") or state["equity_current"])
        equity_end_day = float(state["equity_current"])
        daily_pnl_pct = float(state.get("daily_pnl_pct", 0.0))
        upsert_daily_summary(
            paths["daily"],
            day_id_ftmo=day_id2,
            equity_start_day=equity_start_day,
            equity_end_day=equity_end_day,
            daily_pnl_pct=daily_pnl_pct,
            max_intraday_dd_pct=max_intraday_dd_by_day.get(day_id2, 0.0),
            trades_taken=int(state.get("trades_taken_today", 0)),
            daily_stop_triggered=bool(state.get("daily_stop_triggered", False)),
            hard_stop_triggered=bool(state.get("hard_stop_dd_triggered", False)),
        )

        append_log(paths["logs"], f"[{utc_now_iso()}] CLOSE trade_id={trade.trade_id} reason={trade.exit_reason} pnl={trade.pnl:.6f} R={trade.R:.3f} {reason_hint}")

    # Main loop
    try:
        for bar in prov.iter_bars():
            manifest["events"]["bars"] += 1

            # Load + rollover + optional manual reset
            state = rsm.load_state()
            before_day = state.get("current_day_id")
            state = rsm.rollover_if_needed(state, run_id=run_id, now_utc=bar.ts_utc)
            after_day = state.get("current_day_id")
            if before_day != after_day:
                manifest["events"]["day_rollovers"] += 1
                append_log(paths["logs"], f"[{utc_now_iso()}] DAY_ROLLOVER day_id={after_day}")

            state = rsm.try_manual_reset_hard_stop(state, run_id=run_id, now_utc=bar.ts_utc)
            rsm.save_state(state)

            day_id = state["current_day_id"] or rsm.current_day_id(bar.ts_utc)

            # Track max intraday dd
            max_intraday_dd_by_day[day_id] = max(
                max_intraday_dd_by_day.get(day_id, 0.0),
                float(state.get("dd_current_pct", 0.0)),
            )

            # Equity point per bar
            append_equity_csv(
                paths["equity"],
                ts_utc=bar.ts_utc,
                equity=float(state["equity_current"]),
                equity_peak=float(state["equity_peak"]),
                dd_pct=float(state["dd_current_pct"]),
                day_id_ftmo=day_id,
            )

            # Context
            ctx = StrategyContext(
                day_id_ftmo=day_id,
                equity_current=float(state["equity_current"]),
                dd_current_pct=float(state["dd_current_pct"]),
                trades_taken_today=int(state["trades_taken_today"]),
                trading_enabled=bool(state["trading_enabled"]),
                account_mode=args.account_mode,
                instrument=instrument,
                pip_size=float(args.pip_size),
            )

            # ----------------------------
            # 1) Pending intent fill (next_open)
            # ----------------------------
            if engine.has_pending_intent() and (not engine.has_open_position()):
                before_fill = engine.has_open_position()
                closed = engine.on_bar(bar)  # fills at bar.open
                after_fill = engine.has_open_position()
                if (not before_fill) and after_fill:
                    manifest["events"]["fills"] += 1
                    append_log(paths["logs"], f"[{utc_now_iso()}] FILL opened_at={bar.ts_utc.isoformat().replace('+00:00','Z')}")
                # If it somehow also closed same bar:
                if closed is not None:
                    close_and_persist(closed, bar.ts_utc, reason_hint="(filled+closed)")
                continue

            # ----------------------------
            # 2) Open position management
            # ----------------------------
            if engine.has_open_position():
                # Strategy may request discretionary EXIT (time-stop / anchor-touch)
                intent = strat.on_bar(bar, ctx)
                if intent is not None:
                    manifest["events"]["intents"] += 1

                    if intent.action == "EXIT":
                        closed = engine.force_close(
                            ts_utc=bar.ts_utc,
                            price=float(bar.close),
                            reason=intent.exit_reason or "FORCE_EXIT",
                        )
                        if closed is not None:
                            manifest["events"]["forced_exits"] += 1
                            close_and_persist(closed, bar.ts_utc, reason_hint="(forced)")
                        continue

                # Otherwise engine handles TP/SL intrabar
                closed = engine.on_bar(bar)
                if closed is not None:
                    close_and_persist(closed, bar.ts_utc, reason_hint="(tp/sl)")
                continue

            # ----------------------------
            # 3) No open position, no pending intent -> entry evaluation
            # ----------------------------
            if not rsm.can_trade(state, run_id=run_id, now_utc=bar.ts_utc):
                continue

            intent = strat.on_bar(bar, ctx)
            if intent is None:
                continue

            manifest["events"]["intents"] += 1

            # Ignore EXIT if no position (safety)
            if intent.action == "EXIT":
                continue

            # ENTER validation
            if intent.direction is None or intent.sl_price is None or intent.tp_price is None:
                # Runner v1 requires bracket orders for deterministic R + intrabar exits
                continue

            # Compute effective risk (DD throttle)
            dd_pct = float(state["dd_current_pct"])
            mult = throttle_multiplier_from_schedule(dd_pct, throttle_schedule)
            r_eff = r_base * mult
            if r_eff <= 0:
                continue

            # Add audit meta (without mutation)
            intent = _intent_with_meta(intent, {
                "_dd_at_entry_pct": dd_pct,
                "_daily_pnl_pct_at_entry": float(state.get("daily_pnl_pct", 0.0)),
            })

            engine.submit_intent(
                intent,
                risk_base_pct=r_base,
                risk_multiplier=mult,
                risk_effective_pct=r_eff,
            )
            manifest["events"]["entries_submitted"] += 1

            append_log(
                paths["logs"],
                f"[{utc_now_iso()}] ENTER_INTENT id={intent.intent_id} dir={intent.direction} "
                f"sl={float(intent.sl_price):.6f} tp={float(intent.tp_price):.6f} "
                f"r_eff={r_eff:.6f} dd={dd_pct:.2f}%",
            )

    except DataValidationError as e:
        append_log(paths["logs"], f"[{utc_now_iso()}] DATA_VALIDATION_ERROR: {e}")
        raise
    finally:
        manifest["ended_at_utc"] = utc_now_iso()
        write_json(paths["manifest"], manifest)
        append_log(
            paths["logs"],
            f"[{utc_now_iso()}] RUN_END run_id={run_id} bars={manifest['events']['bars']} "
            f"closes={manifest['events']['closes']} entries_submitted={manifest['events']['entries_submitted']} fills={manifest['events']['fills']}",
        )

    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
