from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from src.data.providers import CSVBarProvider, CSVProviderConfig, DataValidationError
from src.execution.paper_engine import PaperEngine
from src.paper.risk.risk_state_manager import RiskStateManager, RiskConfig

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
    for row in schedule:
        if dd_pct >= float(row["dd_min_pct"]) and dd_pct < float(row["dd_max_pct"]):
            return float(row["risk_multiplier"])
    return 0.0


def _intent_with_meta(intent: OrderIntent, extra_meta: Dict[str, Any]) -> OrderIntent:
    merged = dict(intent.meta or {})
    merged.update(extra_meta)
    return replace(intent, meta=merged)


def _compute_equity_after(equity_before: float, trade_pnl: float, pnl_mode: str) -> float:
    if pnl_mode == "return":
        return float(equity_before) * (1.0 + float(trade_pnl))
    return float(equity_before) + float(trade_pnl)


def _shape_hint(obj: Any) -> str:
    if not isinstance(obj, dict):
        return f"type={type(obj)}"
    return f"dict_keys={sorted(list(obj.keys()))}"


def _resolve_execution_policy(exec_cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Returns (ep_full, meta, dbg)

    Your runtime shape (from logs):
      root has: internal_limits, order_execution, risk_mode, throttle, trade_controls, ...
      and root["execution_policy"] has: name, instrument, timezone_day_rollover, etc.
    """
    dbg: Dict[str, Any] = {"exec_cfg_shape": _shape_hint(exec_cfg)}

    if not isinstance(exec_cfg, dict):
        raise TypeError(f"Execution policy config must be a dict, got {type(exec_cfg)}")

    # Case: flattened/root policy (YOUR CURRENT REALITY)
    if ("internal_limits" in exec_cfg) and ("order_execution" in exec_cfg) and ("risk_mode" in exec_cfg):
        meta = exec_cfg.get("execution_policy", {})
        if not isinstance(meta, dict):
            meta = {}
        dbg["selected"] = "ROOT_FLATTENED"
        dbg["meta_shape"] = _shape_hint(meta)
        return exec_cfg, meta, dbg

    # Case: canonical YAML shape
    if "execution_policy" in exec_cfg and isinstance(exec_cfg["execution_policy"], dict):
        ep = exec_cfg["execution_policy"]
        dbg["selected"] = "CANONICAL"
        dbg["meta_shape"] = _shape_hint(ep)
        return ep, ep, dbg

    dbg["selected"] = "UNKNOWN"
    return exec_cfg, {}, dbg


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True)
    ap.add_argument("--time_col", default="time")
    ap.add_argument("--tz_in", default="auto")
    ap.add_argument("--assume_utc_if_naive", action="store_true")

    ap.add_argument("--execution_policy", required=True)
    ap.add_argument("--state_path", required=True)
    ap.add_argument("--override_path", required=True)
    ap.add_argument("--account_mode", choices=["challenge", "funded"], required=True)

    ap.add_argument("--results_dir", default="results/runs")
    ap.add_argument("--run_tag", default=None)

    ap.add_argument("--anchor_col", default="ny_open")
    ap.add_argument("--entry_threshold_pips", type=float, default=8.0)
    ap.add_argument("--exit_threshold_pips", type=float, default=0.0)
    ap.add_argument("--sl_pips", type=float, default=20.0)
    ap.add_argument("--tp_pips", type=float, default=12.0)
    ap.add_argument("--warmup_bars", type=int, default=0)
    ap.add_argument("--max_hold_bars", type=int, default=24)
    ap.add_argument("--pip_size", type=float, default=0.0001)

    args = ap.parse_args()

    # Create run dir + logs early
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

    append_log(paths["logs"], f"[{utc_now_iso()}] run_id={run_id}")
    append_log(paths["logs"], f"[{utc_now_iso()}] execution_policy_path={args.execution_policy}")
    append_log(paths["logs"], f"[{utc_now_iso()}] execution_policy_sha256={policy_hash}")
    append_log(paths["logs"], f"[{utc_now_iso()}] csv={args.csv}")

    exec_cfg = load_config(args.execution_policy)
    ep, meta, dbg = _resolve_execution_policy(exec_cfg)

    append_log(paths["logs"], f"[{utc_now_iso()}] exec_cfg_shape={dbg.get('exec_cfg_shape')}")
    append_log(paths["logs"], f"[{utc_now_iso()}] policy_shape_selected={dbg.get('selected')}")
    append_log(paths["logs"], f"[{utc_now_iso()}] ep_shape={_shape_hint(ep)}")
    append_log(paths["logs"], f"[{utc_now_iso()}] meta_shape={dbg.get('meta_shape')}")

    policy_name = str(meta.get("name", ep.get("name", "UNKNOWN_POLICY_NAME")))
    append_log(paths["logs"], f"[{utc_now_iso()}] policy_name={policy_name} account_mode={args.account_mode}")

    # Required sections (strict)
    if "internal_limits" not in ep:
        raise KeyError("internal_limits not found in execution policy (see logs.txt)")
    if "order_execution" not in ep:
        raise KeyError("order_execution not found in execution policy (see logs.txt)")
    if "risk_mode" not in ep:
        raise KeyError("risk_mode not found in execution policy (see logs.txt)")
    if "throttle" not in ep:
        raise KeyError("throttle not found in execution policy (see logs.txt)")

    tz_name = str(meta.get("timezone_day_rollover", ep.get("timezone_day_rollover", "Europe/Prague")))
    instrument = str(meta.get("instrument", ep.get("instrument", "EURUSD")))

    max_daily_loss_pct = float(ep["internal_limits"]["daily_stop_pct"]) * 100.0
    max_overall_loss_pct = float(ep["internal_limits"]["hard_stop_total_dd_pct"]) * 100.0

    oe = ep["order_execution"]
    intrabar_path_mode = oe["intrabar"]["path_mode"]
    intrabar_tie_break = oe["intrabar"]["tie_break"]
    fill_mode = oe.get("fill_mode", "next_open")

    trade_controls = ep.get("trade_controls", {})
    max_trades_per_day = int(trade_controls.get("max_trades_per_day", 1))

    pnl_mode = str(ep.get("paper", {}).get("pnl_mode", "absolute")).strip().lower()
    if pnl_mode not in ("absolute", "return"):
        pnl_mode = "absolute"

    append_log(paths["logs"], f"[{utc_now_iso()}] tz_day_rollover={tz_name}")
    append_log(
        paths["logs"],
        f"[{utc_now_iso()}] internal_limits daily_stop={max_daily_loss_pct:.3f}% hard_stop_dd={max_overall_loss_pct:.3f}%",
    )
    append_log(
        paths["logs"],
        f"[{utc_now_iso()}] order_execution fill_mode={fill_mode} intrabar_path={intrabar_path_mode} tie={intrabar_tie_break}",
    )
    append_log(paths["logs"], f"[{utc_now_iso()}] trade_controls max_trades_per_day={max_trades_per_day} pnl_mode={pnl_mode}")

    # Risk manager
    rsm = RiskStateManager(
        state_path=args.state_path,
        override_path=args.override_path,
        config=RiskConfig(
            tz_name=tz_name,
            max_daily_loss_pct=float(abs(max_daily_loss_pct)),
            max_overall_loss_pct=float(abs(max_overall_loss_pct)),
            max_audit=200,
        ),
    )

    state = rsm.load_state()
    state["account_mode"] = args.account_mode
    state["max_trades_per_day"] = max_trades_per_day
    rsm.save_state(state)

    prov = CSVBarProvider(
        CSVProviderConfig(
            csv_path=args.csv,
            time_col=args.time_col,
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            volume_col=None,
            tz_in=args.tz_in,
            assume_utc_if_naive=args.assume_utc_if_naive,
        )
    )

    engine = PaperEngine(
        intrabar_path=intrabar_path_mode,
        tie_break=intrabar_tie_break,
        allow_same_bar_exit=True,
    )

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

    r_base = float(ep["risk_mode"][args.account_mode]["base_risk_pct_per_trade"]) / 100.0
    throttle_schedule = ep["throttle"]["schedule"]

    max_intraday_dd_by_day: Dict[str, float] = {}

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": utc_now_iso(),
        "ended_at_utc": None,
        "mode": "paper",
        "account_mode": args.account_mode,
        "instrument": instrument,
        "policy_name": policy_name,
        "execution_policy_path": args.execution_policy,
        "execution_policy_sha256": policy_hash,
        "state_path": args.state_path,
        "override_path": args.override_path,
        "engine": {
            "fill_mode": fill_mode,
            "intrabar_path": intrabar_path_mode,
            "intrabar_tie_break": intrabar_tie_break,
        },
        "paper": {
            "pnl_mode": pnl_mode,
            "internal_limits_pct": {
                "max_daily_loss_pct": float(abs(max_daily_loss_pct)),
                "max_overall_loss_pct": float(abs(max_overall_loss_pct)),
            },
            "trade_controls": {"max_trades_per_day": max_trades_per_day},
        },
        "events": {"bars": 0, "intents": 0, "entries_submitted": 0, "fills": 0, "forced_exits": 0, "closes": 0, "day_rollovers": 0},
    }
    write_json(paths["manifest"], manifest)

    def close_and_persist(trade, bar_ts_utc: datetime, reason_hint: str) -> None:
        nonlocal state

        if hasattr(strat, "on_trade_closed_reset"):
            strat.on_trade_closed_reset()

        manifest["events"]["closes"] += 1

        state = rsm.load_state()
        equity_before = float(state["equity_current"])
        equity_after = _compute_equity_after(equity_before, float(trade.pnl), pnl_mode=pnl_mode)

        state = rsm.on_trade_closed(state, equity_after=equity_after, run_id=run_id, now_utc=bar_ts_utc)
        rsm.save_state(state)

        # Persist the FTMO day_id as the DAY OF ENTRY, not close.
        entry_day_id = None
        try:
            entry_day_id = (getattr(trade, "meta", None) or {}).get("_day_id_ftmo_entry")
        except Exception:
            entry_day_id = None

        day_id2 = entry_day_id or state.get("current_day_id") or rsm.current_day_id(bar_ts_utc)

        append_trade_csv(
            paths["trades"],
            trade,
            day_id_ftmo=day_id2,
            dd_at_entry_pct=float((getattr(trade, "meta", None) or {}).get("_dd_at_entry_pct", 0.0)),
            dd_at_exit_pct=float(state.get("dd_from_peak_pct", 0.0)),
            daily_pnl_pct_at_entry=float((getattr(trade, "meta", None) or {}).get("_daily_pnl_pct_at_entry", 0.0)),
        )

        trades_taken_for_day = 1 if entry_day_id else int(state.get("trades_taken_today", 0))

        upsert_daily_summary(
            paths["daily"],
            day_id_ftmo=day_id2,
            equity_start_day=float(state.get("equity_start_day") or state["equity_current"]),
            equity_end_day=float(state["equity_current"]),
            daily_pnl_pct=float(state.get("daily_pnl_pct", 0.0)),
            max_intraday_dd_pct=max_intraday_dd_by_day.get(day_id2, 0.0),
            trades_taken=int(trades_taken_for_day),
            daily_stop_triggered=bool(state.get("daily_stop_triggered", False)),
            hard_stop_triggered=bool(state.get("hard_stop_dd_triggered", False)),
        )

        append_log(
            paths["logs"],
            f"[{utc_now_iso()}] CLOSE trade_id={trade.trade_id} reason={trade.exit_reason} pnl={trade.pnl:.6f} R={trade.R:.3f} {reason_hint}",
        )

    last_bar = None  # <- needed for FORCE_EOF

    try:
        for bar in prov.iter_bars():
            last_bar = bar
            manifest["events"]["bars"] += 1

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

            max_intraday_dd_by_day[day_id] = max(
                max_intraday_dd_by_day.get(day_id, 0.0),
                float(state.get("dd_from_peak_pct", 0.0)),
            )

            append_equity_csv(
                paths["equity"],
                ts_utc=bar.ts_utc,
                equity=float(state["equity_current"]),
                equity_peak=float(state["equity_peak"]),
                dd_pct=float(state.get("dd_from_peak_pct", 0.0)),
                day_id_ftmo=day_id,
            )

            ctx = StrategyContext(
                day_id_ftmo=day_id,
                equity_current=float(state["equity_current"]),
                dd_current_pct=float(state.get("dd_from_peak_pct", 0.0)),
                trades_taken_today=int(state.get("trades_taken_today", 0)),
                trading_enabled=bool(state.get("trading_enabled", True)),
                account_mode=args.account_mode,
                instrument=instrument,
                pip_size=float(args.pip_size),
            )

            # Pending intent path: MUST count fills even if same-bar close.
            if engine.has_pending_intent() and (not engine.has_open_position()):
                closed = engine.on_bar(bar)

                # If engine produced a closed trade here, it implies fill occurred.
                if closed is not None:
                    manifest["events"]["fills"] += 1
                    state = rsm.load_state()
                    state = rsm.on_trade_opened(state, run_id=run_id, now_utc=bar.ts_utc)
                    rsm.save_state(state)

                    close_and_persist(closed, bar.ts_utc, reason_hint="(filled+closed)")
                    continue

                # Otherwise, if it filled and remains open, count fill normally.
                if engine.has_open_position():
                    manifest["events"]["fills"] += 1
                    state = rsm.load_state()
                    state = rsm.on_trade_opened(state, run_id=run_id, now_utc=bar.ts_utc)
                    rsm.save_state(state)

                continue

            if engine.has_open_position():
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

                closed = engine.on_bar(bar)
                if closed is not None:
                    close_and_persist(closed, bar.ts_utc, reason_hint="(tp/sl)")
                continue

            if not rsm.can_trade(state, run_id=run_id, now_utc=bar.ts_utc):
                continue

            intent = strat.on_bar(bar, ctx)
            if intent is None:
                continue

            manifest["events"]["intents"] += 1
            if intent.action == "EXIT":
                continue
            if intent.direction is None or intent.sl_price is None or intent.tp_price is None:
                continue

            dd_pct = float(state.get("dd_from_peak_pct", 0.0))
            mult = throttle_multiplier_from_schedule(dd_pct, throttle_schedule)
            r_eff = r_base * mult
            if r_eff <= 0:
                continue

            # Stamp entry day_id (FTMO/Prague local date) into intent meta.
            intent = _intent_with_meta(
                intent,
                {
                    "_day_id_ftmo_entry": day_id,
                    "_dd_at_entry_pct": dd_pct,
                    "_daily_pnl_pct_at_entry": float(state.get("daily_pnl_pct", 0.0)),
                },
            )

            append_log(
                paths["logs"],
                f"[{utc_now_iso()}] SUBMIT "
                f"day_id_ftmo_entry={day_id} dd_from_peak_pct={dd_pct:.6f} "
                f"risk_mult={mult:.3f} r_base={r_base:.6f} r_eff={r_eff:.6f} "
                f"dir={intent.direction} sl={intent.sl_price} tp={intent.tp_price}"
            )
            
            engine.submit_intent(intent, risk_base_pct=r_base, risk_multiplier=mult, risk_effective_pct=r_eff)
            manifest["events"]["entries_submitted"] += 1

        # ------------------------------------------------------------
        # EOF HANDLING (CRITICAL FIX):
        # If there is an open position when the dataset ends, FORCE_EOF close it
        # on the last bar close, persist trade + update risk state.
        # This prevents risk_state.json from ending at TRADE_OPENED with no close.
        # ------------------------------------------------------------
        if last_bar is not None and engine.has_open_position():
            append_log(paths["logs"], f"[{utc_now_iso()}] EOF_DETECTED open_position=True -> FORCE_EOF close")
            closed = engine.force_close(
                ts_utc=last_bar.ts_utc,
                price=float(last_bar.close),
                reason="FORCE_EOF",
            )
            if closed is not None:
                manifest["events"]["forced_exits"] += 1
                close_and_persist(closed, last_bar.ts_utc, reason_hint="(forced_eof)")
            else:
                append_log(paths["logs"], f"[{utc_now_iso()}] EOF_FORCE_CLOSE returned None (unexpected)")

        # Optional: log pending intent at EOF (no state change)
        if last_bar is not None and engine.has_pending_intent() and (not engine.has_open_position()):
            append_log(paths["logs"], f"[{utc_now_iso()}] EOF_DETECTED pending_intent=True open_position=False (intent not filled before EOF)")

    except DataValidationError as e:
        append_log(paths["logs"], f"[{utc_now_iso()}] DATA_VALIDATION_ERROR: {e}")
        raise
    finally:
        manifest["ended_at_utc"] = utc_now_iso()
        write_json(paths["manifest"], manifest)

    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
