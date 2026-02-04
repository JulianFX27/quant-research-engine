# src/backtester/orchestrator/run.py
from __future__ import annotations

import hashlib
import json
import math
import yaml
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from backtester.core.config import validate_run_config
from backtester.data.dataset_fingerprint import build_dataset_id
from backtester.data.dataset_registry import register_or_validate_dataset
from backtester.data.loader import load_bars_csv
from backtester.execution.engine import SimpleBarEngine
from backtester.execution.policies import apply_execution_policy
from backtester.metrics.basic import summarize_trades, trades_to_dicts
from backtester.strategies.registry import make_strategy


def _sanitize_for_json(obj: Any) -> Any:
    """Avoid NaN/Infinity in JSON outputs."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _build_equity_curve(df: pd.DataFrame, trades: list[dict]) -> pd.DataFrame:
    """
    Minimal equity curve:
      - step equity at each trade exit_time (cum pnl)
      - if no trades: single flat point at first bar time (if available)
    """
    if trades:
        tdf = pd.DataFrame(trades)
        if "exit_time" in tdf.columns and "pnl" in tdf.columns:
            tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], utc=True, errors="coerce")
            tdf = tdf.dropna(subset=["exit_time"]).sort_values("exit_time")
            tdf["equity"] = tdf["pnl"].cumsum()
            out = tdf[["exit_time", "equity"]].rename(columns={"exit_time": "time"})
            return out.reset_index(drop=True)

    if len(df) > 0:
        t0 = df.index[0]
        return pd.DataFrame([{"time": t0, "equity": 0.0}])

    return pd.DataFrame(columns=["time", "equity"])


def _canonical_cfg_json(cfg: Dict[str, Any]) -> str:
    """Stable JSON string for hashing (sorted keys, compact)."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _make_run_id(cfg: Dict[str, Any]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    h = hashlib.sha256(_canonical_cfg_json(cfg).encode("utf-8")).hexdigest()[:8]
    return f"{ts}_{h}"


def _get_dataset_registry_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("dataset_registry", {}) or {}


def _get_registry_dir(cfg: Dict[str, Any]) -> str:
    dsreg = _get_dataset_registry_cfg(cfg)
    return str(dsreg.get("registry_dir") or "data/registry")


def _load_bars_and_register_dataset(cfg: Dict[str, Any]) -> tuple[pd.DataFrame, Any, str]:
    """
    Returns:
      (df, dataset_meta_final, dataset_id_final)

    Design:
      - loader may require a dataset_id; we pass provisional to compute metadata
      - then compute dataset_id_final from canonical start/end and rebind meta
      - only dataset_id_final is ever registered
    """
    instrument = cfg.get("instrument", {}) or {}
    symbol = str(cfg.get("symbol") or "UNKNOWN")
    timeframe = str(cfg.get("timeframe") or "UNKNOWN")
    source = str(instrument.get("data_source") or instrument.get("source") or "csv")

    dsreg = _get_dataset_registry_cfg(cfg)
    allow_override = bool(dsreg.get("allow_override", False))
    override_reason = str(dsreg.get("override_reason", "") or "")
    append_match_event = bool(dsreg.get("append_match_event", False))
    registry_dir = _get_registry_dir(cfg)

    Path(registry_dir).mkdir(parents=True, exist_ok=True)

    dataset_id_prov = build_dataset_id(
        instrument=symbol,
        timeframe=timeframe,
        start_ts="unknown",
        end_ts="unknown",
        source=source,
    )

    df, dataset_meta = load_bars_csv(
        cfg["data_path"],
        return_fingerprint=True,
        dataset_id=dataset_id_prov,
    )

    dataset_id_final = build_dataset_id(
        instrument=symbol,
        timeframe=timeframe,
        start_ts=str(dataset_meta.start_ts),
        end_ts=str(dataset_meta.end_ts),
        source=source,
    )

    dataset_meta_final = replace(dataset_meta, dataset_id=dataset_id_final)

    register_or_validate_dataset(
        dataset_meta_final,
        registry_dir=registry_dir,
        allow_new_fingerprint=allow_override,
        override_reason=override_reason,
        append_match_event=append_match_event,
    )

    return df, dataset_meta_final, dataset_id_final


def _flatten_entry_gate_into_metrics(metrics: Dict[str, Any], risk_report: Dict[str, Any]) -> None:
    """
    Flatten entry-gate counters into 'metrics' so leaderboard can compare gating regimes.

    Writes:
      - entry_gate_* (v1)
      - entry_gate_v2_* (guardrails v2)
      - per-reason wide columns: entry_gate_blocked_by_reason__<REASON>
      - per-reason wide columns: entry_gate_v2_blocked_by_reason__<REASON>
    """
    rr = risk_report or {}

    eg1 = (rr.get("entry_gate") or {})
    metrics["entry_gate_attempted_entries"] = int(eg1.get("attempted_entries", 0) or 0)
    metrics["entry_gate_blocked_total"] = int(eg1.get("blocked_total", 0) or 0)
    metrics["entry_gate_blocked_unique_bars"] = int(eg1.get("blocked_unique_bars", 0) or 0)

    bbr1 = eg1.get("blocked_by_reason") or {}
    if isinstance(bbr1, dict):
        for k, v in bbr1.items():
            metrics[f"entry_gate_blocked_by_reason__{k}"] = int(v or 0)

    gr = (rr.get("guardrails") or {})
    eg2 = (gr.get("entry_gate") or {})
    metrics["entry_gate_v2_attempted_entries"] = int(eg2.get("attempted_entries", 0) or 0)
    metrics["entry_gate_v2_blocked_total"] = int(eg2.get("blocked_total", 0) or 0)
    metrics["entry_gate_v2_blocked_unique_bars"] = int(eg2.get("blocked_unique_bars", 0) or 0)

    bbr2 = eg2.get("blocked_v2_by_reason") or {}
    if isinstance(bbr2, dict):
        for k, v in bbr2.items():
            metrics[f"entry_gate_v2_blocked_by_reason__{k}"] = int(v or 0)


def _flatten_fill_dropped_into_metrics(metrics: Dict[str, Any], risk_report: Dict[str, Any]) -> None:
    """
    Flatten 'entry_fill_dropped' diagnostics (not gated, but unfillable).

    Writes:
      - entry_fill_dropped_total
      - entry_fill_dropped_by_reason__<REASON>
    """
    rr = risk_report or {}
    d = (rr.get("entry_fill_dropped") or {})
    metrics["entry_fill_dropped_total"] = int(d.get("dropped_total", 0) or 0)

    by_reason = d.get("dropped_by_reason") or {}
    if isinstance(by_reason, dict):
        for k, v in by_reason.items():
            metrics[f"entry_fill_dropped_by_reason__{k}"] = int(v or 0)


def _flatten_engine_util_into_metrics(metrics: Dict[str, Any], risk_report: Dict[str, Any]) -> None:
    """
    Flatten engine utilization metrics.

    Writes:
      - engine_bars_total
      - engine_time_in_position_bars
      - engine_time_in_position_rate
    """
    rr = risk_report or {}
    e = (rr.get("engine") or {})
    metrics["engine_bars_total"] = int(e.get("bars_total", 0) or 0)
    metrics["engine_time_in_position_bars"] = int(e.get("time_in_position_bars", 0) or 0)

    tir = e.get("time_in_position_rate")
    try:
        metrics["engine_time_in_position_rate"] = float(tir) if tir is not None else None
    except Exception:
        metrics["engine_time_in_position_rate"] = None


def _flatten_forced_exits_into_metrics(metrics: Dict[str, Any], risk_report: Dict[str, Any]) -> None:
    """
    Flatten forced exits audit.

    Writes:
      - forced_exits_total
      - forced_exits__<REASON>
    """
    rr = risk_report or {}
    fx = (rr.get("forced_exits") or {})
    if isinstance(fx, dict):
        total = 0
        for k, v in fx.items():
            n = int(v or 0)
            total += n
            metrics[f"forced_exits__{k}"] = n
        metrics["forced_exits_total"] = int(total)
    else:
        metrics["forced_exits_total"] = 0


def run_from_config(cfg: Dict[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    # Apply execution policy BEFORE validation
    pol_res = apply_execution_policy(cfg)
    cfg_resolved = pol_res.cfg_resolved if pol_res is not None else cfg

    validate_run_config(cfg_resolved)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = _make_run_id(cfg_resolved)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    started_at_utc = datetime.now(timezone.utc).isoformat()
    instrument = cfg_resolved.get("instrument", {}) or {}

    # Load + dataset registry (global)
    df, dataset_meta, dataset_id_final = _load_bars_and_register_dataset(cfg_resolved)

    strat_cfg = cfg_resolved["strategy"]
    strategy = make_strategy(strat_cfg["name"], strat_cfg.get("params", {}))

    intents_by_bar = []
    context = {
        "symbol": cfg_resolved["symbol"],
        "timeframe": cfg_resolved["timeframe"],
        "instrument": instrument,
    }
    for i in range(len(df)):
        intents_by_bar.append(strategy.on_bar(i, df, context))

    # Resolve pip-based costs into price units for the engine
    costs_cfg = dict(cfg_resolved.get("costs", {}))
    pip_size = float(instrument.get("pip_size", 0.0) or 0.0)

    # Keep "effective" costs in pips for leaderboard traceability (cast to float if present)
    spread_pips_eff = costs_cfg.get("spread_pips")
    slippage_pips_eff = costs_cfg.get("slippage_pips")
    try:
        if spread_pips_eff is not None:
            spread_pips_eff = float(spread_pips_eff)
        if slippage_pips_eff is not None:
            slippage_pips_eff = float(slippage_pips_eff)
    except Exception:
        pass

    costs = dict(costs_cfg)
    if pip_size > 0:
        if "spread_pips" in costs and "spread_price" not in costs:
            costs["spread_price"] = float(costs.get("spread_pips", 0.0)) * pip_size
        if "slippage_pips" in costs and "slippage_price" not in costs:
            costs["slippage_price"] = float(costs.get("slippage_pips", 0.0)) * pip_size

    # Guardrails / risk policy layer
    risk_cfg = cfg_resolved.get("risk", {}) or {}

    engine = SimpleBarEngine(costs=costs, exec_cfg=cfg_resolved.get("execution", {}), risk_cfg=risk_cfg)
    trades = engine.run(df, intents_by_bar)

    # Risk report from engine (auditability)
    risk_report = getattr(engine, "last_risk_report", {}) or {}
    blocked = (risk_report.get("blocked") or {})
    risk_cfg_resolved = (risk_report.get("risk_cfg") or risk_cfg or {})

    metrics_raw = summarize_trades(trades)
    pf = metrics_raw.get("profit_factor")
    metrics_raw["profit_factor_is_inf"] = isinstance(pf, float) and (not math.isfinite(pf))

    # ---- Execution policy / effective execution ----
    exe = (cfg_resolved.get("execution", {}) or {})
    metrics_raw["execution_policy_id"] = (exe.get("policy_id") if isinstance(exe, dict) else None)
    metrics_raw["execution_fill_mode"] = (exe.get("fill_mode") if isinstance(exe, dict) else None)
    metrics_raw["execution_intrabar_path"] = (
        str(exe.get("intrabar_path")).replace(" ", "").upper()
        if isinstance(exe, dict) and exe.get("intrabar_path")
        else None
    )
    metrics_raw["execution_intrabar_tie"] = (exe.get("intrabar_tie") if isinstance(exe, dict) else None)
    metrics_raw["costs_spread_pips_effective"] = spread_pips_eff
    metrics_raw["costs_slippage_pips_effective"] = slippage_pips_eff

    # Dataset identity fields (flat) for quick traceability/leaderboard
    metrics_raw["dataset_id"] = dataset_id_final
    metrics_raw["dataset_fp8"] = dataset_meta.fingerprint_short
    metrics_raw["dataset_rows"] = dataset_meta.rows
    metrics_raw["dataset_start_time_utc"] = dataset_meta.start_ts
    metrics_raw["dataset_end_time_utc"] = dataset_meta.end_ts

    # Dataset provenance (flat)
    metrics_raw["dataset_fingerprint_version"] = getattr(dataset_meta, "fingerprint_version", None)
    metrics_raw["dataset_file_sha256"] = getattr(dataset_meta, "file_sha256", None)
    metrics_raw["dataset_file_bytes"] = getattr(dataset_meta, "file_bytes", None)
    metrics_raw["dataset_source_path"] = getattr(dataset_meta, "source_path", None)

    # Risk (flat) for leaderboard / quick inspection
    metrics_raw["risk_max_daily_loss_R"] = risk_cfg_resolved.get("max_daily_loss_R")
    metrics_raw["risk_max_trades_per_day"] = risk_cfg_resolved.get("max_trades_per_day")
    metrics_raw["risk_cooldown_bars"] = risk_cfg_resolved.get("cooldown_bars")

    metrics_raw["risk_blocked_by_daily_stop"] = blocked.get("by_daily_stop")
    metrics_raw["risk_blocked_by_max_trades_per_day"] = blocked.get("by_max_trades_per_day")
    metrics_raw["risk_blocked_by_cooldown"] = blocked.get("by_cooldown")
    metrics_raw["risk_final_realized_R_today"] = risk_report.get("final_realized_R_today")
    metrics_raw["risk_final_stopped_today"] = risk_report.get("final_stopped_today")

    # Guardrails v2 summary into metrics (optional but useful)
    gr = (risk_report.get("guardrails") or {})
    gr_cfg = (gr.get("guardrails_cfg") or {})
    gr_blocked = (gr.get("blocked") or {})
    gr_forced = (gr.get("forced_exits") or {})

    metrics_raw["gr_time_window_enabled"] = gr_cfg.get("time_window_enabled")
    metrics_raw["gr_window_start_utc"] = gr_cfg.get("window_start_utc")
    metrics_raw["gr_window_end_utc"] = gr_cfg.get("window_end_utc")
    metrics_raw["gr_max_concurrent_positions"] = gr_cfg.get("max_concurrent_positions")
    metrics_raw["gr_max_holding_bars"] = gr_cfg.get("max_holding_bars")

    metrics_raw["gr_blocked_by_max_concurrent_positions"] = gr_blocked.get("by_max_concurrent_positions")
    metrics_raw["gr_blocked_by_time_window"] = gr_blocked.get("by_time_window")
    metrics_raw["gr_forced_exit_by_max_holding_bars"] = gr_forced.get("by_max_holding_bars")

    # Entry gate (v1 + v2) into metrics
    _flatten_entry_gate_into_metrics(metrics_raw, risk_report)

    # NEW: dropped (unfillable) entries
    _flatten_fill_dropped_into_metrics(metrics_raw, risk_report)

    # NEW: engine utilization
    _flatten_engine_util_into_metrics(metrics_raw, risk_report)

    # NEW: forced exits (EOF / max_hold, etc.)
    _flatten_forced_exits_into_metrics(metrics_raw, risk_report)

    metrics = _sanitize_for_json(metrics_raw)

    # Persist trades
    trades_path = run_dir / "trades.csv"
    trades_dicts = trades_to_dicts(trades)
    pd.DataFrame(trades_dicts).to_csv(trades_path, index=False)

    # Persist equity (always)
    equity_path = run_dir / "equity.csv"
    equity_df = _build_equity_curve(df, trades_dicts)
    equity_df.to_csv(equity_path, index=False)

    # Persist metrics
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Persist dataset metadata (full) — enforce dataset_id_final explicitly
    dataset_dict = dataset_meta.to_dict()
    dataset_dict["dataset_id"] = dataset_id_final

    # Persist policy audit block
    execution_policy_block: Dict[str, Any] | None = None
    if pol_res is not None:
        execution_policy_block = {
            "policy_id": pol_res.policy_id,
            "policies_path": pol_res.policies_path,
            "overlay": pol_res.overlay,
            "warnings": list(pol_res.warnings),
        }

    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "name": cfg_resolved["name"],
        "symbol": cfg_resolved["symbol"],
        "timeframe": cfg_resolved["timeframe"],
        "data_path": cfg_resolved["data_path"],
        "instrument": cfg_resolved.get("instrument", {}),
        "dataset_registry": cfg_resolved.get("dataset_registry", {}) or {},
        "strategy": cfg_resolved["strategy"],
        "execution": cfg_resolved.get("execution", {}),
        "execution_policy": execution_policy_block,
        "costs": cfg_resolved.get("costs", {}),
        "resolved_costs": {
            "commission": float(costs.get("commission", 0.0)),
            "spread_price": float(costs.get("spread_price", 0.0)),
            "slippage_price": float(costs.get("slippage_price", 0.0)),
            "spread_pips_effective": spread_pips_eff,
            "slippage_pips_effective": slippage_pips_eff,
        },
        # IMPORTANT: persist RESOLVED risk for auditability
        "risk": risk_cfg_resolved,
        "risk_report": risk_report,
        "dataset": dataset_dict,
        "outputs": {
            "run_dir": str(run_dir),
            "trades": str(trades_path),
            "equity": str(equity_path),
            "metrics": str(metrics_path),
        },
        "cfg_hash_sha256": hashlib.sha256(_canonical_cfg_json(cfg_resolved).encode("utf-8")).hexdigest(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "outputs": {
            "run_dir": str(run_dir),
            "trades": str(trades_path),
            "equity": str(equity_path),
            "metrics": str(metrics_path),
        },
        "run_id": run_id,
    }


def _load_yaml_cfg(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config (expected mapping at top-level): {path}")

    return cfg


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run backtest from YAML config.")
    parser.add_argument("config", help="Path to YAML config (e.g., configs/run_example.yaml)")
    parser.add_argument(
        "--out-dir",
        default="results/runs",
        help="Output root directory for runs (default: results/runs)",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Print metrics JSON to stdout (optional).",
    )

    args = parser.parse_args(argv)

    try:
        cfg = _load_yaml_cfg(args.config)
        out = run_from_config(cfg, out_dir=args.out_dir)

        print(f"RUN_ID: {out['run_id']}")
        print(f"RUN_DIR: {out['outputs']['run_dir']}")

        if args.print_metrics:
            print(json.dumps(out["metrics"], indent=2))

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
