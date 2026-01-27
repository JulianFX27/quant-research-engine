from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from backtester.core.config import validate_run_config
from backtester.data.loader import load_bars_csv
from backtester.data.dataset_fingerprint import build_dataset_id
from backtester.execution.engine import SimpleBarEngine
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

    # no trades: flat equity
    if len(df) > 0:
        t0 = df.index[0]
        return pd.DataFrame([{"time": t0, "equity": 0.0}])

    return pd.DataFrame(columns=["time", "equity"])


def _canonical_cfg_json(cfg: Dict[str, Any]) -> str:
    """Stable JSON string for hashing (sorted keys, compact)."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _make_run_id(cfg: Dict[str, Any]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha256(_canonical_cfg_json(cfg).encode("utf-8")).hexdigest()[:8]
    return f"{ts}_{h}"


def run_from_config(cfg: Dict[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    validate_run_config(cfg)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = _make_run_id(cfg)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    instrument = cfg.get("instrument", {}) or {}
    symbol = str(cfg.get("symbol") or "UNKNOWN")
    timeframe = str(cfg.get("timeframe") or "UNKNOWN")
    source = str(instrument.get("data_source") or instrument.get("source") or "csv")

    # Provisional dataset_id (will be aligned after load using real start/end)
    dataset_id_prov = build_dataset_id(
        instrument=symbol,
        timeframe=timeframe,
        start_ts="unknown",
        end_ts="unknown",
        source=source,
    )

    # Load bars + dataset metadata (identity & integrity)
    df, dataset_meta = load_bars_csv(cfg["data_path"], return_fingerprint=True, dataset_id=dataset_id_prov)

    # Align dataset_id with real start/end (post-normalization)
    dataset_id_final = build_dataset_id(
        instrument=symbol,
        timeframe=timeframe,
        start_ts=str(dataset_meta.start_ts),
        end_ts=str(dataset_meta.end_ts),
        source=source,
    )

    strat_cfg = cfg["strategy"]
    strategy = make_strategy(strat_cfg["name"], strat_cfg.get("params", {}))

    intents_by_bar = []
    context = {
        "symbol": cfg["symbol"],
        "timeframe": cfg["timeframe"],
        "instrument": cfg.get("instrument", {}),
    }
    for i in range(len(df)):
        intents_by_bar.append(strategy.on_bar(i, df, context))

    # Resolve pip-based costs into price units for the engine
    costs = dict(cfg.get("costs", {}))
    pip_size = float(instrument.get("pip_size", 0.0) or 0.0)
    if pip_size > 0:
        if "spread_pips" in costs and "spread_price" not in costs:
            costs["spread_price"] = float(costs.get("spread_pips", 0.0)) * pip_size
        if "slippage_pips" in costs and "slippage_price" not in costs:
            costs["slippage_price"] = float(costs.get("slippage_pips", 0.0)) * pip_size

    # Guardrails / risk policy layer
    risk_cfg = cfg.get("risk", {}) or {}

    engine = SimpleBarEngine(costs=costs, exec_cfg=cfg.get("execution", {}), risk_cfg=risk_cfg)
    trades = engine.run(df, intents_by_bar)

    # Risk report from engine (auditability)
    risk_report = getattr(engine, "last_risk_report", {}) or {}
    blocked = (risk_report.get("blocked") or {})
    risk_cfg_resolved = (risk_report.get("risk_cfg") or risk_cfg or {})

    metrics_raw = summarize_trades(trades)
    pf = metrics_raw.get("profit_factor")
    metrics_raw["profit_factor_is_inf"] = isinstance(pf, float) and (not math.isfinite(pf))

    # Dataset identity fields (flat) for quick traceability/leaderboard
    metrics_raw["dataset_id"] = dataset_id_final
    metrics_raw["dataset_fp8"] = dataset_meta.fingerprint_short
    metrics_raw["dataset_rows"] = dataset_meta.rows
    metrics_raw["dataset_start_time_utc"] = dataset_meta.start_ts
    metrics_raw["dataset_end_time_utc"] = dataset_meta.end_ts

    # Risk (flat) for leaderboard / quick inspection
    metrics_raw["risk_max_daily_loss_R"] = risk_cfg_resolved.get("max_daily_loss_R")
    metrics_raw["risk_max_trades_per_day"] = risk_cfg_resolved.get("max_trades_per_day")
    metrics_raw["risk_cooldown_bars"] = risk_cfg_resolved.get("cooldown_bars")

    metrics_raw["risk_blocked_by_daily_stop"] = blocked.get("by_daily_stop")
    metrics_raw["risk_blocked_by_max_trades_per_day"] = blocked.get("by_max_trades_per_day")
    metrics_raw["risk_blocked_by_cooldown"] = blocked.get("by_cooldown")
    metrics_raw["risk_final_realized_R_today"] = risk_report.get("final_realized_R_today")
    metrics_raw["risk_final_stopped_today"] = risk_report.get("final_stopped_today")

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
    metrics_path.write_text(json.dumps(metrics, indent=2))

    started_at_utc = datetime.now(timezone.utc).isoformat()

    # Persist dataset metadata (full)
    dataset_dict = dataset_meta.to_dict()
    dataset_dict["dataset_id"] = dataset_id_final  # enforce aligned id

    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "name": cfg["name"],
        "symbol": cfg["symbol"],
        "timeframe": cfg["timeframe"],
        "data_path": cfg["data_path"],
        "instrument": cfg.get("instrument", {}),
        "strategy": cfg["strategy"],
        "execution": cfg.get("execution", {}),
        "costs": cfg.get("costs", {}),
        "resolved_costs": {
            "commission": float(costs.get("commission", 0.0)),
            "spread_price": float(costs.get("spread_price", 0.0)),
            "slippage_price": float(costs.get("slippage_price", 0.0)),
        },
        "risk": risk_cfg,
        "risk_report": risk_report,
        "dataset": dataset_dict,
        "outputs": {
            "run_dir": str(run_dir),
            "trades": str(trades_path),
            "equity": str(equity_path),
            "metrics": str(metrics_path),
        },
        "cfg_hash_sha256": hashlib.sha256(_canonical_cfg_json(cfg).encode("utf-8")).hexdigest(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

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
