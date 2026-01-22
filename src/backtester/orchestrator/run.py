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

    df = load_bars_csv(cfg["data_path"])

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
    instrument = cfg.get("instrument", {}) or {}
    pip_size = float(instrument.get("pip_size", 0.0) or 0.0)
    if pip_size > 0:
        if "spread_pips" in costs and "spread_price" not in costs:
            costs["spread_price"] = float(costs.get("spread_pips", 0.0)) * pip_size
        if "slippage_pips" in costs and "slippage_price" not in costs:
            costs["slippage_price"] = float(costs.get("slippage_pips", 0.0)) * pip_size

    engine = SimpleBarEngine(costs=costs, exec_cfg=cfg.get("execution", {}))
    trades = engine.run(df, intents_by_bar)

    metrics_raw = summarize_trades(trades)
    pf = metrics_raw.get("profit_factor")
    metrics_raw["profit_factor_is_inf"] = isinstance(pf, float) and (not math.isfinite(pf))
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
        "outputs": {"run_dir": str(run_dir), "trades": str(trades_path), "equity": str(equity_path), "metrics": str(metrics_path)},
        "run_id": run_id,
    }
