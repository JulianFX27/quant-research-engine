# src/backtester/orchestrator/run.py
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml

from backtester.core.config import validate_run_config
from backtester.data.dataset_fingerprint import build_dataset_id
from backtester.data.dataset_registry import register_or_validate_dataset
from backtester.data.loader import load_bars_csv
from backtester.execution.engine import SimpleBarEngine
from backtester.execution.policies import apply_execution_policy
from backtester.metrics.basic import summarize_trades, trades_to_dicts
from backtester.orchestrator.entry_delay import apply_entry_delay
from backtester.strategies.registry import make_strategy


# ----------------------------
# CLI overrides (Option A)
# ----------------------------

def _parse_scalar(s: str) -> Any:
    """
    Parse a scalar string to bool/int/float/None/str.

    Rules:
      - true/false/yes/no/y/n (case-insensitive) -> bool
      - null/none/~ -> None
      - int-like -> int
      - float-like -> float
      - else -> string (stripped)

    NOTE: We intentionally do NOT treat "1"/"0" as bool to avoid ambiguity.
    """
    if s is None:
        return None
    if not isinstance(s, str):
        return s

    raw = s.strip()
    if raw == "":
        return ""

    low = raw.lower()
    if low in ("true", "yes", "y"):
        return True
    if low in ("false", "no", "n"):
        return False
    if low in ("null", "none", "~"):
        return None

    # int?
    try:
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
    except Exception:
        pass

    # float?
    try:
        v = float(raw)
        if math.isfinite(v):
            return v
    except Exception:
        pass

    return raw


def _parse_set_kv(item: str) -> Tuple[str, Any]:
    """Parse 'a.b.c=value' into ('a.b.c', value)."""
    if not isinstance(item, str) or "=" not in item:
        raise ValueError(
            f"Invalid --set {item!r}. Expected format: --set key=value (e.g., --set risk.force_exit_on_eof=false)"
        )
    k, v = item.split("=", 1)
    key = k.strip()
    if not key:
        raise ValueError(f"Invalid --set {item!r}: empty key.")
    return key, _parse_scalar(v)


def _set_deep(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    Set cfg['a']['b']['c']=value given dotted_key='a.b.c'.
    Creates intermediate dicts if needed.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a dict to apply --set overrides.")

    parts = [p for p in dotted_key.split(".") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")

    cur = cfg
    for p in parts[:-1]:
        if p not in cur or cur[p] is None:
            cur[p] = {}
        if not isinstance(cur[p], dict):
            raise ValueError(
                f"Cannot set '{dotted_key}': '{p}' is not a dict (found {type(cur[p]).__name__})."
            )
        cur = cur[p]

    cur[parts[-1]] = value


def _apply_cli_overrides(cfg: Dict[str, Any], set_items: list[str] | None) -> Dict[str, Any]:
    """Apply --set overrides in order. Returns dict of applied overrides."""
    applied: Dict[str, Any] = {}
    if not set_items:
        return applied

    for item in set_items:
        k, v = _parse_set_kv(item)
        _set_deep(cfg, k, v)
        applied[k] = v

    return applied


# ----------------------------
# Existing helpers
# ----------------------------

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


def _norm_forced_dataset_id(x: Any) -> str | None:
    """
    Normalize cfg.dataset_id:
      - None/"" -> None
      - strip spaces
      - reject non-strings
    """
    if x is None:
        return None
    if not isinstance(x, str):
        return None
    s = x.strip()
    return s or None


def _norm_int(x: Any) -> int | None:
    """Best-effort normalize int-like values. Returns None if cannot parse."""
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(x)
    except Exception:
        return None


def _capture_policy_sensitive_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture overrides that must remain stable even if an execution policy overlay runs.
    """
    risk = cfg.get("risk", {}) or {}
    return {
        "risk.eof_buffer_bars": _norm_int(risk.get("eof_buffer_bars", None)),
    }


def _apply_policy_sensitive_overrides(cfg_resolved: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Re-apply captured overrides onto cfg_resolved with highest precedence."""
    if not isinstance(cfg_resolved, dict):
        return

    eofb = overrides.get("risk.eof_buffer_bars", None)
    if eofb is not None:
        cfg_resolved.setdefault("risk", {})
        if not isinstance(cfg_resolved["risk"], dict):
            cfg_resolved["risk"] = {}
        cfg_resolved["risk"]["eof_buffer_bars"] = int(eofb)


def _get_validity_mode(cfg: Dict[str, Any]) -> str | None:
    """Best-effort: validity mode can live at top-level or inside risk."""
    v = cfg.get("validity_mode")
    if isinstance(v, str) and v.strip():
        return v.strip()
    risk = cfg.get("risk", {}) or {}
    v2 = risk.get("validity_mode")
    if isinstance(v2, str) and v2.strip():
        return v2.strip()
    return None


def _extract_max_holding_bars_from_cfg(cfg: Dict[str, Any]) -> int | None:
    """
    Best-effort: max_holding_bars can be declared in several places depending on wiring.
    """
    risk = cfg.get("risk", {}) or {}

    g = risk.get("guardrails")
    if isinstance(g, dict):
        v = _norm_int(g.get("max_holding_bars"))
        if v is not None:
            return v
        gcfg = g.get("guardrails_cfg")
        if isinstance(gcfg, dict):
            v2 = _norm_int(gcfg.get("max_holding_bars"))
            if v2 is not None:
                return v2

    g2 = cfg.get("guardrails")
    if isinstance(g2, dict):
        v3 = _norm_int(g2.get("max_holding_bars"))
        if v3 is not None:
            return v3

    gcfg2 = cfg.get("guardrails_cfg")
    if isinstance(gcfg2, dict):
        v4 = _norm_int(gcfg2.get("max_holding_bars"))
        if v4 is not None:
            return v4

    return None


def _enforce_strict_no_eof_entry_safety(cfg_resolved: Dict[str, Any]) -> None:
    """
    If validity_mode=strict_no_eof, enforce that eof_buffer_bars is at least max_holding_bars.
    """
    mode = _get_validity_mode(cfg_resolved)
    if mode != "strict_no_eof":
        return

    cfg_resolved.setdefault("risk", {})
    if not isinstance(cfg_resolved["risk"], dict):
        cfg_resolved["risk"] = {}

    risk = cfg_resolved["risk"]

    eofb = _norm_int(risk.get("eof_buffer_bars"))
    max_hold = _extract_max_holding_bars_from_cfg(cfg_resolved)

    if max_hold is None:
        return

    eofb_eff = int(eofb or 0)
    if eofb_eff < int(max_hold):
        risk["eof_buffer_bars"] = int(max_hold)


def _ts_to_ymd(ts: Any) -> str:
    """
    Normalize a timestamp-ish (iso string, pandas Timestamp, datetime) to YYYY-MM-DD.
    If parsing fails, fallback to str(ts).
    """
    try:
        t = pd.to_datetime(ts, utc=True, errors="raise")
        return t.strftime("%Y-%m-%d")
    except Exception:
        s = str(ts)
        # best-effort: if iso-like, take date portion
        if "T" in s:
            return s.split("T", 1)[0]
        return s


def _load_bars_and_register_dataset(cfg: Dict[str, Any]) -> tuple[pd.DataFrame, Any, str]:
    """
    Returns:
      (df, dataset_meta_final, dataset_id_final)

    Design:
      - loader may require a dataset_id; we pass provisional to compute metadata
      - then compute dataset_id_final from canonical start/end OR respect cfg.dataset_id if provided
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
        keep_extra_cols=True,  # <-- CLAVE para estrategias con features
    )

    # IMPORTANT: dataset_id should be stable and match tests: YYYY-MM-DD only
    start_ymd = _ts_to_ymd(dataset_meta.start_ts)
    end_ymd = _ts_to_ymd(dataset_meta.end_ts)

    dataset_id_default = build_dataset_id(
        instrument=symbol,
        timeframe=timeframe,
        start_ts=start_ymd,
        end_ts=end_ymd,
        source=source,
    )

    forced_id = _norm_forced_dataset_id(cfg.get("dataset_id"))
    dataset_id_final = forced_id if forced_id else dataset_id_default

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
    rr = risk_report or {}
    d = (rr.get("entry_fill_dropped") or {})
    metrics["entry_fill_dropped_total"] = int(d.get("dropped_total", 0) or 0)

    by_reason = d.get("dropped_by_reason") or {}
    if isinstance(by_reason, dict):
        for k, v in by_reason.items():
            metrics[f"entry_fill_dropped_by_reason__{k}"] = int(v or 0)


def _flatten_engine_util_into_metrics(metrics: Dict[str, Any], risk_report: Dict[str, Any]) -> None:
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


def _flatten_entry_delay_into_metrics(metrics: Dict[str, Any], entry_delay_report: Any | None) -> None:
    if entry_delay_report is None:
        metrics["execution_entry_delay_bars"] = None
        metrics["execution_entry_delay_attempted"] = None
        metrics["execution_entry_delay_shifted"] = None
        metrics["execution_entry_delay_dropped_out_of_range"] = None
        return

    metrics["execution_entry_delay_bars"] = int(getattr(entry_delay_report, "delay_bars", 0) or 0)
    metrics["execution_entry_delay_attempted"] = int(getattr(entry_delay_report, "attempted", 0) or 0)
    metrics["execution_entry_delay_shifted"] = int(getattr(entry_delay_report, "shifted", 0) or 0)
    metrics["execution_entry_delay_dropped_out_of_range"] = int(
        getattr(entry_delay_report, "dropped_out_of_range", 0) or 0
    )


# ----------------------------
# Validity / status policy
# ----------------------------

def _allowed_forced_exits_total(metrics: Dict[str, Any], cfg_resolved: Dict[str, Any]) -> int:
    """
    Allowed forced exits are those that are part of the strategy's intended lifecycle.

    Current rule:
      - If max_holding_bars > 0, then FORCE_MAX_HOLD is considered a valid (allowed) exit.
        (It remains "forced" for auditability, but should NOT invalidate trades.)
    """
    max_hold_cfg = _extract_max_holding_bars_from_cfg(cfg_resolved)
    if max_hold_cfg is None or int(max_hold_cfg) <= 0:
        return 0

    return int(metrics.get("exit_reason_count__FORCE_MAX_HOLD", 0) or 0)


def _compute_run_status(metrics: Dict[str, Any], cfg_resolved: Dict[str, Any]) -> Dict[str, Any]:
    forced_total = int(metrics.get("forced_exits_total", 0) or 0)

    eof_forced = int(metrics.get("forced_exits__EOF", 0) or 0)
    eof_skipped = int(metrics.get("forced_exits__EOF_SKIPPED_BY_POLICY", 0) or 0)

    forced_eof_total = int(metrics.get("forced_eof_total", 0) or 0)
    eof_exits_total = int(metrics.get("eof_exits_total", 0) or 0)

    invalid_eof = (eof_forced + eof_skipped + forced_eof_total + eof_exits_total) > 0

    allowed_forced = _allowed_forced_exits_total(metrics, cfg_resolved)
    forced_non_allowed = max(0, forced_total - allowed_forced)

    status = "OK"
    if invalid_eof:
        status = "INVALID_SAMPLE_EOF"
    elif forced_non_allowed > 0:
        status = "OK_FORCED_EXITS_PRESENT"

    return {
        "run_status": status,
        "invalid_eof": bool(invalid_eof),
        "eof_incomplete_count": int(eof_skipped),
        "eof_forced_count": int(eof_forced),
        "allowed_forced_exits_total": int(allowed_forced),
        "forced_exits_non_allowed_total": int(forced_non_allowed),
    }


def run_from_config(
    cfg: Dict[str, Any],
    out_dir: str | Path,
    *,
    cli_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    # capture policy-sensitive overrides AFTER CLI overrides have been applied
    policy_overrides = _capture_policy_sensitive_overrides(cfg)

    pol_res = apply_execution_policy(cfg)
    cfg_resolved = pol_res.cfg_resolved if pol_res is not None else cfg

    _apply_policy_sensitive_overrides(cfg_resolved, policy_overrides)
    _enforce_strict_no_eof_entry_safety(cfg_resolved)

    validate_run_config(cfg_resolved)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = _make_run_id(cfg_resolved)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    started_at_utc = datetime.now(timezone.utc).isoformat()
    instrument = cfg_resolved.get("instrument", {}) or {}

    # Load + dataset registry
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

    # ---- APPLY ENTRY DELAY (orchestrator-layer) ----
    exe_cfg = cfg_resolved.get("execution", {}) or {}
    delay_bars = _norm_int(exe_cfg.get("entry_delay_bars"))
    if delay_bars is None:
        delay_bars = _norm_int(exe_cfg.get("entry_delay"))  # backward compatibility
    delay_bars = int(delay_bars or 0)

    intents_by_bar, entry_delay_report = apply_entry_delay(intents_by_bar, delay_bars=delay_bars)

    # Resolve pip-based costs into price units for the engine
    costs_cfg = dict(cfg_resolved.get("costs", {}))
    pip_size = float(instrument.get("pip_size", 0.0) or 0.0)

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

    risk_cfg = cfg_resolved.get("risk", {}) or {}

    engine = SimpleBarEngine(costs=costs, exec_cfg=exe_cfg, risk_cfg=risk_cfg)
    trades = engine.run(df, intents_by_bar)

    risk_report = getattr(engine, "last_risk_report", {}) or {}
    blocked = (risk_report.get("blocked") or {})
    risk_cfg_resolved = (risk_report.get("risk_cfg") or risk_cfg or {})

    metrics_raw = summarize_trades(trades)
    pf = metrics_raw.get("profit_factor")
    metrics_raw["profit_factor_is_inf"] = isinstance(pf, float) and (not math.isfinite(pf))

    # Execution / effective execution
    metrics_raw["execution_policy_id"] = (exe_cfg.get("policy_id") if isinstance(exe_cfg, dict) else None)
    metrics_raw["execution_fill_mode"] = (exe_cfg.get("fill_mode") if isinstance(exe_cfg, dict) else None)
    metrics_raw["execution_intrabar_path"] = (
        str(exe_cfg.get("intrabar_path")).replace(" ", "").upper()
        if isinstance(exe_cfg, dict) and exe_cfg.get("intrabar_path")
        else None
    )
    metrics_raw["execution_intrabar_tie"] = (exe_cfg.get("intrabar_tie") if isinstance(exe_cfg, dict) else None)
    metrics_raw["costs_spread_pips_effective"] = spread_pips_eff
    metrics_raw["costs_slippage_pips_effective"] = slippage_pips_eff

    # Entry delay diagnostics
    _flatten_entry_delay_into_metrics(metrics_raw, entry_delay_report)

    # Dataset identity fields
    metrics_raw["dataset_id"] = dataset_id_final
    metrics_raw["dataset_fp8"] = dataset_meta.fingerprint_short
    metrics_raw["dataset_rows"] = dataset_meta.rows
    metrics_raw["dataset_start_time_utc"] = dataset_meta.start_ts
    metrics_raw["dataset_end_time_utc"] = dataset_meta.end_ts

    metrics_raw["dataset_fingerprint_version"] = getattr(dataset_meta, "fingerprint_version", None)
    metrics_raw["dataset_file_sha256"] = getattr(dataset_meta, "file_sha256", None)
    metrics_raw["dataset_file_bytes"] = getattr(dataset_meta, "file_bytes", None)
    metrics_raw["dataset_source_path"] = getattr(dataset_meta, "source_path", None)

    # Risk (flat)
    metrics_raw["risk_max_daily_loss_R"] = risk_cfg_resolved.get("max_daily_loss_R")
    metrics_raw["risk_max_trades_per_day"] = risk_cfg_resolved.get("max_trades_per_day")
    metrics_raw["risk_cooldown_bars"] = risk_cfg_resolved.get("cooldown_bars")
    metrics_raw["risk_eof_buffer_bars"] = risk_cfg_resolved.get("eof_buffer_bars")
    metrics_raw["risk_force_exit_on_eof"] = risk_cfg_resolved.get("force_exit_on_eof")
    metrics_raw["validity_mode"] = _get_validity_mode(cfg_resolved)
    metrics_raw["cfg_max_holding_bars_hint"] = _extract_max_holding_bars_from_cfg(cfg_resolved)

    metrics_raw["risk_blocked_by_daily_stop"] = blocked.get("by_daily_stop")
    metrics_raw["risk_blocked_by_max_trades_per_day"] = blocked.get("by_max_trades_per_day")
    metrics_raw["risk_blocked_by_cooldown"] = blocked.get("by_cooldown")
    metrics_raw["risk_final_realized_R_today"] = risk_report.get("final_realized_R_today")
    metrics_raw["risk_final_stopped_today"] = risk_report.get("final_stopped_today")

    # Guardrails v2 summary into metrics
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

    # Flatten audits
    _flatten_entry_gate_into_metrics(metrics_raw, risk_report)
    _flatten_fill_dropped_into_metrics(metrics_raw, risk_report)
    _flatten_engine_util_into_metrics(metrics_raw, risk_report)
    _flatten_forced_exits_into_metrics(metrics_raw, risk_report)

    # ----------------------------
    # Research-grade validity layer
    # ----------------------------
    n_trades = int(metrics_raw.get("n_trades", 0) or 0)
    forced_total = int(metrics_raw.get("forced_exits_total", 0) or 0)
    non_forced_total = int(metrics_raw.get("non_forced_exits_total", 0) or max(0, n_trades - forced_total))

    allowed_forced = _allowed_forced_exits_total(metrics_raw, cfg_resolved)
    valid_trades = non_forced_total + allowed_forced

    metrics_raw["valid_trades"] = int(valid_trades)

    attempted = metrics_raw.get("entry_gate_v2_attempted_entries")
    if attempted is None:
        attempted = metrics_raw.get("entry_gate_attempted_entries")
    if attempted is None:
        attempted = n_trades

    try:
        attempted_i = int(attempted or 0)
    except Exception:
        attempted_i = 0

    metrics_raw["valid_trade_ratio"] = (valid_trades / attempted_i) if attempted_i > 0 else None

    # status (needs cfg_resolved for allowed forced exits policy)
    status_block = _compute_run_status(metrics_raw, cfg_resolved)
    metrics_raw.update(status_block)

    metrics = _sanitize_for_json(metrics_raw)

    # Persist trades
    trades_path = run_dir / "trades.csv"
    trades_dicts = trades_to_dicts(trades)
    pd.DataFrame(trades_dicts).to_csv(trades_path, index=False)

    # Persist equity
    equity_path = run_dir / "equity.csv"
    equity_df = _build_equity_curve(df, trades_dicts)
    equity_df.to_csv(equity_path, index=False)

    # Persist metrics
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Persist dataset metadata
    dataset_dict = dataset_meta.to_dict()
    dataset_dict["dataset_id"] = dataset_id_final

    # Persist policy audit
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
        "dataset_id_forced": _norm_forced_dataset_id(cfg_resolved.get("dataset_id")),
        "cli_overrides": (cli_overrides or {}),
        "policy_sensitive_overrides": {
            "risk.eof_buffer_bars": policy_overrides.get("risk.eof_buffer_bars", None),
        },
        "instrument": cfg_resolved.get("instrument", {}),
        "dataset_registry": cfg_resolved.get("dataset_registry", {}) or {},
        "strategy": cfg_resolved["strategy"],
        "execution": cfg_resolved.get("execution", {}) or {},
        "execution_policy": execution_policy_block,
        "costs": cfg_resolved.get("costs", {}),
        "resolved_costs": {
            "commission": float(costs.get("commission", 0.0)),
            "spread_price": float(costs.get("spread_price", 0.0)),
            "slippage_price": float(costs.get("slippage_price", 0.0)),
            "spread_pips_effective": spread_pips_eff,
            "slippage_pips_effective": slippage_pips_eff,
        },
        "risk": risk_cfg_resolved,
        "risk_report": risk_report,
        "entry_delay_report": {
            "delay_bars": int(getattr(entry_delay_report, "delay_bars", 0) or 0),
            "attempted": int(getattr(entry_delay_report, "attempted", 0) or 0),
            "shifted": int(getattr(entry_delay_report, "shifted", 0) or 0),
            "dropped_out_of_range": int(getattr(entry_delay_report, "dropped_out_of_range", 0) or 0),
        },
        "dataset": dataset_dict,
        "run_status": metrics.get("run_status"),
        "run_invalid_eof": metrics.get("invalid_eof"),
        "run_valid_trades": metrics.get("valid_trades"),
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
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config value using dotted path. Repeatable. Example: --set risk.force_exit_on_eof=false",
    )

    args = parser.parse_args(argv)

    try:
        cfg = _load_yaml_cfg(args.config)

        # Apply CLI overrides BEFORE policy application & validation
        cli_overrides = _apply_cli_overrides(cfg, args.set)

        out = run_from_config(cfg, out_dir=args.out_dir, cli_overrides=cli_overrides)

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
