from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass, replace
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
from backtester.orchestrator.dist_to_anchor_hook import attach_dist_to_anchor_if_enabled
from backtester.orchestrator.entry_delay import apply_entry_delay
from backtester.orchestrator.regime_fx_hook import attach_regime_fx_if_enabled
from backtester.orchestrator.shock_z_hook import attach_shock_z_if_enabled
from backtester.strategies.registry import make_strategy

from backtester.policies.policy_gate import PolicyGate, PolicyGateConfig


# ----------------------------
# CLI overrides (Option A)
# ----------------------------

def _parse_scalar(s: str) -> Any:
    if s is None:
        return None
    if not isinstance(s, str):
        return s

    raw = s.strip()
    if raw == "":
        return ""

    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except Exception:
            pass

    low = raw.lower()
    if low in ("true", "yes", "y"):
        return True
    if low in ("false", "no", "n"):
        return False
    if low in ("null", "none", "~"):
        return None

    try:
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
    except Exception:
        pass

    try:
        v = float(raw)
        if math.isfinite(v):
            return v
    except Exception:
        pass

    return raw


def _parse_set_kv(item: str) -> Tuple[str, Any]:
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
    applied: Dict[str, Any] = {}
    if not set_items:
        return applied

    for item in set_items:
        k, v = _parse_set_kv(item)
        _set_deep(cfg, k, v)
        applied[k] = v

    return applied


# ----------------------------
# Deep merge (CRÍTICO)
# ----------------------------

def _deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = dict(base)
        for k, v in overlay.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return overlay if overlay is not None else base


# ----------------------------
# Helpers
# ----------------------------

def _json_safe(obj: Any) -> Any:
    if is_dataclass(obj):
        try:
            return _json_safe(asdict(obj))
        except Exception:
            return str(obj)

    try:
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
    except Exception:
        pass

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_safe(v) for v in obj], key=lambda x: str(x))

    if isinstance(obj, Path):
        return str(obj)

    return str(obj)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _build_equity_curve(df: pd.DataFrame, trades: list[dict]) -> pd.DataFrame:
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
    if x is None:
        return None
    if not isinstance(x, str):
        return None
    s = x.strip()
    return s or None


def _norm_int(x: Any) -> int | None:
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
    risk = cfg.get("risk", {}) or {}

    v = None
    if isinstance(cfg.get("validity_mode"), str) and str(cfg.get("validity_mode")).strip():
        v = str(cfg.get("validity_mode")).strip()
    elif isinstance(risk.get("validity_mode"), str) and str(risk.get("validity_mode")).strip():
        v = str(risk.get("validity_mode")).strip()

    return {
        "risk.eof_buffer_bars": _norm_int(risk.get("eof_buffer_bars", None)),
        "risk.validity_mode": v,
    }


def _apply_policy_sensitive_overrides(cfg_resolved: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    eofb = overrides.get("risk.eof_buffer_bars", None)
    if eofb is not None:
        cfg_resolved.setdefault("risk", {})
        if not isinstance(cfg_resolved["risk"], dict):
            cfg_resolved["risk"] = {}
        cfg_resolved["risk"]["eof_buffer_bars"] = int(eofb)

    v = overrides.get("risk.validity_mode", None)
    if isinstance(v, str) and v.strip():
        cfg_resolved.setdefault("risk", {})
        if not isinstance(cfg_resolved["risk"], dict):
            cfg_resolved["risk"] = {}
        cfg_resolved["risk"]["validity_mode"] = v.strip()


def _get_validity_mode(cfg: Dict[str, Any]) -> str | None:
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
    Canon order:
      1) strategy.params.max_hold_bars (if present)
      2) risk.guardrails_cfg.max_holding_bars (your canonical guardrails block)
      3) risk.max_holding_bars (legacy/overlay)
    """
    strat = cfg.get("strategy", {}) or {}
    if isinstance(strat, dict):
        params = strat.get("params", {}) or {}
        if isinstance(params, dict):
            v0 = _norm_int(params.get("max_hold_bars"))
            if v0 is not None:
                return v0

    risk = cfg.get("risk", {}) or {}
    if isinstance(risk, dict):
        gcfg = risk.get("guardrails_cfg")
        if isinstance(gcfg, dict):
            v1 = _norm_int(gcfg.get("max_holding_bars"))
            if v1 is not None:
                return v1

        v2 = _norm_int(risk.get("max_holding_bars"))
        if v2 is not None:
            return v2

    return None


def _resolve_risk_cfg_for_engine(cfg_resolved: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make risk_cfg deterministic for the engine:
      - Keep original risk dict for audit.
      - If risk.guardrails_cfg exists, flatten its keys to risk top-level for Guardrails().
      - Do NOT drop unknown keys.
    """
    risk = cfg_resolved.get("risk", {}) or {}
    if not isinstance(risk, dict):
        return {}

    out = dict(risk)

    gcfg = risk.get("guardrails_cfg")
    if isinstance(gcfg, dict) and gcfg:
        # Flatten known guardrails keys
        for k in (
            "max_concurrent_positions",
            "time_window_enabled",
            "window_start_utc",
            "window_end_utc",
            "max_holding_bars",
            "weekend_exit_enabled",
            "weekend_exit_cutoff_utc",
            "weekend_exit_weekday",
        ):
            if k in gcfg and gcfg.get(k) is not None:
                out[k] = gcfg.get(k)

    return out


def _enforce_strict_no_eof_entry_safety(cfg_resolved: Dict[str, Any]) -> None:
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

    margin = 12
    required = int(max_hold) + margin
    if eofb_eff < required:
        risk["eof_buffer_bars"] = required


def _ts_to_ymd(ts: Any) -> str:
    try:
        t = pd.to_datetime(ts, utc=True, errors="raise")
        return t.strftime("%Y-%m-%d")
    except Exception:
        s = str(ts)
        if "T" in s:
            return s.split("T", 1)[0]
        return s


# ----------------------------
# Dataset slicing (Option A)
# ----------------------------

def _parse_iso_utc_any(x: Any) -> pd.Timestamp | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return pd.to_datetime(s, utc=True, errors="raise")
    except Exception:
        return None


def _apply_dataset_slice(df: pd.DataFrame, cfg: Dict[str, Any]) -> tuple[pd.DataFrame, Dict[str, Any] | None]:
    ds = cfg.get("dataset_slice", None)
    if not isinstance(ds, dict) or not ds:
        return df, None

    start = _parse_iso_utc_any(ds.get("start_utc"))
    end = _parse_iso_utc_any(ds.get("end_utc"))

    if start is None and end is None:
        return df, None

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("DATASET_SLICE_REQUIRES_DATETIME_INDEX: df.index must be DatetimeIndex(UTC)")

    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= (df.index >= start)
    if end is not None:
        mask &= (df.index <= end)

    out = df.loc[mask.values].copy()
    manifest = {
        "start_utc": (start.isoformat() if start is not None else None),
        "end_utc": (end.isoformat() if end is not None else None),
        "rows_before": int(len(df)),
        "rows_after": int(len(out)),
    }

    if len(out) == 0:
        raise ValueError(f"DATASET_SLICE_EMPTY: slice produced 0 rows. slice={manifest}")

    return out, manifest


def _slice_fingerprint_short(df: pd.DataFrame) -> str:
    tmp = df.copy()
    tmp.insert(0, "__time__", tmp.index.astype("datetime64[ns, UTC]").astype(str))
    b = tmp.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:8]


def _load_bars_and_register_dataset(cfg: Dict[str, Any]) -> tuple[pd.DataFrame, Any, str, Dict[str, Any] | None, str | None]:
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
        keep_extra_cols=True,
    )

    df, slice_manifest = _apply_dataset_slice(df, cfg)
    slice_fp8 = _slice_fingerprint_short(df) if slice_manifest else None

    start_ymd = _ts_to_ymd(df.index.min())
    end_ymd = _ts_to_ymd(df.index.max())

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

    try:
        dataset_meta_final = replace(
            dataset_meta_final,
            rows=int(len(df)),
            start_ts=str(pd.to_datetime(df.index.min(), utc=True).isoformat()),
            end_ts=str(pd.to_datetime(df.index.max(), utc=True).isoformat()),
        )
    except Exception:
        pass

    register_or_validate_dataset(
        dataset_meta_final,
        registry_dir=registry_dir,
        allow_new_fingerprint=allow_override,
        override_reason=override_reason,
        append_match_event=append_match_event,
    )

    return df, dataset_meta_final, dataset_id_final, slice_manifest, slice_fp8


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
# Forced exit accounting
# ----------------------------

def _is_regime_fx_enabled(cfg_resolved: Dict[str, Any]) -> bool:
    rx = cfg_resolved.get("regime_fx", {}) or {}
    if isinstance(rx, dict):
        return bool(rx.get("enabled", False))
    return False


def _is_weekend_exit_enabled(cfg_resolved: Dict[str, Any]) -> bool:
    risk = cfg_resolved.get("risk", {}) or {}
    if not isinstance(risk, dict):
        return False
    gcfg = risk.get("guardrails_cfg")
    if isinstance(gcfg, dict):
        v = gcfg.get("weekend_exit_enabled")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() not in ("false", "0", "no", "n")
    # fallback legacy
    v2 = risk.get("weekend_exit_enabled")
    if isinstance(v2, bool):
        return v2
    if isinstance(v2, str):
        return v2.strip().lower() not in ("false", "0", "no", "n")
    return False


def _allowed_forced_exit_reasons(cfg_resolved: Dict[str, Any]) -> set[str]:
    risk = cfg_resolved.get("risk", {}) or {}
    reasons_raw = risk.get("allowed_forced_exit_reasons", None)

    out: set[str] = set()

    if isinstance(reasons_raw, (list, tuple)):
        for x in reasons_raw:
            if isinstance(x, str) and x.strip():
                out.add(x.strip())

    if out:
        return out

    max_hold_cfg = _extract_max_holding_bars_from_cfg(cfg_resolved)
    if max_hold_cfg is not None and int(max_hold_cfg) > 0:
        out.add("FORCE_MAX_HOLD")

    if _is_weekend_exit_enabled(cfg_resolved):
        out.add("FORCE_WEEKEND")

    if _is_regime_fx_enabled(cfg_resolved):
        out.add("FORCE_REGIME_EXIT")

    return out


def _allowed_forced_exits_total(metrics: Dict[str, Any], cfg_resolved: Dict[str, Any]) -> int:
    allowed = _allowed_forced_exit_reasons(cfg_resolved)
    total = 0
    for r in sorted(allowed):
        k = f"exit_reason_count__{r}"
        total += int(metrics.get(k, 0) or 0)
    return int(total)


def _compute_run_status(metrics: Dict[str, Any], cfg_resolved: Dict[str, Any]) -> Dict[str, Any]:
    forced_total = int(metrics.get("forced_exits_total", 0) or 0)

    eof_forced = int(metrics.get("forced_exits__EOF", 0) or 0)
    eof_skipped = int(metrics.get("forced_exits__EOF_SKIPPED_BY_POLICY", 0) or 0)

    forced_eof_total = int(metrics.get("forced_eof_total", 0) or 0)
    eof_exits_total = int(metrics.get("eof_exits_total", 0) or 0)

    allowed = _allowed_forced_exit_reasons(cfg_resolved)
    eof_present = (eof_forced + eof_skipped + forced_eof_total + eof_exits_total) > 0

    invalid_eof = bool(eof_present and ("FORCE_EOF" not in allowed))

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


def _intent_is_empty(intent: Any) -> bool:
    if intent is None:
        return True
    if isinstance(intent, (list, tuple, dict, set)):
        return len(intent) == 0
    return False


def _summarize_intents(intents_by_bar: list[Any], *, sample_max: int = 5) -> Dict[str, Any]:
    non_null = 0
    non_empty = 0
    sample: list[dict] = []

    for i, it in enumerate(intents_by_bar):
        if it is not None:
            non_null += 1
        if not _intent_is_empty(it):
            non_empty += 1
            if len(sample) < sample_max:
                sample.append({"bar_index": i, "intent": _json_safe(it)})

    return {
        "intents_total_bars": int(len(intents_by_bar)),
        "intents_non_null_bars": int(non_null),
        "intents_non_empty_bars": int(non_empty),
        "intents_sample": sample,
    }


# ----------------------------
# Policy gate wiring (PolicyGate API)
# ----------------------------

def _apply_policy_gate_to_intents(
    intents_by_bar: list[Any],
    cfg_resolved: Dict[str, Any],
) -> tuple[list[Any], Dict[str, Any] | None]:
    pg = cfg_resolved.get("policy_gate", None)
    if not isinstance(pg, dict) or not pg:
        return intents_by_bar, None

    enabled = pg.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in ("false", "0", "no", "n")
    if not bool(enabled):
        return intents_by_bar, {"enabled": False}

    policy_path = pg.get("policy_path")
    features_path = pg.get("features_path")
    tbm = pg.get("time_bucket_min", 30)

    if not isinstance(policy_path, str) or not policy_path.strip():
        raise ValueError("policy_gate.policy_path is required and must be a non-empty string")
    if not isinstance(features_path, str) or not features_path.strip():
        raise ValueError("policy_gate.features_path is required and must be a non-empty string")

    gate = PolicyGate(PolicyGateConfig(
        policy_path=str(policy_path),
        features_path=str(features_path),
        time_bucket_min=int(tbm or 30),
    ))

    attempted = 0
    blocked = 0
    blocked_unique_bars = 0

    out = list(intents_by_bar)
    for i, it in enumerate(intents_by_bar):
        if _intent_is_empty(it):
            continue

        attempted += 1
        ok = gate.evaluate_entry_idx(int(i))
        if not ok:
            out[i] = None
            blocked += 1
            blocked_unique_bars += 1

    stats = gate.stats()
    report = {
        "enabled": True,
        "policy_id": stats.get("policy_id"),
        "policy_path": stats.get("policy_path"),
        "features_path": stats.get("features_path"),
        "time_bucket_min": int(tbm or 30),
        "attempted_entries": int(attempted),
        "blocked_total": int(blocked),
        "blocked_unique_bars": int(blocked_unique_bars),
        "stats": stats,
    }
    return out, report


def run_from_config(
    cfg: Dict[str, Any],
    out_dir: str | Path,
    *,
    cli_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    policy_overrides = _capture_policy_sensitive_overrides(cfg)

    pol_res = apply_execution_policy(cfg)
    cfg_resolved = pol_res.cfg_resolved if pol_res is not None else cfg

    cfg_resolved = _deep_merge(cfg, cfg_resolved)

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

    df, dataset_meta, dataset_id_final, dataset_slice_manifest, dataset_slice_fp8 = _load_bars_and_register_dataset(cfg_resolved)

    pip_size = float(instrument.get("pip_size", 0.0) or 0.0)
    df, dist_anchor_manifest = attach_dist_to_anchor_if_enabled(df, cfg_resolved, pip_size=pip_size)
    df, shock_manifest = attach_shock_z_if_enabled(df, cfg_resolved, dataset_id=dataset_id_final)
    df, regime_fx_manifest = attach_regime_fx_if_enabled(df, cfg_resolved, dataset_id=dataset_id_final)

    strat_cfg = cfg_resolved["strategy"]
    strategy = make_strategy(strat_cfg["name"], strat_cfg.get("params", {}))

    context = {
        "symbol": cfg_resolved["symbol"],
        "timeframe": cfg_resolved["timeframe"],
        "instrument": instrument,
        "pip_size": float(instrument.get("pip_size", 0.0) or 0.0),
    }

    intents_by_bar: list[Any] = []
    for i in range(len(df)):
        intents_by_bar.append(strategy.on_bar(i, df, context))

    intent_diag = _summarize_intents(intents_by_bar, sample_max=5)

    exe_cfg = cfg_resolved.get("execution", {}) or {}
    delay_bars = _norm_int(exe_cfg.get("entry_delay_bars"))
    if delay_bars is None:
        delay_bars = _norm_int(exe_cfg.get("entry_delay"))
    delay_bars = int(delay_bars or 0)

    intents_by_bar, entry_delay_report = apply_entry_delay(intents_by_bar, delay_bars=delay_bars)

    intents_by_bar, policy_gate_report = _apply_policy_gate_to_intents(intents_by_bar, cfg_resolved)

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

    # IMPORTANT: resolve risk cfg for engine deterministically (flatten guardrails_cfg)
    risk_cfg_raw = cfg_resolved.get("risk", {}) or {}
    risk_cfg = _resolve_risk_cfg_for_engine(cfg_resolved)

    engine = SimpleBarEngine(costs=costs, exec_cfg=exe_cfg, risk_cfg=risk_cfg)
    trades = engine.run(df, intents_by_bar)

    risk_report = getattr(engine, "last_risk_report", {}) or {}
    blocked = (risk_report.get("blocked") or {})
    risk_cfg_resolved = (risk_report.get("risk_cfg") or risk_cfg or {})

    metrics_raw = summarize_trades(trades)
    pf = metrics_raw.get("profit_factor")
    metrics_raw["profit_factor_is_inf"] = isinstance(pf, float) and (not math.isfinite(pf))

    metrics_raw.update({
        "intents_total_bars": intent_diag["intents_total_bars"],
        "intents_non_null_bars": intent_diag["intents_non_null_bars"],
        "intents_non_empty_bars": intent_diag["intents_non_empty_bars"],
    })

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

    _flatten_entry_delay_into_metrics(metrics_raw, entry_delay_report)

    metrics_raw["dataset_id"] = dataset_id_final
    metrics_raw["dataset_fp8"] = getattr(dataset_meta, "fingerprint_short", None)
    metrics_raw["dataset_rows"] = getattr(dataset_meta, "rows", None)
    metrics_raw["dataset_start_time_utc"] = getattr(dataset_meta, "start_ts", None)
    metrics_raw["dataset_end_time_utc"] = getattr(dataset_meta, "end_ts", None)

    metrics_raw["dataset_fingerprint_version"] = getattr(dataset_meta, "fingerprint_version", None)
    metrics_raw["dataset_file_sha256"] = getattr(dataset_meta, "file_sha256", None)
    metrics_raw["dataset_file_bytes"] = getattr(dataset_meta, "file_bytes", None)
    metrics_raw["dataset_source_path"] = getattr(dataset_meta, "source_path", None)

    metrics_raw["dataset_slice_start_utc"] = (dataset_slice_manifest.get("start_utc") if dataset_slice_manifest else None)
    metrics_raw["dataset_slice_end_utc"] = (dataset_slice_manifest.get("end_utc") if dataset_slice_manifest else None)
    metrics_raw["dataset_slice_fp8"] = dataset_slice_fp8

    if policy_gate_report and policy_gate_report.get("enabled"):
        stats = (policy_gate_report.get("stats") or {})
        metrics_raw["policy_gate_enabled"] = True
        metrics_raw["policy_gate_policy_id"] = stats.get("policy_id")
        metrics_raw["policy_gate_policy_path"] = stats.get("policy_path")
        metrics_raw["policy_gate_features_path"] = stats.get("features_path")
        metrics_raw["policy_gate_time_bucket_min"] = policy_gate_report.get("time_bucket_min")
        metrics_raw["policy_gate_attempted_entries"] = policy_gate_report.get("attempted_entries")
        metrics_raw["policy_gate_blocked_total"] = policy_gate_report.get("blocked_total")
        metrics_raw["policy_gate_blocked_unique_bars"] = policy_gate_report.get("blocked_unique_bars")
        metrics_raw["policy_allowed"] = stats.get("policy_allowed")
        metrics_raw["policy_blocked"] = stats.get("policy_blocked")
        metrics_raw["policy_blocked_by_time"] = stats.get("policy_blocked_by_time")
        metrics_raw["policy_blocked_by_shock_vol"] = stats.get("policy_blocked_by_shock_vol")
        metrics_raw["policy_coverage_allowed"] = stats.get("policy_coverage_allowed")
    elif policy_gate_report and policy_gate_report.get("enabled") is False:
        metrics_raw["policy_gate_enabled"] = False
    else:
        metrics_raw["policy_gate_enabled"] = False

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

    status_block = _compute_run_status(metrics_raw, cfg_resolved)
    metrics_raw.update(status_block)

    metrics = _sanitize_for_json(metrics_raw)

    trades_path = run_dir / "trades.csv"
    trades_dicts = trades_to_dicts(trades)
    pd.DataFrame(trades_dicts).to_csv(trades_path, index=False)

    equity_path = run_dir / "equity.csv"
    equity_df = _build_equity_curve(df, trades_dicts)
    equity_df.to_csv(equity_path, index=False)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    dataset_dict = dataset_meta.to_dict() if hasattr(dataset_meta, "to_dict") else {}
    dataset_dict["dataset_id"] = dataset_id_final

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
            "risk.validity_mode": policy_overrides.get("risk.validity_mode", None),
        },
        "instrument": cfg_resolved.get("instrument", {}),
        "dataset_registry": cfg_resolved.get("dataset_registry", {}) or {},
        "dataset_slice": dataset_slice_manifest,
        "dataset_slice_fp8": dataset_slice_fp8,
        "policy_gate": policy_gate_report,
        "shock_z": shock_manifest,
        "regime_fx": regime_fx_manifest,
        "dist_to_anchor": dist_anchor_manifest,
        "intent_diagnostics": intent_diag,
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
        "risk_raw": risk_cfg_raw,
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
    (run_dir / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

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
    parser.add_argument("--out-dir", default="results/runs", help="Output root directory (default: results/runs)")
    parser.add_argument("--print-metrics", action="store_true", help="Print metrics JSON to stdout.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config value using dotted path. Repeatable.",
    )

    args = parser.parse_args(argv)

    try:
        cfg = _load_yaml_cfg(args.config)
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
