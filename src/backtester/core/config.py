from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunConfig:
    name: str
    symbol: str
    timeframe: str
    data_path: str
    instrument: Dict[str, Any]
    strategy: Dict[str, Any]
    execution: Dict[str, Any]
    costs: Dict[str, Any]


def _as_float(d: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    if key not in d:
        return default
    try:
        return float(d[key])
    except Exception as e:
        raise ValueError(f"Invalid '{key}': expected float-like, got {d[key]!r}") from e


def _as_int(d: Dict[str, Any], key: str, default: Optional[int] = None) -> Optional[int]:
    if key not in d:
        return default
    try:
        return int(d[key])
    except Exception as e:
        raise ValueError(f"Invalid '{key}': expected int-like, got {d[key]!r}") from e


def _as_bool(d: Dict[str, Any], key: str, default: Optional[bool] = None) -> Optional[bool]:
    if key not in d:
        return default
    v = d[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
    raise ValueError(f"Invalid '{key}': expected bool-like, got {v!r}")


_HHMM_RE = re.compile(r"^\d{2}:\d{2}$")


def _validate_hhmm_utc(s: str, *, field: str) -> None:
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"Invalid {field}: must be a non-empty string in 'HH:MM'")
    s = s.strip()
    if not _HHMM_RE.match(s):
        raise ValueError(f"Invalid {field}: must match 'HH:MM', got {s!r}")
    hh = int(s[:2])
    mm = int(s[3:])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Invalid {field}: out of range, got {s!r}")


def validate_run_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"Run config must be a dict, got {type(cfg).__name__}")

    # ---- Required top-level keys ----
    required = ["name", "symbol", "timeframe", "data_path", "strategy", "execution", "costs"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing keys in run config: {missing}")

    # ---- Basic types ----
    for k in ["name", "symbol", "timeframe", "data_path"]:
        if not isinstance(cfg.get(k), str) or not cfg[k].strip():
            raise ValueError(f"Invalid '{k}': must be a non-empty string")

    if not isinstance(cfg["strategy"], dict):
        raise ValueError("Invalid 'strategy': must be a mapping/dict")
    if not isinstance(cfg["execution"], dict):
        raise ValueError("Invalid 'execution': must be a mapping/dict")
    if not isinstance(cfg["costs"], dict):
        raise ValueError("Invalid 'costs': must be a mapping/dict")

    # ---- Data path exists ----
    data_path = Path(cfg["data_path"])
    if not data_path.exists():
        raise ValueError(f"data_path not found: {cfg['data_path']}")

    # ---- Instrument (optional but recommended) ----
    instrument = cfg.get("instrument", {})
    if instrument is not None and not isinstance(instrument, dict):
        raise ValueError("Invalid 'instrument': must be a mapping/dict if provided")

    pip_size = None
    if isinstance(instrument, dict) and "pip_size" in instrument:
        pip_size = _as_float(instrument, "pip_size")
        if pip_size is None or pip_size <= 0:
            raise ValueError(f"Invalid instrument.pip_size: must be > 0, got {instrument.get('pip_size')!r}")

    if isinstance(instrument, dict) and "data_source" in instrument:
        if not isinstance(instrument["data_source"], str) or not instrument["data_source"].strip():
            raise ValueError("Invalid instrument.data_source: must be a non-empty string")

    # ---- Dataset registry policy (optional) ----
    dsreg = cfg.get("dataset_registry", {}) or {}
    if dsreg is not None and not isinstance(dsreg, dict):
        raise ValueError("Invalid 'dataset_registry': must be a mapping/dict if provided")

    if isinstance(dsreg, dict) and dsreg:
        allow_override = _as_bool(dsreg, "allow_override", default=False)
        override_reason = str(dsreg.get("override_reason", "") or "")

        if allow_override and not override_reason.strip():
            raise ValueError(
                "DATASET_ID_OVERRIDE_REASON_REQUIRED: override requested but override_reason is empty.\n"
                "Fix: set dataset_registry.override_reason to a non-empty string."
            )

    # ---- Strategy ----
    strat = cfg["strategy"]
    if "name" not in strat or not isinstance(strat["name"], str) or not strat["name"].strip():
        raise ValueError("Invalid strategy.name: must be a non-empty string")

    params = strat.get("params", {})
    if params is not None and not isinstance(params, dict):
        raise ValueError("Invalid strategy.params: must be a mapping/dict if provided")

    if isinstance(params, dict):
        qty = _as_float(params, "qty", default=None)
        if qty is not None and qty <= 0:
            raise ValueError(f"Invalid strategy.params.qty: must be > 0, got {params.get('qty')!r}")

        warmup = _as_int(params, "warmup_bars", default=None)
        if warmup is not None and warmup < 0:
            raise ValueError(f"Invalid strategy.params.warmup_bars: must be >= 0, got {params.get('warmup_bars')!r}")

        sma_n = _as_int(params, "sma_n", default=None)
        if sma_n is not None and sma_n <= 0:
            raise ValueError(f"Invalid strategy.params.sma_n: must be > 0, got {params.get('sma_n')!r}")

        if warmup is not None and sma_n is not None and warmup < sma_n:
            raise ValueError(f"Invalid strategy.params: warmup_bars ({warmup}) must be >= sma_n ({sma_n})")

        sl_pips = _as_float(params, "sl_pips", default=None)
        tp_pips = _as_float(params, "tp_pips", default=None)
        if (sl_pips is not None or tp_pips is not None) and (pip_size is None):
            raise ValueError(
                "strategy.params includes sl_pips/tp_pips but instrument.pip_size is missing. "
                "Add instrument.pip_size to the run config."
            )
        if sl_pips is not None and sl_pips <= 0:
            raise ValueError(f"Invalid strategy.params.sl_pips: must be > 0, got {params.get('sl_pips')!r}")
        if tp_pips is not None and tp_pips <= 0:
            raise ValueError(f"Invalid strategy.params.tp_pips: must be > 0, got {params.get('tp_pips')!r}")

    # ---- Execution ----
    exe = cfg["execution"]

    fill_mode = exe.get("fill_mode", "close")
    if fill_mode not in ("close", "next_open"):
        raise ValueError("Invalid execution.fill_mode: must be one of ['close', 'next_open']")

    intrabar_tie = exe.get("intrabar_tie", "sl_first")
    if intrabar_tie not in ("sl_first", "tp_first"):
        raise ValueError("Invalid execution.intrabar_tie: must be one of ['sl_first', 'tp_first']")

    intrabar_path = str(exe.get("intrabar_path", "OHLC")).replace(" ", "").upper()
    if intrabar_path not in ("OHLC", "OLHC"):
        raise ValueError("Invalid execution.intrabar_path: must be 'OHLC' or 'OLHC' (spaces ignored)")

    # ---- Costs ----
    costs = cfg["costs"]

    commission = _as_float(costs, "commission", default=0.0)
    if commission is not None and commission < 0:
        raise ValueError(f"Invalid costs.commission: must be >= 0, got {costs.get('commission')!r}")

    spread_pips = _as_float(costs, "spread_pips", default=None)
    slippage_pips = _as_float(costs, "slippage_pips", default=None)
    if spread_pips is not None and spread_pips < 0:
        raise ValueError(f"Invalid costs.spread_pips: must be >= 0, got {costs.get('spread_pips')!r}")
    if slippage_pips is not None and slippage_pips < 0:
        raise ValueError(f"Invalid costs.slippage_pips: must be >= 0, got {costs.get('slippage_pips')!r}")

    # ---- Risk / Guardrails ----
    risk = cfg.get("risk", {}) or {}
    if risk is not None and not isinstance(risk, dict):
        raise ValueError("Invalid 'risk': must be a mapping/dict if provided")

    if isinstance(risk, dict) and risk:
        # v1 guardrails
        mdl = _as_float(risk, "max_daily_loss_R", default=None)
        if mdl is not None and mdl <= 0:
            raise ValueError(f"Invalid risk.max_daily_loss_R: must be > 0, got {risk.get('max_daily_loss_R')!r}")

        mtd = _as_int(risk, "max_trades_per_day", default=None)
        if mtd is not None and mtd <= 0:
            raise ValueError(f"Invalid risk.max_trades_per_day: must be > 0, got {risk.get('max_trades_per_day')!r}")

        cd = _as_int(risk, "cooldown_bars", default=None)
        if cd is not None and cd < 0:
            raise ValueError(f"Invalid risk.cooldown_bars: must be >= 0, got {risk.get('cooldown_bars')!r}")

        # v2 guardrails (single-position engine policy)
        mcp = _as_int(risk, "max_concurrent_positions", default=1)
        if mcp is None or int(mcp) < 1:
            raise ValueError(
                f"Invalid risk.max_concurrent_positions: must be >= 1, got {risk.get('max_concurrent_positions')!r}"
            )
        if int(mcp) != 1:
            raise ValueError(
                "Invalid risk.max_concurrent_positions: engine currently supports only 1 concurrent position.\n"
                "Fix: set risk.max_concurrent_positions: 1"
            )

        tw = _as_bool(risk, "time_window_enabled", default=False)
        if tw:
            ws = risk.get("window_start_utc")
            we = risk.get("window_end_utc")
            if not ws or not we:
                raise ValueError("time_window_enabled=true requires window_start_utc and window_end_utc")
            _validate_hhmm_utc(str(ws), field="risk.window_start_utc")
            _validate_hhmm_utc(str(we), field="risk.window_end_utc")

        mhb = _as_int(risk, "max_holding_bars", default=0)
        if mhb is not None and mhb < 0:
            raise ValueError(f"Invalid risk.max_holding_bars: must be >= 0, got {risk.get('max_holding_bars')!r}")
