# src/backtester/orchestrator/regime_fx_hook.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from backtester.regime_fx.contract import load_regime_fx_config, RegimeFXConfig
from backtester.regime_fx.pipeline import build_regime_frame


def _sha256_text(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def _norm_int(x: Any, default: int) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, str) and x.strip() == "":
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _apply_warmup_nan(df: pd.DataFrame, cols: list[str], warmup_bars: int) -> None:
    if warmup_bars <= 0:
        return
    n = min(int(warmup_bars), len(df))
    if n <= 0:
        return
    for c in cols:
        if c in df.columns:
            df.loc[df.index[:n], c] = pd.NA


def attach_regime_fx_if_enabled(
    df: pd.DataFrame,
    cfg_resolved: Dict[str, Any],
    *,
    dataset_id: str,
) -> Tuple[pd.DataFrame, Dict[str, Any] | None]:
    """
    Deterministic enrichment hook: attaches regime_fx columns to df.

    Requires in run YAML:
      regime_fx:
        enabled: true
        config_path: "configs/regime_fx.yaml"
        warmup_bars: 300   # optional extra warmup on top of feature warmups

    Output columns are defined by RegimeFXConfig.output_cols:
      - er
      - dp
      - raw_regime
      - state_regime
    """
    rx = (cfg_resolved.get("regime_fx") or {}) if isinstance(cfg_resolved, dict) else {}
    if not bool(rx.get("enabled", False)):
        return df, None

    config_path = rx.get("config_path")
    if not config_path:
        raise ValueError("regime_fx.enabled=true requires regime_fx.config_path")

    p = Path(str(config_path))
    if not p.exists():
        raise FileNotFoundError(f"regime_fx config_path not found: {p.resolve()}")

    cfg: RegimeFXConfig = load_regime_fx_config(p)

    # Build full regime frame (this must handle thresholds internally)
    df2 = build_regime_frame(df, cfg)

    # Optional additional warmup at orchestrator layer
    warmup_bars = _norm_int(rx.get("warmup_bars"), default=0)
    if warmup_bars > 0:
        warm_cols = [
            cfg.output_cols.er,
            cfg.output_cols.dp,
            cfg.output_cols.raw_regime,
            cfg.output_cols.state_regime,
        ]
        _apply_warmup_nan(df2, warm_cols, warmup_bars)

    # Manifest for auditability
    cfg_text = Path(p).read_text(encoding="utf-8")
    manifest = {
        "enabled": True,
        "dataset_id": dataset_id,
        "status": "computed",
        "config_path": str(p),
        "config_sha256": _sha256_text(cfg_text),
        "config_version": cfg.version,
        "lookbacks": {"er": int(cfg.er.lookback), "dp": int(cfg.dp.lookback)},
        "thresholds_mode": cfg.thresholds.mode,
        "state_machine": {
            "enabled": bool(cfg.state_machine.enabled),
            "confirm_bars": int(cfg.state_machine.confirm_bars),
            "min_state_bars": int(cfg.state_machine.min_state_bars),
            "cooldown_bars": int(cfg.state_machine.cooldown_bars),
            "hysteresis": {
                "enabled": bool(cfg.state_machine.hysteresis.enabled),
                "margin_frac": float(cfg.state_machine.hysteresis.margin_frac),
            },
        },
        "output_cols": asdict(cfg.output_cols),
        "hook_warmup_bars": int(warmup_bars),
    }

    return df2, manifest
