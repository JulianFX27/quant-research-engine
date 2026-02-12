from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml


WarmupPolicy = Literal["nan", "zero"]
ThresholdMode = Literal["artifact", "fixed"]


@dataclass(frozen=True)
class FeatureSpec:
    lookback: int
    warmup: WarmupPolicy = "nan"


@dataclass(frozen=True)
class ThresholdsFixed:
    er_trend: float
    er_range: float
    dp_trend: float
    dp_range: float


@dataclass(frozen=True)
class ThresholdsSpec:
    mode: ThresholdMode
    artifact_path: Optional[str] = None
    fixed: Optional[ThresholdsFixed] = None


@dataclass(frozen=True)
class HysteresisSpec:
    enabled: bool = True
    margin_frac: float = 0.10


@dataclass(frozen=True)
class StateMachineSpec:
    enabled: bool = True
    confirm_bars: int = 2
    min_state_bars: int = 9
    cooldown_bars: int = 9
    hysteresis: HysteresisSpec = HysteresisSpec()


@dataclass(frozen=True)
class OutputColumns:
    er: str = "er"
    dp: str = "dp"
    raw_regime: str = "raw_regime"
    state_regime: str = "state_regime"


@dataclass(frozen=True)
class RegimeFXConfig:
    version: str
    er: FeatureSpec
    dp: FeatureSpec
    thresholds: ThresholdsSpec
    state_machine: StateMachineSpec
    output_cols: OutputColumns

    @property
    def lookback_max(self) -> int:
        return max(self.er.lookback, self.dp.lookback)


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key: {key}")
    return d[key]


def load_regime_fx_config(path: str | Path) -> RegimeFXConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Regime FX config not found: {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Invalid YAML: expected mapping at top level")

    version = str(_require(raw, "version"))

    feats = _require(raw, "features")
    er_raw = _require(feats, "er")
    dp_raw = _require(feats, "dp")

    er = FeatureSpec(
        lookback=int(_require(er_raw, "lookback")),
        warmup=str(er_raw.get("warmup", "nan")),
    )
    dp = FeatureSpec(
        lookback=int(_require(dp_raw, "lookback")),
        warmup=str(dp_raw.get("warmup", "nan")),
    )
    if er.lookback <= 1 or dp.lookback <= 1:
        raise ValueError("lookback must be > 1")

    thr_raw = _require(raw, "thresholds")
    mode = str(_require(thr_raw, "mode"))
    if mode not in ("artifact", "fixed"):
        raise ValueError("thresholds.mode must be 'artifact' or 'fixed'")

    fixed_obj = None
    artifact_path = thr_raw.get("artifact_path")
    if mode == "fixed":
        fx = _require(thr_raw, "fixed")
        fixed_obj = ThresholdsFixed(
            er_trend=float(_require(fx, "er_trend")),
            er_range=float(_require(fx, "er_range")),
            dp_trend=float(_require(fx, "dp_trend")),
            dp_range=float(_require(fx, "dp_range")),
        )
    else:
        if not artifact_path:
            raise ValueError("thresholds.artifact_path required when mode='artifact'")

    thresholds = ThresholdsSpec(mode=mode, artifact_path=artifact_path, fixed=fixed_obj)

    sm_raw = raw.get("state_machine", {})
    hyst_raw = sm_raw.get("hysteresis", {})
    hysteresis = HysteresisSpec(
        enabled=bool(hyst_raw.get("enabled", True)),
        margin_frac=float(hyst_raw.get("margin_frac", 0.10)),
    )
    if not (0.0 <= hysteresis.margin_frac <= 1.0):
        raise ValueError("hysteresis.margin_frac must be in [0, 1]")

    state_machine = StateMachineSpec(
        enabled=bool(sm_raw.get("enabled", True)),
        confirm_bars=int(sm_raw.get("confirm_bars", 2)),
        min_state_bars=int(sm_raw.get("min_state_bars", 9)),
        cooldown_bars=int(sm_raw.get("cooldown_bars", 9)),
        hysteresis=hysteresis,
    )
    if state_machine.confirm_bars < 1:
        raise ValueError("confirm_bars must be >= 1")
    if state_machine.min_state_bars < 0 or state_machine.cooldown_bars < 0:
        raise ValueError("min_state_bars/cooldown_bars must be >= 0")

    out_raw = raw.get("output", {}).get("columns", {})
    output_cols = OutputColumns(
        er=str(out_raw.get("er", "er")),
        dp=str(out_raw.get("dp", "dp")),
        raw_regime=str(out_raw.get("raw_regime", "raw_regime")),
        state_regime=str(out_raw.get("state_regime", "state_regime")),
    )

    return RegimeFXConfig(
        version=version,
        er=er,
        dp=dp,
        thresholds=thresholds,
        state_machine=state_machine,
        output_cols=output_cols,
    )
