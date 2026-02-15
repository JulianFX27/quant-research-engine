# src/backtester/regime_fx/contract.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _as_int(x: Any, *, field: str, default: int) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, str) and x.strip() == "":
            return int(default)
        return int(x)
    except Exception as e:
        raise ValueError(f"Invalid {field}: expected int-like, got {x!r}") from e


def _as_float(x: Any, *, field: str, default: float) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, str) and x.strip() == "":
            return float(default)
        return float(x)
    except Exception as e:
        raise ValueError(f"Invalid {field}: expected float-like, got {x!r}") from e


def _as_str(x: Any, *, field: str, default: str) -> str:
    if x is None:
        return str(default)
    s = str(x)
    if not s.strip():
        return str(default)
    return s


@dataclass(frozen=True)
class FeatureConfig:
    lookback: int = 12

    def __post_init__(self) -> None:
        if int(self.lookback) <= 0:
            raise ValueError(f"lookback must be > 0, got {self.lookback}")


@dataclass(frozen=True)
class ThresholdsFixed:
    er_trend: float = 0.60
    dp_trend: float = 0.60
    er_range: float = 0.30
    dp_range: float = 0.30


@dataclass(frozen=True)
class ThresholdsConfig:
    # mode: "artifact" or "fixed"
    mode: str = "fixed"
    artifact_path: str = ""
    fixed: ThresholdsFixed = ThresholdsFixed()

    def __post_init__(self) -> None:
        m = str(self.mode).strip().lower()
        if m not in ("artifact", "fixed"):
            raise ValueError(f"thresholds.mode must be 'artifact' or 'fixed', got {self.mode!r}")
        object.__setattr__(self, "mode", m)
        if m == "artifact":
            ap = str(self.artifact_path).strip()
            if not ap:
                raise ValueError("thresholds.mode='artifact' requires thresholds.artifact_path non-empty")
            object.__setattr__(self, "artifact_path", ap)


@dataclass(frozen=True)
class HysteresisConfig:
    enabled: bool = True
    margin_frac: float = 0.10

    def __post_init__(self) -> None:
        if float(self.margin_frac) < 0:
            raise ValueError(f"hysteresis.margin_frac must be >= 0, got {self.margin_frac}")


@dataclass(frozen=True)
class StateMachineConfig:
    enabled: bool = True
    confirm_bars: int = 2
    min_state_bars: int = 9
    cooldown_bars: int = 9
    hysteresis: HysteresisConfig = HysteresisConfig()

    def __post_init__(self) -> None:
        if int(self.confirm_bars) < 0:
            raise ValueError("state_machine.confirm_bars must be >= 0")
        if int(self.min_state_bars) < 0:
            raise ValueError("state_machine.min_state_bars must be >= 0")
        if int(self.cooldown_bars) < 0:
            raise ValueError("state_machine.cooldown_bars must be >= 0")


@dataclass(frozen=True)
class OutputCols:
    er: str = "er"
    dp: str = "dp"
    raw_regime: str = "raw_regime"
    state_regime: str = "state_regime"


@dataclass(frozen=True)
class RegimeFXConfig:
    version: str = "v0.1.0"
    er: FeatureConfig = FeatureConfig(lookback=12)
    dp: FeatureConfig = FeatureConfig(lookback=12)
    thresholds: ThresholdsConfig = ThresholdsConfig()
    state_machine: StateMachineConfig = StateMachineConfig()
    output_cols: OutputCols = OutputCols()


def load_regime_fx_config(path: Path) -> RegimeFXConfig:
    """
    Load a regime_fx YAML config from disk.
    Expected top-level keys:
      version, er, dp, thresholds, state_machine, output_cols
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"RegimeFX config not found: {p.resolve()}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("RegimeFX config file must parse to a dict at top-level")

    version = _as_str(raw.get("version"), field="version", default="v0.1.0")

    er_d = raw.get("er", {}) or {}
    dp_d = raw.get("dp", {}) or {}
    if not isinstance(er_d, dict) or not isinstance(dp_d, dict):
        raise ValueError("er/dp must be dicts")

    er = FeatureConfig(lookback=_as_int(er_d.get("lookback"), field="er.lookback", default=12))
    dp = FeatureConfig(lookback=_as_int(dp_d.get("lookback"), field="dp.lookback", default=12))

    thr_d = raw.get("thresholds", {}) or {}
    if not isinstance(thr_d, dict):
        raise ValueError("thresholds must be dict")

    thr_mode = _as_str(thr_d.get("mode"), field="thresholds.mode", default="fixed").lower()
    thr_artifact = _as_str(thr_d.get("artifact_path"), field="thresholds.artifact_path", default="")

    fixed_d = thr_d.get("fixed", {}) or {}
    if not isinstance(fixed_d, dict):
        raise ValueError("thresholds.fixed must be dict")

    fixed = ThresholdsFixed(
        er_trend=_as_float(fixed_d.get("er_trend"), field="thresholds.fixed.er_trend", default=0.60),
        dp_trend=_as_float(fixed_d.get("dp_trend"), field="thresholds.fixed.dp_trend", default=0.60),
        er_range=_as_float(fixed_d.get("er_range"), field="thresholds.fixed.er_range", default=0.30),
        dp_range=_as_float(fixed_d.get("dp_range"), field="thresholds.fixed.dp_range", default=0.30),
    )
    thresholds = ThresholdsConfig(mode=thr_mode, artifact_path=thr_artifact, fixed=fixed)

    sm_d = raw.get("state_machine", {}) or {}
    if not isinstance(sm_d, dict):
        raise ValueError("state_machine must be dict")

    hyst_d = sm_d.get("hysteresis", {}) or {}
    if not isinstance(hyst_d, dict):
        raise ValueError("state_machine.hysteresis must be dict")

    hysteresis = HysteresisConfig(
        enabled=bool(hyst_d.get("enabled", True)),
        margin_frac=_as_float(hyst_d.get("margin_frac"), field="state_machine.hysteresis.margin_frac", default=0.10),
    )

    state_machine = StateMachineConfig(
        enabled=bool(sm_d.get("enabled", True)),
        confirm_bars=_as_int(sm_d.get("confirm_bars"), field="state_machine.confirm_bars", default=2),
        min_state_bars=_as_int(sm_d.get("min_state_bars"), field="state_machine.min_state_bars", default=9),
        cooldown_bars=_as_int(sm_d.get("cooldown_bars"), field="state_machine.cooldown_bars", default=9),
        hysteresis=hysteresis,
    )

    oc_d = raw.get("output_cols", {}) or {}
    if not isinstance(oc_d, dict):
        raise ValueError("output_cols must be dict")

    output_cols = OutputCols(
        er=_as_str(oc_d.get("er"), field="output_cols.er", default="er"),
        dp=_as_str(oc_d.get("dp"), field="output_cols.dp", default="dp"),
        raw_regime=_as_str(oc_d.get("raw_regime"), field="output_cols.raw_regime", default="raw_regime"),
        state_regime=_as_str(oc_d.get("state_regime"), field="output_cols.state_regime", default="state_regime"),
    )

    return RegimeFXConfig(
        version=version,
        er=er,
        dp=dp,
        thresholds=thresholds,
        state_machine=state_machine,
        output_cols=output_cols,
    )
