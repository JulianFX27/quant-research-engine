# src/backtester/regime_fx/pipeline.py
from __future__ import annotations

import pandas as pd

from .contract import RegimeFXConfig
from .features import compute_dp, compute_er
from .thresholds import Thresholds, load_thresholds_from_artifact, thresholds_from_fixed
from .state_machine import StateMachineParams, apply_state_machine


def classify_raw_regime(er: pd.Series, dp: pd.Series, thresholds: Thresholds) -> pd.Series:
    """
    Map ER/DP to raw regime labels using threshold bands.

    Output labels:
      - "TREND"
      - "RANGE"
      - "NO_TRADE"
    """
    regime = pd.Series(index=er.index, dtype="object")

    trend = (er >= thresholds.er_trend) & (dp >= thresholds.dp_trend)
    rng = (er <= thresholds.er_range) & (dp <= thresholds.dp_range)

    regime.loc[trend] = "TREND"
    regime.loc[rng] = "RANGE"
    regime = regime.fillna("NO_TRADE")
    return regime


def build_regime_frame(df: pd.DataFrame, cfg: RegimeFXConfig) -> pd.DataFrame:
    """
    Build deterministic regime enrichment frame:
      - compute ER/DP features
      - load thresholds (artifact/fixed)
      - compute raw_regime
      - apply state machine (optional)
      - write output columns as per cfg.output_cols
    """
    df2 = df.copy()

    # 0) Preconditions
    if "close" not in df2.columns:
        raise ValueError("regime_fx: df must contain 'close' column")

    # 1) Features
    er = compute_er(df2["close"], lookback=cfg.er.lookback)
    dp = compute_dp(df2["close"], lookback=cfg.dp.lookback)

    df2[cfg.output_cols.er] = er
    df2[cfg.output_cols.dp] = dp

    # 2) Thresholds (artifact must match feature lookbacks)
    if cfg.thresholds.mode == "artifact":
        thresholds = load_thresholds_from_artifact(
            cfg.thresholds.artifact_path,
            expected_lookback_er=cfg.er.lookback,
            expected_lookback_dp=cfg.dp.lookback,
        )
    else:
        thresholds = thresholds_from_fixed(cfg.thresholds.fixed)

    # 3) Raw regime
    raw_regime = classify_raw_regime(
        er=df2[cfg.output_cols.er],
        dp=df2[cfg.output_cols.dp],
        thresholds=thresholds,
    )
    df2[cfg.output_cols.raw_regime] = raw_regime

    # 4) State machine (NOTE: signature requires er/dp/thresholds too)
    if cfg.state_machine.enabled:
        params = StateMachineParams(
            confirm_bars=int(cfg.state_machine.confirm_bars),
            min_state_bars=int(cfg.state_machine.min_state_bars),
            cooldown_bars=int(cfg.state_machine.cooldown_bars),
            hysteresis_enabled=bool(cfg.state_machine.hysteresis.enabled),
            hysteresis_margin_frac=float(cfg.state_machine.hysteresis.margin_frac),
        )
        state_regime = apply_state_machine(
            raw_regime=raw_regime,
            er=df2[cfg.output_cols.er],
            dp=df2[cfg.output_cols.dp],
            thr=thresholds,
            params=params,
        )
    else:
        state_regime = raw_regime

    df2[cfg.output_cols.state_regime] = state_regime
    return df2
