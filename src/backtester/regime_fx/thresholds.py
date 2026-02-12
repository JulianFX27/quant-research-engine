from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Thresholds:
    er_trend: float
    er_range: float
    dp_trend: float
    dp_range: float

    def validate(self) -> None:
        if not (0.0 <= self.er_range <= 1.0 and 0.0 <= self.er_trend <= 1.0):
            # ER puede ser [0,1] típicamente; si tu ER sale de rango por definiciones, ajusta aquí.
            pass
        if self.er_trend < self.er_range:
            raise ValueError("er_trend must be >= er_range")
        if self.dp_trend < self.dp_range:
            raise ValueError("dp_trend must be >= dp_range")


def load_thresholds_from_artifact(path: str | Path, expected_lookback_er: int, expected_lookback_dp: int) -> Thresholds:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Threshold artifact not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0":
        raise ValueError("Unsupported thresholds artifact schema_version")

    feats = data.get("features", {})
    er_lb = int(feats.get("er", {}).get("lookback", -1))
    dp_lb = int(feats.get("dp", {}).get("lookback", -1))
    if er_lb != expected_lookback_er or dp_lb != expected_lookback_dp:
        raise ValueError(
            f"Artifact lookbacks mismatch. artifact(er={er_lb},dp={dp_lb}) "
            f"expected(er={expected_lookback_er},dp={expected_lookback_dp})"
        )

    thr = data.get("thresholds", {})
    t = Thresholds(
        er_trend=float(thr["er_trend"]),
        er_range=float(thr["er_range"]),
        dp_trend=float(thr["dp_trend"]),
        dp_range=float(thr["dp_range"]),
    )
    t.validate()
    return t


def thresholds_from_fixed(fixed: Dict[str, Any]) -> Thresholds:
    t = Thresholds(
        er_trend=float(fixed["er_trend"]),
        er_range=float(fixed["er_range"]),
        dp_trend=float(fixed["dp_trend"]),
        dp_range=float(fixed["dp_range"]),
    )
    t.validate()
    return t
