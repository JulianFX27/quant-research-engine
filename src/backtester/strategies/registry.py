from __future__ import annotations

from typing import Any, Dict, Type

from backtester.strategies.base import Strategy
from backtester.strategies.example_momentum import ExampleMomentum
from backtester.strategies.baseline_v1_fixed_rr import BaselineV1FixedRR
from backtester.strategies.baseline_v2_random_dir import BaselineV2RandomDir
from backtester.strategies.baseline_v4_vol_bucket_buy import BaselineV4VolBucketBuy


REGISTRY: Dict[str, Type[Strategy]] = {
    "ExampleMomentum": ExampleMomentum,
    "BaselineV1FixedRR": BaselineV1FixedRR,
    "BaselineV2RandomDir": BaselineV2RandomDir,
    "BaselineV4VolBucketBuy": BaselineV4VolBucketBuy,
}


def make_strategy(name: str, params: Dict[str, Any]) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](params=params)
