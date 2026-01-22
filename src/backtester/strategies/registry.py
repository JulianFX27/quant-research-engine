from __future__ import annotations

from typing import Any, Dict, Type

from backtester.strategies.base import Strategy
from backtester.strategies.example_momentum import ExampleMomentum


REGISTRY: Dict[str, Type[Strategy]] = {
    "ExampleMomentum": ExampleMomentum,
}


def make_strategy(name: str, params: Dict[str, Any]) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](params=params)
