from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from backtester.core.contracts import OrderIntent


class Strategy(ABC):
    """Strategy plug-in interface."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @property
    def warmup_bars(self) -> int:
        return int(self.params.get("warmup_bars", 0))

    @abstractmethod
    def on_bar(self, i: int, df, context: Dict[str, Any]) -> List[OrderIntent]:
        """Called for each bar index i. Return zero or more order intents."""
        raise NotImplementedError
