from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from backtester.core.contracts import OrderIntent
from backtester.strategies.base import Strategy


class ExampleMomentum(Strategy):
    """Toy example: if close > SMA(n) -> buy; if close < SMA(n) -> sell.

    This is only a scaffold to prove wiring end-to-end.
    """

    def on_bar(self, i: int, df, context: Dict[str, Any]) -> List[OrderIntent]:
        n = int(self.params.get("sma_n", 20))
        qty = float(self.params.get("qty", 1.0))
        sl_pips = float(self.params.get("sl_pips", 10))
        tp_pips = float(self.params.get("tp_pips", 20))
        pip = float(self.params.get("pip", 0.0001))

        if i < max(self.warmup_bars, n):
            return []

        close = float(df.iloc[i]["close"])
        sma = float(np.mean(df.iloc[i - n + 1 : i + 1]["close"]))

        if close > sma:
            return [
                OrderIntent(
                    side="BUY",
                    qty=qty,
                    sl_price=close - sl_pips * pip,
                    tp_price=close + tp_pips * pip,
                    tag=f"sma{n}_long",
                )
            ]
        if close < sma:
            return [
                OrderIntent(
                    side="SELL",
                    qty=qty,
                    sl_price=close + sl_pips * pip,
                    tp_price=close - tp_pips * pip,
                    tag=f"sma{n}_short",
                )
            ]
        return []
