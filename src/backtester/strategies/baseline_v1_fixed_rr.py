from __future__ import annotations

from typing import Any, Dict, List

from backtester.core.contracts import OrderIntent
from backtester.strategies.base import Strategy


class BaselineV1FixedRR(Strategy):
    """
    Baseline control:
      - Emite 1 intención por DÍA (en el primer bar de cada día) después del warmup.
      - Dirección fija (param side: BUY/SELL).
      - SL/TP fijo en pips (RR 1:1 por defecto).
      - Determinista, sin IO, sin lookahead, sin estado oculto (decisión solo por df/index).
    """

    def on_bar(self, i: int, df, context: Dict[str, Any]) -> List[OrderIntent]:
        qty = float(self.params.get("qty", 1.0))
        sl_pips = float(self.params.get("sl_pips", 10.0))
        tp_pips = float(self.params.get("tp_pips", 10.0))
        side = str(self.params.get("side", "BUY")).upper()
        tag = str(self.params.get("tag", "BASELINE_V1"))

        instrument = (context.get("instrument") or {})
        pip_size = float(instrument.get("pip_size", 0.0) or 0.0)
        if pip_size <= 0:
            raise ValueError("BaselineV1FixedRR requires instrument.pip_size > 0 when using sl_pips/tp_pips")

        # Warmup: no intenciones antes de warmup_bars
        if i < self.warmup_bars:
            return []

        # Emitir solo en el "primer bar del día" (UTC) para no spamear intents.
        # Esto reduce ruido en métricas de entry_gate_attempted_entries.
        if i > 0:
            d0 = df.index[i - 1].date()
            d1 = df.index[i].date()
            if d1 == d0:
                return []

        close = float(df.iloc[i]["close"])

        if side == "BUY":
            sl = close - sl_pips * pip_size
            tp = close + tp_pips * pip_size
        elif side == "SELL":
            sl = close + sl_pips * pip_size
            tp = close - tp_pips * pip_size
        else:
            raise ValueError(f"Invalid side: {side}. Expected BUY or SELL.")

        return [
            OrderIntent(
                side=side,
                qty=qty,
                sl_price=sl,
                tp_price=tp,
                tag=tag,
            )
        ]
