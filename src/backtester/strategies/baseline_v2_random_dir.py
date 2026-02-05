from __future__ import annotations

from typing import Any, Dict, List
import hashlib

from backtester.core.contracts import OrderIntent
from backtester.strategies.base import Strategy


class BaselineV2RandomDir(Strategy):
    """
    Control baseline (direction-neutral):
      - 1 trade por día (primer bar del día UTC después de warmup).
      - Dirección pseudo-aleatoria PERO determinista (seed fija + fecha).
      - SL/TP fijo en pips.
      - Sin estado oculto, sin IO, sin lookahead.
    """

    def on_bar(self, i: int, df, context: Dict[str, Any]) -> List[OrderIntent]:
        qty = float(self.params.get("qty", 1.0))
        sl_pips = float(self.params.get("sl_pips", 10.0))
        tp_pips = float(self.params.get("tp_pips", 10.0))
        seed = str(self.params.get("seed", "BASELINE_V2_SEED"))
        tag = str(self.params.get("tag", "BASELINE_V2_RANDOM_DIR"))

        instrument = (context.get("instrument") or {})
        pip_size = float(instrument.get("pip_size", 0.0) or 0.0)
        if pip_size <= 0:
            raise ValueError("BaselineV2RandomDir requires instrument.pip_size > 0 when using sl_pips/tp_pips")

        if i < self.warmup_bars:
            return []

        # Solo el primer bar del día UTC
        if i > 0:
            if df.index[i].date() == df.index[i - 1].date():
                return []

        # Dirección determinista por día: hash(seed + YYYY-MM-DD) -> bit
        day = df.index[i].date().isoformat()  # 'YYYY-MM-DD'
        h = hashlib.sha256(f"{seed}|{day}".encode("utf-8")).digest()
        side = "BUY" if (h[0] & 1) == 0 else "SELL"

        close = float(df.iloc[i]["close"])

        if side == "BUY":
            sl = close - sl_pips * pip_size
            tp = close + tp_pips * pip_size
        else:
            sl = close + sl_pips * pip_size
            tp = close - tp_pips * pip_size

        return [
            OrderIntent(
                side=side,
                qty=qty,
                sl_price=sl,
                tp_price=tp,
                tag=f"{tag}|{side}",
            )
        ]
