from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np

from backtester.execution.engine import Trade


def summarize_trades(trades: List[Trade]) -> Dict[str, Any]:
    if not trades:
        return {"n_trades": 0}

    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    out: Dict[str, Any] = {
        "n_trades": int(len(trades)),
        "total_pnl": float(pnls.sum()),
        "winrate": float((pnls > 0).mean()),
        "avg_pnl": float(pnls.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 1e-12 else float("inf"),
    }
    return out


def trades_to_dicts(trades: List[Trade]) -> List[Dict[str, Any]]:
    return [asdict(t) for t in trades]
