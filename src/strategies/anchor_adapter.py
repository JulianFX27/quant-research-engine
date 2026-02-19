from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime

from src.runner.interfaces import Bar, StrategyContext, OrderIntent, Direction


@dataclass
class AnchorAdapterConfig:
    anchor_col: str = "anchor"
    entry_threshold_pips: float = 5.0
    exit_threshold_pips: float = 0.0
    sl_pips: float = 20.0
    tp_pips: float = 20.0
    warmup_bars: int = 0
    max_hold_bars: int = 0
    tag: str = "AnchorReversionAdapter"


class AnchorReversionAdapter:
    """
    Bar-by-bar adapter:

    ENTRY:
      dist = close - anchor
      dist >= entry_thr -> SHORT
      dist <= -entry_thr -> LONG

    EXIT:
      - TIME_STOP if held >= max_hold_bars
      - ANCHOR_TOUCH if abs(close-anchor) <= exit_thr
      - SL/TP handled by engine intrabar

    Requires anchor in Bar.extras[anchor_col].
    """

    name = "AnchorReversionAdapter"
    version = "0.1.0"
    instrument = "EURUSD"

    def __init__(self, cfg: Optional[AnchorAdapterConfig] = None):
        self.cfg = cfg or AnchorAdapterConfig()

        self._i = -1
        self._in_pos = False
        self._entry_i: Optional[int] = None
        self._side: Optional[Direction] = None

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> Optional[OrderIntent]:
        self._i += 1
        cfg = self.cfg

        if self._i < cfg.warmup_bars:
            return None

        anchor_raw = bar.extras.get(cfg.anchor_col, None)
        anchor = float(anchor_raw) if anchor_raw is not None else None

        pip = float(ctx.pip_size)
        entry_thr = float(cfg.entry_threshold_pips) * pip
        exit_thr = float(cfg.exit_threshold_pips) * pip

        c = float(bar.close)

        # ---- EXIT logic (can run even if anchor missing for TIME_STOP)
        if self._in_pos:
            if cfg.max_hold_bars > 0 and self._entry_i is not None:
                held = self._i - self._entry_i
                if held >= cfg.max_hold_bars:
                    # discretionary exit at bar close
                    return OrderIntent(
                        intent_id=f"EXIT_{bar.ts_utc.isoformat()}",
                        ts_utc=bar.ts_utc,
                        action="EXIT",
                        exit_reason="TIME_STOP",
                        meta={"tag": cfg.tag},
                    )

            # anchor-touch exit requires anchor
            if anchor is None:
                return None

            if abs(c - anchor) <= exit_thr:
                return OrderIntent(
                    intent_id=f"EXIT_{bar.ts_utc.isoformat()}",
                    ts_utc=bar.ts_utc,
                    action="EXIT",
                    exit_reason="ANCHOR_TOUCH",
                    meta={"tag": cfg.tag},
                )

            return None

        # ---- ENTRY logic (requires anchor)
        if anchor is None:
            return None

        dist = c - anchor  # + above anchor

        if dist >= entry_thr:
            direction: Direction = "SHORT"
            sl = c + float(cfg.sl_pips) * pip
            tp = c - float(cfg.tp_pips) * pip
            self._in_pos = True
            self._entry_i = self._i
            self._side = direction
            return OrderIntent(
                intent_id=f"ENT_{bar.ts_utc.isoformat()}",
                ts_utc=bar.ts_utc,
                action="ENTER",
                direction=direction,
                sl_price=sl,
                tp_price=tp,
                meta={"tag": cfg.tag, "anchor": anchor, "dist": dist},
            )

        if dist <= -entry_thr:
            direction = "LONG"
            sl = c - float(cfg.sl_pips) * pip
            tp = c + float(cfg.tp_pips) * pip
            self._in_pos = True
            self._entry_i = self._i
            self._side = direction
            return OrderIntent(
                intent_id=f"ENT_{bar.ts_utc.isoformat()}",
                ts_utc=bar.ts_utc,
                action="ENTER",
                direction=direction,
                sl_price=sl,
                tp_price=tp,
                meta={"tag": cfg.tag, "anchor": anchor, "dist": dist},
            )

        return None

    def on_trade_closed_reset(self):
        # called by runner when engine closes a trade via TP/SL/force
        self._in_pos = False
        self._entry_i = None
        self._side = None
