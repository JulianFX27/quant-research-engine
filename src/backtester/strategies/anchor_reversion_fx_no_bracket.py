from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backtester.core.contracts import OrderIntent
from backtester.strategies.base import Strategy


class AnchorReversionFXNoBracket(Strategy):
    """
    Mean-reversion around an anchor level:
      - Entry: when price deviates from anchor by >= entry threshold.
        * If price is above anchor by thr -> SELL
        * If price is below anchor by thr -> BUY
      - Exit:
        * ANCHOR_TOUCH: when abs(price - anchor) <= exit threshold
        * TIME_STOP (optional): when held bars >= max_hold_bars

    Important fix:
      - Even if anchor is NaN/missing on some bars, we STILL enforce TIME_STOP
        while in position. Previously, early-return on NaN anchor could block exits.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params=params or {})

        # ---- required / common params
        self.anchor_col: str = str(self.params.get("anchor_col", "anchor"))
        self.qty: float = float(self.params.get("qty", 1.0))
        self.tag: str = str(self.params.get("tag", self.__class__.__name__))

        # ---- thresholds (either in price or pips for entry; exit in pips)
        self.entry_threshold_price: Optional[float] = self._opt_float(self.params.get("entry_threshold_price"))
        self.entry_threshold_pips: Optional[float] = self._opt_float(self.params.get("entry_threshold_pips"))
        self.exit_threshold_pips: Optional[float] = self._opt_float(self.params.get("exit_threshold_pips", 0.0))

        # ---- bracket (optional)
        self.sl_pips: Optional[float] = self._opt_float(self.params.get("sl_pips"))
        self.tp_pips: Optional[float] = self._opt_float(self.params.get("tp_pips"))

        # ---- warmup / time stop
        self._warmup_bars: int = int(self.params.get("warmup_bars", 0) or 0)
        self.max_hold_bars: int = int(self.params.get("max_hold_bars", 0) or 0)
        if self.max_hold_bars < 0:
            self.max_hold_bars = 0

        # ---- internal state
        self._in_pos: bool = False
        self._pos_side: Optional[str] = None  # "BUY" or "SELL"
        self._entry_i: Optional[int] = None

        # cached pip size if available
        self._pip_size: Optional[float] = None

        # Validate entry threshold resolvability early if user provided in PRICE units
        if self.entry_threshold_price is not None and float(self.entry_threshold_price) <= 0:
            raise ValueError(
                "AnchorReversionFXNoBracket requires params.entry_threshold_price > 0 (price units). "
                f"Got {self.entry_threshold_price!r}"
            )

    # -----------------------------
    # Strategy interface
    # -----------------------------
    def warmup_bars(self) -> int:
        return int(self._warmup_bars)

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _opt_float(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        try:
            return float(x)
        except Exception:
            return None

    def _ensure_pip_size(self, context: Dict[str, Any]) -> Optional[float]:
        """
        Resolve pip_size from context.instrument.pip_size OR context["pip_size"].
        Cache it once found.
        """
        if self._pip_size is not None and self._pip_size > 0:
            return self._pip_size

        ps: Optional[float] = None
        inst = context.get("instrument") if isinstance(context, dict) else None
        if isinstance(inst, dict):
            ps = self._opt_float(inst.get("pip_size"))

        if ps is None:
            ps = self._opt_float(context.get("pip_size"))

        if ps is not None and ps > 0:
            self._pip_size = float(ps)
            return self._pip_size
        return None

    def _entry_thr_price(self, context: Dict[str, Any]) -> Optional[float]:
        """
        Entry threshold in PRICE units.
        Preference:
          1) params.entry_threshold_price
          2) params.entry_threshold_pips * pip_size
        """
        if self.entry_threshold_price is not None:
            return float(self.entry_threshold_price)

        if self.entry_threshold_pips is None:
            return None

        ps = self._ensure_pip_size(context)
        if ps is None:
            return None
        return float(self.entry_threshold_pips) * float(ps)

    def _exit_thr_price(self, context: Dict[str, Any]) -> float:
        """
        Exit threshold in PRICE units. If missing, defaults to 0 (exact touch).
        """
        out = 0.0
        if self.exit_threshold_pips is not None:
            ps = self._ensure_pip_size(context)
            if ps is not None:
                out = float(self.exit_threshold_pips) * float(ps)
        return max(0.0, float(out))

    def _compute_bracket_prices(
        self, side: str, ref_price: float, context: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute SL/TP in PRICE units around ref_price.
        Uses pip_size from context. If missing or params not provided, returns (None, None).
        """
        if self.sl_pips is None or self.tp_pips is None:
            return (None, None)

        ps = self._ensure_pip_size(context)
        if ps is None:
            return (None, None)

        sl_dist = float(self.sl_pips) * float(ps)
        tp_dist = float(self.tp_pips) * float(ps)

        if side == "BUY":
            sl = ref_price - sl_dist
            tp = ref_price + tp_dist
        else:  # SELL
            sl = ref_price + sl_dist
            tp = ref_price - tp_dist

        return (float(sl), float(tp))

    # -----------------------------
    # Main loop
    # -----------------------------
    def on_bar(self, i: int, df: pd.DataFrame, context: Dict[str, Any]) -> List[OrderIntent]:
        if i < self.warmup_bars():
            return []

        # basic column checks
        if "close" not in df.columns:
            raise ValueError("AnchorReversionFXNoBracket: df missing required column 'close'")

        if self.anchor_col not in df.columns:
            raise ValueError(f"AnchorReversionFXNoBracket: missing required column {self.anchor_col!r} in df")

        row = df.iloc[i]
        c = float(row["close"])

        # resolve entry threshold (price units) lazily
        thr_price = self._entry_thr_price(context)
        if thr_price is None or thr_price <= 0:
            raise ValueError(
                "AnchorReversionFXNoBracket could not resolve entry threshold in PRICE units.\n"
                "Fix:\n"
                "  - Provide params.entry_threshold_price > 0, OR\n"
                "  - Provide params.entry_threshold_pips > 0 AND instrument.pip_size > 0.\n"
                f"Got entry_threshold_price={self.entry_threshold_price!r}, entry_threshold_pips={self.entry_threshold_pips!r}, "
                f"pip_size={self._pip_size!r}"
            )

        exit_thr_price = self._exit_thr_price(context)

        intents: List[OrderIntent] = []

        # ----------------
        # EXIT (must run even if anchor is NaN)
        # ----------------
        if self._in_pos:
            # time stop (optional) — DO NOT depend on anchor being present
            if self.max_hold_bars > 0 and (self._entry_i is not None):
                held = int(i - self._entry_i)
                if held >= int(self.max_hold_bars):
                    intents.append(
                        OrderIntent(
                            action="EXIT",
                            exit_reason="TIME_STOP",
                            tag=self.tag,
                        )
                    )
                    self._in_pos = False
                    self._pos_side = None
                    self._entry_i = None
                    return intents

            # anchor-touch exit requires anchor value
            a_raw = row[self.anchor_col]
            if a_raw is None or pd.isna(a_raw):
                # can't evaluate anchor-touch; stay in trade
                return []

            anchor = float(a_raw)
            dist = c - anchor
            if abs(dist) <= float(exit_thr_price):
                intents.append(
                    OrderIntent(
                        action="EXIT",
                        exit_reason="ANCHOR_TOUCH",
                        tag=self.tag,
                    )
                )
                self._in_pos = False
                self._pos_side = None
                self._entry_i = None
                return intents

            return []

        # ----------------
        # ENTRY (requires anchor)
        # ----------------
        a_raw = row[self.anchor_col]
        if a_raw is None or pd.isna(a_raw):
            return []

        anchor = float(a_raw)
        dist = c - anchor  # positive => above anchor

        if dist >= float(thr_price):
            side = "SELL"
            sl_price, tp_price = self._compute_bracket_prices(side=side, ref_price=c, context=context)
            intents.append(
                OrderIntent(
                    action="ENTER",
                    side=side,
                    qty=float(self.qty),
                    sl_price=sl_price,
                    tp_price=tp_price,
                    tag=self.tag,
                )
            )
            self._in_pos = True
            self._pos_side = side
            self._entry_i = i
            return intents

        if dist <= -float(thr_price):
            side = "BUY"
            sl_price, tp_price = self._compute_bracket_prices(side=side, ref_price=c, context=context)
            intents.append(
                OrderIntent(
                    action="ENTER",
                    side=side,
                    qty=float(self.qty),
                    sl_price=sl_price,
                    tp_price=tp_price,
                    tag=self.tag,
                )
            )
            self._in_pos = True
            self._pos_side = side
            self._entry_i = i
            return intents

        return intents
