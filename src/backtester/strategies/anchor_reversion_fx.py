from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from backtester.core.contracts import OrderIntent
from backtester.strategies.base import Strategy


@dataclass(frozen=True)
class AnchorReversionFXConfig:
    # Position sizing
    qty: float = 1.0

    # Entry thresholds
    shock_z_entry: float = 2.0
    min_dist_pips: float = 2.0  # minimum distance from anchor proxy

    # Horizon (hint only; enforcement should be via engine guardrails max_holding_bars)
    horizon_bars: int = 24

    # Sessions gating (only applies if session column exists)
    sessions: tuple[str, ...] = ("LONDON", "NY")
    col_session: str = "session"

    # --- SL MODE CONTROL (CRITICAL) ---
    # "none"                 => time-stop pure: sl_price=None always
    # "model_only"           => use model SL columns if provided; else None
    # "model_or_catastrophic"=> current behavior: model SL else catastrophic SL fallback
    sl_mode: str = "model_or_catastrophic"

    # Catastrophic SL: ONLY used in sl_mode="model_or_catastrophic"
    catastrophic_sl_pips: float = 300.0  # far by default to avoid interfering with edge
    pip_size: float = 0.0001

    # Required market column
    col_close: str = "close"

    # Required feature column
    col_shock_z: str = "shock_z"

    # Distance-to-anchor proxy:
    # - If col_dist_pips exists, interpreted as PIPS by default
    # - If missing, fallback uses abs_close_diff (price) / pip_size
    col_dist_pips: str = "dist_to_anchor_pips"
    dist_is_price: bool = False  # if True, col_dist_pips is in price units and will be converted to pips

    # Optional: model-provided SL (either in pips or price)
    col_model_sl_pips: Optional[str] = None
    col_model_sl_price: Optional[str] = None


class AnchorReversionFX(Strategy):
    name = "AnchorReversionFX"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        p = params or {}
        self.cfg = AnchorReversionFXConfig(**p)

    def on_bar(self, i: int, df: pd.DataFrame, context: Dict[str, Any]) -> List[OrderIntent]:
        cfg = self.cfg
        out: List[OrderIntent] = []

        # ---- Hard requirements ----
        if cfg.col_close not in df.columns:
            raise ValueError(f"AnchorReversionFX requires column '{cfg.col_close}' in df")
        if cfg.col_shock_z not in df.columns:
            raise ValueError(f"AnchorReversionFX requires column '{cfg.col_shock_z}' in df")

        row = df.iloc[i]

        # ---- Session gating (only if session column exists) ----
        if cfg.col_session in df.columns:
            sess = str(row.get(cfg.col_session, "")).upper().strip()
            if sess and cfg.sessions and (sess not in cfg.sessions):
                return out

        # ---- Compute signal inputs ----
        shock_z = _safe_float(row.get(cfg.col_shock_z, 0.0), default=0.0)
        if shock_z is None or abs(shock_z) < cfg.shock_z_entry:
            return out

        dist_pips = self._compute_dist_pips(row, df)
        if dist_pips is None or abs(dist_pips) < cfg.min_dist_pips:
            return out

        # ---- Direction: contrarian vs shock sign ----
        side = "SELL" if float(shock_z) > 0 else "BUY"
        close = _safe_float(row.get(cfg.col_close), default=None)
        if close is None:
            return out

        # ---- SL resolution (depends on sl_mode) ----
        sl_price = self._resolve_sl_price(row, entry_price=float(close), side=side)

        out.append(
            OrderIntent(
                side=side,
                qty=float(cfg.qty),
                sl_price=sl_price,  # may be None in sl_mode="none" or "model_only" with no model SL
                tp_price=None,
                tag=f"ANCHOR_MR_ENTRY_z{float(shock_z):.2f}_d{float(dist_pips):.2f}",
            )
        )
        return out

    def _compute_dist_pips(self, row: pd.Series, df: pd.DataFrame) -> float:
        """
        dist_pips resolution order:
          1) If cfg.col_dist_pips exists: use it (pips by default; price->pips if dist_is_price=True)
          2) Else fallback to abs_close_diff if present (assumed PRICE) => / pip_size
          3) Else 0.0
        """
        cfg = self.cfg

        # 1) Primary
        if cfg.col_dist_pips in df.columns:
            v = _safe_float(row.get(cfg.col_dist_pips, 0.0), default=0.0) or 0.0
            if cfg.dist_is_price:
                return float(v) / cfg.pip_size if cfg.pip_size > 0 else 0.0
            return float(v)

        # 2) Fallback: abs_close_diff (appears in your dataset)
        if "abs_close_diff" in df.columns:
            v_price = _safe_float(row.get("abs_close_diff", 0.0), default=0.0) or 0.0
            return float(v_price) / cfg.pip_size if cfg.pip_size > 0 else 0.0

        return 0.0

    def _resolve_sl_price(self, row: pd.Series, entry_price: float, side: str) -> Optional[float]:
        """
        SL logic depends on cfg.sl_mode:

          - "none":
              returns None always (time-stop pure). Engine must allow missing SL + use risk_proxy_price.

          - "model_only":
              uses model SL if available; else None.

          - "model_or_catastrophic":
              uses model SL if available; else catastrophic fallback.

        Supported model SL:
          - col_model_sl_price: absolute price level
          - col_model_sl_pips: pips distance from entry (directional)
        """
        cfg = self.cfg
        mode = str(cfg.sl_mode or "").lower().strip()

        if mode == "none":
            return None

        # A) model absolute price SL
        if cfg.col_model_sl_price and cfg.col_model_sl_price in row.index:
            v = _safe_float(row.get(cfg.col_model_sl_price), default=None)
            if v is not None and v > 0:
                return float(v)

        # B) model pips SL (distance)
        if cfg.col_model_sl_pips and cfg.col_model_sl_pips in row.index:
            sl_pips = _safe_float(row.get(cfg.col_model_sl_pips), default=None)
            if sl_pips is not None and sl_pips > 0:
                return _sl_from_pips(entry_price, side, float(sl_pips), cfg.pip_size)

        if mode == "model_only":
            return None

        # C) catastrophic fallback (far) only in model_or_catastrophic
        if mode in ("model_or_catastrophic", "catastrophic", "default"):
            if cfg.catastrophic_sl_pips and cfg.catastrophic_sl_pips > 0:
                return _sl_from_pips(entry_price, side, float(cfg.catastrophic_sl_pips), cfg.pip_size)

        return None


def _safe_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    if x is None:
        return default
    try:
        v = float(x)
        return v
    except Exception:
        return default


def _sl_from_pips(entry_price: float, side: str, sl_pips: float, pip_size: float) -> float:
    s = str(side).upper().strip()
    if s == "BUY":
        return float(entry_price) - float(sl_pips) * float(pip_size)
    return float(entry_price) + float(sl_pips) * float(pip_size)
