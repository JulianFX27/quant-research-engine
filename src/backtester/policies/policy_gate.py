from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import pandas as pd
import numpy as np
import json


# ============================================================
# Utilities
# ============================================================

def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"policy_path not found: {path}")

    if p.suffix.lower() in [".yaml", ".yml"]:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("PyYAML not installed but policy is YAML.") from e

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError("Policy file must contain a top-level mapping.")

    return data


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class PolicyGateConfig:
    policy_path: str
    features_path: str
    time_bucket_min: int = 30

    allowed_hours_utc: Optional[Iterable[int]] = None
    allowed_time_ranges_utc: Optional[Iterable[Tuple[int, int]]] = None


# ============================================================
# PolicyGate
# ============================================================

class PolicyGate:
    """
    Deterministic gating based on:
        - session_bucket
        - shock_sign
        - shock_mag_bin (quartiles of |shock_z|)
        - vol_bucket (tertiles of atr_14)
        - optional UTC hour filters
    """

    def __init__(self, cfg: PolicyGateConfig):
        self.cfg = cfg

        # -----------------------
        # Load policy rules
        # -----------------------
        pol = _load_yaml_or_json(cfg.policy_path)

        if "policy_id" not in pol:
            raise ValueError("Policy file missing 'policy_id'.")

        if "rules" not in pol or not isinstance(pol["rules"], dict):
            raise ValueError("Policy file missing valid 'rules' section.")

        self.policy_id = str(pol["policy_id"])
        rules = pol["rules"]

        self.disable_session: Set[int] = set(
            int(x) for x in rules.get("disable_session_buckets", [])
        )

        enable = rules.get("enable_shock_vol", [])
        if not isinstance(enable, list):
            raise ValueError("'enable_shock_vol' must be list")

        self.enable_set: Set[Tuple[str, str, str]] = set()
        for r in enable:
            self.enable_set.add((
                str(r["shock_sign"]),
                str(r["shock_mag_bin"]),
                str(r["vol_bucket"]),
            ))

        # -----------------------
        # Load features
        # -----------------------
        required_cols = ["time", "shock_z", "shock_log_ret", "atr_14"]
        fx = pd.read_csv(cfg.features_path, usecols=required_cols)

        if len(fx) == 0:
            raise ValueError("Features file is empty.")

        fx = fx.reset_index(drop=True)

        # ---- Compute bins safely ----

        # shock magnitude bins
        shock_abs = fx["shock_z"].abs()

        if shock_abs.nunique() >= 4:
            fx["shock_mag_bin"] = pd.qcut(
                shock_abs,
                4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            )
        else:
            fx["shock_mag_bin"] = "Q1"

        # shock sign
        fx["shock_sign"] = np.where(
            fx["shock_log_ret"] > 0,
            "+",
            np.where(fx["shock_log_ret"] < 0, "-", "0"),
        )

        # volatility bins
        if fx["atr_14"].nunique() >= 3:
            fx["vol_bucket"] = pd.qcut(
                fx["atr_14"],
                3,
                labels=["VOL_LOW", "VOL_MED", "VOL_HIGH"],
                duplicates="drop",
            )
        else:
            fx["vol_bucket"] = "VOL_MED"

        # Time parsing
        t = pd.to_datetime(fx["time"], utc=True, errors="coerce")
        fx["_ts_utc"] = t

        fx["session_bucket"] = (
            (t.dt.hour * 60 + t.dt.minute) // int(cfg.time_bucket_min)
        ).astype("Int64")

        self._fx = fx

        # -----------------------
        # UTC hour gating
        # -----------------------
        self.allowed_hours_utc: Optional[Set[int]] = None
        if cfg.allowed_hours_utc:
            self.allowed_hours_utc = set(int(h) for h in cfg.allowed_hours_utc)

        self.allowed_time_ranges_utc: Optional[Tuple[Tuple[int, int], ...]] = None
        if cfg.allowed_time_ranges_utc:
            self.allowed_time_ranges_utc = tuple(
                (int(a), int(b)) for a, b in cfg.allowed_time_ranges_utc
            )

        # Stats
        self.allowed = 0
        self.blocked = 0
        self.blocked_by_time = 0
        self.blocked_by_sv = 0
        self.blocked_by_time_window = 0

    # ============================================================

    def evaluate_entry_idx(self, entry_idx: int) -> bool:

        if entry_idx < 0 or entry_idx >= len(self._fx):
            self.blocked += 1
            self.blocked_by_sv += 1
            return False

        row = self._fx.iloc[int(entry_idx)]

        # -------------------------
        # UTC time window gate
        # -------------------------
        ts = row["_ts_utc"]
        hour = None
        if pd.notna(ts):
            hour = int(ts.hour)

        if self.allowed_hours_utc is not None:
            if hour not in self.allowed_hours_utc:
                self.blocked += 1
                self.blocked_by_time_window += 1
                return False

        if self.allowed_time_ranges_utc is not None:
            ok = False
            if hour is not None:
                for a, b in self.allowed_time_ranges_utc:
                    if a <= hour < b:
                        ok = True
                        break
            if not ok:
                self.blocked += 1
                self.blocked_by_time_window += 1
                return False

        # -------------------------
        # Session bucket disable
        # -------------------------
        sb = row["session_bucket"]
        if pd.notna(sb) and int(sb) in self.disable_session:
            self.blocked += 1
            self.blocked_by_time += 1
            return False

        # -------------------------
        # Shock/Vol rule
        # -------------------------
        key = (
            str(row["shock_sign"]),
            str(row["shock_mag_bin"]),
            str(row["vol_bucket"]),
        )

        if key in self.enable_set:
            self.allowed += 1
            return True

        self.blocked += 1
        self.blocked_by_sv += 1
        return False

    # ============================================================

    def stats(self) -> Dict[str, Any]:
        total = self.allowed + self.blocked
        return {
            "policy_id": self.policy_id,
            "policy_path": self.cfg.policy_path,
            "features_path": self.cfg.features_path,
            "time_bucket_min": self.cfg.time_bucket_min,
            "policy_allowed": self.allowed,
            "policy_blocked": self.blocked,
            "policy_blocked_by_time_window": self.blocked_by_time_window,
            "policy_blocked_by_time": self.blocked_by_time,
            "policy_blocked_by_shock_vol": self.blocked_by_sv,
            "policy_coverage_allowed": (
                self.allowed / total if total > 0 else None
            ),
        }
