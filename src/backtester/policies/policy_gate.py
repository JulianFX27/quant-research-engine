from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import json
import os

import numpy as np
import pandas as pd


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


def _parse_utc_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid UTC datetime in time_range: {s}")
    return ts


def _labels_q4() -> list[str]:
    return ["Q1", "Q2", "Q3", "Q4"]


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

    STRICT OOS FIX:
      - Engine currently does NOT slice bars by data.time_range (loader has no slicing).
      - PolicyGate MUST therefore enforce time_range itself by blocking entries outside range.
      - PolicyGate MUST keep features indexed consistently with bar indexing (entry_idx).
        => Do NOT shrink the features dataframe.
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

        self.disable_session: Set[int] = set(int(x) for x in rules.get("disable_session_buckets", []))

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
        # Resolve time_range (env-driven, injected by research runner)
        # -----------------------
        env_start = os.getenv("QRE_TIME_RANGE_START")
        env_end = os.getenv("QRE_TIME_RANGE_END")
        self._time_range_start_utc = _parse_utc_ts(env_start)
        self._time_range_end_utc = _parse_utc_ts(env_end)

        # -----------------------
        # Load features (FULL LENGTH) + compute derived columns
        # -----------------------
        required_cols = ["time", "shock_z", "shock_log_ret", "atr_14"]
        fx = pd.read_csv(cfg.features_path, usecols=required_cols)

        if len(fx) == 0:
            raise ValueError("Features file is empty.")

        # Parse time (UTC) and drop invalid timestamps deterministically
        t = pd.to_datetime(fx["time"], utc=True, errors="coerce")
        fx["_ts_utc"] = t
        fx = fx[pd.notna(fx["_ts_utc"])].copy()

        # Sort by timestamp to match bar loader behavior (bar loader sorts by time)
        fx = fx.sort_values("_ts_utc").reset_index(drop=True)

        # shock sign (row-wise, no leakage)
        fx["shock_sign"] = np.where(
            fx["shock_log_ret"] > 0,
            "+",
            np.where(fx["shock_log_ret"] < 0, "-", "0"),
        )

        # session bucket
        fx["session_bucket"] = (
            (fx["_ts_utc"].dt.hour * 60 + fx["_ts_utc"].dt.minute) // int(cfg.time_bucket_min)
        ).astype("Int64")

        # -----------------------
        # STRICT: Calibrate bins on time_range slice ONLY (avoid cross-period leakage)
        # but assign bins for ALL rows (do not change length / index alignment).
        # -----------------------
        slice_mask = pd.Series(True, index=fx.index)
        if self._time_range_start_utc is not None:
            slice_mask &= fx["_ts_utc"] >= self._time_range_start_utc
        if self._time_range_end_utc is not None:
            slice_mask &= fx["_ts_utc"] <= self._time_range_end_utc

        fx_slice = fx.loc[slice_mask]

        if len(fx_slice) == 0:
            raise ValueError(
                "Time-range slice produced 0 feature rows. "
                f"start={env_start}, end={env_end}, path={cfg.features_path}"
            )

        # ---- shock_mag_bin thresholds from slice
        shock_abs_slice = fx_slice["shock_z"].abs()
        if shock_abs_slice.nunique() >= 4:
            q1, q2, q3 = shock_abs_slice.quantile([0.25, 0.50, 0.75]).tolist()
            shock_bins = [-np.inf, float(q1), float(q2), float(q3), np.inf]
            fx["shock_mag_bin"] = pd.cut(
                fx["shock_z"].abs(),
                bins=shock_bins,
                labels=_labels_q4(),
                include_lowest=True,
            )
            self._shock_mag_edges = {"q25": float(q1), "q50": float(q2), "q75": float(q3)}
        else:
            fx["shock_mag_bin"] = "Q1"
            self._shock_mag_edges = {"q25": None, "q50": None, "q75": None}

        # ---- vol_bucket thresholds from slice (tertiles)
        atr_slice = fx_slice["atr_14"]
        if atr_slice.nunique() >= 3:
            v1, v2 = atr_slice.quantile([1/3, 2/3]).tolist()
            vol_bins = [-np.inf, float(v1), float(v2), np.inf]
            fx["vol_bucket"] = pd.cut(
                fx["atr_14"],
                bins=vol_bins,
                labels=["VOL_LOW", "VOL_MED", "VOL_HIGH"],
                include_lowest=True,
            )
            self._vol_edges = {"q33": float(v1), "q66": float(v2)}
        else:
            fx["vol_bucket"] = "VOL_MED"
            self._vol_edges = {"q33": None, "q66": None}

        self._fx = fx

        # -----------------------
        # UTC hour gating
        # -----------------------
        self.allowed_hours_utc: Optional[Set[int]] = None
        if cfg.allowed_hours_utc:
            self.allowed_hours_utc = set(int(h) for h in cfg.allowed_hours_utc)

        self.allowed_time_ranges_utc: Optional[Tuple[Tuple[int, int], ...]] = None
        if cfg.allowed_time_ranges_utc:
            self.allowed_time_ranges_utc = tuple((int(a), int(b)) for a, b in cfg.allowed_time_ranges_utc)

        # Stats
        self.allowed = 0
        self.blocked = 0
        self.blocked_by_time = 0
        self.blocked_by_sv = 0
        self.blocked_by_time_window = 0
        self.blocked_by_timerange = 0  # new: outside [start,end]

    # ============================================================

    def _in_time_range(self, ts: pd.Timestamp) -> bool:
        if pd.isna(ts):
            return False
        if self._time_range_start_utc is not None and ts < self._time_range_start_utc:
            return False
        if self._time_range_end_utc is not None and ts > self._time_range_end_utc:
            return False
        return True

    def evaluate_entry_idx(self, entry_idx: int) -> bool:
        if entry_idx < 0 or entry_idx >= len(self._fx):
            self.blocked += 1
            self.blocked_by_sv += 1
            return False

        row = self._fx.iloc[int(entry_idx)]

        # -------------------------
        # STRICT time_range enforcement
        # -------------------------
        ts = row["_ts_utc"]
        if (self._time_range_start_utc is not None) or (self._time_range_end_utc is not None):
            if not self._in_time_range(ts):
                self.blocked += 1
                self.blocked_by_timerange += 1
                return False

        # -------------------------
        # UTC time window gate (hour filters)
        # -------------------------
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

            # strict slice audit
            "time_range_start_utc": os.getenv("QRE_TIME_RANGE_START"),
            "time_range_end_utc": os.getenv("QRE_TIME_RANGE_END"),
            "features_rows_total": int(len(self._fx)),
            "features_min_ts_utc": str(self._fx["_ts_utc"].min()),
            "features_max_ts_utc": str(self._fx["_ts_utc"].max()),
            "shock_mag_edges": getattr(self, "_shock_mag_edges", None),
            "vol_edges": getattr(self, "_vol_edges", None),

            # gate stats
            "policy_allowed": self.allowed,
            "policy_blocked": self.blocked,
            "policy_blocked_by_timerange": self.blocked_by_timerange,
            "policy_blocked_by_time_window": self.blocked_by_time_window,
            "policy_blocked_by_time": self.blocked_by_time,
            "policy_blocked_by_shock_vol": self.blocked_by_sv,
            "policy_coverage_allowed": (self.allowed / total if total > 0 else None),
        }
