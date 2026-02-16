from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import pandas as pd
import numpy as np
import json


def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"policy_path not found: {path}")
    if p.suffix.lower() in [".yaml", ".yml"]:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML not installed but policy is YAML. pip install pyyaml") from e
        with open(p, "r") as f:
            return yaml.safe_load(f)
    with open(p, "r") as f:
        return json.load(f)


@dataclass(frozen=True)
class PolicyGateConfig:
    policy_path: str
    features_path: str
    time_bucket_min: int = 30


class PolicyGate:
    """
    Deterministic gating based on:
      - session_bucket (computed from features['time'])
      - shock_sign (from shock_log_ret)
      - shock_mag_bin (Q1..Q4 from abs(shock_z) via qcut on FULL features)
      - vol_bucket (VOL_LOW/MED/HIGH from atr_14 via qcut on FULL features)

    Notes:
      - For v1 we recompute bins using the same dataset used by the run.
      - This is deterministic as long as features file is identical (dataset contract).
    """

    def __init__(self, cfg: PolicyGateConfig):
        self.cfg = cfg
        pol = _load_yaml_or_json(cfg.policy_path)

        self.policy_id: str = pol["policy_id"]
        rules = pol["rules"]

        self.disable_session: Set[int] = set(int(x) for x in rules.get("disable_session_buckets", []))

        enable = rules.get("enable_shock_vol", [])
        self.enable_set: Set[Tuple[str, str, str]] = set(
            (str(r["shock_sign"]), str(r["shock_mag_bin"]), str(r["vol_bucket"])) for r in enable
        )

        # load features columns needed; we rely on row index as join key
        usecols = ["time", "shock_z", "shock_log_ret", "atr_14"]
        fx = pd.read_csv(cfg.features_path, usecols=usecols).reset_index(drop=True)

        # precompute bins for ALL rows -> O(1) lookup per idx
        shock_abs = fx["shock_z"].abs()
        fx["shock_mag_bin"] = pd.qcut(shock_abs, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")

        fx["shock_sign"] = np.where(
            fx["shock_log_ret"] > 0,
            "+",
            np.where(fx["shock_log_ret"] < 0, "-", "0"),
        )

        fx["vol_bucket"] = pd.qcut(fx["atr_14"], 3, labels=["VOL_LOW", "VOL_MED", "VOL_HIGH"], duplicates="drop")

        t = pd.to_datetime(fx["time"], utc=True, errors="coerce")
        fx["session_bucket"] = ((t.dt.hour * 60 + t.dt.minute) // int(cfg.time_bucket_min)).astype("Int64")

        self._fx = fx  # keep for lookup

        self.allowed: int = 0
        self.blocked: int = 0
        self.blocked_by_time: int = 0
        self.blocked_by_sv: int = 0

    def evaluate_entry_idx(self, entry_idx: int) -> bool:
        if entry_idx < 0 or entry_idx >= len(self._fx):
            # conservative: block if out of range
            self.blocked += 1
            self.blocked_by_sv += 1
            return False

        row = self._fx.iloc[int(entry_idx)]
        sb = row["session_bucket"]
        if pd.notna(sb) and int(sb) in self.disable_session:
            self.blocked += 1
            self.blocked_by_time += 1
            return False

        key = (str(row["shock_sign"]), str(row["shock_mag_bin"]), str(row["vol_bucket"]))
        if key in self.enable_set:
            self.allowed += 1
            return True

        self.blocked += 1
        self.blocked_by_sv += 1
        return False

    def stats(self) -> Dict[str, Any]:
        tot = self.allowed + self.blocked
        return {
            "policy_id": self.policy_id,
            "policy_path": self.cfg.policy_path,
            "features_path": self.cfg.features_path,
            "policy_allowed": self.allowed,
            "policy_blocked": self.blocked,
            "policy_blocked_by_time": self.blocked_by_time,
            "policy_blocked_by_shock_vol": self.blocked_by_sv,
            "policy_coverage_allowed": (self.allowed / tot) if tot > 0 else None,
        }
