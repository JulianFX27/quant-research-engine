from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ExecutionPolicy:
    policy_id: str
    description: str
    overrides: Dict[str, Any]


# NOTE:
# Overrides soportados en C2:
#   - "execution": dict (merge profundo)
#   - "costs": dict (merge profundo)
#
# No toques "strategy" ni "risk" aquí. Execution policies = microestructura / fills / costos.
_POLICIES: Dict[str, ExecutionPolicy] = {
    "baseline": ExecutionPolicy(
        policy_id="baseline",
        description="No overrides. Uses config as-is.",
        overrides={},
    ),
    "stress_spread_2": ExecutionPolicy(
        policy_id="stress_spread_2",
        description="Increase spread to 2.0 pips.",
        overrides={"costs": {"spread_pips": 2.0}},
    ),
    "stress_slip_0_5": ExecutionPolicy(
        policy_id="stress_slip_0_5",
        description="Increase slippage to 0.5 pips (adverse on fills).",
        overrides={"costs": {"slippage_pips": 0.5}},
    ),
    "stress_spread_2_slip_0_5": ExecutionPolicy(
        policy_id="stress_spread_2_slip_0_5",
        description="Increase spread to 2.0 pips and slippage to 0.5 pips.",
        overrides={"costs": {"spread_pips": 2.0, "slippage_pips": 0.5}},
    ),
    "intrabar_olhc": ExecutionPolicy(
        policy_id="intrabar_olhc",
        description="Use intrabar path OLHC (open->low->high->close).",
        overrides={"execution": {"intrabar_path": "OLHC"}},
    ),
    "intrabar_ohlc": ExecutionPolicy(
        policy_id="intrabar_ohlc",
        description="Use intrabar path OHLC (open->high->low->close).",
        overrides={"execution": {"intrabar_path": "OHLC"}},
    ),
    "tie_sl_first": ExecutionPolicy(
        policy_id="tie_sl_first",
        description="If both SL/TP reachable, prefer SL first (conservative).",
        overrides={"execution": {"intrabar_tie": "sl_first"}},
    ),
    "tie_tp_first": ExecutionPolicy(
        policy_id="tie_tp_first",
        description="If both SL/TP reachable, prefer TP first (optimistic).",
        overrides={"execution": {"intrabar_tie": "tp_first"}},
    ),
    "fill_close": ExecutionPolicy(
        policy_id="fill_close",
        description="Fill entry at signal bar close.",
        overrides={"execution": {"fill_mode": "close"}},
    ),
    "fill_next_open": ExecutionPolicy(
        policy_id="fill_next_open",
        description="Fill entry at next bar open.",
        overrides={"execution": {"fill_mode": "next_open"}},
    ),
}


def list_policies() -> List[str]:
    return sorted(_POLICIES.keys())


def get_policy(policy_id: str) -> ExecutionPolicy:
    pid = str(policy_id or "").strip()
    if not pid:
        pid = "baseline"
    if pid not in _POLICIES:
        raise ValueError(
            "UNKNOWN_EXECUTION_POLICY_ID: policy_id not found.\n"
            f"policy_id={pid!r}\n"
            f"available={list_policies()}"
        )
    return _POLICIES[pid]
