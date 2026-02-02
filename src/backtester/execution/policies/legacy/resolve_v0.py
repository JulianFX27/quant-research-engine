# src/backtester/execution/policies/resolve.py
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from backtester.execution.policies.legacy.registry_v0 import get_policy


@dataclass(frozen=True)
class ExecutionPolicyMeta:
    """
    Meta audit block for execution policy resolution.

    - policy_id: resolved policy id (always explicit)
    - overrides_applied: raw override tree pulled from registry (deep-copied)
    - execution_effective: final effective execution subtree after merge
    - costs_effective: final effective costs subtree after merge
    """
    policy_id: str
    overrides_applied: Dict[str, Any]
    execution_effective: Dict[str, Any]
    costs_effective: Dict[str, Any]


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge dictionaries:
      - dict values are merged recursively
      - non-dict values overwrite
    Mutates and returns dst.
    """
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def resolve_execution_policy(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      (cfg_effective, policy_meta_dict)

    Contract:
      - Reads cfg.execution.policy_id (optional). Default: "baseline".
      - Applies overrides only to cfg['execution'] and cfg['costs'] subtrees.
      - Does NOT mutate cfg input.
      - Always stamps cfg_effective['execution']['policy_id'] with resolved policy_id.
    """
    cfg_in = cfg or {}
    exe_in = cfg_in.get("execution", {}) or {}

    # Default policy
    policy_id = str(exe_in.get("policy_id") or "baseline").strip() or "baseline"

    policy = get_policy(policy_id)
    overrides = copy.deepcopy(policy.overrides) if getattr(policy, "overrides", None) else {}

    cfg_eff: Dict[str, Any] = copy.deepcopy(cfg_in)

    # Ensure expected subtrees exist
    if "execution" not in cfg_eff or cfg_eff["execution"] is None:
        cfg_eff["execution"] = {}
    if "costs" not in cfg_eff or cfg_eff["costs"] is None:
        cfg_eff["costs"] = {}

    if not isinstance(cfg_eff["execution"], dict):
        raise ValueError("Run config 'execution' must be a dict")
    if not isinstance(cfg_eff["costs"], dict):
        raise ValueError("Run config 'costs' must be a dict")

    # Apply overrides (only supported subtrees)
    if "execution" in overrides:
        if not isinstance(overrides["execution"], dict):
            raise ValueError("Invalid policy override: overrides['execution'] must be a dict")
        _deep_merge(cfg_eff["execution"], overrides["execution"])

    if "costs" in overrides:
        if not isinstance(overrides["costs"], dict):
            raise ValueError("Invalid policy override: overrides['costs'] must be a dict")
        _deep_merge(cfg_eff["costs"], overrides["costs"])

    # Always set policy_id explicitly (auditability)
    cfg_eff["execution"]["policy_id"] = getattr(policy, "policy_id", policy_id)

    meta = ExecutionPolicyMeta(
        policy_id=cfg_eff["execution"]["policy_id"],
        overrides_applied=overrides,
        execution_effective=copy.deepcopy(cfg_eff.get("execution", {}) or {}),
        costs_effective=copy.deepcopy(cfg_eff.get("costs", {}) or {}),
    )

    return cfg_eff, {
        "policy_id": meta.policy_id,
        "overrides_applied": meta.overrides_applied,
        "execution_effective": meta.execution_effective,
        "costs_effective": meta.costs_effective,
    }
