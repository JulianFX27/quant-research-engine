# src/backtester/execution/policies.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


DEFAULT_POLICIES_PATH = "configs/execution_policies.yaml"


@dataclass(frozen=True)
class ExecutionPolicyResult:
    policy_id: str
    policies_path: str
    overlay: Dict[str, Any]            # what was applied (execution/costs/risk only)
    cfg_resolved: Dict[str, Any]       # full resolved cfg (copy)
    warnings: list[str]


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge dictionaries. patch overwrites base.
    Only merges dict-to-dict; other types overwrite.
    """
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_policies_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"EXECUTION_POLICIES_FILE_NOT_FOUND: {p}")

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid policies YAML: expected mapping at top-level, got {type(data).__name__}")

    # Accept either:
    #   policies: { ... }
    # or:
    #   { ... }  (direct map)
    if "policies" in data:
        policies = data.get("policies")
    else:
        policies = data

    if not isinstance(policies, dict):
        raise ValueError("Invalid policies YAML: 'policies' must be a mapping/dict")

    return policies


def _normalize_policy_id(policy_id: Optional[str]) -> Optional[str]:
    if policy_id is None:
        return None
    s = str(policy_id).strip()
    return s if s else None


def _extract_policy_id_from_cfg(cfg: Dict[str, Any]) -> Optional[str]:
    exe = cfg.get("execution", {}) or {}
    if not isinstance(exe, dict):
        return None
    return _normalize_policy_id(exe.get("policy_id"))


def _extract_policies_path_from_cfg(cfg: Dict[str, Any]) -> str:
    """
    Allows override via:
      execution.policies_path
    Otherwise uses DEFAULT_POLICIES_PATH.
    """
    exe = cfg.get("execution", {}) or {}
    if isinstance(exe, dict) and exe.get("policies_path"):
        return str(exe["policies_path"])
    return DEFAULT_POLICIES_PATH


def apply_execution_policy(cfg: Dict[str, Any]) -> Optional[ExecutionPolicyResult]:
    """
    Opción B (research-friendly, baseline-safe):
      - If no policy_id: return None (cfg unchanged).
      - If policy_id == "baseline": treat as no-op but still record policy meta.
      - Otherwise:
          - load policies YAML
          - apply overlay only into: execution/costs/risk
          - return resolved cfg + overlay for auditing

    The caller should run validate_run_config() on cfg_resolved after applying.
    """
    policy_id = _extract_policy_id_from_cfg(cfg)
    if not policy_id:
        return None

    policies_path = _extract_policies_path_from_cfg(cfg)
    policies = _load_policies_yaml(policies_path)

    if policy_id not in policies:
        raise ValueError(
            "EXECUTION_POLICY_NOT_FOUND: "
            f"{policy_id!r} not present in {policies_path}. "
            "Fix: add it to execution_policies.yaml or remove execution.policy_id."
        )

    raw_pol = policies[policy_id]
    if raw_pol is None:
        raw_pol = {}
    if not isinstance(raw_pol, dict):
        raise ValueError(f"Invalid policy '{policy_id}': expected mapping/dict, got {type(raw_pol).__name__}")

    # Only allow these top-level overlay sections (baseline safety).
    allowed_top = {"execution", "costs", "risk"}
    overlay: Dict[str, Any] = {}
    warnings: list[str] = []

    for k, v in raw_pol.items():
        if k in allowed_top:
            if v is None:
                overlay[k] = {}
            elif isinstance(v, dict):
                overlay[k] = v
            else:
                raise ValueError(f"Invalid policy '{policy_id}': '{k}' must be a dict, got {type(v).__name__}")
        else:
            warnings.append(f"Ignored policy key '{k}' (allowed only: execution/costs/risk)")

    # Build resolved cfg
    cfg_resolved = dict(cfg)

    # Deep merge only in allowed sections
    for sec in ("execution", "costs", "risk"):
        base_sec = cfg_resolved.get(sec, {}) or {}
        if base_sec is None:
            base_sec = {}
        if not isinstance(base_sec, dict):
            raise ValueError(f"Run config '{sec}' must be a dict to apply policies")

        patch = overlay.get(sec, {}) or {}
        cfg_resolved[sec] = _deep_merge(base_sec, patch)

    return ExecutionPolicyResult(
        policy_id=policy_id,
        policies_path=policies_path,
        overlay=overlay,
        cfg_resolved=cfg_resolved,
        warnings=warnings,
    )
