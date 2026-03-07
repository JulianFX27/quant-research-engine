# src/backtester/orchestrator/policy_gate_hook.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backtester.policies.policy_gate import PolicyGate, PolicyGateConfig


def apply_policy_gate_to_intents(
    *,
    policy_cfg: PolicyGateConfig,
    features_df: pd.DataFrame,
    intents_by_bar: List[Optional[list]],
) -> Tuple[List[Optional[list]], Dict[str, Any], Dict[str, Any]]:
    """
    Apply PolicyGate to entry intents.

    Backtester compatibility:
    - Current backtester OrderIntent does NOT use action="ENTRY".
    - Entry intent is inferred by presence of a valid side ("BUY"/"SELL").

    Returns:
        (filtered_intents, metrics_extra, manifest_extra)
    """
    if len(features_df) != len(intents_by_bar):
        raise ValueError(
            f"PolicyGate: features_df len ({len(features_df)}) != intents_by_bar len ({len(intents_by_bar)})"
        )

    gate = PolicyGate(policy_cfg)

    attempted_entries = 0
    blocked_entries_total = 0
    blocked_unique_bars = 0

    out: List[Optional[list]] = []

    def _is_entry_intent(x: Any) -> bool:
        # dict-style
        if isinstance(x, dict):
            action = x.get("action")
            if action == "ENTRY":
                return True
            side = str(x.get("side", "")).upper().strip()
            return side in {"BUY", "SELL"}

        # live/paper style
        action = getattr(x, "action", None)
        if action == "ENTRY":
            return True

        # backtester OrderIntent style
        side = str(getattr(x, "side", "")).upper().strip()
        return side in {"BUY", "SELL"}

    for i, intents in enumerate(intents_by_bar):
        if not intents:
            out.append(intents)
            continue

        has_entry = any(_is_entry_intent(x) for x in intents)
        if not has_entry:
            out.append(intents)
            continue

        attempted_entries += 1

        ok = gate.evaluate_entry_idx(int(i))
        if ok:
            out.append(intents)
            continue

        blocked_unique_bars += 1
        new_intents: list = []
        for it in intents:
            if _is_entry_intent(it):
                blocked_entries_total += 1
                continue
            new_intents.append(it)

        out.append(new_intents if new_intents else [])

    stats = gate.stats()

    metrics_extra: Dict[str, Any] = {}
    metrics_extra["policy_gate_enabled"] = True
    metrics_extra["policy_gate_policy_id"] = stats.get("policy_id")
    metrics_extra["policy_gate_policy_path"] = stats.get("policy_path")
    metrics_extra["policy_gate_features_path"] = stats.get("features_path")
    metrics_extra["policy_gate_time_bucket_min"] = stats.get(
        "time_bucket_min", getattr(policy_cfg, "time_bucket_min", None)
    )

    metrics_extra["policy_gate_attempted_entries"] = int(attempted_entries)
    metrics_extra["policy_gate_blocked_total"] = int(blocked_entries_total)
    metrics_extra["policy_gate_blocked_unique_bars"] = int(blocked_unique_bars)

    metrics_extra["policy_allowed"] = stats.get("policy_allowed")
    metrics_extra["policy_blocked"] = stats.get("policy_blocked")
    metrics_extra["policy_blocked_by_time"] = stats.get("policy_blocked_by_time")
    metrics_extra["policy_blocked_by_shock_vol"] = stats.get("policy_blocked_by_shock_vol")
    metrics_extra["policy_blocked_by_timerange"] = stats.get("policy_blocked_by_timerange")
    metrics_extra["policy_blocked_by_time_window"] = stats.get("policy_blocked_by_time_window")
    metrics_extra["policy_coverage_allowed"] = stats.get("policy_coverage_allowed")

    manifest_extra: Dict[str, Any] = {
        "policy_gate": {
            "enabled": True,
            "config": asdict(policy_cfg),
            "attempted_entries": int(attempted_entries),
            "blocked_total": int(blocked_entries_total),
            "blocked_unique_bars": int(blocked_unique_bars),
            "stats": dict(stats),
        }
    }

    return out, metrics_extra, manifest_extra