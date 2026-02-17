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
    Aplica PolicyGate a ENTRY intents (bloquea entradas en barras no permitidas).
    - NO toca EXIT intents.
    - Devuelve:
        (intents filtrados, metrics_extra (flatten-ready), manifest_extra)

    Requisitos:
    - features_df indexado 1:1 con bars del dataset (mismo orden / longitud).
    - intents_by_bar tiene misma longitud que features_df.

    Importante:
    - Este hook usa PolicyGate.evaluate_entry_idx(i), que opera por índice.
    - La PolicyGate interna carga su propio features_path y usa el índice como join key.
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

    for i, intents in enumerate(intents_by_bar):
        if not intents:
            out.append(intents)
            continue

        # Detect if bar has ENTRY intents
        def _is_entry_intent(x: Any) -> bool:
            if isinstance(x, dict):
                return (x.get("action") == "ENTRY")
            return (getattr(x, "action", None) == "ENTRY")

        has_entry = any(_is_entry_intent(x) for x in intents)
        if not has_entry:
            out.append(intents)
            continue

        attempted_entries += 1

        ok = gate.evaluate_entry_idx(int(i))
        if ok:
            out.append(intents)
            continue

        # blocked -> remove ENTRY intents only
        blocked_unique_bars += 1
        new_intents: list = []
        for it in intents:
            if _is_entry_intent(it):
                blocked_entries_total += 1
                continue
            new_intents.append(it)

        out.append(new_intents if new_intents else [])

    stats = gate.stats()  # canonical API in your PolicyGate

    # ---- metrics_extra (flatten-ready) ----
    metrics_extra: Dict[str, Any] = {}

    # keep your previous metric keys stable (so downstream doesn't break)
    metrics_extra["policy_gate_enabled"] = True
    metrics_extra["policy_gate_policy_id"] = stats.get("policy_id")
    metrics_extra["policy_gate_policy_path"] = stats.get("policy_path")
    metrics_extra["policy_gate_features_path"] = stats.get("features_path")
    metrics_extra["policy_gate_time_bucket_min"] = stats.get("time_bucket_min", getattr(policy_cfg, "time_bucket_min", None))

    metrics_extra["policy_gate_attempted_entries"] = int(attempted_entries)
    metrics_extra["policy_gate_blocked_total"] = int(blocked_entries_total)
    metrics_extra["policy_gate_blocked_unique_bars"] = int(blocked_unique_bars)

    # passthrough core gate counters (your run.py expects these names)
    metrics_extra["policy_allowed"] = stats.get("policy_allowed")
    metrics_extra["policy_blocked"] = stats.get("policy_blocked")
    metrics_extra["policy_blocked_by_time"] = stats.get("policy_blocked_by_time")
    metrics_extra["policy_blocked_by_shock_vol"] = stats.get("policy_blocked_by_shock_vol")
    metrics_extra["policy_coverage_allowed"] = stats.get("policy_coverage_allowed")

    # ---- manifest_extra ----
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
