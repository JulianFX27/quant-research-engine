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
    - No toca EXIT intents.
    - Devuelve: intents filtrados, metrics_extra (flatten-ready), manifest_extra.

    Requisitos:
    - features_df indexado 1:1 con bars del dataset (mismo orden / longitud).
    - intents_by_bar tiene misma longitud que features_df.
    """

    if len(features_df) != len(intents_by_bar):
        raise ValueError(
            f"PolicyGate: features_df len ({len(features_df)}) != intents_by_bar len ({len(intents_by_bar)})"
        )

    gate = PolicyGate(policy_cfg)
    # Este método debe existir en tu PolicyGate.
    # Si el nombre difiere, lo ajustamos una vez mires el archivo.
    allowed_mask = gate.allowed_mask(features_df)  # -> pd.Series[bool] len N

    if len(allowed_mask) != len(intents_by_bar):
        raise ValueError(
            f"PolicyGate: allowed_mask len ({len(allowed_mask)}) != intents_by_bar len ({len(intents_by_bar)})"
        )

    blocked_total = 0
    blocked_unique_bars = 0

    out: List[Optional[list]] = []
    for i, intents in enumerate(intents_by_bar):
        if not intents:
            out.append(intents)
            continue

        # Si la barra NO está permitida, removemos intents de ENTRY
        if not bool(allowed_mask.iloc[i]):
            # cuenta solo si había algún ENTRY
            had_entry = any(getattr(x, "action", None) == "ENTRY" or (isinstance(x, dict) and x.get("action") == "ENTRY")
                            for x in intents)
            if had_entry:
                blocked_unique_bars += 1

            new_intents = []
            for it in intents:
                action = getattr(it, "action", None) if not isinstance(it, dict) else it.get("action")
                if action == "ENTRY":
                    blocked_total += 1
                    continue
                new_intents.append(it)

            out.append(new_intents if new_intents else [])
            continue

        out.append(intents)

    # gate.summary() ya existe en tu archivo (por tus rg hits)
    # y devuelve policy_path, policy_allowed, policy_blocked, blocked_by_time, blocked_by_shock_vol, etc.
    summary = gate.summary()

    metrics_extra: Dict[str, Any] = {}
    metrics_extra.update(summary)
    metrics_extra["policy_gate_blocked_total"] = blocked_total
    metrics_extra["policy_gate_blocked_unique_bars"] = blocked_unique_bars

    manifest_extra: Dict[str, Any] = {
        "policy_gate": {
            "config": asdict(policy_cfg),
            "summary": summary,
        }
    }

    return out, metrics_extra, manifest_extra
