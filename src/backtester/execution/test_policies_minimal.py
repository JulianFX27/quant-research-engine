from __future__ import annotations

from pathlib import Path

import pytest

from backtester.execution.policies import apply_execution_policy


def _base_cfg(policy_id: str) -> dict:
    # OJO: data_path no se usa por apply_execution_policy, pero lo dejamos realista.
    return {
        "name": "t",
        "symbol": "EURUSD",
        "timeframe": "M5",
        "data_path": "data/raw/eurusd_m5_2022-01-03__2024-12-31.csv",
        "strategy": {"name": "ExampleMomentum", "params": {}},
        "execution": {"policy_id": policy_id},
        "costs": {"commission": 0.0, "spread_pips": 1.0, "slippage_pips": 0.0},
        "risk": {"eof_buffer_bars": 50},
    }


def test_no_policy_id_returns_none():
    cfg = _base_cfg("baseline")
    cfg["execution"].pop("policy_id", None)
    assert apply_execution_policy(cfg) is None


def test_missing_policies_file_raises(tmp_path, monkeypatch):
    # fuerza a que no encuentre ningún policies file
    from backtester.execution.policies import policies as polmod
    monkeypatch.setattr(polmod, "DEFAULT_POLICIES_PATHS", ["_does_not_exist_.yaml"])
    cfg = _base_cfg("baseline")
    with pytest.raises(FileNotFoundError):
        apply_execution_policy(cfg)


def test_baseline_policy_applies_overlay_fields():
    cfg = _base_cfg("baseline")
    res = apply_execution_policy(cfg)
    assert res is not None
    assert Path(res.policies_path).exists()
    assert res.policy_id == "baseline"

    exe = res.cfg_resolved["execution"]
    # Baseline expected overlay (según tus manifests ya generados)
    assert exe["fill_mode"] == "next_open"
    assert str(exe["intrabar_path"]).replace(" ", "").upper() == "OHLC"
    assert exe["intrabar_tie"] == "sl_first"


def test_unknown_policy_id_raises_value_error():
    cfg = _base_cfg("___this_policy_should_not_exist___")
    with pytest.raises(ValueError) as e:
        apply_execution_policy(cfg)
    assert "EXECUTION_POLICY_NOT_FOUND" in str(e.value)


def test_overlay_only_touches_execution_costs_risk():
    # Si el YAML trae llaves extra, deben quedar en warnings y no tocar cfg fuera de execution/costs/risk
    cfg = _base_cfg("baseline")
    cfg["foo"] = {"bar": 1}
    before = dict(cfg["foo"])

    res = apply_execution_policy(cfg)
    assert res is not None
    assert cfg["foo"] == before
