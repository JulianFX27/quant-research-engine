# tests/test_smoke.py
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "configs" / "run_example.yaml"
REGISTRY_DIR = REPO_ROOT / "data" / "registry"
DATASET_CSV = REPO_ROOT / "data" / "example_bars.csv"


def _run(cmd: list[str], *, cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    return p


def _read_latest_registry() -> dict:
    latest = REGISTRY_DIR / "datasets_latest.json"
    assert latest.exists(), f"Expected registry latest file at {latest}"
    return json.loads(latest.read_text(encoding="utf-8"))


def setup_function() -> None:
    # Clean registry before each test for determinism
    if REGISTRY_DIR.exists():
        shutil.rmtree(REGISTRY_DIR)


def teardown_function() -> None:
    # Optional: keep registry for debugging by commenting this out.
    # We clean it to avoid cross-test interference.
    if REGISTRY_DIR.exists():
        shutil.rmtree(REGISTRY_DIR)


def test_smoke_two_runs_registers_dataset(tmp_path: Path) -> None:
    assert CFG_PATH.exists()
    assert DATASET_CSV.exists()

    out_dir = tmp_path / "runs"

    # Run twice (should not collide, even within same second if run_id includes microseconds)
    p1 = _run(
        ["python", "-m", "backtester.orchestrator.run", str(CFG_PATH), "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
    )
    assert p1.returncode == 0, f"run1 failed\nSTDOUT:\n{p1.stdout}\nSTDERR:\n{p1.stderr}"

    p2 = _run(
        ["python", "-m", "backtester.orchestrator.run", str(CFG_PATH), "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
    )
    assert p2.returncode == 0, f"run2 failed\nSTDOUT:\n{p2.stdout}\nSTDERR:\n{p2.stderr}"

    # Ensure registry exists and contains at least one dataset_id
    latest = _read_latest_registry()
    assert isinstance(latest, dict) and latest, "registry latest map is empty"

    # Assert expected dataset key is present (the one you're seeing in your logs)
    expected_key = "EURUSD_M5_2026-01-01__2026-01-02__csv_example"
    assert expected_key in latest, f"Expected dataset_id {expected_key} in registry keys={list(latest.keys())}"

    # Basic sanity fields
    rec = latest[expected_key]
    assert rec.get("fingerprint_version") == "v1.1"
    assert rec.get("fingerprint_sha256", "").startswith("sha256:")
    assert rec.get("file_sha256", "").startswith("sha256:")
    assert isinstance(rec.get("file_bytes"), int) and rec["file_bytes"] > 0
    assert rec.get("source_path") == "data/example_bars.csv"
    assert rec.get("time_col") == "__index__"

    # Ensure two run dirs exist
    run_dirs = [p for p in out_dir.glob("*") if p.is_dir()]
    assert len(run_dirs) >= 2, f"Expected >=2 run dirs in {out_dir}, got {len(run_dirs)}"


def test_mutation_triggers_dataset_id_fingerprint_mismatch(tmp_path: Path) -> None:
    """
    1) Run once to create registry binding dataset_id -> fingerprint
    2) Mutate the CSV (content changes)
    3) Run again => must fail with DATASET_ID_FINGERPRINT_MISMATCH
    4) Restore CSV
    """
    out_dir = tmp_path / "runs"

    # First run: registers dataset
    p1 = _run(
        ["python", "-m", "backtester.orchestrator.run", str(CFG_PATH), "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
    )
    assert p1.returncode == 0, f"baseline run failed\nSTDOUT:\n{p1.stdout}\nSTDERR:\n{p1.stderr}"

    # Mutate CSV (in-place) and restore afterwards
    bak = DATASET_CSV.with_suffix(".csv.bak")
    if bak.exists():
        bak.unlink()

    shutil.copy2(DATASET_CSV, bak)

    try:
        # Deterministic small mutation: append a space to file OR nudge a value.
        # We'll nudge close[0] by +1e-5 using pandas one-liner in a subprocess.
        mut = _run(
            [
                "python",
                "-c",
                (
                    "import pandas as pd; "
                    f"p=r'{DATASET_CSV.as_posix()}'; "
                    "df=pd.read_csv(p); "
                    "df.loc[0,'close']=float(df.loc[0,'close'])+0.00001; "
                    "df.to_csv(p,index=False); "
                    "print('mutated', df.loc[0,'close'])"
                ),
            ],
            cwd=REPO_ROOT,
        )
        assert mut.returncode == 0, f"mutation helper failed\nSTDOUT:\n{mut.stdout}\nSTDERR:\n{mut.stderr}"

        # Second run: must fail (registry mismatch)
        p2 = _run(
            ["python", "-m", "backtester.orchestrator.run", str(CFG_PATH), "--out-dir", str(out_dir)],
            cwd=REPO_ROOT,
        )
        assert p2.returncode != 0, "Expected failure after dataset mutation, but run succeeded"

        combined = (p2.stdout or "") + "\n" + (p2.stderr or "")
        assert "DATASET_ID_FINGERPRINT_MISMATCH" in combined, (
            "Expected DATASET_ID_FINGERPRINT_MISMATCH in output\n"
            f"STDOUT:\n{p2.stdout}\nSTDERR:\n{p2.stderr}"
        )

    finally:
        # Restore original CSV
        shutil.move(str(bak), str(DATASET_CSV))
