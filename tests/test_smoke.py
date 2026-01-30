# tests/test_smoke.py
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "configs" / "run_example.yaml"
REGISTRY_DIR = REPO_ROOT / "data" / "registry"
DATASET_CSV = REPO_ROOT / "data" / "example_bars.csv"


def _run(cmd: list[str], *, cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )


def _read_latest_registry() -> dict:
    latest = REGISTRY_DIR / "datasets_latest.json"
    assert latest.exists(), f"Expected registry latest file at {latest}"
    return json.loads(latest.read_text(encoding="utf-8"))


def setup_function() -> None:
    if REGISTRY_DIR.exists():
        shutil.rmtree(REGISTRY_DIR)


def teardown_function() -> None:
    if REGISTRY_DIR.exists():
        shutil.rmtree(REGISTRY_DIR)


def test_smoke_two_runs_registers_dataset_and_manifest(tmp_path: Path) -> None:
    assert CFG_PATH.exists()
    assert DATASET_CSV.exists()

    out_dir = tmp_path / "runs"

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

    # Registry contains dataset_id
    latest = _read_latest_registry()
    assert isinstance(latest, dict) and latest, "registry latest map is empty"

    expected_key = "EURUSD_M5_2026-01-01__2026-01-02__csv_example"
    assert expected_key in latest, f"Expected dataset_id {expected_key} in registry keys={list(latest.keys())}"

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

    # Validate each run dir has manifest + required dataset fields
    for rd in run_dirs:
        manifest_path = rd / "run_manifest.json"
        assert manifest_path.exists(), f"Missing run_manifest.json in {rd}"
        man = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert "cfg_hash_sha256" in man and isinstance(man["cfg_hash_sha256"], str) and man["cfg_hash_sha256"]
        assert "dataset" in man and isinstance(man["dataset"], dict)
        ds = man["dataset"]

        for k in ["dataset_id", "fingerprint_sha256", "schema_sha256", "fingerprint_version"]:
            assert k in ds and ds[k], f"Missing dataset.{k} in manifest {manifest_path}"

        assert str(ds["dataset_id"]) == expected_key
        assert str(ds["fingerprint_sha256"]).startswith("sha256:")
        assert str(ds["schema_sha256"]).startswith("sha256:")


def test_mutation_triggers_dataset_id_fingerprint_mismatch(tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"

    p1 = _run(
        ["python", "-m", "backtester.orchestrator.run", str(CFG_PATH), "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
    )
    assert p1.returncode == 0, f"baseline run failed\nSTDOUT:\n{p1.stdout}\nSTDERR:\n{p1.stderr}"

    bak = DATASET_CSV.with_suffix(".csv.bak")
    if bak.exists():
        bak.unlink()

    shutil.copy2(DATASET_CSV, bak)

    try:
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
        shutil.move(str(bak), str(DATASET_CSV))
