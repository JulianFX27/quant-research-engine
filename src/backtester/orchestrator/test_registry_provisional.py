from __future__ import annotations

from backtester.data.dataset_registry import register_or_validate_dataset, DatasetIdentityError
from backtester.data.dataset_fingerprint import DatasetMetadata


def test_registry_rejects_provisional_dataset_id(tmp_path):
    meta = DatasetMetadata(
        dataset_id="EURUSD_M5_unknown__unknown__csv",
        fingerprint_sha256="sha256:deadbeef",
        schema_sha256="sha256:cafebabe",
        rows=1,
        start_ts="2026-01-01T00:00:00+00:00",
        end_ts="2026-01-01T00:05:00+00:00",
    )

    try:
        register_or_validate_dataset(meta, registry_dir=tmp_path / "registry")
        assert False, "Expected DatasetIdentityError"
    except DatasetIdentityError as e:
        assert "DATASET_ID_PROVISIONAL_NOT_ALLOWED" in str(e)
