from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backtester.data.dataset_fingerprint import DatasetMetadata


class DatasetIdentityError(RuntimeError):
    pass


_UNKNOWN_SENTINEL = "unknown__unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_latest_map(latest_path: Path) -> Dict[str, Dict[str, Any]]:
    if not latest_path.exists():
        return {}
    obj = json.loads(latest_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid registry latest file (expected dict): {latest_path}")
    return obj


def _write_latest_map(latest_path: Path, m: Dict[str, Dict[str, Any]]) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(jsonl_path: Path, record: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _meta_to_record(meta: DatasetMetadata) -> Dict[str, Any]:
    d = meta.to_dict()
    d["registered_at_utc"] = _utc_now_iso()
    return d


def _assert_dataset_id_is_final(dsid: str) -> None:
    # Guardrail: nunca permitimos registrar dataset_id provisional.
    if _UNKNOWN_SENTINEL in dsid:
        raise DatasetIdentityError(
            "DATASET_ID_PROVISIONAL_NOT_ALLOWED: refusing to register provisional dataset_id.\n"
            f"dataset_id={dsid}\n"
            "Fix: compute dataset_id using real start/end (post-load) BEFORE calling registry.\n"
        )


def register_or_validate_dataset(
    meta: DatasetMetadata,
    *,
    registry_dir: str | Path = "data/registry",
    allow_new_fingerprint: bool = False,
    override_reason: str = "",
    append_match_event: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Returns:
      (is_new, message)

    Policy:
      - dataset_id is a semantic identity (instrument/timeframe/start/end/source, etc.)
      - dataset_id MUST map to a stable content fingerprint (fingerprint_sha256)
      - if dataset_id exists and fingerprint differs -> error unless allow_new_fingerprint=True (override)

    Notes:
      - append_match_event: if True, also appends MATCH events to datasets.jsonl for full audit trails.
      - override_reason required if allow_new_fingerprint=True
    """
    dsid = meta.dataset_id
    _assert_dataset_id_is_final(dsid)

    reg_dir = Path(registry_dir)
    latest_path = reg_dir / "datasets_latest.json"
    jsonl_path = reg_dir / "datasets.jsonl"

    latest = _load_latest_map(latest_path)
    record = _meta_to_record(meta)

    # New dataset_id
    if dsid not in latest:
        latest[dsid] = record
        _write_latest_map(latest_path, latest)
        _append_jsonl(jsonl_path, {"event": "REGISTER", **record})
        return True, f"registered new dataset_id={dsid}"

    prev = latest[dsid]
    prev_fp = prev.get("fingerprint_sha256")
    new_fp = meta.fingerprint_sha256

    # Same fingerprint -> OK
    if prev_fp == new_fp:
        if append_match_event:
            _append_jsonl(jsonl_path, {"event": "MATCH", **record})
        return False, "dataset_id matches existing fingerprint"

    # fingerprint mismatch -> error or override
    msg = (
        "DATASET_ID_FINGERPRINT_MISMATCH: dataset_id points to different content.\n"
        f"dataset_id={dsid}\n"
        f"prev_fingerprint={prev_fp}\n"
        f"new_fingerprint={new_fp}\n"
        f"prev_source_path={prev.get('source_path')}\n"
        f"new_source_path={meta.source_path}\n"
        "Fix: change dataset_id semantics OR store as new dataset_id.\n"
    )

    if not allow_new_fingerprint:
        raise DatasetIdentityError(msg)

    # Override requires reason (contract-stable error string)
    if not str(override_reason or "").strip():
        raise DatasetIdentityError(
            "DATASET_ID_OVERRIDE_REASON_REQUIRED: override requested but override_reason is empty.\n"
            f"dataset_id={dsid}\n"
            "Fix: set dataset_registry.override_reason to a non-empty string.\n"
        )

    # allowed override: update latest snapshot + append audit
    latest[dsid] = record
    _write_latest_map(latest_path, latest)
    _append_jsonl(
        jsonl_path,
        {
            "event": "UPDATE_FINGERPRINT_ALLOWED",
            "override_reason": str(override_reason),
            "prev": prev,
            "new": record,
        },
    )
    return False, "fingerprint mismatch but override allowed; registry updated"
