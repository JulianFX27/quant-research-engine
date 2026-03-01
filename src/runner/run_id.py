from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def file_sha256(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "MISSING"
    return sha256_bytes(p.read_bytes())


def dict_sha256(d: Dict[str, Any]) -> str:
    # stable representation
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(blob)


def make_run_id(tag: str | None, policy_hash: str) -> str:
    base = f"{utc_now_compact()}_{policy_hash[:8]}"
    if tag:
        safe = "".join(ch for ch in tag if ch.isalnum() or ch in ("-", "_"))[:24]
        return f"{base}_{safe}"
    return base
