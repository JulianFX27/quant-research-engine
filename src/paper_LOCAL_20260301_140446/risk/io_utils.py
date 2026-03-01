from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict


def json_read(path: str) -> Dict[str, Any]:
    """
    Read JSON file robustly.

    Windows/PowerShell sometimes writes UTF-8 with BOM. The standard json module
    will fail if the file is opened with plain 'utf-8'. We support both:
      - utf-8
      - utf-8-sig (strips BOM if present)

    Also raises a clearer error if the JSON is invalid.
    """
    # Fast path: try normal utf-8
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e_utf8:
        # Retry with utf-8-sig to handle BOM
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Re-raise original but with context
            raise json.JSONDecodeError(
                f"{e_utf8.msg} (also tried utf-8-sig; file may be corrupted)",
                e_utf8.doc,
                e_utf8.pos,
            ) from e_utf8


def atomic_json_write(path: str, obj: Dict[str, Any]) -> None:
    """
    Best-effort atomic JSON write.

    On Windows, os.replace() can fail with PermissionError if the destination file
    is temporarily locked (editor, AV, indexing, sync). We:
      1) write to tmp in same directory
      2) fsync tmp
      3) try os.replace with short retries
      4) as last resort (paper-mode pragmatic): overwrite destination directly
    """
    dirpath = os.path.dirname(os.path.abspath(path))
    os.makedirs(dirpath, exist_ok=True)

    tmp_name = f".tmp_{random.randint(10**6, 10**9 - 1)}.json"
    tmp_path = os.path.join(dirpath, tmp_name)

    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

    # Write temp
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    # Try atomic replace with retries (Windows locks)
    last_err: Exception | None = None
    for _ in range(12):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.05)  # 50ms backoff

    # Fallback: direct overwrite (non-atomic) — should be acceptable for paper runner.
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # cleanup tmp if still exists
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return
    except Exception as e:
        # cleanup tmp if still exists
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        if last_err is not None:
            raise last_err
        raise e
