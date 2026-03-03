from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union


def _read_text_robust(path: Union[str, Path]) -> str:
    p = Path(path)
    # UTF-8 with BOM tolerance first
    try:
        return p.read_text(encoding="utf-8-sig")
    except UnicodeError:
        # fallback for accidental UTF-16 saves
        return p.read_text(encoding="utf-16")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    text = _read_text_robust(p).strip()
    if not text:
        return {}

    suffix = p.suffix.lower()

    # YAML
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PyYAML is required to load .yaml configs. Install with: pip install pyyaml"
            ) from e

        try:
            obj = yaml.safe_load(text)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML (file={p}): {type(e).__name__}: {e}") from e

    # JSON (default)
    else:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"{e.msg} (file={p})", e.doc, e.pos) from e

    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a dict/object (file={p}), got {type(obj).__name__}")
    return obj