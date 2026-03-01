from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def load_config(path: str) -> Dict[str, Any]:
    """
    Loads YAML if PyYAML is available; otherwise expects JSON.
    """
    text = _read_text(path)
    suffix = Path(path).suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML not available. Install pyyaml or use JSON config.") from e
        obj = yaml.safe_load(text)
        if not isinstance(obj, dict):
            raise ValueError("Config root must be a mapping/dict.")
        return obj

    if suffix == ".json":
        import json
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("Config root must be a mapping/dict.")
        return obj

    raise ValueError(f"Unsupported config extension: {suffix}")
