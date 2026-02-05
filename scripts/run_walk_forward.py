# scripts/run_walk_forward.py
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from backtester.orchestrator.run import run_from_config, _canonical_cfg_json
from backtester.data.loader import load_bars_csv
from backtester.data.dataset_fingerprint import build_dataset_id
from backtester.data.dataset_registry import register_or_validate_dataset


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _hash8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a top-level mapping")
    return cfg


def _as_df_only(res: Any) -> pd.DataFrame:
    """
    Robustly extract df from loader output.
    Some loaders return df only; others return (df, meta) or (df, meta, warnings).
    """
    if isinstance(res, tuple):
        if len(res) == 0:
            raise ValueError("load_bars_csv returned an empty tuple")
        return res[0]
    return res


def _as_df_meta(res: Any) -> tuple[pd.DataFrame, Any]:
    """
    Extract (df, meta) robustly from loader output when fingerprint is requested.
    We require meta to be present as second item (consistent with your run.py usage).
    """
    if not isinstance(res, tuple) or len(res) < 2:
        raise ValueError("load_bars_csv did not return (df, meta, ...) when return_fingerprint=True")
    return res[0], res[1]


def _make_chunks(index_utc: pd.DatetimeIndex, chunk_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Produce contiguous [start, end] chunks inclusive by time, using calendar days.
    End aligned to last available bar if shorter than chunk.
    """
    if len(index_utc) == 0:
        return []

    # Ensure UTC tz-aware
    if index_utc.tz is None:
        index_utc = index_utc.tz_localize("UTC")
    else:
        index_utc = index_utc.tz_convert("UTC")

    t0 = index_utc[0].floor("D")
    t1 = index_utc[-1].floor("D")

    starts = pd.date_range(start=t0, end=t1, freq=f"{chunk_days}D", tz="UTC")
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    for s in starts:
        # Inclusive end: chunk_days of 5m bars => end = s + chunk_days - 5m
        e = (s + pd.Timedelta(days=chunk_days)) - pd.Timedelta(minutes=5)
        if e > index_utc[-1]:
            e = index_utc[-1]
        if s > index_utc[-1]:
            break
        chunks.append((s, e))

    return chunks


def _slice_df(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index <= end)].copy()


def _write_slice_csv(df_slice: pd.DataFrame, path: str) -> None:
    """
    Persist slice to CSV with explicit time column (UTC ISO string),
    keeping all original columns (including tick_volume/spread/etc) for auditability.
    """
    out = df_slice.copy()

    # Ensure index is tz-aware UTC
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    # Insert time column as first col
    out.insert(0, "time", out.index.astype("datetime64[ns, UTC]").astype(str))

    # Do not drop extra columns; order OHLC early if present (optional)
    preferred = ["time", "open", "high", "low", "close"]
    cols = preferred + [c for c in out.columns if c not in preferred]
    out = out[cols]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward harness (chunked time splits).")
    ap.add_argument("config", help="Base YAML config (e.g., configs/baseline_v1_long.yaml)")
    ap.add_argument("--chunk-days", type=int, default=90, help="Chunk size in days (default 90)")
    ap.add_argument("--out-dir", default="results/walk_forward", help="Output root for walk-forward")
    ap.add_argument("--runs-out-dir", default="results/runs", help="Where each fold run is stored")
    ap.add_argument("--derived-dir", default="data/derived/walk_forward", help="Where slice csvs are written")
    args = ap.parse_args()

    base_cfg = _load_yaml(args.config)

    wf_id = f"{_utc_now_id()}_{_hash8(_canonical_cfg_json(base_cfg))}"
    wf_dir = Path(args.out_dir) / wf_id
    wf_dir.mkdir(parents=True, exist_ok=False)

    derived_dir = Path(args.derived_dir) / wf_id
    derived_dir.mkdir(parents=True, exist_ok=True)

    # Registry config (from base cfg)
    dsreg = base_cfg.get("dataset_registry", {}) or {}
    registry_dir = str(dsreg.get("registry_dir") or "data/registry")
    allow_override = bool(dsreg.get("allow_override", False))
    override_reason = str(dsreg.get("override_reason", "") or "")
    append_match_event = bool(dsreg.get("append_match_event", False))

    instrument = base_cfg.get("instrument", {}) or {}
    symbol = str(base_cfg.get("symbol") or "UNKNOWN")
    timeframe = str(base_cfg.get("timeframe") or "UNKNOWN")
    source = str(instrument.get("data_source") or instrument.get("source") or "csv")

    # Load full dataset once (no fingerprint required here)
    res_full = load_bars_csv(
        base_cfg["data_path"],
        return_fingerprint=False,
        dataset_id="prov",
    )
    df_full = _as_df_only(res_full)

    # Ensure df_full is indexed by UTC datetime
    if not isinstance(df_full.index, pd.DatetimeIndex):
        raise ValueError("Loaded dataframe index must be a DatetimeIndex (UTC).")
    if df_full.index.tz is None:
        df_full.index = df_full.index.tz_localize("UTC")
    else:
        df_full.index = df_full.index.tz_convert("UTC")

    # Build folds
    chunks = _make_chunks(df_full.index, chunk_days=args.chunk_days)

    folds_rows = []
    leaderboard_rows = []

    for k, (start, end) in enumerate(chunks, start=1):
        df_slice = _slice_df(df_full, start, end)
        if len(df_slice) == 0:
            continue

        slice_fname = (
            f"{symbol}_{timeframe}_"
            f"{start.date().isoformat()}__{end.date().isoformat()}__{source}__fold{k:03d}.csv"
        )
        slice_path = str(derived_dir / slice_fname)

        _write_slice_csv(df_slice, slice_path)

        # Register slice dataset using fingerprint pipeline
        dataset_id_prov = build_dataset_id(
            instrument=symbol,
            timeframe=timeframe,
            start_ts="unknown",
            end_ts="unknown",
            source=source,
        )

        res2 = load_bars_csv(slice_path, return_fingerprint=True, dataset_id=dataset_id_prov)
        df_loaded, meta = _as_df_meta(res2)

        dataset_id_final = build_dataset_id(
            instrument=symbol,
            timeframe=timeframe,
            start_ts=str(meta.start_ts),
            end_ts=str(meta.end_ts),
            source=source,
        )
        meta_final = replace(meta, dataset_id=dataset_id_final)

        register_or_validate_dataset(
            meta_final,
            registry_dir=registry_dir,
            allow_new_fingerprint=allow_override,
            override_reason=override_reason,
            append_match_event=append_match_event,
        )

        # Fold config (only change name + data_path)
        fold_cfg = json.loads(json.dumps(base_cfg))  # deep copy
        fold_cfg["name"] = f"{base_cfg.get('name','run')}_WF_FOLD_{k:03d}"
        fold_cfg["data_path"] = slice_path

        out = run_from_config(fold_cfg, out_dir=args.runs_out_dir)
        m = out["metrics"]

        folds_rows.append({
            "fold": k,
            "start_utc": str(meta_final.start_ts),
            "end_utc": str(meta_final.end_ts),
            "slice_path": slice_path,
            "dataset_id": m.get("dataset_id"),
            "dataset_fp8": m.get("dataset_fp8"),
            "run_id": out["run_id"],
            "run_dir": out["outputs"]["run_dir"],
        })

        leaderboard_rows.append({
            "fold": k,
            "run_id": out["run_id"],
            "n_trades": m.get("n_trades"),
            "expectancy_R": m.get("expectancy_R"),
            "winrate_R": m.get("winrate_R"),
            "profit_factor": m.get("profit_factor"),
            "max_drawdown_R_abs": m.get("max_drawdown_R_abs"),
            "max_consecutive_losses_R": m.get("max_consecutive_losses_R"),
        })

    folds_df = pd.DataFrame(folds_rows)
    lb_df = pd.DataFrame(leaderboard_rows)

    folds_csv = wf_dir / "folds.csv"
    lb_csv = wf_dir / "leaderboard.csv"
    folds_df.to_csv(folds_csv, index=False)
    lb_df.to_csv(lb_csv, index=False)

    wf_manifest = {
        "wf_id": wf_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config_path": args.config,
        "base_cfg_hash_sha256": hashlib.sha256(_canonical_cfg_json(base_cfg).encode("utf-8")).hexdigest(),
        "chunk_days": args.chunk_days,
        "derived_dir": str(derived_dir),
        "runs_out_dir": args.runs_out_dir,
        "n_folds": int(len(folds_df)),
        "folds_csv": str(folds_csv),
        "leaderboard_csv": str(lb_csv),
        "notes": {
            "dataset_policy": "Slices are derived CSVs; raw source remains immutable.",
            "registry_policy": "Each slice registers as its own dataset_id/fingerprint (no overrides).",
        },
    }
    (wf_dir / "wf_manifest.json").write_text(json.dumps(wf_manifest, indent=2), encoding="utf-8")

    print(f"WF_ID: {wf_id}")
    print(f"WF_DIR: {wf_dir}")
    print(f"FOLDS: {len(folds_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
