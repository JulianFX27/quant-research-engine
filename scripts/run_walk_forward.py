# scripts/run_walk_forward.py
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from backtester.data.loader import load_bars_csv
from backtester.orchestrator.run import _canonical_cfg_json, run_from_config


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _hash8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fp8_from_file(path: str) -> str:
    return _file_sha256(path)[:8]


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
    if isinstance(res, tuple):
        if len(res) == 0:
            raise ValueError("load_bars_csv returned an empty tuple")
        return res[0]
    return res


def _to_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def _parse_timeframe_to_minutes(tf: str) -> int:
    s = str(tf).strip().upper()
    if not s:
        raise ValueError("timeframe is empty")
    if s.startswith("M"):
        n = int(s[1:])
        if n <= 0:
            raise ValueError(f"Invalid timeframe minutes: {tf}")
        return n
    if s.startswith("H"):
        n = int(s[1:])
        if n <= 0:
            raise ValueError(f"Invalid timeframe hours: {tf}")
        return n * 60
    if s.startswith("D"):
        n = int(s[1:])
        if n <= 0:
            raise ValueError(f"Invalid timeframe days: {tf}")
        return n * 1440
    raise ValueError(f"Unsupported timeframe format: {tf}")


def _make_chunks(
    index_utc: pd.DatetimeIndex,
    chunk_days: int,
    bar_minutes: int,
    eof_buffer_bars: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if len(index_utc) == 0:
        return []

    if index_utc.tz is None:
        index_utc = index_utc.tz_localize("UTC")
    else:
        index_utc = index_utc.tz_convert("UTC")

    t0 = index_utc[0].floor("D")
    t1 = index_utc[-1].floor("D")

    starts = pd.date_range(start=t0, end=t1, freq=f"{chunk_days}D", tz="UTC")
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    bar_delta = pd.Timedelta(minutes=bar_minutes)
    buffer_delta = pd.Timedelta(minutes=bar_minutes * max(int(eof_buffer_bars), 0))

    for s in starts:
        # inclusive end: s + chunk_days - 1 bar, then remove EOF buffer
        e = (s + pd.Timedelta(days=chunk_days)) - bar_delta
        e = e - buffer_delta

        if e > index_utc[-1]:
            e = index_utc[-1]
        if e < s:
            continue
        if s > index_utc[-1]:
            break

        chunks.append((s, e))

    return chunks


def _slice_df(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index <= end)].copy()


def _write_slice_csv(df_slice: pd.DataFrame, path: str) -> None:
    out = df_slice.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    out.insert(0, "time", out.index.astype("datetime64[ns, UTC]").astype(str))

    preferred = ["time", "open", "high", "low", "close"]
    cols = preferred + [c for c in out.columns if c not in preferred]
    out = out[cols]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _valid_mask(lb_df: pd.DataFrame, validity_mode: str) -> pd.Series:
    """
    validity_mode:
      - strict_no_eof: only folds with invalid_eof == False are valid
      - allow_eof: all folds are usable (but still marked for filtering)
    """
    if len(lb_df) == 0:
        return pd.Series([], dtype=bool)

    if validity_mode not in {"strict_no_eof", "allow_eof"}:
        validity_mode = "strict_no_eof"

    if validity_mode == "allow_eof":
        return pd.Series([True] * len(lb_df), index=lb_df.index)

    if "invalid_eof" not in lb_df.columns:
        return pd.Series([True] * len(lb_df), index=lb_df.index)

    inv = lb_df["invalid_eof"].apply(_to_bool).fillna(False)
    return inv == False


def _wf_quality_flag(n_trades: int, forced_exits_total: int, non_forced_exits_total: int) -> str:
    if int(n_trades) <= 0:
        return "NO_TRADES"
    if int(forced_exits_total) <= 0:
        return "NO_FORCED_EXITS"
    if int(non_forced_exits_total) <= 0:
        return "ALL_EXITS_FORCED"
    return "MIXED_EXITS"


def _deepcopy_json(x: Any) -> Any:
    return json.loads(json.dumps(x))


def _ensure_fold_eof_alignment(fold_cfg: Dict[str, Any], eof_buffer_bars: int) -> None:
    """
    Critical WF correctness:
      - Slicing CSV removes last `eof_buffer_bars` bars, but the engine/guardrails
        only gate entries using risk.eof_buffer_bars. If it stays at the base YAML
        value (e.g., 50) while slices are cut by 120, you'll still get FORCE_EOF.
      - Therefore, propagate eof_buffer_bars into fold_cfg.risk.eof_buffer_bars.

    Optional safety:
      - If max_holding_bars is missing/0, set it to eof_buffer_bars so the WF strict mode
        has deterministic "can this trade finish before EOF?" semantics.
    """
    fold_cfg.setdefault("risk", {})
    if not isinstance(fold_cfg["risk"], dict):
        fold_cfg["risk"] = {}

    fold_cfg["risk"]["eof_buffer_bars"] = int(eof_buffer_bars)

    mh = fold_cfg["risk"].get("max_holding_bars")
    try:
        mh_i = int(mh) if mh is not None else 0
    except Exception:
        mh_i = 0

    if mh_i <= 0:
        fold_cfg["risk"]["max_holding_bars"] = int(eof_buffer_bars)


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward harness (chunked time splits).")
    ap.add_argument("config", help="Base YAML config")
    ap.add_argument("--chunk-days", type=int, default=90, help="Chunk size in days (default 90)")
    ap.add_argument("--out-dir", default="results/walk_forward", help="Output root for walk-forward")
    ap.add_argument("--runs-out-dir", default="results/runs", help="Where each fold run is stored")
    ap.add_argument("--derived-dir", default="data/derived/walk_forward", help="Where slice csvs are written")
    ap.add_argument(
        "--eof-buffer-bars",
        type=int,
        default=-1,
        help="Bars to cut from end of each fold. If -1, uses risk.max_holding_bars from config.",
    )
    ap.add_argument(
        "--validity-mode",
        choices=["strict_no_eof", "allow_eof"],
        default="strict_no_eof",
        help="How to count folds as 'valid' in wf_summary.",
    )
    args = ap.parse_args()

    base_cfg = _load_yaml(args.config)

    wf_id = f"{_utc_now_id()}_{_hash8(_canonical_cfg_json(base_cfg))}"
    wf_dir = Path(args.out_dir) / wf_id
    wf_dir.mkdir(parents=True, exist_ok=False)

    derived_dir = Path(args.derived_dir) / wf_id
    derived_dir.mkdir(parents=True, exist_ok=True)

    instrument = base_cfg.get("instrument", {}) or {}
    symbol = str(base_cfg.get("symbol") or "UNKNOWN")
    timeframe = str(base_cfg.get("timeframe") or "UNKNOWN")
    source = str(instrument.get("data_source") or instrument.get("source") or "csv")

    bar_minutes = _parse_timeframe_to_minutes(timeframe)

    risk_cfg = base_cfg.get("risk", {}) or {}
    cfg_max_hold = risk_cfg.get("max_holding_bars", 0)
    try:
        cfg_max_hold_int = int(cfg_max_hold) if cfg_max_hold is not None else 0
    except Exception:
        cfg_max_hold_int = 0

    eof_buffer_bars = int(args.eof_buffer_bars)
    if eof_buffer_bars < 0:
        eof_buffer_bars = max(cfg_max_hold_int, 0)

    # Load full dataset once
    res_full = load_bars_csv(
        base_cfg["data_path"],
        return_fingerprint=False,
        dataset_id="prov",
    )
    df_full = _as_df_only(res_full)

    if not isinstance(df_full.index, pd.DatetimeIndex):
        raise ValueError("Loaded dataframe index must be a DatetimeIndex (UTC).")
    if df_full.index.tz is None:
        df_full.index = df_full.index.tz_localize("UTC")
    else:
        df_full.index = df_full.index.tz_convert("UTC")

    chunks = _make_chunks(
        df_full.index,
        chunk_days=args.chunk_days,
        bar_minutes=bar_minutes,
        eof_buffer_bars=eof_buffer_bars,
    )

    folds_rows: List[Dict[str, Any]] = []
    leaderboard_rows: List[Dict[str, Any]] = []

    for k, (start, end) in enumerate(chunks, start=1):
        df_slice = _slice_df(df_full, start, end)
        if len(df_slice) == 0:
            continue

        # NOTE: file name uses *requested* date range (start/end days), not necessarily exact last timestamp
        slice_fname = (
            f"{symbol}_{timeframe}_"
            f"{start.date().isoformat()}__{end.date().isoformat()}__{source}__fold{k:03d}.csv"
        )
        slice_path = str(derived_dir / slice_fname)

        _write_slice_csv(df_slice, slice_path)

        # ✅ Content-addressed identity for WF slice (fp8 from file bytes)
        wf_slice_fp8 = _fp8_from_file(slice_path)

        # ✅ Force a dataset_id that cannot collide across regenerated slices
        wf_dataset_id_forced = (
            f"{symbol}_{timeframe}_"
            f"{start.date().isoformat()}__{end.date().isoformat()}__{source}"
            f"__fold{k:03d}__fp8_{wf_slice_fp8}"
        )

        fold_cfg = _deepcopy_json(base_cfg)
        fold_cfg["name"] = f"{base_cfg.get('name','run')}_WF_FOLD_{k:03d}"
        fold_cfg["data_path"] = slice_path

        # ✅ Force fold dataset_id (required to avoid registry collisions on regenerated slices)
        fold_cfg["dataset_id"] = wf_dataset_id_forced

        # ✅ CRITICAL: align engine EOF gate with WF truncation (fixes INVALID_SAMPLE_EOF)
        _ensure_fold_eof_alignment(fold_cfg, eof_buffer_bars=eof_buffer_bars)

        out = run_from_config(fold_cfg, out_dir=args.runs_out_dir)
        m = out.get("metrics", {}) or {}

        # Canonical counters from metrics.json
        n_trades = int(m.get("n_trades") or 0)
        forced_exits_total = int(m.get("forced_exits_total") or 0)
        forced_eof_total = int(m.get("forced_eof_total") or 0)
        non_forced_exits_total = int(m.get("non_forced_exits_total") or 0)

        m_valid_trades_raw = max(n_trades - forced_eof_total, 0)
        m_valid_trade_ratio_raw = (float(m_valid_trades_raw) / float(n_trades)) if n_trades > 0 else 0.0

        valid_trades = int(m.get("valid_trades") or 0)
        valid_trade_ratio = float(m.get("valid_trade_ratio") or 0.0)

        wf_quality_flag = _wf_quality_flag(
            n_trades=n_trades,
            forced_exits_total=forced_exits_total,
            non_forced_exits_total=non_forced_exits_total,
        )

        folds_rows.append(
            {
                "fold": k,
                "start_utc": m.get("dataset_start_time_utc"),
                "end_utc": m.get("dataset_end_time_utc"),
                "slice_path": slice_path,
                "dataset_id": m.get("dataset_id"),
                "dataset_fp8": m.get("dataset_fp8"),
                "run_id": out.get("run_id"),
                "run_dir": (out.get("outputs", {}) or {}).get("run_dir"),
                "wf_slice_fp8": wf_slice_fp8,
                "wf_dataset_id_forced": wf_dataset_id_forced,
                # helpful: what WF intended for this fold
                "wf_eof_buffer_bars": int(eof_buffer_bars),
            }
        )

        # ✅ Pull-through entry-gate v2 EOF buffer diagnostics (from metrics.json)
        eg_v2_attempted = m.get("entry_gate_v2_attempted_entries")
        eg_v2_blocked_total = m.get("entry_gate_v2_blocked_total")
        eg_v2_blocked_unique = m.get("entry_gate_v2_blocked_unique_bars")
        eg_v2_by_eof_buffer = m.get("entry_gate_v2_blocked_by_reason__by_eof_buffer")

        leaderboard_rows.append(
            {
                "fold": k,
                "run_id": out.get("run_id"),
                "run_status": m.get("run_status"),
                "invalid_eof": _to_bool(m.get("invalid_eof")),

                "m_valid_trades_raw": m_valid_trades_raw,
                "m_valid_trade_ratio_raw": m_valid_trade_ratio_raw,
                "valid_trades": valid_trades,
                "valid_trade_ratio": valid_trade_ratio,

                "non_forced_trades": non_forced_exits_total,
                "forced_trades_total": forced_exits_total,
                "forced_trades_non_eof": max(forced_exits_total - forced_eof_total, 0),

                "eof_incomplete_count": m.get("eof_incomplete_count"),
                "eof_forced_count": m.get("eof_forced_count"),
                "forced_exits_total": forced_exits_total,
                "forced_eof_total": forced_eof_total,
                "eof_exits_total": m.get("eof_exits_total"),
                "non_forced_exits_total": non_forced_exits_total,

                "n_trades": n_trades,
                "expectancy_R": m.get("expectancy_R"),
                "winrate_R": m.get("winrate_R"),
                "avg_R": m.get("avg_R"),
                "profit_factor": m.get("profit_factor"),
                "max_drawdown_R_pct": m.get("max_drawdown_R_pct"),
                "max_consecutive_losses_R": m.get("max_consecutive_losses_R"),

                # Entry-gate v2 diagnostics (so it exists in WF leaderboard.csv)
                "entry_gate_v2_attempted_entries": eg_v2_attempted,
                "entry_gate_v2_blocked_total": eg_v2_blocked_total,
                "entry_gate_v2_blocked_unique_bars": eg_v2_blocked_unique,
                "entry_gate_v2_blocked_by_reason__by_eof_buffer": eg_v2_by_eof_buffer,

                "execution_policy_id": m.get("execution_policy_id"),
                "execution_fill_mode": m.get("execution_fill_mode"),
                "execution_intrabar_path": m.get("execution_intrabar_path"),
                "execution_intrabar_tie": m.get("execution_intrabar_tie"),
                "costs_spread_pips_effective": m.get("costs_spread_pips_effective"),
                "costs_slippage_pips_effective": m.get("costs_slippage_pips_effective"),

                "dataset_id": m.get("dataset_id"),
                "dataset_fp8": m.get("dataset_fp8"),
                "dataset_rows": m.get("dataset_rows"),
                "dataset_start_time_utc": m.get("dataset_start_time_utc"),
                "dataset_end_time_utc": m.get("dataset_end_time_utc"),
                "dataset_file_sha256": m.get("dataset_file_sha256"),
                "dataset_source_path": m.get("dataset_source_path"),

                "risk_max_daily_loss_R": m.get("risk_max_daily_loss_R"),
                "risk_max_trades_per_day": m.get("risk_max_trades_per_day"),
                "risk_cooldown_bars": m.get("risk_cooldown_bars"),
                "risk_final_realized_R_today": m.get("risk_final_realized_R_today"),
                "risk_final_stopped_today": m.get("risk_final_stopped_today"),

                # helpful: effective engine EOF settings (from run.py metrics)
                "risk_eof_buffer_bars": m.get("risk_eof_buffer_bars"),
                "risk_force_exit_on_eof": m.get("risk_force_exit_on_eof"),

                "wf_quality_flag": wf_quality_flag,
            }
        )

    folds_df = pd.DataFrame(folds_rows)
    lb_df = pd.DataFrame(leaderboard_rows)

    folds_csv = wf_dir / "folds.csv"
    lb_csv = wf_dir / "leaderboard.csv"
    folds_df.to_csv(folds_csv, index=False)
    lb_df.to_csv(lb_csv, index=False)

    mask = _valid_mask(lb_df, args.validity_mode)
    valid_df = lb_df[mask].copy() if len(lb_df) else lb_df.copy()

    quality_counts = valid_df["wf_quality_flag"].value_counts(dropna=False).to_dict() if len(valid_df) else {}
    folds_valid = int(len(valid_df))
    pct_all_forced = (
        float(quality_counts.get("ALL_EXITS_FORCED", 0)) / float(folds_valid)
        if folds_valid > 0 else 0.0
    )

    summary = {
        "wf_id": wf_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_days": args.chunk_days,
        "bar_minutes": bar_minutes,
        "eof_buffer_bars": int(eof_buffer_bars),
        "validity_mode": args.validity_mode,
        "folds_total": int(len(lb_df)),
        "folds_valid": int(len(valid_df)),
        "median_expectancy_R_valid": _safe_float(valid_df["expectancy_R"].median()) if len(valid_df) else None,
        "min_expectancy_R_valid": _safe_float(valid_df["expectancy_R"].min()) if len(valid_df) else None,
        "max_drawdown_R_pct_worst_valid": _safe_float(valid_df["max_drawdown_R_pct"].max()) if len(valid_df) else None,
        "max_consecutive_losses_R_worst_valid": _safe_float(valid_df["max_consecutive_losses_R"].max()) if len(valid_df) else None,
        "quality_valid": {
            "NO_TRADES": int(quality_counts.get("NO_TRADES", 0)),
            "ALL_EXITS_FORCED": int(quality_counts.get("ALL_EXITS_FORCED", 0)),
            "MIXED_EXITS": int(quality_counts.get("MIXED_EXITS", 0)),
            "NO_FORCED_EXITS": int(quality_counts.get("NO_FORCED_EXITS", 0)),
            "UNKNOWN": int(quality_counts.get("UNKNOWN", 0)),
            "pct_ALL_EXITS_FORCED": float(pct_all_forced),
        },
    }
    (wf_dir / "wf_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    wf_manifest = {
        "wf_id": wf_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config_path": args.config,
        "base_cfg_hash_sha256": hashlib.sha256(_canonical_cfg_json(base_cfg).encode("utf-8")).hexdigest(),
        "chunk_days": args.chunk_days,
        "bar_minutes": bar_minutes,
        "eof_buffer_bars": int(eof_buffer_bars),
        "validity_mode": args.validity_mode,
        "derived_dir": str(derived_dir),
        "runs_out_dir": args.runs_out_dir,
        "n_folds": int(len(folds_df)),
        "folds_csv": str(folds_csv),
        "leaderboard_csv": str(lb_csv),
        "wf_summary_json": str(wf_dir / "wf_summary.json"),
        "notes": {
            "dataset_policy": "Slices are derived CSVs; raw source remains immutable.",
            "registry_policy": "Slices are registered by run_from_config (single source of truth).",
            "validity_policy": "invalid_eof/run_status preserved in leaderboard; wf_summary counts folds via validity_mode.",
            "eof_policy": (
                "WF truncates the last eof_buffer_bars AND forces fold_cfg.risk.eof_buffer_bars to the same value "
                "(critical) so the engine/guardrails gate entries consistently."
            ),
            "wf_dataset_id_policy": "Per-fold dataset_id is content-addressed with fp8(file_sha256) to avoid collisions.",
        },
    }
    (wf_dir / "wf_manifest.json").write_text(json.dumps(wf_manifest, indent=2), encoding="utf-8")

    print(f"WF_ID: {wf_id}")
    print(f"WF_DIR: {wf_dir}")
    print(f"FOLDS: {len(folds_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
