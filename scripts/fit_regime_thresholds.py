from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# ---- Dataset Contract v1.1 (tu repo) ----
from backtester.data.dataset_fingerprint import build_dataset_id
from backtester.data.dataset_registry import register_or_validate_dataset
from backtester.data.loader import load_bars_csv

# ---- Regime FX module (dentro del paquete backtester) ----
from backtester.regime_fx.contract import load_regime_fx_config
from backtester.regime_fx.features import compute_er, compute_dp


def _canonical_json_dumps(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _r9(x: float) -> float:
    return float(f"{x:.9f}")


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _build_dataset_id_compat(
    *,
    symbol: str,
    timeframe: str,
    start: Any,
    end: Any,
    source: str,
) -> str:
    """
    Wrapper robusto: adapta argumentos a la firma real de build_dataset_id()
    sin asumir nombres (symbol vs instrument vs pair vs ticker, etc).

    Estrategia:
    - Intenta kwargs por intersección de nombres
    - Si no coincide, usa posicionales (orden: [id, timeframe, start, end, source]) según aridad
    """
    sig = inspect.signature(build_dataset_id)
    params = list(sig.parameters.values())

    # Mapeo semántico -> posibles nombres en tu función
    candidates = {
        "symbol": symbol,
        "instrument": symbol,
        "pair": symbol,
        "ticker": symbol,
        "symbol_or_pair": symbol,
        "timeframe": timeframe,
        "tf": timeframe,
        "granularity": timeframe,
        "start": start,
        "start_ts": start,
        "start_utc": start,
        "end": end,
        "end_ts": end,
        "end_utc": end,
        "source": source,
        "data_source": source,
        "provider": source,
    }

    # 1) Try keyword call using signature names
    kwargs: Dict[str, Any] = {}
    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        name = p.name
        if name in candidates:
            kwargs[name] = candidates[name]

    try:
        return str(build_dataset_id(**kwargs))
    except TypeError:
        # 2) Fallback: positional by arity (best-effort)
        # We assume first param corresponds to symbol/instrument/pair
        arity = sum(
            1
            for p in params
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )

        base = [symbol, timeframe, start, end, source]
        if arity <= 0:
            return str(build_dataset_id())  # unlikely, but safe
        return str(build_dataset_id(*base[:arity]))


def load_and_register_dataset(
    *,
    data_path: str,
    symbol: str,
    timeframe: str,
    data_source: str,
    registry_dir: str,
    dataset_id_forced: str | None,
    allow_override: bool,
    override_reason: str,
    append_match_event: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Mirrors orchestrator/register_dataset.py behavior:
      - Load CSV with provisional dataset_id to compute fingerprint + metadata
      - Compute FINAL stable dataset_id from meta.start_ts/end_ts (unless forced provided)
      - Register/validate with Dataset Contract v1.1
    """
    _ensure_dir(registry_dir)

    # 1) Provisional dataset_id (never registered)
    dataset_id_prov = _build_dataset_id_compat(
        symbol=symbol,
        timeframe=timeframe,
        start="unknown",
        end="unknown",
        source=data_source,
    )

    # 2) Load bars + fingerprint (loader returns df + meta dataclass)
    df, meta = load_bars_csv(str(data_path), return_fingerprint=True, dataset_id=dataset_id_prov)

    # 3) Compute FINAL dataset_id from meta.start_ts/end_ts
    start_ts = getattr(meta, "start_ts", None)
    end_ts = getattr(meta, "end_ts", None)

    dataset_id_default = _build_dataset_id_compat(
        symbol=symbol,
        timeframe=timeframe,
        start=start_ts,
        end=end_ts,
        source=data_source,
    )

    dataset_id_final = dataset_id_forced.strip() if dataset_id_forced else dataset_id_default

    # Guardrail: refuse provisional ids
    if "unknown__unknown" in str(dataset_id_final):
        raise SystemExit(
            "ERROR: refusing to use provisional dataset_id containing 'unknown__unknown'. "
            "Fix: ensure loader produced start_ts/end_ts, or force a FINAL dataset_id."
        )

    # 4) Stamp FINAL dataset_id into meta for registry call
    meta_final = replace(meta, dataset_id=dataset_id_final)

    ok, msg = register_or_validate_dataset(
        meta_final,
        registry_dir=registry_dir,
        allow_new_fingerprint=allow_override,
        override_reason=override_reason,
        append_match_event=append_match_event,
    )

    # fingerprint contract
    fingerprint = getattr(meta_final, "fingerprint_sha256", None) or getattr(meta_final, "fingerprint", None)
    if not fingerprint:
        raise ValueError("Dataset meta missing fingerprint_sha256 (dataset_hash). Check loader/meta contract.")

    if not start_ts or not end_ts:
        # recalc from meta_final in case loader sets there
        start_ts = getattr(meta_final, "start_ts", None)
        end_ts = getattr(meta_final, "end_ts", None)
    if not start_ts or not end_ts:
        raise ValueError("Dataset meta missing start_ts/end_ts. Fit requires valid time range.")

    meta_out = {
        "dataset_id": str(dataset_id_final),
        "dataset_hash": f"sha256:{fingerprint}" if not str(fingerprint).startswith("sha256:") else str(fingerprint),
        "time_range_utc": [str(start_ts), str(end_ts)],
        "registry_dir": str(registry_dir),
        "registry_msg": str(msg),
    }
    return df, meta_out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit Regime FX thresholds artifact using Dataset Contract v1.1 (deterministic, reproducible)."
    )
    ap.add_argument("--config", required=True, help="Path to configs/regime_fx.yaml")
    ap.add_argument("--data-path", required=True, help="CSV path (same format expected by load_bars_csv)")
    ap.add_argument("--symbol", required=True, help="Symbol, e.g. EURUSD")
    ap.add_argument("--timeframe", required=True, help="Timeframe, e.g. M5")
    ap.add_argument("--data-source", default="csv", help="Dataset source label used in dataset_id semantics")
    ap.add_argument("--registry-dir", default="data/registry", help="Registry dir (default: data/registry)")

    ap.add_argument("--dataset-id", default="", help="Optional FINAL dataset_id override (advanced)")
    ap.add_argument(
        "--allow-override",
        action="store_true",
        help="Allow updating an existing dataset_id with a new fingerprint (writes UPDATE event).",
    )
    ap.add_argument("--override-reason", default="", help="Required if --allow-override is set")
    ap.add_argument(
        "--append-match-event",
        action="store_true",
        help="Also append MATCH events to datasets.jsonl when fingerprint matches.",
    )

    ap.add_argument("--out", required=True, help="Output thresholds artifact path (json)")
    ap.add_argument("--p-low", type=float, default=0.30, help="Low percentile (range)")
    ap.add_argument("--p-high", type=float, default=0.70, help="High percentile (trend)")

    args = ap.parse_args()

    if args.allow_override and not args.override_reason.strip():
        raise SystemExit("ERROR: --allow-override requires --override-reason (non-empty).")

    cfg = load_regime_fx_config(args.config)

    df, meta = load_and_register_dataset(
        data_path=args.data_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_source=args.data_source,
        registry_dir=args.registry_dir,
        dataset_id_forced=(args.dataset_id.strip() or None),
        allow_override=bool(args.allow_override),
        override_reason=str(args.override_reason or ""),
        append_match_event=bool(args.append_match_event),
    )

    if "close" not in df.columns:
        raise ValueError("Loaded DF missing 'close' column (required).")

    er = compute_er(df["close"], cfg.er.lookback, warmup=cfg.er.warmup)
    dp = compute_dp(df["close"], cfg.dp.lookback, warmup=cfg.dp.warmup)

    # drop warmup NaNs
    m = (~er.isna()) & (~dp.isna())
    er_fit = er.loc[m].astype("float64").values
    dp_fit = dp.loc[m].astype("float64").values

    if len(er_fit) < 1000:
        raise ValueError(f"Too few rows for percentile fit: {len(er_fit)} (need >= 1000)")

    p_low = float(args.p_low)
    p_high = float(args.p_high)
    if not (0.0 < p_low < p_high < 1.0):
        raise ValueError("Percentiles must satisfy 0 < p_low < p_high < 1")

    er_lo = _r9(float(np.quantile(er_fit, p_low)))
    er_hi = _r9(float(np.quantile(er_fit, p_high)))
    dp_lo = _r9(float(np.quantile(dp_fit, p_low)))
    dp_hi = _r9(float(np.quantile(dp_fit, p_high)))

    artifact = {
        "schema_version": "1.0",
        "module": "regime_fx",
        "created_utc": _utc_now_iso(),
        "dataset": {
            "dataset_id": meta["dataset_id"],
            "dataset_hash": meta["dataset_hash"],
            "time_range_utc": meta["time_range_utc"],
        },
        "features": {
            "er": {"lookback": cfg.er.lookback, "definition": "abs(c_t-c_{t-n})/sum(abs(diff(close)))"},
            "dp": {"lookback": cfg.dp.lookback, "definition": "abs(sum(sign(diff(close))))/n"},
        },
        "percentiles": {
            f"p{int(p_low*100)}": {"er": er_lo, "dp": dp_lo},
            f"p{int(p_high*100)}": {"er": er_hi, "dp": dp_hi},
        },
        "thresholds": {
            "er_trend": er_hi,
            "er_range": er_lo,
            "dp_trend": dp_hi,
            "dp_range": dp_lo,
        },
        "fit_policy": {
            "mode": "static_percentiles",
            "warmup": cfg.er.warmup,
            "nan_policy": "drop_warmup_rows",
        },
        "provenance": {
            "registry_dir": meta["registry_dir"],
            "registry_msg": meta["registry_msg"],
            "config_version": cfg.version,
        },
    }

    out_path = Path(args.out)
    _ensure_dir(out_path.parent)
    out_path.write_text(_canonical_json_dumps(artifact) + "\n", encoding="utf-8")

    print(f"[OK] thresholds artifact written: {out_path}")
    print(f"     dataset_id:   {meta['dataset_id']}")
    print(f"     dataset_hash: {meta['dataset_hash']}")
    print(f"     ER range/trend: {er_lo} / {er_hi}")
    print(f"     DP range/trend: {dp_lo} / {dp_hi}")
    print(f"     registry: {meta['registry_msg']}")


if __name__ == "__main__":
    main()
