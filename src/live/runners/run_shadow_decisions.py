from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from src.live.csv_tail_provider import TailCSVBarProvider, TailCSVConfig, TailBar
from src.live.state.shadow_state_store import ShadowStateStore
from src.runner.config_loader import load_config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha1_16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


@dataclass(frozen=True)
class ShadowDecision:
    bar_ts_utc: str
    row_idx: int
    bar_key: str

    decision: str               # NO_TRADE / WOULD_TRADE / ERROR
    side: str                   # LONG/SHORT/"" if none
    sl_pips: float
    tp_pips: float
    max_hold_min: int

    reason: str                 # reason_code
    strategy_id: str
    strategy_version: str

    frozen_config_path: str
    execution_policy_path: str
    override_path: str


class ShadowDecisionsLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "schema_version","ts_utc","bar_ts_utc","row_idx","bar_key",
                    "decision","side","sl_pips","tp_pips","max_hold_min","reason",
                    "strategy_id","strategy_version",
                    "frozen_config_path","execution_policy_path","override_path"
                ])

    def append(self, d: ShadowDecision) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "shadow_decisions_v1", utc_now_iso(), d.bar_ts_utc, d.row_idx, d.bar_key,
                d.decision, d.side, f"{d.sl_pips:.4f}", f"{d.tp_pips:.4f}", int(d.max_hold_min), d.reason,
                d.strategy_id, d.strategy_version,
                d.frozen_config_path, d.execution_policy_path, d.override_path
            ])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--time_col", default="ts_utc")
    ap.add_argument("--poll_seconds", type=float, default=2.0)
    ap.add_argument("--start_from_last_row", action="store_true")

    ap.add_argument("--state_path", required=True)
    ap.add_argument("--results_dir", default=r".\results\shadow_runs")
    ap.add_argument("--run_tag", default="FTMO_SHADOW")

    ap.add_argument("--frozen_config", required=True)
    ap.add_argument("--execution_policy", required=True)
    ap.add_argument("--override_path", required=True)
    ap.add_argument("--account_mode", choices=["challenge","funded"], required=True)
    return ap.parse_args()


# ---------- STRATEGY HOOK (placeholder) ----------
def decide_readonly(
    *,
    bar: Dict[str, Any],
    frozen_cfg: Dict[str, Any],
    exec_cfg: Dict[str, Any],
    override_cfg: Dict[str, Any],
    account_mode: str,
) -> Tuple[str, str, float, float, int, str, str]:
    """
    Returns:
      decision, side, sl_pips, tp_pips, max_hold_min, reason, strategy_version
    """
    strategy_version = str(frozen_cfg.get("strategy_version", frozen_cfg.get("version", "UNKNOWN")))
    return ("NO_TRADE", "", 0.0, 0.0, 0, "PLACEHOLDER_NOT_WIRED", strategy_version)


def _tailbar_to_dict(event: TailBar, *, time_col: str) -> Dict[str, Any]:
    """
    Convert TailBar into a dict contract expected by decide_readonly() and for stable hashing.
    - Keep ALL CSV columns in extras (strings)
    - Add _row_idx for deterministic sequencing
    - Ensure time_col exists as ISO string
    """
    d = dict(event.extras) if getattr(event, "extras", None) else {}
    d["_row_idx"] = int(getattr(event, "row_idx", -1))

    # Ensure time_col value is present and string
    ts = getattr(event, "ts_utc", None)
    if ts is not None:
        d[time_col] = ts.isoformat()

    # Optional: if you ever want raw OHLC available as numeric
    d.setdefault("open", getattr(event, "open", None))
    d.setdefault("high", getattr(event, "high", None))
    d.setdefault("low", getattr(event, "low", None))
    d.setdefault("close", getattr(event, "close", None))

    return d


def main() -> None:
    args = parse_args()

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = ensure_dir(Path(args.results_dir) / run_id)
    logs_path = run_dir / "logs.txt"
    decisions_path = run_dir / "shadow_decisions.csv"

    append_line(logs_path, f"[{utc_now_iso()}] shadow_run_id={run_id}")
    append_line(logs_path, f"[{utc_now_iso()}] csv={args.csv}")
    append_line(logs_path, f"[{utc_now_iso()}] time_col={args.time_col}")
    append_line(logs_path, f"[{utc_now_iso()}] state_path={args.state_path}")
    append_line(logs_path, f"[{utc_now_iso()}] frozen_config={args.frozen_config}")
    append_line(logs_path, f"[{utc_now_iso()}] execution_policy={args.execution_policy}")
    append_line(logs_path, f"[{utc_now_iso()}] override_path={args.override_path}")
    append_line(logs_path, f"[{utc_now_iso()}] account_mode={args.account_mode}")

    frozen_cfg = load_config(args.frozen_config)
    exec_cfg = load_config(args.execution_policy)
    override_cfg = load_config(args.override_path)

    state = ShadowStateStore(args.state_path)
    last_idx = state.get_last_row_idx()
    append_line(logs_path, f"[{utc_now_iso()}] state_last_row_idx={last_idx}")

    cfg = TailCSVConfig(
        csv_path=args.csv,
        time_col=args.time_col,
        poll_seconds=float(args.poll_seconds),
        start_from_last_row=bool(args.start_from_last_row),
    )
    provider = TailCSVBarProvider(cfg)
    out = ShadowDecisionsLog(decisions_path)

    append_line(logs_path, f"[{utc_now_iso()}] STATUS=RUNNING decisions_loop")

    for event in provider.iter_bars_live():
        # event is TailBar
        bar = _tailbar_to_dict(event, time_col=args.time_col)

        row_idx = int(bar.get("_row_idx", -1))
        if row_idx < 0:
            # fallback if something went weird
            row_idx = last_idx + 1

        if row_idx <= last_idx:
            continue

        # Prefer provider bar_key (stable). Fallback to sha1 over full row dict.
        bar_key = getattr(event, "bar_key", "")
        if not bar_key:
            row_repr = "|".join([f"{k}={bar.get(k)}" for k in sorted(bar.keys())])
            bar_key = sha1_16(row_repr)

        if state.is_seen(bar_key):
            continue

        bar_ts = str(bar.get(args.time_col) or "")

        # Decision (read-only)
        try:
            decision, side, sl_pips, tp_pips, max_hold_min, reason, strategy_version = decide_readonly(
                bar=bar,
                frozen_cfg=frozen_cfg,
                exec_cfg=exec_cfg,
                override_cfg=override_cfg,
                account_mode=args.account_mode,
            )
            strategy_id = str(frozen_cfg.get("strategy_id", frozen_cfg.get("strategy", "UNKNOWN")))
        except Exception as e:
            decision, side, sl_pips, tp_pips, max_hold_min = "ERROR", "", 0.0, 0.0, 0
            reason = f"EXC:{type(e).__name__}"
            strategy_id = str(frozen_cfg.get("strategy_id", frozen_cfg.get("strategy", "UNKNOWN")))
            strategy_version = str(frozen_cfg.get("strategy_version", frozen_cfg.get("version", "UNKNOWN")))

        out.append(
            ShadowDecision(
                bar_ts_utc=bar_ts,
                row_idx=row_idx,
                bar_key=bar_key,
                decision=str(decision),
                side=str(side),
                sl_pips=float(sl_pips),
                tp_pips=float(tp_pips),
                max_hold_min=int(max_hold_min),
                reason=str(reason),
                strategy_id=strategy_id,
                strategy_version=str(strategy_version),
                frozen_config_path=str(args.frozen_config),
                execution_policy_path=str(args.execution_policy),
                override_path=str(args.override_path),
            )
        )

        append_line(
            logs_path,
            f"[{utc_now_iso()}] DECISION row_idx={row_idx} ts={bar_ts} {decision} {side} reason={reason}"
        )

        state.mark_seen(bar_key)
        state.set_last_row_idx(row_idx)
        last_idx = row_idx


if __name__ == "__main__":
    main()