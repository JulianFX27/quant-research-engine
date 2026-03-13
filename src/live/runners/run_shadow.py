# src/live/runners/run_shadow.py
from __future__ import annotations

# ------------------------------------------------------------
# Shadow runner v2 — Strategy-exact, read-only mirror
#
# Goals:
#   - Consume live feature CSV (eurusd_m5_features_latest.csv)
#   - Build Bar(extras=ALL CSV cols) EXACTLY like paper runner expects
#   - Run the SAME strategy logic (AnchorReversionAdapter)
#   - Evaluate the SAME PolicyGate logic (optional, from frozen config)
#   - Log one row per bar with intent + gate decision + key diagnostics
#
# IMPORTANT:
#   - Shadow NEVER submits to engine, never affects paper
#   - Shadow keeps its own durable state (ShadowStateStore)
#   - Risk gate (max trades/day) is informational here, isolated from paper
#   - FIX: do NOT mutate adapter state via on_intent_submitted(), because
#          without a real engine lifecycle the adapter can get stuck in
#          _pending=True and suppress future signals in backfill/live audit.
# ------------------------------------------------------------

import argparse
import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from zoneinfo import ZoneInfo

from src.live.csv_tail_provider import TailCSVBarProvider, TailCSVConfig, TailBar
from src.live.state.shadow_state_store import ShadowStateStore
from src.runner.config_loader import load_config
from src.runner.interfaces import Bar, StrategyContext, OrderIntent

from src.strategies.anchor_adapter import AnchorReversionAdapter, AnchorAdapterConfig


NY_TZ = ZoneInfo("America/New_York")


# ----------------------------
# small utils
# ----------------------------
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


def _shape_hint(obj: Any) -> str:
    if not isinstance(obj, dict):
        return f"type={type(obj)}"
    return f"dict_keys={sorted(list(obj.keys()))}"


def _resolve_execution_policy(exec_cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Mirror of run_ftmo_paper_live._resolve_execution_policy for shape-compat.
    Returns (ep, meta, dbg)
    """
    dbg: Dict[str, Any] = {"exec_cfg_shape": _shape_hint(exec_cfg)}

    if not isinstance(exec_cfg, dict):
        raise TypeError(f"Execution policy config must be a dict, got {type(exec_cfg)}")

    if ("internal_limits" in exec_cfg) and ("order_execution" in exec_cfg) and ("risk_mode" in exec_cfg):
        meta = exec_cfg.get("execution_policy", {})
        if not isinstance(meta, dict):
            meta = {}
        dbg["selected"] = "ROOT_FLATTENED"
        dbg["meta_shape"] = _shape_hint(meta)
        return exec_cfg, meta, dbg

    if "execution_policy" in exec_cfg and isinstance(exec_cfg["execution_policy"], dict):
        ep = exec_cfg["execution_policy"]
        dbg["selected"] = "CANONICAL"
        dbg["meta_shape"] = _shape_hint(ep)
        return ep, ep, dbg

    dbg["selected"] = "UNKNOWN"
    return exec_cfg, {}, dbg


def _strategy_id_from_cfg(frozen_cfg: Dict[str, Any]) -> str:
    return str(
        frozen_cfg.get("strategy_id")
        or frozen_cfg.get("name")
        or frozen_cfg.get("strategy", {}).get("name")
        or "UNKNOWN"
    )


def _strategy_version_from_cfg(frozen_cfg: Dict[str, Any]) -> str:
    return str(
        frozen_cfg.get("strategy_version")
        or frozen_cfg.get("version")
        or "UNKNOWN"
    )


def _ny_day_id(ts_utc: datetime) -> str:
    return ts_utc.astimezone(NY_TZ).strftime("%Y-%m-%d")


def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            v = float(x)
            if v != v:
                return default
            return v
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return default
        v = float(s)
        if v != v:
            return default
        return v
    except Exception:
        return default


# ----------------------------
# PolicyGate (copied, minimal, deterministic) — same as paper runner
# ----------------------------
class PolicyGate:
    def __init__(self, gate_cfg: Dict[str, Any]):
        self.cfg = gate_cfg
        self.policy_id = str(gate_cfg.get("policy_id", "UNKNOWN_GATE"))
        self.shock_col = str(gate_cfg["binning"]["shock_col"])
        self.shock_ret_col = str(gate_cfg["binning"]["shock_ret_col"])
        self.vol_col = str(gate_cfg["binning"]["vol_col"])
        self.session_bucket_min = int(gate_cfg["binning"].get("session_bucket_min", 30))

        self.rules = gate_cfg.get("rules", {}) or {}
        self.enable = list(self.rules.get("enable_shock_vol", []) or [])
        self.disable = list(self.rules.get("disable_shock_vol", []) or [])
        self.disable_session_buckets = set(self.rules.get("disable_session_buckets", []) or [])

    @staticmethod
    def _session_bucket_30m_ny(ts_utc: datetime) -> int:
        ny = ts_utc.astimezone(NY_TZ)
        return ny.hour * 2 + (1 if ny.minute >= 30 else 0)

    @staticmethod
    def _shock_mag_bin_from_z(shock_z: float) -> str:
        z = abs(float(shock_z))
        if z < 0.5:
            return "Q1"
        if z < 1.0:
            return "Q2"
        if z < 1.5:
            return "Q3"
        return "Q4"

    @staticmethod
    def _vol_bucket_from_atr(atr_14: float) -> str:
        a = float(atr_14)
        if a <= 0.00022:
            return "VOL_LOW"
        if a <= 0.00030:
            return "VOL_MED"
        return "VOL_HIGH"

    @staticmethod
    def _is_missing(x: Any) -> bool:
        if x is None:
            return True
        if isinstance(x, str):
            s = x.strip().lower()
            if s == "" or s in ("nan", "none"):
                return True
        try:
            v = float(x)
            return v != v
        except Exception:
            return True

    def evaluate_bar(self, ts_utc: datetime, feats: Dict[str, Any]) -> Tuple[Optional[bool], Dict[str, Any]]:
        session_bucket = self._session_bucket_30m_ny(ts_utc)

        if session_bucket in self.disable_session_buckets:
            return False, {"gate": "BLOCK", "reason": "DISABLED_SESSION_BUCKET", "session_bucket": session_bucket}

        shock_z = feats.get(self.shock_col)
        shock_ret = feats.get(self.shock_ret_col)
        atr_14 = feats.get(self.vol_col)

        if self._is_missing(shock_z) or self._is_missing(shock_ret) or self._is_missing(atr_14):
            return None, {
                "gate": "SKIP",
                "reason": "MISSING_REQUIRED_FEATURES_WARMUP",
                "session_bucket": session_bucket,
                "missing": {
                    self.shock_col: self._is_missing(shock_z),
                    self.shock_ret_col: self._is_missing(shock_ret),
                    self.vol_col: self._is_missing(atr_14),
                },
            }

        shock_sign = "+" if float(shock_ret) >= 0 else "-"
        shock_mag_bin = self._shock_mag_bin_from_z(float(shock_z))
        vol_bucket = self._vol_bucket_from_atr(float(atr_14))
        key = {"shock_sign": shock_sign, "shock_mag_bin": shock_mag_bin, "vol_bucket": vol_bucket}

        for row in self.disable:
            if (
                str(row.get("shock_sign")) == shock_sign
                and str(row.get("shock_mag_bin")) == shock_mag_bin
                and str(row.get("vol_bucket")) == vol_bucket
            ):
                return False, {"gate": "BLOCK", "reason": "DISABLED_RULE_MATCH", "session_bucket": session_bucket, **key}

        for row in self.enable:
            if (
                str(row.get("shock_sign")) == shock_sign
                and str(row.get("shock_mag_bin")) == shock_mag_bin
                and str(row.get("vol_bucket")) == vol_bucket
            ):
                return True, {"gate": "ALLOW", "reason": "ENABLED_RULE_MATCH", "session_bucket": session_bucket, **key}

        return False, {"gate": "BLOCK", "reason": "NO_MATCH", "session_bucket": session_bucket, **key}


# ----------------------------
# Logging contract
# ----------------------------
@dataclass(frozen=True)
class ShadowDecisionRow:
    bar_ts_utc: str
    row_idx: int
    bar_key: str

    intent_action: str           # NONE / ENTER / EXIT
    intent_id: str
    direction: str
    sl_price: float
    tp_price: float

    gate_result: str             # NONE / SKIP / ALLOW / BLOCK
    gate_reason: str

    anchor: float
    dist: float
    event_i_day: str
    event_ts_utc: str

    strategy_id: str
    strategy_version: str

    frozen_config_path: str
    execution_policy_path: str
    override_path: str

    risk_gate: str               # ALLOW / BLOCKED_MAX_TRADES / SKIPPED


class ShadowDecisionsLog:
    """
    CSV schema matches what you already printed (shadow_decisions_v2).
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "schema_version", "ts_utc", "bar_ts_utc", "row_idx", "bar_key",
                    "intent_action", "intent_id", "direction", "sl_price", "tp_price",
                    "gate_result", "gate_reason",
                    "anchor", "dist", "event_i_day", "event_ts_utc",
                    "strategy_id", "strategy_version",
                    "frozen_config_path", "execution_policy_path", "override_path",
                    "risk_gate",
                ])

    def append(self, d: ShadowDecisionRow) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "shadow_decisions_v2", utc_now_iso(), d.bar_ts_utc, d.row_idx, d.bar_key,
                d.intent_action, d.intent_id, d.direction,
                f"{d.sl_price:.8f}", f"{d.tp_price:.8f}",
                d.gate_result, d.gate_reason,
                f"{d.anchor:.8f}", f"{d.dist:.8f}",
                d.event_i_day, d.event_ts_utc,
                d.strategy_id, d.strategy_version,
                d.frozen_config_path, d.execution_policy_path, d.override_path,
                d.risk_gate,
            ])


# ----------------------------
# args
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Shadow runner v2 (strategy-exact, read-only)")

    ap.add_argument("--csv", required=True)
    ap.add_argument("--time_col", default="ts_utc")
    ap.add_argument("--poll_seconds", type=float, default=2.0)
    ap.add_argument("--start_from_last_row", action="store_true")

    ap.add_argument("--state_path", required=True)
    ap.add_argument("--results_dir", default=r".\results\shadow\runs")
    ap.add_argument("--run_tag", default="FTMO_SHADOW")

    ap.add_argument("--frozen_config", required=True)
    ap.add_argument("--execution_policy", required=True)
    ap.add_argument("--override_path", required=True)
    ap.add_argument("--account_mode", choices=["challenge", "funded"], required=True)
    return ap.parse_args()


# ----------------------------
# core
# ----------------------------
def _build_strategy_from_frozen(frozen: Dict[str, Any]) -> tuple[AnchorReversionAdapter, float, Dict[str, Any]]:
    strat_cfg = frozen.get("strategy", {}) or {}
    params = strat_cfg.get("params", {}) or {}

    anchor_col = str(params.get("anchor_col", "ny_open"))
    warmup_bars = int(params.get("warmup_bars", 0))
    entry_threshold_pips = float(params.get("entry_threshold_pips", 8.0))
    exit_threshold_pips = float(params.get("exit_threshold_pips", 0.3))
    sl_pips = float(params.get("sl_pips", 10.0))
    tp_pips = float(params.get("tp_pips", 15.0))
    max_hold_bars = int(params.get("max_hold_bars", params.get("max_hold", params.get("max_holding_bars", 96))))
    tag = str(params.get("tag", "ANCHOR_ADAPTER_FROZEN"))

    # FULL freeze-v2 alignment
    require_event = bool(params.get("require_event", True))
    event_col = str(params.get("event_col", "shock_z"))
    event_z_threshold = float(params.get("event_z_threshold", 2.0))
    event_window_bars = int(params.get("event_window_bars", 96))
    one_trade_per_event = bool(params.get("one_trade_per_event", True))
    guard_friday_entries = bool(params.get("guard_friday_entries", True))
    bar_minutes = int(params.get("bar_minutes", 5))
    friday_buffer_minutes = int(params.get("friday_buffer_minutes", 10))

    pip_size = float((frozen.get("instrument", {}) or {}).get("pip_size", 0.0001))

    strat = AnchorReversionAdapter(
        AnchorAdapterConfig(
            anchor_col=anchor_col,
            entry_threshold_pips=entry_threshold_pips,
            exit_threshold_pips=exit_threshold_pips,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            warmup_bars=warmup_bars,
            max_hold_bars=max_hold_bars,
            tag=tag,
            require_event=require_event,
            event_col=event_col,
            event_z_threshold=event_z_threshold,
            event_window_bars=event_window_bars,
            one_trade_per_event=one_trade_per_event,
            guard_friday_entries=guard_friday_entries,
            bar_minutes=bar_minutes,
            friday_buffer_minutes=friday_buffer_minutes,
        )
    )

    debug = {
        "anchor_col": anchor_col,
        "warmup_bars": warmup_bars,
        "entry_threshold_pips": entry_threshold_pips,
        "exit_threshold_pips": exit_threshold_pips,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "max_hold_bars": max_hold_bars,
        "require_event": require_event,
        "event_col": event_col,
        "event_z_threshold": event_z_threshold,
        "event_window_bars": event_window_bars,
        "one_trade_per_event": one_trade_per_event,
        "guard_friday_entries": guard_friday_entries,
        "bar_minutes": bar_minutes,
        "friday_buffer_minutes": friday_buffer_minutes,
        "tag": tag,
        "pip_size": pip_size,
    }
    return strat, pip_size, debug


def _tailbar_to_bar(event: TailBar) -> Bar:
    """
    CRITICAL: preserve ALL CSV columns in extras.
    AnchorReversionAdapter reads:
      - extras['ny_open'] (or anchor_col)
      - extras['shock_z'] for event conditioning
    """
    extras = dict(event.extras or {})  # strings, OK
    return Bar(
        ts_utc=event.ts_utc,
        open=float(event.open),
        high=float(event.high),
        low=float(event.low),
        close=float(event.close),
        volume=None,
        extras=extras,
    )


def _extract_anchor_dist(bar: Bar, anchor_col: str) -> tuple[float, float]:
    anchor = _to_float((bar.extras or {}).get(anchor_col), default=None)
    if anchor is None:
        return 0.0, 0.0
    dist = float(bar.close) - float(anchor)
    return float(anchor), float(dist)


def _extract_event_meta_from_intent(intent: Optional[OrderIntent]) -> tuple[str, str]:
    if intent is None:
        return "", ""
    m = intent.meta or {}
    return str(m.get("event_i_day") or ""), str(m.get("event_ts_utc") or "")


def main() -> None:
    args = parse_args()

    csv_path = str(Path(args.csv).expanduser())
    if not Path(csv_path).exists():
        raise SystemExit(f"CSV does not exist: {csv_path}")

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir = ensure_dir(Path(args.results_dir) / run_id)
    logs_path = run_dir / "logs.txt"
    decisions_path = run_dir / "shadow_decisions.csv"

    append_line(logs_path, f"[{utc_now_iso()}] shadow_run_id={run_id}")
    append_line(logs_path, f"[{utc_now_iso()}] csv={csv_path}")
    append_line(logs_path, f"[{utc_now_iso()}] time_col={args.time_col}")
    append_line(logs_path, f"[{utc_now_iso()}] poll_seconds={args.poll_seconds}")
    append_line(logs_path, f"[{utc_now_iso()}] start_from_last_row={bool(args.start_from_last_row)}")
    append_line(logs_path, f"[{utc_now_iso()}] state_path={args.state_path}")
    append_line(logs_path, f"[{utc_now_iso()}] frozen_config={args.frozen_config}")
    append_line(logs_path, f"[{utc_now_iso()}] execution_policy={args.execution_policy}")
    append_line(logs_path, f"[{utc_now_iso()}] override_path={args.override_path}")
    append_line(logs_path, f"[{utc_now_iso()}] account_mode={args.account_mode}")

    frozen_cfg = load_config(args.frozen_config)
    exec_cfg_raw = load_config(args.execution_policy)
    override_cfg = load_config(args.override_path)

    ep, meta, dbg = _resolve_execution_policy(exec_cfg_raw)
    append_line(logs_path, f"[{utc_now_iso()}] exec_policy_shape_selected={dbg.get('selected')}")

    # Trade controls (informational risk gate for shadow)
    trade_controls = ep.get("trade_controls", {}) or {}
    max_trades_per_day = int(trade_controls.get("max_trades_per_day", 1))

    # Optional override for shadow audit runs.
    # This is a shallow override by design for a small number of fields.
    if isinstance(override_cfg, dict) and isinstance(override_cfg.get("trade_controls"), dict):
        otc = override_cfg.get("trade_controls", {}) or {}
        if "max_trades_per_day" in otc:
            try:
                max_trades_per_day = int(otc["max_trades_per_day"])
            except Exception:
                pass

    append_line(logs_path, f"[{utc_now_iso()}] shadow_trade_controls max_trades_per_day={max_trades_per_day}")

    # Strategy
    strat, pip_size, strat_dbg = _build_strategy_from_frozen(frozen_cfg)
    append_line(logs_path, f"[{utc_now_iso()}] strategy_loaded {strat_dbg}")

    # Optional PolicyGate (from frozen)
    gate: Optional[PolicyGate] = None
    policy_gate_path = None
    if isinstance(frozen_cfg.get("policy_gate"), dict):
        policy_gate_path = frozen_cfg["policy_gate"].get("policy_path")
    if policy_gate_path:
        gate_cfg = load_config(policy_gate_path)
        gate = PolicyGate(gate_cfg)
        append_line(
            logs_path,
            f"[{utc_now_iso()}] policy_gate_enabled path={policy_gate_path} policy_id={gate.policy_id}",
        )
    else:
        append_line(logs_path, f"[{utc_now_iso()}] policy_gate_disabled")

    state = ShadowStateStore(args.state_path)
    last_idx = state.get_last_row_idx()
    append_line(logs_path, f"[{utc_now_iso()}] state_last_row_idx={last_idx}")

    provider = TailCSVBarProvider(
        TailCSVConfig(
            csv_path=csv_path,
            time_col=args.time_col,
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            tick_volume_col="tick_volume",
            poll_seconds=float(args.poll_seconds),
            start_from_last_row=bool(args.start_from_last_row),
        )
    )

    out = ShadowDecisionsLog(decisions_path)
    append_line(logs_path, f"[{utc_now_iso()}] STATUS=RUNNING decisions_loop")

    # Shadow-local counters for informational risk gate
    shadow_trades_taken_by_day: Dict[str, int] = {}

    # Pre-resolve anchor_col for diagnostics
    anchor_col = str(((frozen_cfg.get("strategy") or {}).get("params") or {}).get("anchor_col", "ny_open"))
    strategy_id = _strategy_id_from_cfg(frozen_cfg)
    strategy_version = _strategy_version_from_cfg(frozen_cfg) or "freeze_v1"

    for event in provider.iter_bars_live():
        row_idx = int(getattr(event, "row_idx", -1))
        if row_idx < 0:
            row_idx = last_idx + 1

        if row_idx <= last_idx:
            continue

        bar_key = getattr(event, "bar_key", "") or sha1_16(f"{event.ts_utc.isoformat()}|{row_idx}")
        if state.is_seen(bar_key):
            continue

        bar_ts = event.ts_utc.isoformat()
        bar = _tailbar_to_bar(event)

        # Build StrategyContext (shadow has no equity; keep stable dummies)
        day_id = _ny_day_id(bar.ts_utc)
        trades_today = int(shadow_trades_taken_by_day.get(day_id, 0))

        ctx = StrategyContext(
            day_id_ftmo=day_id,
            equity_current=0.0,
            dd_current_pct=0.0,
            trades_taken_today=trades_today,
            trading_enabled=True,
            account_mode=args.account_mode,
            instrument=str(meta.get("instrument", (frozen_cfg.get("symbol") or "EURUSD"))),
            pip_size=float(pip_size),
        )

        # Run exact strategy logic (read-only)
        intent: Optional[OrderIntent] = None
        try:
            intent = strat.on_bar(bar, ctx)
        except Exception as e:
            # On strategy exception, log a row and move on (shadow must not crash)
            append_line(logs_path, f"[{utc_now_iso()}] STRAT_EXC row_idx={row_idx} ts={bar_ts} {type(e).__name__}: {e}")
            intent = None

        intent_action = "NONE"
        intent_id = ""
        direction = ""
        sl_price = 0.0
        tp_price = 0.0

        if intent is not None:
            intent_action = str(intent.action)
            intent_id = str(intent.intent_id or "")
            direction = str(intent.direction or "")
            sl_price = float(intent.sl_price or 0.0)
            tp_price = float(intent.tp_price or 0.0)

        # PolicyGate evaluation only relevant for ENTER intents
        gate_result = "NONE"
        gate_reason = ""
        if gate is not None and intent is not None and intent.action == "ENTER":
            feats = {
                gate.shock_col: (bar.extras or {}).get(gate.shock_col),
                gate.shock_ret_col: (bar.extras or {}).get(gate.shock_ret_col),
                gate.vol_col: (bar.extras or {}).get(gate.vol_col),
            }
            allow, gmeta = gate.evaluate_bar(bar.ts_utc, feats)
            if allow is None:
                gate_result = "SKIP"
                gate_reason = str(gmeta.get("reason", "SKIP"))
            elif allow:
                gate_result = "ALLOW"
                gate_reason = str(gmeta.get("reason", "ALLOW"))
            else:
                gate_result = "BLOCK"
                gate_reason = str(gmeta.get("reason", "BLOCK"))
        elif gate is not None:
            gate_result = "NONE"

        # Informational risk gate: max_trades_per_day (shadow-local)
        risk_gate = "SKIPPED"
        if intent is not None and intent.action == "ENTER":
            if trades_today >= max_trades_per_day:
                risk_gate = "BLOCKED_MAX_TRADES"
            else:
                risk_gate = "ALLOW"

        # Diagnostics: anchor/dist + event meta
        anchor, dist = _extract_anchor_dist(bar, anchor_col)
        event_i_day, event_ts_utc = _extract_event_meta_from_intent(intent)

        out.append(
            ShadowDecisionRow(
                bar_ts_utc=bar_ts,
                row_idx=row_idx,
                bar_key=bar_key,
                intent_action=intent_action,
                intent_id=intent_id,
                direction=direction,
                sl_price=sl_price,
                tp_price=tp_price,
                gate_result=gate_result,
                gate_reason=gate_reason,
                anchor=anchor,
                dist=dist,
                event_i_day=event_i_day,
                event_ts_utc=event_ts_utc,
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                frozen_config_path=str(args.frozen_config),
                execution_policy_path=str(args.execution_policy),
                override_path=str(args.override_path),
                risk_gate=risk_gate,
            )
        )

        append_line(
            logs_path,
            f"[{utc_now_iso()}] row_idx={row_idx} ts={bar_ts} "
            f"intent={intent_action} dir={direction} gate={gate_result}:{gate_reason} risk={risk_gate} "
            f"anchor={anchor:.5f} dist={dist:.5f}"
        )

        # If this bar WOULD submit an ENTER and passes shadow risk gate, count it as "taken" for the day
        # (informational; does not affect the strategy object state)
        #
        # IMPORTANT FIX:
        #   DO NOT call strat.on_intent_submitted(...)
        #   because shadow has no engine/fill lifecycle and leaving the adapter
        #   in _pending=True suppresses future signals in backfill/live audit.
        if intent is not None and intent.action == "ENTER" and risk_gate == "ALLOW":
            shadow_trades_taken_by_day[day_id] = trades_today + 1

        state.mark_seen(bar_key)
        state.set_last_row_idx(row_idx)
        last_idx = row_idx


if __name__ == "__main__":
    main()