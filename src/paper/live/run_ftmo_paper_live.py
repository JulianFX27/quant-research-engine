from __future__ import annotations

# ------------------------------------------------------------
# BOOTSTRAP: ensure repo root is on sys.path so `import src...` works
# Works when running: python .\src\paper\live\run_ftmo_paper_live.py
# ------------------------------------------------------------
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ------------------------------------------------------------
# Standard imports
# ------------------------------------------------------------
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone, time as dtime
from typing import Any, Dict, Optional, Tuple, List

from zoneinfo import ZoneInfo

from src.execution.paper_engine import PaperEngine
from src.paper.risk.risk_state_manager import RiskStateManager, RiskConfig
from src.runner.interfaces import StrategyContext, OrderIntent
from src.runner.config_loader import load_config
from src.runner.run_id import file_sha256, make_run_id
from src.runner.artifacts import (
    ensure_dir,
    write_json,
    append_log,
    append_trade_csv,
    append_equity_csv,
    upsert_daily_summary,
)

from src.strategies.anchor_adapter import AnchorReversionAdapter, AnchorAdapterConfig

# LIVE tail provider (persistent)
from src.live.csv_tail_provider import TailCSVBarProvider, TailCSVConfig


# ------------------------------------------------------------
# Weekend flatten (NY close) — DST-safe
# - Force close open positions on/after Friday 16:55 NY.
# - Cancel pending intents at/after cutoff to prevent Monday fills.
# ------------------------------------------------------------
NY_TZ = ZoneInfo("America/New_York")
WEEKEND_CUTOFF_NY = dtime(16, 55, 0)  # 16:55 NY on Friday


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def throttle_multiplier_from_schedule(dd_pct: float, schedule: list[dict]) -> float:
    for row in schedule:
        if dd_pct >= float(row["dd_min_pct"]) and dd_pct < float(row["dd_max_pct"]):
            return float(row["risk_multiplier"])
    return 0.0


def _intent_with_meta(intent: OrderIntent, extra_meta: Dict[str, Any]) -> OrderIntent:
    merged = dict(intent.meta or {})
    merged.update(extra_meta)
    return replace(intent, meta=merged)


def _compute_equity_after(equity_before: float, trade_pnl: float, pnl_mode: str) -> float:
    if pnl_mode == "return":
        return float(equity_before) * (1.0 + float(trade_pnl))
    return float(equity_before) + float(trade_pnl)


def _shape_hint(obj: Any) -> str:
    if not isinstance(obj, dict):
        return f"type={type(obj)}"
    return f"dict_keys={sorted(list(obj.keys()))}"


def _resolve_execution_policy(exec_cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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


def _bar_feature(bar: Any, key: str) -> Any:
    if hasattr(bar, key):
        return getattr(bar, key)

    for container_name in ("extras", "extra", "features", "feature", "meta", "data", "row", "payload"):
        c = getattr(bar, container_name, None)
        if isinstance(c, dict) and key in c:
            return c.get(key)

    d = getattr(bar, "__dict__", None)
    if isinstance(d, dict) and key in d:
        return d.get(key)

    return None


def _is_weekend_cutoff(ts_utc: datetime) -> bool:
    ny = ts_utc.astimezone(NY_TZ)
    if ny.weekday() != 4:
        return False
    return ny.time() >= WEEKEND_CUTOFF_NY


class PolicyGate:
    def __init__(self, gate_cfg: Dict[str, Any]):
        self.cfg = gate_cfg
        self.policy_id = str(gate_cfg.get("policy_id", "UNKNOWN_GATE"))
        self.shock_col = str(gate_cfg["binning"]["shock_col"])
        self.shock_ret_col = str(gate_cfg["binning"]["shock_ret_col"])
        self.vol_col = str(gate_cfg["binning"]["vol_col"])
        self.session_bucket_min = int(gate_cfg["binning"].get("session_bucket_min", 30))

        self.thresholds = gate_cfg.get("thresholds", {})
        self.rules = gate_cfg.get("rules", {})

        self.enable = list(self.rules.get("enable_shock_vol", []) or [])
        self.disable = list(self.rules.get("disable_shock_vol", []) or [])
        self.disable_session_buckets = set(self.rules.get("disable_session_buckets", []) or [])

    @staticmethod
    def _is_missing(x: Any) -> bool:
        # Treat None / "" / "nan" / NaN as missing.
        if x is None:
            return True
        if isinstance(x, str):
            s = x.strip().lower()
            if s == "" or s == "nan" or s == "none":
                return True
        try:
            v = float(x)
            return v != v  # NaN
        except Exception:
            # If it can't be converted to float, consider it missing for gating
            return True

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


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _parse_iso_dt(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None
    try:
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        return datetime.fromisoformat(txt)
    except Exception:
        return None


def summarize_trades_csv(trades_path: str) -> Dict[str, Any]:
    p = Path(trades_path)
    if (not p.exists()) or p.stat().st_size == 0:
        return {"n_trades": 0}

    rows: List[Dict[str, Any]] = []
    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    n = len(rows)
    if n == 0:
        return {"n_trades": 0}

    pnls = [_to_float(r.get("pnl"), 0.0) for r in rows]
    Rs = [_to_float(r.get("R"), 0.0) for r in rows]

    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]

    winrate = float(sum(1 for x in pnls if x > 0)) / float(n)

    gross_profit = float(sum(wins)) if wins else 0.0
    gross_loss = float(abs(sum(losses))) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    avg_R = float(sum(Rs)) / float(n)
    expectancy_R = avg_R

    win_Rs = [r for r in Rs if r > 0]
    loss_Rs = [r for r in Rs if r < 0]
    avg_win_R = float(sum(win_Rs)) / float(len(win_Rs)) if win_Rs else 0.0
    avg_loss_R = float(sum(loss_Rs)) / float(len(loss_Rs)) if loss_Rs else 0.0

    holds: List[float] = []
    for r in rows:
        et = _parse_iso_dt(r.get("entry_time_utc"))
        xt = _parse_iso_dt(r.get("exit_time_utc"))
        if et is None or xt is None:
            continue
        holds.append((xt - et).total_seconds() / 60.0)

    holds_sorted = sorted(holds)

    def _pct(arr: List[float], q: float) -> float:
        if not arr:
            return 0.0
        if q <= 0:
            return float(arr[0])
        if q >= 1:
            return float(arr[-1])
        idx = int(round((len(arr) - 1) * q))
        idx = max(0, min(idx, len(arr) - 1))
        return float(arr[idx])

    avg_hold = float(sum(holds)) / float(len(holds)) if holds else 0.0
    med_hold = _pct(holds_sorted, 0.50)
    p95_hold = _pct(holds_sorted, 0.95)

    by_exit: Dict[str, int] = {}
    by_dir: Dict[str, int] = {}
    by_day: Dict[str, int] = {}

    for r in rows:
        ex = str(r.get("exit_reason") or "UNKNOWN")
        by_exit[ex] = by_exit.get(ex, 0) + 1

        d = str(r.get("direction") or "UNKNOWN")
        by_dir[d] = by_dir.get(d, 0) + 1

        day = str(r.get("day_id_ftmo") or "UNKNOWN")
        by_day[day] = by_day.get(day, 0) + 1

    per_day_counts = list(by_day.values())
    avg_trades_per_day = float(sum(per_day_counts)) / float(len(per_day_counts)) if per_day_counts else 0.0
    max_trades_day = int(max(per_day_counts)) if per_day_counts else 0
    min_trades_day = int(min(per_day_counts)) if per_day_counts else 0

    return {
        "n_trades": n,
        "winrate": winrate,
        "profit_factor_pnl": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss_abs": gross_loss,
        "expectancy_R": expectancy_R,
        "avg_R": avg_R,
        "avg_win_R": avg_win_R,
        "avg_loss_R": avg_loss_R,
        "hold_time": {
            "n_with_times": int(len(holds)),
            "avg_min": avg_hold,
            "median_min": med_hold,
            "p95_min": p95_hold,
        },
        "counts": {
            "by_exit_reason": dict(sorted(by_exit.items(), key=lambda kv: (-kv[1], kv[0]))),
            "by_direction": dict(sorted(by_dir.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "per_day": {
            "n_days": int(len(by_day)),
            "avg_trades_per_day": avg_trades_per_day,
            "max_trades_in_a_day": max_trades_day,
            "min_trades_in_a_day": min_trades_day,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="FTMO-aware Paper Live Runner (CSV tail, persistent state)")

    ap.add_argument("--frozen_config", required=True)
    ap.add_argument("--execution_policy", required=True)
    ap.add_argument("--state_path", required=True)
    ap.add_argument("--override_path", required=True)
    ap.add_argument("--account_mode", choices=["challenge", "funded"], required=True)

    ap.add_argument("--results_dir", default="results/runs")
    ap.add_argument("--run_tag", default=None)

    ap.add_argument("--csv", default=None)
    ap.add_argument("--time_col", default="time")

    # live tail
    ap.add_argument("--poll_seconds", type=float, default=1.0)
    ap.add_argument("--start_from_last_row", action="store_true")

    args = ap.parse_args()

    frozen = load_config(args.frozen_config)
    exec_cfg = load_config(args.execution_policy)
    ep, meta, dbg = _resolve_execution_policy(exec_cfg)

    frozen_hash = file_sha256(args.frozen_config)
    policy_hash = file_sha256(args.execution_policy)

    run_id = make_run_id(args.run_tag, f"{frozen_hash}_{policy_hash}")
    run_dir = Path(args.results_dir) / run_id
    ensure_dir(str(run_dir))

    paths = {
        "manifest": str(run_dir / "run_manifest.json"),
        "trades": str(run_dir / "trades.csv"),
        "equity": str(run_dir / "equity.csv"),
        "daily": str(run_dir / "daily_summary.csv"),
        "logs": str(run_dir / "logs.txt"),
        "metrics": str(run_dir / "metrics.json"),
    }

    append_log(paths["logs"], f"[{utc_now_iso()}] run_id={run_id}")
    append_log(paths["logs"], f"[{utc_now_iso()}] frozen_config={args.frozen_config}")
    append_log(paths["logs"], f"[{utc_now_iso()}] frozen_sha256={frozen_hash}")
    append_log(paths["logs"], f"[{utc_now_iso()}] execution_policy={args.execution_policy}")
    append_log(paths["logs"], f"[{utc_now_iso()}] execution_policy_sha256={policy_hash}")
    append_log(paths["logs"], f"[{utc_now_iso()}] exec_cfg_shape={dbg.get('exec_cfg_shape')}")
    append_log(paths["logs"], f"[{utc_now_iso()}] policy_shape_selected={dbg.get('selected')}")

    for k in ("internal_limits", "order_execution", "risk_mode", "throttle"):
        if k not in ep:
            raise KeyError(f"{k} not found in execution policy (check logs.txt for shape)")

    tz_name = str(meta.get("timezone_day_rollover", ep.get("timezone_day_rollover", "Europe/Prague")))
    instrument = str(meta.get("instrument", ep.get("instrument", frozen.get("symbol", "EURUSD"))))
    policy_name = str(meta.get("name", ep.get("name", ep.get("execution_policy", {}).get("name", "UNKNOWN_POLICY"))))

    max_daily_loss_pct = float(ep["internal_limits"]["daily_stop_pct"]) * 100.0
    max_overall_loss_pct = float(ep["internal_limits"]["hard_stop_total_dd_pct"]) * 100.0

    oe = ep["order_execution"]
    fill_mode = oe.get("fill_mode", "next_open")
    intrabar_path_mode = oe["intrabar"]["path_mode"]
    intrabar_tie_break = oe["intrabar"]["tie_break"]

    trade_controls = ep.get("trade_controls", {})
    max_trades_per_day = int(trade_controls.get("max_trades_per_day", 1))

    pnl_mode = str(ep.get("paper", {}).get("pnl_mode", "absolute")).strip().lower()
    if pnl_mode not in ("absolute", "return"):
        pnl_mode = "absolute"

    append_log(paths["logs"], f"[{utc_now_iso()}] policy_name={policy_name} account_mode={args.account_mode}")
    append_log(paths["logs"], f"[{utc_now_iso()}] tz_day_rollover={tz_name}")
    append_log(
        paths["logs"],
        f"[{utc_now_iso()}] internal_limits daily_stop={max_daily_loss_pct:.3f}% hard_stop_dd={max_overall_loss_pct:.3f}%",
    )
    append_log(
        paths["logs"],
        f"[{utc_now_iso()}] order_execution fill_mode={fill_mode} intrabar_path={intrabar_path_mode} tie={intrabar_tie_break}",
    )
    append_log(paths["logs"], f"[{utc_now_iso()}] trade_controls max_trades_per_day={max_trades_per_day} pnl_mode={pnl_mode}")
    append_log(
        paths["logs"],
        f"[{utc_now_iso()}] weekend_flatten tz=America/New_York cutoff={WEEKEND_CUTOFF_NY.strftime('%H:%M:%S')} (Friday)",
    )

    csv_path = args.csv or str(frozen.get("data_path"))
    if not csv_path:
        raise ValueError("CSV path is missing: provide --csv or ensure frozen config has data_path")

    prov = TailCSVBarProvider(
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
    append_log(
        paths["logs"],
        f"[{utc_now_iso()}] paper_live tail_config poll_seconds={args.poll_seconds} start_from_last_row={bool(args.start_from_last_row)}",
    )

    engine = PaperEngine(
        intrabar_path=intrabar_path_mode,
        tie_break=intrabar_tie_break,
        allow_same_bar_exit=True,
    )

    strat_cfg = frozen.get("strategy", {}) or {}
    params = strat_cfg.get("params", {}) or {}

    anchor_col = str(params.get("anchor_col", "ny_open"))
    warmup_bars = int(params.get("warmup_bars", 0))
    entry_threshold_pips = float(params.get("entry_threshold_pips", 8.0))
    exit_threshold_pips = float(params.get("exit_threshold_pips", 0.3))
    sl_pips = float(params.get("sl_pips", 10.0))
    tp_pips = float(params.get("tp_pips", 15.0))
    max_hold_bars = int(params.get("max_hold_bars", params.get("max_hold", params.get("max_holding_bars", 96))))

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
            tag=str(params.get("tag", "ANCHOR_ADAPTER_FROZEN")),
        )
    )

    gate: Optional[PolicyGate] = None
    policy_gate_path = None
    if isinstance(frozen.get("policy_gate"), dict):
        policy_gate_path = frozen["policy_gate"].get("policy_path")
    if policy_gate_path:
        gate_cfg = load_config(policy_gate_path)
        gate = PolicyGate(gate_cfg)
        append_log(paths["logs"], f"[{utc_now_iso()}] policy_gate_enabled path={policy_gate_path} policy_id={gate.policy_id}")

    rsm = RiskStateManager(
        state_path=args.state_path,
        override_path=args.override_path,
        config=RiskConfig(
            tz_name=tz_name,
            max_daily_loss_pct=float(abs(max_daily_loss_pct)),
            max_overall_loss_pct=float(abs(max_overall_loss_pct)),
            max_audit=200,
        ),
    )

    state = rsm.load_state()
    state["account_mode"] = args.account_mode
    state["max_trades_per_day"] = max_trades_per_day
    rsm.save_state(state)

    r_base = float(ep["risk_mode"][args.account_mode]["base_risk_pct_per_trade"]) / 100.0
    throttle_schedule = ep["throttle"]["schedule"]

    max_intraday_dd_by_day: Dict[str, float] = {}

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": utc_now_iso(),
        "ended_at_utc": None,
        "mode": "paper_live",
        "account_mode": args.account_mode,
        "instrument": instrument,
        "policy_name": policy_name,
        "frozen_config_path": args.frozen_config,
        "frozen_config_sha256": frozen_hash,
        "execution_policy_path": args.execution_policy,
        "execution_policy_sha256": policy_hash,
        "state_path": args.state_path,
        "override_path": args.override_path,
        "csv": csv_path,
        "engine": {
            "fill_mode": fill_mode,
            "intrabar_path": intrabar_path_mode,
            "intrabar_tie_break": intrabar_tie_break,
        },
        "paper": {
            "pnl_mode": pnl_mode,
            "internal_limits_pct": {
                "max_daily_loss_pct": float(abs(max_daily_loss_pct)),
                "max_overall_loss_pct": float(abs(max_overall_loss_pct)),
            },
            "trade_controls": {"max_trades_per_day": max_trades_per_day},
            "live_tail": {
                "enabled": True,
                "poll_seconds": float(args.poll_seconds),
                "start_from_last_row": bool(args.start_from_last_row),
            },
        },
        "strategy": {
            "anchor_col": anchor_col,
            "entry_threshold_pips": entry_threshold_pips,
            "exit_threshold_pips": exit_threshold_pips,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "warmup_bars": warmup_bars,
            "max_hold_bars": max_hold_bars,
            "pip_size": pip_size,
        },
        "gate": {
            "enabled": bool(gate is not None),
            "policy_gate_path": policy_gate_path,
            "policy_id": getattr(gate, "policy_id", None),
            "gate_allow": 0,
            "gate_block": 0,
            "gate_skip": 0,
        },
        "events": {
            "bars": 0,
            "intents": 0,
            "entries_submitted": 0,
            "fills": 0,
            "forced_exits": 0,
            "weekend_forced_exits": 0,
            "closes": 0,
            "day_rollovers": 0,
            "intent_cancels": 0,
        },
        "summary": None,
    }
    write_json(paths["manifest"], manifest)

    def close_and_persist(trade, bar_ts_utc: datetime, reason_hint: str) -> None:
        nonlocal state

        if hasattr(strat, "on_trade_closed_reset"):
            strat.on_trade_closed_reset()

        manifest["events"]["closes"] += 1

        state = rsm.load_state()
        equity_before = float(state["equity_current"])
        equity_after = _compute_equity_after(equity_before, float(trade.pnl), pnl_mode=pnl_mode)

        state = rsm.on_trade_closed(state, equity_after=equity_after, run_id=run_id, now_utc=bar_ts_utc)
        rsm.save_state(state)

        entry_day_id = None
        try:
            entry_day_id = (getattr(trade, "meta", None) or {}).get("_day_id_ftmo_entry")
        except Exception:
            entry_day_id = None

        day_id2 = entry_day_id or state.get("current_day_id") or rsm.current_day_id(bar_ts_utc)

        append_trade_csv(
            paths["trades"],
            trade,
            day_id_ftmo=day_id2,
            dd_at_entry_pct=float((getattr(trade, "meta", None) or {}).get("_dd_at_entry_pct", 0.0)),
            dd_at_exit_pct=float(state.get("dd_from_peak_pct", 0.0)),
            daily_pnl_pct_at_entry=float((getattr(trade, "meta", None) or {}).get("_daily_pnl_pct_at_entry", 0.0)),
        )

        trades_taken_for_day = 1 if entry_day_id else int(state.get("trades_taken_today", 0))

        upsert_daily_summary(
            paths["daily"],
            day_id_ftmo=day_id2,
            equity_start_day=float(state.get("equity_start_day") or state["equity_current"]),
            equity_end_day=float(state["equity_current"]),
            daily_pnl_pct=float(state.get("daily_pnl_pct", 0.0)),
            max_intraday_dd_pct=max_intraday_dd_by_day.get(day_id2, 0.0),
            trades_taken=int(trades_taken_for_day),
            daily_stop_triggered=bool(state.get("daily_stop_triggered", False)),
            hard_stop_triggered=bool(state.get("hard_stop_dd_triggered", False)),
        )

        append_log(
            paths["logs"],
            f"[{utc_now_iso()}] CLOSE trade_id={trade.trade_id} reason={trade.exit_reason} pnl={trade.pnl:.6f} R={trade.R:.3f} {reason_hint}",
        )

    try:
        for bar in prov.iter_bars_live():
            manifest["events"]["bars"] += 1

            state = rsm.load_state()
            before_day = state.get("current_day_id")

            state = rsm.rollover_if_needed(state, run_id=run_id, now_utc=bar.ts_utc)
            after_day = state.get("current_day_id")
            if before_day != after_day:
                manifest["events"]["day_rollovers"] += 1
                append_log(paths["logs"], f"[{utc_now_iso()}] DAY_ROLLOVER day_id={after_day}")

            state = rsm.try_manual_reset_hard_stop(state, run_id=run_id, now_utc=bar.ts_utc)
            rsm.save_state(state)

            day_id = state["current_day_id"] or rsm.current_day_id(bar.ts_utc)

            max_intraday_dd_by_day[day_id] = max(
                max_intraday_dd_by_day.get(day_id, 0.0),
                float(state.get("dd_from_peak_pct", 0.0)),
            )

            append_equity_csv(
                paths["equity"],
                ts_utc=bar.ts_utc,
                equity=float(state["equity_current"]),
                equity_peak=float(state["equity_peak"]),
                dd_pct=float(state.get("dd_from_peak_pct", 0.0)),
                day_id_ftmo=day_id,
            )

            # WEEKEND FLATTEN / CUTOFF
            if _is_weekend_cutoff(bar.ts_utc):
                if engine.has_open_position():
                    append_log(paths["logs"], f"[{utc_now_iso()}] WEEKEND_CUTOFF open_position=True -> FORCE_WEEKEND close")
                    closed = engine.force_close(ts_utc=bar.ts_utc, price=float(bar.close), reason="FORCE_WEEKEND")
                    if closed is not None:
                        manifest["events"]["forced_exits"] += 1
                        manifest["events"]["weekend_forced_exits"] += 1
                        close_and_persist(closed, bar.ts_utc, reason_hint="(forced_weekend)")
                    continue

                if engine.has_pending_intent() and (not engine.has_open_position()):
                    engine.cancel_pending_intent(reason="WEEKEND_CUTOFF_CANCEL")
                    manifest["events"]["intent_cancels"] += 1
                    append_log(paths["logs"], f"[{utc_now_iso()}] WEEKEND_CUTOFF pending_intent=True -> CANCEL")
                    if hasattr(strat, "on_intent_cancelled"):
                        strat.on_intent_cancelled()
                    continue

                continue

            ctx = StrategyContext(
                day_id_ftmo=day_id,
                equity_current=float(state["equity_current"]),
                dd_current_pct=float(state.get("dd_from_peak_pct", 0.0)),
                trades_taken_today=int(state.get("trades_taken_today", 0)),
                trading_enabled=bool(state.get("trading_enabled", True)),
                account_mode=args.account_mode,
                instrument=instrument,
                pip_size=float(pip_size),
            )

            # Pending intent path
            if engine.has_pending_intent() and (not engine.has_open_position()):
                closed = engine.on_bar(bar)

                if closed is not None:
                    manifest["events"]["fills"] += 1
                    if hasattr(strat, "on_trade_opened_confirmed"):
                        strat.on_trade_opened_confirmed(bar.ts_utc)

                    state = rsm.load_state()
                    state = rsm.on_trade_opened(state, run_id=run_id, now_utc=bar.ts_utc)
                    rsm.save_state(state)

                    close_and_persist(closed, bar.ts_utc, reason_hint="(filled+closed)")
                    continue

                if engine.has_open_position():
                    manifest["events"]["fills"] += 1
                    if hasattr(strat, "on_trade_opened_confirmed"):
                        strat.on_trade_opened_confirmed(bar.ts_utc)

                    state = rsm.load_state()
                    state = rsm.on_trade_opened(state, run_id=run_id, now_utc=bar.ts_utc)
                    rsm.save_state(state)

                continue

            # Open position path
            if engine.has_open_position():
                intent = strat.on_bar(bar, ctx)
                if intent is not None:
                    manifest["events"]["intents"] += 1
                    if intent.action == "EXIT":
                        closed = engine.force_close(
                            ts_utc=bar.ts_utc,
                            price=float(bar.close),
                            reason=intent.exit_reason or "FORCE_EXIT",
                        )
                        if closed is not None:
                            manifest["events"]["forced_exits"] += 1
                            close_and_persist(closed, bar.ts_utc, reason_hint="(forced)")
                        continue

                closed = engine.on_bar(bar)
                if closed is not None:
                    close_and_persist(closed, bar.ts_utc, reason_hint="(tp/sl)")
                continue

            # Entry gating by risk state manager
            if not rsm.can_trade(state, run_id=run_id, now_utc=bar.ts_utc):
                continue

            intent = strat.on_bar(bar, ctx)
            if intent is None:
                continue

            manifest["events"]["intents"] += 1
            if intent.action == "EXIT":
                continue
            if intent.direction is None or intent.sl_price is None or intent.tp_price is None:
                continue

            # POLICY GATE
            if gate is not None:
                feats = {
                    gate.shock_col: _bar_feature(bar, gate.shock_col),
                    gate.shock_ret_col: _bar_feature(bar, gate.shock_ret_col),
                    gate.vol_col: _bar_feature(bar, gate.vol_col),
                }
                allow, gmeta = gate.evaluate_bar(bar.ts_utc, feats)

                if allow is None:
                    manifest["gate"]["gate_skip"] += 1
                    append_log(paths["logs"], f"[{utc_now_iso()}] POLICY_GATE SKIP {gmeta}")
                    continue

                if allow:
                    manifest["gate"]["gate_allow"] += 1
                    append_log(paths["logs"], f"[{utc_now_iso()}] POLICY_GATE ALLOW {gmeta}")
                else:
                    manifest["gate"]["gate_block"] += 1
                    append_log(paths["logs"], f"[{utc_now_iso()}] POLICY_GATE BLOCK {gmeta}")
                    continue

            dd_pct = float(state.get("dd_from_peak_pct", 0.0))
            mult = throttle_multiplier_from_schedule(dd_pct, throttle_schedule)
            r_eff = r_base * mult
            if r_eff <= 0:
                continue

            intent = _intent_with_meta(
                intent,
                {
                    "_day_id_ftmo_entry": day_id,
                    "_dd_at_entry_pct": dd_pct,
                    "_daily_pnl_pct_at_entry": float(state.get("daily_pnl_pct", 0.0)),
                    "_pip_size": float(pip_size),
                    "_sl_pips": float(sl_pips),
                    "_tp_pips": float(tp_pips),
                },
            )

            append_log(
                paths["logs"],
                f"[{utc_now_iso()}] SUBMIT "
                f"day_id_ftmo_entry={day_id} dd_from_peak_pct={dd_pct:.6f} "
                f"risk_mult={mult:.3f} r_base={r_base:.6f} r_eff={r_eff:.6f} "
                f"dir={intent.direction} sl={intent.sl_price} tp={intent.tp_price}"
            )

            if hasattr(strat, "on_intent_submitted"):
                strat.on_intent_submitted(intent.direction)

            engine.submit_intent(intent, risk_base_pct=r_base, risk_multiplier=mult, risk_effective_pct=r_eff)
            manifest["events"]["entries_submitted"] += 1

    except KeyboardInterrupt:
        append_log(paths["logs"], f"[{utc_now_iso()}] STOP KeyboardInterrupt received; shutting down runner.")
    finally:
        manifest["ended_at_utc"] = utc_now_iso()

        # Snapshot final state (no “final metrics” in live, but this is useful)
        try:
            st = rsm.load_state()
        except Exception:
            st = {}

        manifest["summary"] = {
            "n_trades": None,
            "final_state": {
                "equity_current": float(st.get("equity_current", 0.0)) if isinstance(st, dict) else 0.0,
                "equity_peak": float(st.get("equity_peak", 0.0)) if isinstance(st, dict) else 0.0,
                "dd_from_peak_pct": float(st.get("dd_from_peak_pct", 0.0)) if isinstance(st, dict) else 0.0,
                "daily_pnl_pct": float(st.get("daily_pnl_pct", 0.0)) if isinstance(st, dict) else 0.0,
                "daily_stop_triggered": bool(st.get("daily_stop_triggered", False)) if isinstance(st, dict) else False,
                "hard_stop_dd_triggered": bool(st.get("hard_stop_dd_triggered", False)) if isinstance(st, dict) else False,
            },
        }
        write_json(paths["manifest"], manifest)

        # OPTIONAL: you can write a rolling metrics snapshot if you want
        try:
            summary = summarize_trades_csv(paths["trades"])
            summary["final_state"] = manifest["summary"]["final_state"]
            write_json(paths["metrics"], summary)
        except Exception as e:
            append_log(paths["logs"], f"[{utc_now_iso()}] METRICS_WRITE_ERROR: {e}")

    print(f"Runner stopped: {run_dir}")


if __name__ == "__main__":
    main()