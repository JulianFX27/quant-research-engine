from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_iso(s: str) -> datetime:
    txt = (s or "").strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class PaperTrade:
    trade_id: str
    entry_time_utc: datetime
    direction: str


@dataclass
class ShadowSignal:
    bar_ts_utc: datetime
    row_idx: int
    intent_action: str
    direction: str
    gate_result: str
    gate_reason: str


def load_paper_trades(path: Path) -> List[PaperTrade]:
    out: List[PaperTrade] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            et = row.get("entry_time_utc") or ""
            d = (row.get("direction") or "").strip()
            tid = (row.get("trade_id") or row.get("id") or "").strip()
            if not et or not d:
                continue
            out.append(PaperTrade(trade_id=tid, entry_time_utc=_parse_iso(et), direction=d))
    return out


def load_shadow_signals(path: Path) -> List[ShadowSignal]:
    out: List[ShadowSignal] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            act = (row.get("intent_action") or "").strip()
            if act != "ENTER":
                continue
            ts = row.get("bar_ts_utc") or ""
            if not ts:
                continue
            out.append(
                ShadowSignal(
                    bar_ts_utc=_parse_iso(ts),
                    row_idx=int(float(row.get("row_idx") or 0)),
                    intent_action=act,
                    direction=(row.get("direction") or "").strip(),
                    gate_result=(row.get("gate_result") or "").strip(),
                    gate_reason=(row.get("gate_reason") or "").strip(),
                )
            )
    out.sort(key=lambda x: (x.bar_ts_utc, x.row_idx))
    return out


def find_match(
    trade: PaperTrade,
    signals: List[ShadowSignal],
    bar_minutes: int = 5,
    tol_bars: int = 1,
) -> Optional[ShadowSignal]:
    """
    Paper entry happens at next bar open. Shadow ENTER intent happens at signal bar close.
    So ideal match is around entry_time_utc - bar_minutes.

    We accept +/- tol_bars around that expected signal time.
    """
    expected = trade.entry_time_utc - timedelta(minutes=bar_minutes)

    lo = expected - timedelta(minutes=bar_minutes * tol_bars)
    hi = expected + timedelta(minutes=bar_minutes * tol_bars)

    cands = [s for s in signals if lo <= s.bar_ts_utc <= hi]
    if not cands:
        return None

    # choose closest by absolute time difference
    cands.sort(key=lambda s: abs((s.bar_ts_utc - expected).total_seconds()))
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_trades", required=True)
    ap.add_argument("--shadow_decisions", required=True)
    ap.add_argument("--out", default="decision_diff_report.csv")
    ap.add_argument("--bar_minutes", type=int, default=5)
    ap.add_argument("--tol_bars", type=int, default=1)
    args = ap.parse_args()

    paper_path = Path(args.paper_trades)
    shadow_path = Path(args.shadow_decisions)

    trades = load_paper_trades(paper_path)
    sigs = load_shadow_signals(shadow_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "paper_trade_id", "paper_entry_time_utc", "paper_direction",
            "shadow_signal_ts_utc", "shadow_direction", "shadow_gate_result", "shadow_gate_reason",
            "match", "mismatch_reason"
        ])

        for t in trades:
            s = find_match(t, sigs, bar_minutes=args.bar_minutes, tol_bars=args.tol_bars)
            if s is None:
                w.writerow([t.trade_id, t.entry_time_utc.isoformat(), t.direction, "", "", "", "", "FALSE", "NO_SHADOW_SIGNAL_IN_WINDOW"])
                continue

            match = (t.direction == s.direction) and (s.gate_result == "ALLOW")
            if not match:
                reason = []
                if t.direction != s.direction:
                    reason.append("DIRECTION_MISMATCH")
                if s.gate_result != "ALLOW":
                    reason.append(f"GATE_{s.gate_result}")
                mismatch_reason = "|".join(reason) if reason else "UNKNOWN"
                w.writerow([t.trade_id, t.entry_time_utc.isoformat(), t.direction,
                            s.bar_ts_utc.isoformat(), s.direction, s.gate_result, s.gate_reason,
                            "FALSE", mismatch_reason])
            else:
                w.writerow([t.trade_id, t.entry_time_utc.isoformat(), t.direction,
                            s.bar_ts_utc.isoformat(), s.direction, s.gate_result, s.gate_reason,
                            "TRUE", ""])

    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()