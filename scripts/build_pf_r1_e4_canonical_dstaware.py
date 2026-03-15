from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


# Permite importar desde src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.opening_momentum.session_times import ny_open_utc  # noqa: E402


DEFAULT_PIP_SIZE = 0.0001


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pip_to_price(pips: float, pip_size: float) -> float:
    return pips * pip_size


def resolve_trade(
    direction: str,
    tp_price: float,
    sl_price: float,
    bars: pd.DataFrame,
    time_stop_bars: int,
):
    hold = 0

    for _, row in bars.iterrows():
        hold += 1
        high = float(row["high"])
        low = float(row["low"])

        # Convención conservadora: si ambos se tocan en la misma barra, prioriza SL
        if direction == "short":
            if high >= sl_price:
                return sl_price, "SL", hold
            if low <= tp_price:
                return tp_price, "TP", hold
        else:
            if low <= sl_price:
                return sl_price, "SL", hold
            if high >= tp_price:
                return tp_price, "TP", hold

        if hold >= time_stop_bars:
            return float(row["close"]), "TIME", hold

    return float(bars.iloc[-1]["close"]), "EOF", hold


def build_ny_session_rows(df: pd.DataFrame, session_minutes: int) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["time"].dt.date
    px = df.set_index("time").sort_index()

    session_rows = []

    for d in sorted(df["date"].unique()):
        t0 = ny_open_utc(d)
        t1 = t0 + pd.Timedelta(minutes=session_minutes)

        window = px.loc[t0 : t1 - pd.Timedelta(minutes=5)].copy()
        if window.empty:
            continue

        window = window.reset_index()
        window["session_date"] = d
        window["session_open_time"] = t0
        session_rows.append(window)

    if not session_rows:
        raise ValueError("No session rows built. Check dataset/timeframe/session logic.")

    out = pd.concat(session_rows, ignore_index=True)
    out = out.sort_values("time").reset_index(drop=True)
    return out


def add_event_features(df: pd.DataFrame, window_min: int, lookback_bars: int, pip_size: float) -> pd.DataFrame:
    if window_min % 5 != 0:
        raise ValueError("window_min must be compatible with M5 data.")

    bars = window_min // 5
    out = df.copy()

    out["event_return_log"] = np.log(out["close"] / out["close"].shift(bars))
    out["rolling_std"] = (
        out["event_return_log"]
        .rolling(window=lookback_bars, min_periods=lookback_bars)
        .std(ddof=0)
    )
    out["event_zscore"] = out["event_return_log"] / out["rolling_std"]
    out["event_return_pips"] = (out["close"] - out["close"].shift(bars)) / pip_size

    return out


def deduplicate_events(events: pd.DataFrame, cooldown_bars: int = 2) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    events = events.sort_values("source_index").reset_index(drop=True)
    kept = []

    block = []
    prev_idx = None

    for _, row in events.iterrows():
        idx = int(row["source_index"])

        if prev_idx is None:
            block = [row.to_dict()]
        else:
            if idx - prev_idx <= cooldown_bars:
                block.append(row.to_dict())
            else:
                best = max(block, key=lambda x: abs(float(x["event_zscore"])))
                kept.append(best)
                block = [row.to_dict()]

        prev_idx = idx

    if block:
        best = max(block, key=lambda x: abs(float(x["event_zscore"])))
        kept.append(best)

    out = pd.DataFrame(kept).reset_index(drop=True)
    out["event_id"] = [f"evt_{i:08d}" for i in range(len(out))]
    return out


def compute_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
    wins = trades[trades["pnl_pips"] > 0]
    losses = trades[trades["pnl_pips"] <= 0]

    winrate = len(wins) / len(trades) if len(trades) else 0.0
    expectancy = float(trades["pnl_pips"].mean()) if len(trades) else None
    avg_win = float(wins["pnl_pips"].mean()) if len(wins) else None
    avg_loss = float(losses["pnl_pips"].mean()) if len(losses) else None
    pf = (
        float(wins["pnl_pips"].sum() / abs(losses["pnl_pips"].sum()))
        if len(losses) and abs(losses["pnl_pips"].sum()) > 0
        else None
    )

    return {
        "trades": int(len(trades)),
        "winrate": winrate,
        "expectancy_pips": expectancy,
        "avg_win_pips": avg_win,
        "avg_loss_pips": avg_loss,
        "profit_factor": pf,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_json(cfg_path)

    dataset_path = Path(cfg["dataset"]["path"])
    out_dir = Path(cfg["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pip_size = float(cfg["dataset"].get("pip_size", DEFAULT_PIP_SIZE))

    print("Loading canonical dataset...")
    df = pd.read_csv(dataset_path)
    ts_col = cfg["dataset"]["timestamp_column"]
    df["time"] = pd.to_datetime(df[ts_col], errors="raise")
    df = df.sort_values("time").reset_index(drop=True)

    # Filtrado temporal
    t_start = pd.to_datetime(cfg["time_frame"]["start"])
    t_end = pd.to_datetime(cfg["time_frame"]["end"])
    df = df[(df["time"] >= t_start) & (df["time"] <= t_end)].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset empty after time_frame filter.")

    print("Building DST-aware NY session rows...")
    session_minutes = int(cfg["session"]["session_minutes"])
    session_df = build_ny_session_rows(df, session_minutes=session_minutes)

    print("Adding event features...")
    window_min = int(cfg["event_definition"]["window_min"])
    lookback_bars = int(cfg["event_definition"]["rolling_vol_lookback_bars"])
    z_threshold = float(cfg["event_definition"]["z_threshold"])
    direction = cfg["event_definition"]["direction"]

    feat_df = add_event_features(
        session_df,
        window_min=window_min,
        lookback_bars=lookback_bars,
        pip_size=pip_size,
    )
    feat_df = feat_df.reset_index(drop=True)
    feat_df["source_index"] = feat_df.index

    print("Selecting candidate events...")
    if direction != "UP_EXTREME":
        raise ValueError("This E4 script is currently fixed to UP_EXTREME validation.")

    events = feat_df[feat_df["event_zscore"] >= z_threshold].copy()

    if events.empty:
        raise ValueError("No candidate events found with current canonical/DST-aware config.")

    events = events[
        [
            "source_index",
            "time",
            "session_date",
            "session_open_time",
            "event_return_log",
            "event_return_pips",
            "event_zscore",
            "close",
        ]
    ].rename(
        columns={
            "time": "timestamp",
            "close": "close_at_event",
        }
    ).reset_index(drop=True)

    print("Deduplicating events...")
    events = deduplicate_events(events, cooldown_bars=2)

    events["session"] = "NEW_YORK_DST_AWARE"
    events["window_min"] = window_min
    events["z_threshold"] = z_threshold
    events["event_direction"] = direction

    events_path = out_dir / "events.csv"
    events.to_csv(events_path, index=False)
    print(f"Saved: {events_path}")

    print("Simulating trades...")
    px = df.set_index("time").sort_index()

    entry_delay = int(cfg["execution"]["entry_delay_bars"])
    tp_pips = float(cfg["execution"]["take_profit_pips"])
    sl_pips = float(cfg["execution"]["stop_loss_pips"])
    time_stop_minutes = int(cfg["execution"]["time_stop_minutes"])
    spread = float(cfg["execution"]["spread_pips"])
    slip = float(cfg["execution"]["slippage_pips"])

    trades = []

    for _, ev in events.iterrows():
        event_time = pd.Timestamp(ev["timestamp"])

        if event_time not in px.index:
            continue

        event_idx = px.index.get_loc(event_time)
        entry_idx = event_idx + entry_delay

        if entry_idx >= len(px):
            continue

        entry_time = px.index[entry_idx]
        entry_price = float(px.iloc[entry_idx]["close"])

        trade_direction = "short"  # UP_EXTREME => short

        tp_price = entry_price - pip_to_price(tp_pips, pip_size)
        sl_price = entry_price + pip_to_price(sl_pips, pip_size)

        future = px.iloc[entry_idx + 1 : entry_idx + 1 + 200]
        if future.empty:
            continue

        exit_price, exit_reason, hold_bars = resolve_trade(
            direction=trade_direction,
            tp_price=tp_price,
            sl_price=sl_price,
            bars=future,
            time_stop_bars=time_stop_minutes // 5,
        )

        pnl_price = entry_price - exit_price
        pnl_pips = pnl_price / pip_size
        pnl_pips -= (spread + slip)

        trades.append(
            {
                "event_id": ev["event_id"],
                "event_time": event_time,
                "entry_time": entry_time,
                "direction": trade_direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
                "time_stop_minutes": time_stop_minutes,
                "exit_reason": exit_reason,
                "hold_bars": hold_bars,
                "pnl_pips": pnl_pips,
            }
        )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        raise ValueError("No trades were generated.")

    trades_path = out_dir / "trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"Saved: {trades_path}")

    metrics = compute_metrics(trades_df)
    metrics_df = pd.DataFrame([metrics])

    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    summary = {
        "experiment_id": cfg["experiment_id"],
        "dataset_path": str(dataset_path),
        "time_frame": cfg["time_frame"],
        "session_mode": cfg["session"]["mode"],
        "session_minutes": session_minutes,
        "event_definition": cfg["event_definition"],
        "execution": cfg["execution"],
        "pip_size": pip_size,
        "events_n": int(len(events)),
        "metrics": metrics,
    }

    summary_path = out_dir / "summary.json"
    save_json(summary, summary_path)
    print(f"Saved: {summary_path}")

    audit_lines: List[str] = []
    audit_lines.append("# PF_R1_E4 Audit")
    audit_lines.append("")
    audit_lines.append("## Dataset")
    audit_lines.append(f"- path: {dataset_path}")
    audit_lines.append(f"- pip_size: {pip_size}")
    audit_lines.append(f"- rows_filtered: {len(df)}")
    audit_lines.append(f"- start: {df['time'].min()}")
    audit_lines.append(f"- end: {df['time'].max()}")
    audit_lines.append("")
    audit_lines.append("## Session logic")
    audit_lines.append("- mode: NEW_YORK DST-aware")
    audit_lines.append("- helper: src.research.opening_momentum.session_times.ny_open_utc")
    audit_lines.append(f"- session_minutes: {session_minutes}")
    audit_lines.append("")
    audit_lines.append("## Event logic")
    audit_lines.append(f"- window_min: {window_min}")
    audit_lines.append(f"- z_threshold: {z_threshold}")
    audit_lines.append(f"- direction: {direction}")
    audit_lines.append(f"- events_n: {len(events)}")
    audit_lines.append("")
    audit_lines.append("## Execution")
    audit_lines.append(f"- entry_delay_bars: {entry_delay}")
    audit_lines.append(f"- TP: {tp_pips}")
    audit_lines.append(f"- SL: {sl_pips}")
    audit_lines.append(f"- TS: {time_stop_minutes}")
    audit_lines.append(f"- costs: spread={spread}, slip={slip}")
    audit_lines.append("")
    audit_lines.append("## Metrics")
    for k, v in metrics.items():
        audit_lines.append(f"- {k}: {v}")

    audit_path = out_dir / "audit.md"
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"Saved: {audit_path}")

    print("PF_R1_E4 complete.")
    print(f"Events: {len(events)}")
    print(f"Trades: {len(trades_df)}")


if __name__ == "__main__":
    main()
