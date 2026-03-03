from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def _parse_ts_utc(s: str) -> datetime:
    txt = (s or "").strip()
    if not txt:
        raise ValueError("empty timestamp")
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class TailCSVConfig:
    csv_path: str
    time_col: str = "time"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    tick_volume_col: Optional[str] = None
    poll_seconds: float = 1.0
    start_from_last_row: bool = True  # start paper/shadow at "now" by default


class TailBar:
    """
    Bar contract for live/paper/shadow runners.

    Required:
      - ts_utc (bar timestamp in UTC)
      - open/high/low/close
      - extras: dict with ALL columns (strings as read from CSV)

    Added for runner compatibility:
      - row_idx: monotonic row index within the CSV stream
      - bar_ts_utc: alias of ts_utc (explicit name many runners expect)
      - bar_key: stable id key for idempotency/state
    """
    def __init__(
        self,
        *,
        ts_utc: datetime,
        o: float,
        h: float,
        l: float,
        c: float,
        extras: Dict[str, Any],
        row_idx: int,
    ):
        self.ts_utc = ts_utc
        self.bar_ts_utc = ts_utc
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.extras = extras
        self.row_idx = int(row_idx)
        # stable id: timestamp + row index
        self.bar_key = f"{self.ts_utc.isoformat()}|{self.row_idx}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc,
            "bar_ts_utc": self.bar_ts_utc,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "row_idx": self.row_idx,
            "bar_key": self.bar_key,
            "extras": self.extras,
        }


class TailCSVBarProvider:
    def __init__(self, cfg: TailCSVConfig):
        self.cfg = cfg
        self._path = Path(cfg.csv_path)
        self._last_ts: Optional[datetime] = None
        self._last_row_idx: int = -1

    def _prime_from_file_end(self) -> None:
        """
        If start_from_last_row=True, we want to start from "now".
        We set both:
          - _last_ts to the last row timestamp
          - _last_row_idx to the last row index
        so the next new row increments row_idx and can be committed by runners.
        """
        if not self._path.exists() or self._path.stat().st_size == 0:
            return

        try:
            last_row: Optional[Dict[str, str]] = None
            last_idx = -1
            with self._path.open("r", newline="", encoding="utf-8-sig") as f:
                r = csv.DictReader(f)
                for i, row in enumerate(r):
                    if row:
                        last_row = row
                        last_idx = i
            if last_row and (self.cfg.time_col in last_row):
                self._last_ts = _parse_ts_utc(str(last_row[self.cfg.time_col]))
                self._last_row_idx = last_idx
        except Exception:
            # If anything odd, start from beginning
            self._last_ts = None
            self._last_row_idx = -1

    def iter_bars_live(self) -> Iterator[TailBar]:
        if self.cfg.start_from_last_row and self._last_ts is None:
            self._prime_from_file_end()

        while True:
            if not self._path.exists() or self._path.stat().st_size == 0:
                time.sleep(self.cfg.poll_seconds)
                continue

            try:
                with self._path.open("r", newline="", encoding="utf-8-sig") as f:
                    r = csv.DictReader(f)

                    # NOTE: enumerate() counts rows after header
                    for i, row in enumerate(r):
                        if not row:
                            continue

                        # Required cols check
                        if self.cfg.time_col not in row:
                            continue
                        for col in (self.cfg.open_col, self.cfg.high_col, self.cfg.low_col, self.cfg.close_col):
                            if col not in row:
                                continue

                        # Parse timestamp; if fails, skip row (common when file is mid-write)
                        try:
                            ts = _parse_ts_utc(str(row[self.cfg.time_col]))
                        except Exception:
                            continue

                        # Idempotency gating:
                        # - use row index primarily
                        # - keep ts gating as extra safety
                        if i <= self._last_row_idx:
                            continue
                        if self._last_ts is not None and ts <= self._last_ts and i <= self._last_row_idx:
                            continue

                        # Parse OHLC; if partial line => skip
                        try:
                            o = float(row[self.cfg.open_col])
                            h = float(row[self.cfg.high_col])
                            l = float(row[self.cfg.low_col])
                            c = float(row[self.cfg.close_col])
                        except Exception:
                            continue

                        extras = dict(row)  # keep ALL columns as strings
                        bar = TailBar(
                            ts_utc=ts,
                            o=o,
                            h=h,
                            l=l,
                            c=c,
                            extras=extras,
                            row_idx=i,
                        )
                        yield bar

                        # commit local cursor AFTER yield
                        self._last_ts = ts
                        self._last_row_idx = i

            except Exception:
                # If file is mid-write/locked, retry next tick
                time.sleep(self.cfg.poll_seconds)

            time.sleep(self.cfg.poll_seconds)