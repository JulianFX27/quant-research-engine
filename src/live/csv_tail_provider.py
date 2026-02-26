from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional, Any


def _parse_ts_utc(s: str) -> datetime:
    txt = (s or "").strip()
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
    start_from_last_row: bool = True  # paper_live starts at "now" by default


class TailBar:
    """
    Minimal bar object compatible with your runner expectations:
      - ts_utc
      - open/high/low/close
      - optional extras dict containing all other columns (for PolicyGate features)
    """
    def __init__(self, ts_utc: datetime, o: float, h: float, l: float, c: float, extras: Dict[str, Any]):
        self.ts_utc = ts_utc
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.extras = extras  # IMPORTANT: PolicyGate uses _bar_feature() to read from extras


class TailCSVBarProvider:
    def __init__(self, cfg: TailCSVConfig):
        self.cfg = cfg
        self._path = Path(cfg.csv_path)
        self._last_ts: Optional[datetime] = None

    def _prime_last_ts_from_file_end(self) -> None:
        """
        If we want to start paper_live from the latest available bar,
        we set _last_ts to the last row timestamp so we only process new rows.
        """
        if not self._path.exists() or self._path.stat().st_size == 0:
            return

        try:
            with self._path.open("r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                last = None
                for row in r:
                    last = row
            if last and (self.cfg.time_col in last):
                self._last_ts = _parse_ts_utc(str(last[self.cfg.time_col]))
        except Exception:
            # if any issue, just start from beginning
            self._last_ts = None

    def iter_bars_live(self) -> Iterator[TailBar]:
        if self.cfg.start_from_last_row and self._last_ts is None:
            self._prime_last_ts_from_file_end()

        while True:
            if not self._path.exists() or self._path.stat().st_size == 0:
                time.sleep(self.cfg.poll_seconds)
                continue

            try:
                with self._path.open("r", newline="", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        if not row:
                            continue
                        if self.cfg.time_col not in row:
                            continue

                        ts = _parse_ts_utc(str(row[self.cfg.time_col]))
                        if self._last_ts is not None and ts <= self._last_ts:
                            continue

                        o = float(row[self.cfg.open_col])
                        h = float(row[self.cfg.high_col])
                        l = float(row[self.cfg.low_col])
                        c = float(row[self.cfg.close_col])

                        # store ALL columns (including atr_14/shock_z/etc) for PolicyGate
                        extras = dict(row)
                        yield TailBar(ts, o, h, l, c, extras)

                        self._last_ts = ts
            except Exception:
                # If file is mid-write, just retry next tick
                time.sleep(self.cfg.poll_seconds)

            time.sleep(self.cfg.poll_seconds)