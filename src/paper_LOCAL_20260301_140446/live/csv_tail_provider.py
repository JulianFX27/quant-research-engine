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
    start_from_last_row: bool = True


class TailBar:
    """
    Minimal bar compatible with runner:
      - ts_utc
      - open/high/low/close
      - extras (ALL columns, e.g. atr_14, shock_z, ny_open...)
    """
    def __init__(self, ts_utc: datetime, o: float, h: float, l: float, c: float, extras: Dict[str, Any]):
        self.ts_utc = ts_utc
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.extras = extras


class TailCSVBarProvider:
    def __init__(self, cfg: TailCSVConfig):
        self.cfg = cfg
        self._path = Path(cfg.csv_path)
        self._last_ts: Optional[datetime] = None

    def _prime_last_ts_from_file_end(self) -> None:
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

                        extras = dict(row)
                        yield TailBar(ts, o, h, l, c, extras)

                        self._last_ts = ts
            except Exception:
                time.sleep(self.cfg.poll_seconds)

            time.sleep(self.cfg.poll_seconds)