from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional, Any, Dict, TextIO

from zoneinfo import ZoneInfo

from src.runner.interfaces import Bar


class DataValidationError(Exception):
    pass


@dataclass(frozen=True)
class CSVProviderConfig:
    csv_path: str
    time_col: str = "time"

    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    volume_col: Optional[str] = None

    # tz_in:
    # - "auto" => if timestamps are timezone-aware, keep; if naive, assume UTC (safe default)
    # - any IANA name (e.g., "America/Bogota") => localize naive timestamps to that tz
    tz_in: str = "auto"

    # if True and timestamps are naive -> assume UTC regardless of tz_in
    assume_utc_if_naive: bool = False


def _open_text_auto(path: str) -> TextIO:
    """
    Open CSV with robust encoding handling.
    MT5/MetaEditor sometimes writes CSVs with BOM or UTF-16.
    We detect BOM in binary and choose encoding accordingly.

    Returns an opened text file handle (newline="") suitable for csv module.
    """
    try:
        with open(path, "rb") as fb:
            head = fb.read(4)
    except FileNotFoundError as e:
        raise DataValidationError(f"CSV not found: {path}") from e
    except Exception as e:
        raise DataValidationError(f"Unable to read CSV bytes: {path}. err={e}") from e

    # BOM detection
    # UTF-8 BOM:    EF BB BF
    # UTF-16 LE:    FF FE
    # UTF-16 BE:    FE FF
    # UTF-32 LE:    FF FE 00 00
    # UTF-32 BE:    00 00 FE FF
    enc = "utf-8"
    if head.startswith(b"\xef\xbb\xbf"):
        enc = "utf-8-sig"
    elif head.startswith(b"\xff\xfe\x00\x00"):
        enc = "utf-32-le"
    elif head.startswith(b"\x00\x00\xfe\xff"):
        enc = "utf-32-be"
    elif head.startswith(b"\xff\xfe"):
        enc = "utf-16-le"
    elif head.startswith(b"\xfe\xff"):
        enc = "utf-16-be"
    else:
        # Most common safe default: utf-8-sig handles accidental UTF-8 BOM too.
        enc = "utf-8-sig"

    try:
        return open(path, "r", newline="", encoding=enc)
    except UnicodeDecodeError:
        # Last-resort fallback for weird encodings; keeps process alive.
        # (We prefer parsing over crashing; downstream validation still applies.)
        return open(path, "r", newline="", encoding="latin-1")
    except Exception as e:
        raise DataValidationError(f"Unable to open CSV: {path} (encoding={enc}). err={e}") from e


def _parse_dt(s: Any) -> datetime:
    """
    Parse timestamp. Supports:

    ISO family:
      - "2024-01-01 12:00:00"
      - "2024-01-01T12:00:00"
      - "2024-01-01T12:00:00Z"
      - "2024-01-01T12:00:00+00:00"

    MT5 common:
      - "2026.02.25 22:40"
      - "2026.02.25 22:40:00"

    Returns datetime (may be naive).
    """
    if s is None:
        raise ValueError("timestamp is None")

    txt = str(s).strip()
    if not txt:
        raise ValueError("empty timestamp")

    # normalize common 'Z' suffix to ISO offset
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"

    # Fast path: ISO (fromisoformat accepts " " or "T" in modern Python)
    try:
        return datetime.fromisoformat(txt)
    except ValueError:
        pass

    # MT5 path: "YYYY.MM.DD HH:MM[:SS]"
    # Normalize dots into dashes and parse with strptime.
    mt5 = txt.replace(".", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(mt5, fmt)
        except ValueError:
            continue

    raise ValueError(f"Invalid timestamp format: {txt!r}")


def _ensure_tz(dt: datetime, *, tz_in: str, assume_utc_if_naive: bool) -> datetime:
    """
    Ensure dt is timezone-aware UTC.

    Rules:
      - If dt already has tzinfo -> convert to UTC
      - If dt is naive:
          - if assume_utc_if_naive => treat as UTC
          - elif tz_in == "auto"  => treat as UTC (pragmatic default)
          - else localize using tz_in then convert to UTC
    """
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)

    # dt is naive
    if assume_utc_if_naive:
        return dt.replace(tzinfo=timezone.utc)

    if str(tz_in).strip().lower() == "auto":
        # Pragmatic default: assume naive timestamps represent UTC in research CSVs.
        return dt.replace(tzinfo=timezone.utc)

    # Explicit tz_in given: localize naive dt to tz_in, then convert to UTC
    try:
        tz = ZoneInfo(tz_in)
    except Exception as e:
        raise DataValidationError(
            f"Invalid tz_in={tz_in!r}. Provide a valid IANA timezone like 'America/Bogota' or use tz_in='auto'."
        ) from e

    localized = dt.replace(tzinfo=tz)
    return localized.astimezone(timezone.utc)


def _to_float(row: Dict[str, Any], key: str) -> float:
    if key not in row:
        raise DataValidationError(f"Missing required column: {key}")
    try:
        return float(row[key])
    except Exception as e:
        raise DataValidationError(f"Invalid float in column '{key}': {row.get(key)!r}") from e


class CSVBarProvider:
    """
    Streaming CSV bar provider.

    Output:
      Bar(ts_utc, open, high, low, close, volume?, extras?)
    """

    def __init__(self, cfg: CSVProviderConfig):
        self.cfg = cfg

    def iter_bars(self) -> Iterator[Bar]:
        path = self.cfg.csv_path

        with _open_text_auto(path) as f:
            r = csv.DictReader(f)
            if r.fieldnames is None:
                raise DataValidationError("CSV has no header row.")

            # Basic column presence checks (OHLC)
            required = [
                self.cfg.time_col,
                self.cfg.open_col,
                self.cfg.high_col,
                self.cfg.low_col,
                self.cfg.close_col,
            ]
            for k in required:
                if k not in r.fieldnames:
                    raise DataValidationError(f"CSV missing required column: {k}. Found: {r.fieldnames}")

            for row in r:
                # Skip completely empty lines
                if not row:
                    continue

                # timestamp
                try:
                    dt = _parse_dt(row[self.cfg.time_col])
                except Exception as e:
                    raise DataValidationError(
                        f"Invalid timestamp in {self.cfg.time_col}: {row.get(self.cfg.time_col)!r}"
                    ) from e

                ts_utc = _ensure_tz(dt, tz_in=self.cfg.tz_in, assume_utc_if_naive=self.cfg.assume_utc_if_naive)

                o = _to_float(row, self.cfg.open_col)
                h = _to_float(row, self.cfg.high_col)
                l = _to_float(row, self.cfg.low_col)
                c = _to_float(row, self.cfg.close_col)

                vol = None
                if self.cfg.volume_col and self.cfg.volume_col in row and str(row[self.cfg.volume_col]).strip() != "":
                    try:
                        vol = float(row[self.cfg.volume_col])
                    except Exception:
                        vol = None

                # extras: include all columns not in OHLCV+time
                known = set(required)
                if self.cfg.volume_col:
                    known.add(self.cfg.volume_col)

                extras = {k: row[k] for k in row.keys() if k not in known}

                yield Bar(
                    ts_utc=ts_utc,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=vol,
                    extras=extras,
                )