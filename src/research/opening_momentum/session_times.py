from __future__ import annotations

from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo


UTC = ZoneInfo("UTC")
LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")


def london_open_utc(d: date) -> datetime:
    local_dt = datetime.combine(d, dtime(hour=8, minute=0), tzinfo=LONDON_TZ)
    return local_dt.astimezone(UTC).replace(tzinfo=None)


def ny_open_utc(d: date) -> datetime:
    local_dt = datetime.combine(d, dtime(hour=8, minute=30), tzinfo=NY_TZ)
    return local_dt.astimezone(UTC).replace(tzinfo=None)
