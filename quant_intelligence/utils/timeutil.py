"""Single source of 'now' for the app: Indian Standard Time, regardless of server timezone.

Streamlit Cloud servers run in UTC, so a bare `datetime.now()` there is 5h30 behind the
market. Everything that talks to the market (data end-times, staleness, trade timestamps,
on-screen clocks) must use these helpers instead.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
_OPEN = dt.time(9, 15)
_CLOSE = dt.time(15, 30)


def now_ist() -> dt.datetime:
    """Current IST wall-clock time as a naive datetime (matches the naive-IST bar timestamps)."""
    return dt.datetime.now(IST).replace(tzinfo=None)


def today_ist() -> dt.date:
    return now_ist().date()


def is_market_open(moment: dt.datetime | None = None) -> bool:
    """Weekday 09:15-15:30 IST. Exchange holidays are not modelled."""
    moment = moment or now_ist()
    return moment.weekday() < 5 and _OPEN <= moment.time() < _CLOSE
