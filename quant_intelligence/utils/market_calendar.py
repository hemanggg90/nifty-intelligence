"""NSE trading-session helpers.

Used to judge data staleness against when a new bar is actually expected
(NSE market hours), rather than raw wall-clock time - a pipeline run at
20:00 on a Monday should not treat the 15:30 close as "5 hours stale".
Holidays are not modelled (weekends only); a holiday will be treated as an
ordinary non-trading day, which only makes the staleness check slightly more
lenient than the exact NSE calendar, never stricter.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = dt.time(9, 15)
SESSION_CLOSE = dt.time(15, 30)


def _to_ist(moment: dt.datetime) -> dt.datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=IST)
    return moment.astimezone(IST)


def most_recent_expected_bar_time(now: dt.datetime) -> dt.datetime:
    """The most recent point in time a live feed would have produced a bar.

    Within a weekday trading session, that's `now` itself. Outside session
    hours (or on a weekend), it's the close of the most recent trading day
    on or before `now`. The result is tz-aware in IST.
    """
    ist_now = _to_ist(now)
    candidate = ist_now

    while True:
        is_weekday = candidate.weekday() < 5  # Mon=0 .. Fri=4
        if is_weekday and candidate.time() >= SESSION_OPEN:
            session_close = candidate.replace(
                hour=SESSION_CLOSE.hour, minute=SESSION_CLOSE.minute, second=0, microsecond=0
            )
            if candidate.time() <= SESSION_CLOSE:
                return candidate if candidate == ist_now else session_close
            return session_close

        # Before market open, or a weekend day: step back to the prior day's close.
        candidate = (candidate - dt.timedelta(days=1)).replace(
            hour=SESSION_CLOSE.hour, minute=SESSION_CLOSE.minute, second=0, microsecond=0
        )
