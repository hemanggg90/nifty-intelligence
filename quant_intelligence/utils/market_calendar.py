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

from quant_intelligence.utils.market_profile import NSE, MarketProfile

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = NSE.open
SESSION_CLOSE = NSE.close


def _to_ist(moment: dt.datetime) -> dt.datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=IST)
    return moment.astimezone(IST)


def most_recent_expected_bar_time(now: dt.datetime, profile: MarketProfile = NSE) -> dt.datetime:
    """The most recent point in time a live feed would have produced a bar.

    Within a weekday trading session, that's `now` itself. Outside session
    hours (or on a weekend), it's the close of the most recent trading day
    on or before `now`. The result is tz-aware in IST.
    """
    SESSION_OPEN, SESSION_CLOSE = profile.open, profile.close
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


def expected_last_closed_bar_start(now: dt.datetime, profile: MarketProfile = NSE, tf_minutes: int = 5) -> dt.datetime:
    """Start time (naive IST) of the most recent bar that has CLOSED as of `now`.

    Bars tile the session from its open in steps of `tf_minutes`. Inside a session that is the bar
    that finished most recently; before the first bar of the day has closed, or outside the session,
    it is the last bar of the previous/most recent trading session. Used to decide whether cached
    candles are up to date, so a fresh request is needed only when a new bar has actually closed.
    """
    ref = most_recent_expected_bar_time(now, profile).replace(tzinfo=None)
    open_dt = ref.replace(hour=profile.open.hour, minute=profile.open.minute, second=0, microsecond=0)
    elapsed = min((ref - open_dt).total_seconds() / 60.0, float(profile.session_minutes))
    last_index = int((elapsed - tf_minutes) // tf_minutes)
    if last_index < 0:
        # The day's first bar has not closed yet: the last closed bar belongs to the previous session.
        return expected_last_closed_bar_start(open_dt - dt.timedelta(minutes=1), profile, tf_minutes)
    return open_dt + dt.timedelta(minutes=last_index * tf_minutes)


def session_status(now: dt.datetime, profile: MarketProfile = NSE) -> dict:
    """Is the session open at `now` (naive IST), and how long until that changes?

    Returns {"open": bool, "change_in": timedelta, "label": "closes in 2h 14m" | "opens in 17h 5m"}.
    Weekends are closed; exchange holidays are not modelled.
    """
    now = now.replace(tzinfo=None)
    today_open = dt.datetime.combine(now.date(), profile.open)
    today_close = dt.datetime.combine(now.date(), profile.close)
    if now.weekday() < 5 and today_open <= now < today_close:
        delta = today_close - now
        return {"open": True, "change_in": delta, "label": "closes in " + _short(delta)}
    for offset in range(0, 8):
        day = now.date() + dt.timedelta(days=offset)
        candidate = dt.datetime.combine(day, profile.open)
        if day.weekday() < 5 and candidate > now:
            delta = candidate - now
            return {"open": False, "change_in": delta, "label": "opens in " + _short(delta)}
    return {"open": False, "change_in": dt.timedelta(0), "label": "closed"}


def _short(delta: dt.timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    return f"{hours}h {mins:02d}m" if hours else f"{mins}m"
