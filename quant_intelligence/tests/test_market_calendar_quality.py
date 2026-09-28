import datetime as dt

import pandas as pd

from quant_intelligence.data.quality import QUALITY_DEGRADED, QUALITY_OK, validate_ohlcv
from quant_intelligence.utils.market_calendar import most_recent_expected_bar_time


def _ohlcv(last_ts: dt.datetime, n: int = 10, freq_minutes: int = 5) -> pd.DataFrame:
    timestamps = pd.date_range(end=last_ts, periods=n, freq=f"{freq_minutes}min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1000] * n,
        }
    )


def test_most_recent_expected_bar_time_after_hours_returns_same_day_close():
    # Monday 20:29 -> today's 15:30 close
    now = dt.datetime(2026, 9, 21, 20, 29)
    ref = most_recent_expected_bar_time(now)
    assert ref.hour == 15 and ref.minute == 30
    assert ref.date() == now.date()


def test_most_recent_expected_bar_time_weekend_returns_last_friday_close():
    # Sunday -> prior Friday's 15:30 close
    now = dt.datetime(2026, 9, 20, 10, 0)  # a Sunday
    ref = most_recent_expected_bar_time(now)
    assert ref.weekday() == 4  # Friday
    assert ref.hour == 15 and ref.minute == 30


def test_validate_ohlcv_not_stale_when_last_bar_matches_prior_session_close():
    last_bar = dt.datetime(2026, 9, 21, 15, 30)  # Monday close
    df = _ohlcv(last_bar)
    now = dt.datetime(2026, 9, 21, 20, 29)  # same evening, well past raw 30-min wall-clock threshold
    report = validate_ohlcv(df, timeframe_minutes=5, now=now)
    assert report.status == QUALITY_OK
    assert not report.is_stale


def test_validate_ohlcv_still_stale_when_genuinely_behind_during_market_hours():
    last_bar = dt.datetime(2026, 9, 21, 9, 30)  # early in the session
    df = _ohlcv(last_bar)
    now = dt.datetime(2026, 9, 21, 14, 0)  # same session, hours later, still no new bars
    report = validate_ohlcv(df, timeframe_minutes=5, now=now)
    assert report.status == QUALITY_DEGRADED
    assert report.is_stale
