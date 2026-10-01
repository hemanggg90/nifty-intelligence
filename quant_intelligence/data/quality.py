"""
Market-data quality validation.

Every dataset that enters the pipeline is checked for: missing candles,
duplicate timestamps, timezone consistency, zero/negative prices, abnormal
OHLC relationships (high < low, close outside [low, high], etc.), missing
volume, and staleness. The result feeds market_data_metadata and, at
decision time, the Market State Engine's data_quality_status — poor quality
data must lead to NO TRADE, never a silently-degraded decision.
"""
from __future__ import annotations

import datetime as dt
from quant_intelligence.utils.timeutil import now_ist
from dataclasses import dataclass, field

import pandas as pd

from quant_intelligence.utils.market_calendar import most_recent_expected_bar_time
from quant_intelligence.utils.market_profile import NSE, MarketProfile

SESSION_OPEN_MIN = NSE.open_min
SESSION_CLOSE_MIN = NSE.close_min
# More than this fraction of bars outside the session means the clock/timezone is wrong
# (a real misalignment), as opposed to a few stray pre/post-close prints.
MAX_OUT_OF_SESSION_FRACTION = 0.02

QUALITY_OK = "OK"
QUALITY_DEGRADED = "DEGRADED"
QUALITY_FAIL = "FAIL"


@dataclass
class QualityReport:
    status: str
    issues: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_duplicates: int = 0
    n_missing_candles: int = 0
    n_ohlc_violations: int = 0
    n_zero_or_negative: int = 0
    n_missing_volume: int = 0
    is_stale: bool = False
    last_timestamp: dt.datetime | None = None

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["last_timestamp"] = str(self.last_timestamp) if self.last_timestamp else None
        return d


def out_of_session_mask(timestamps: pd.Series, profile: MarketProfile = NSE) -> pd.Series:
    ts = pd.to_datetime(timestamps)
    minutes = ts.dt.hour * 60 + ts.dt.minute
    return (minutes < profile.open_min) | (minutes >= profile.close_min)


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe_minutes: int,
    staleness_threshold_minutes: int = 30,
    now: dt.datetime | None = None,
    profile: MarketProfile = NSE,
) -> QualityReport:
    issues: list[str] = []
    n_rows = len(df)

    if n_rows == 0:
        return QualityReport(status=QUALITY_FAIL, issues=["no data rows"], n_rows=0)

    df = df.sort_values("timestamp")

    n_duplicates = int(df["timestamp"].duplicated().sum())
    if n_duplicates > 0:
        issues.append(f"{n_duplicates} duplicate timestamps")

    # Missing candles: within each trading session, gaps larger than the expected
    # timeframe are flagged (holidays/weekends across days are expected and excluded).
    ts = pd.to_datetime(df["timestamp"])
    same_day = ts.dt.date == ts.dt.date.shift(1)
    deltas_minutes = ts.diff().dt.total_seconds() / 60.0
    expected = timeframe_minutes
    gap_mask = same_day & (deltas_minutes > expected * 1.5)
    n_missing_candles = int(gap_mask.sum())
    if n_missing_candles > 0:
        issues.append(f"{n_missing_candles} intra-session gaps larger than expected bar interval")

    n_out_of_session = 0
    if timeframe_minutes < 1440:
        n_out_of_session = int(out_of_session_mask(df["timestamp"], profile).sum())
        if n_out_of_session / n_rows > MAX_OUT_OF_SESSION_FRACTION:
            issues.append(f"{n_out_of_session} bars outside the {profile.label} session (timezone/data error)")
        else:
            n_out_of_session = 0  # a few stray pre/post-close prints; removed by clean_ohlcv

    zero_or_neg = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    n_zero_or_negative = int(zero_or_neg.sum())
    if n_zero_or_negative > 0:
        issues.append(f"{n_zero_or_negative} rows with zero/negative price")

    ohlc_violation = (
        (df["high"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
    )
    n_ohlc_violations = int(ohlc_violation.sum())
    if n_ohlc_violations > 0:
        issues.append(f"{n_ohlc_violations} rows with abnormal OHLC relationships")

    n_missing_volume = int((df["volume"].isna() | (df["volume"] < 0)).sum())
    if n_missing_volume > 0:
        issues.append(f"{n_missing_volume} rows with missing/invalid volume")

    last_timestamp = ts.max()
    now = now or now_ist()
    reference = most_recent_expected_bar_time(now, profile)
    last_ts_py = last_timestamp.to_pydatetime()
    if last_ts_py.tzinfo is None:
        last_ts_py = last_ts_py.replace(tzinfo=reference.tzinfo)
    is_stale = bool((reference - last_ts_py) > dt.timedelta(minutes=staleness_threshold_minutes))
    if is_stale:
        behind = reference - last_ts_py
        age = f"{behind.days}d {behind.seconds // 3600}h" if behind.days else f"{behind.seconds // 3600}h {behind.seconds % 3600 // 60}m"
        issues.append(
            f"data is stale: last bar {pd.Timestamp(last_timestamp):%d %b %H:%M}, "
            f"{age} behind the {reference:%d %b %H:%M} IST market clock"
        )

    # Determine overall status.
    hard_fail = n_zero_or_negative > 0 or n_ohlc_violations > 0 or n_rows < 5 or n_out_of_session > 0
    degraded = n_duplicates > 0 or n_missing_candles > 0 or n_missing_volume > 0 or is_stale

    if hard_fail:
        status = QUALITY_FAIL
    elif degraded:
        status = QUALITY_DEGRADED
    else:
        status = QUALITY_OK

    return QualityReport(
        status=status,
        issues=issues,
        n_rows=n_rows,
        n_duplicates=n_duplicates,
        n_missing_candles=n_missing_candles,
        n_ohlc_violations=n_ohlc_violations,
        n_zero_or_negative=n_zero_or_negative,
        n_missing_volume=n_missing_volume,
        is_stale=is_stale,
        last_timestamp=last_timestamp.to_pydatetime() if isinstance(last_timestamp, pd.Timestamp) else last_timestamp,
    )


def clean_ohlcv(df: pd.DataFrame, timeframe_minutes: int | None = None, profile: MarketProfile = NSE) -> pd.DataFrame:
    """Deterministic cleanup: drop duplicate timestamps (keep last) and, for intraday
    data, bars outside the 09:15-15:30 session (stray pre/post-close prints); sort, reset index."""
    if timeframe_minutes is not None and timeframe_minutes < 1440 and len(df):
        df = df[~out_of_session_mask(df["timestamp"], profile)]
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if timeframe_minutes is not None and timeframe_minutes < 1440 and len(df):
        df = _drop_bad_opening_bars(df)
    return df


def _ohlc_violation_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["high"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
    )


def _drop_bad_opening_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drop OHLC-inconsistent bars that are the FIRST bar of a session.

    Dhan occasionally reports an opening bar whose open lies outside its own high-low range
    (seen on MCX silver at 09:00 - the exchange's opening-call price). Dropping it loses one bar
    and leaves no intra-session gap (the next bar simply becomes the first). The same violation
    anywhere else in a session is still left in place so validate_ohlcv hard-fails it.
    """
    is_first = df.groupby(pd.to_datetime(df["timestamp"]).dt.date).cumcount() == 0
    drop = _ohlc_violation_mask(df) & is_first
    if drop.any():
        from quant_intelligence.utils.logging_utils import log_event

        log_event("data_quality", f"Dropped {int(drop.sum())} inconsistent session-opening bar(s)", level="WARNING")
    return df[~drop].reset_index(drop=True)
