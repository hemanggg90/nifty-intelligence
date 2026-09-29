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
from dataclasses import dataclass, field

import pandas as pd

from quant_intelligence.utils.market_calendar import most_recent_expected_bar_time

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


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe_minutes: int,
    staleness_threshold_minutes: int = 30,
    now: dt.datetime | None = None,
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
        minutes_of_day = ts.dt.hour * 60 + ts.dt.minute
        n_out_of_session = int(((minutes_of_day < 9 * 60 + 15) | (minutes_of_day >= 15 * 60 + 30)).sum())
        if n_out_of_session > 0:
            issues.append(f"{n_out_of_session} bars outside the 09:15-15:30 session (timezone/data error)")

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
    now = now or dt.datetime.now()
    reference = most_recent_expected_bar_time(now)
    last_ts_py = last_timestamp.to_pydatetime()
    if last_ts_py.tzinfo is None:
        last_ts_py = last_ts_py.replace(tzinfo=reference.tzinfo)
    is_stale = bool((reference - last_ts_py) > dt.timedelta(minutes=staleness_threshold_minutes))
    if is_stale:
        issues.append(f"data is stale: last bar at {last_timestamp}, expected as of {reference}")

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


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic cleanup: drop exact duplicate timestamps (keep last), sort, reset index."""
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
