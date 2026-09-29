import datetime as dt

import pandas as pd
import pytest

from quant_intelligence.data.quality import QUALITY_FAIL, validate_ohlcv
from quant_intelligence.data_adapters.dhan_adapter import _align_to_ist_session, drop_forming_bar


def _bars(start: str, n: int, freq: str = "5min") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0})


def test_ist_timestamps_kept_as_is():
    df = _bars("2024-06-28 09:15", 20)
    assert _align_to_ist_session(df)["timestamp"].iloc[0] == pd.Timestamp("2024-06-28 09:15")


def test_utc_epoch_is_shifted_to_ist():
    df = _bars("2024-06-28 03:45", 20)  # 09:15 IST expressed in UTC
    assert _align_to_ist_session(df)["timestamp"].iloc[0] == pd.Timestamp("2024-06-28 09:15")


def test_unalignable_timestamps_are_refused():
    df = _bars("2024-06-28 01:00", 20)
    with pytest.raises(RuntimeError):
        _align_to_ist_session(df)


def test_forming_bar_is_dropped_but_closed_bar_kept():
    df = _bars("2024-06-28 10:00", 3)  # 10:00, 10:05, 10:10
    assert len(drop_forming_bar(df, 5, now=dt.datetime(2024, 6, 28, 10, 12))) == 2
    assert len(drop_forming_bar(df, 5, now=dt.datetime(2024, 6, 28, 10, 15))) == 3


def test_out_of_session_bars_fail_quality():
    report = validate_ohlcv(_bars("2024-06-28 03:45", 20), 5, now=dt.datetime(2024, 6, 28, 10, 0))
    assert report.status == QUALITY_FAIL


def _sessions(first_bar: str, days: int = 4) -> pd.DataFrame:
    """`days` full 75-bar sessions starting at `first_bar` on 2024-06-24, 25, ..."""
    hh, mm = first_bar.split(":")
    frames = [_bars(f"2024-06-{24 + d} {hh}:{mm}", 75) for d in range(days)]
    return pd.concat(frames).reset_index(drop=True)


def test_stray_post_close_bars_are_dropped_not_fatal():
    from quant_intelligence.data_adapters.dhan_adapter import _align_to_ist_session

    utc = _sessions("03:45")  # 09:15-15:25 IST expressed in UTC
    strays = _bars("2024-06-24 10:00", 2)  # 15:30, 15:35 IST
    aligned = _align_to_ist_session(pd.concat([utc, strays]).reset_index(drop=True))
    assert len(aligned) == len(utc)  # strays removed
    report = validate_ohlcv(aligned, 5, now=dt.datetime(2024, 6, 27, 15, 30))
    assert report.status != QUALITY_FAIL


def test_finalize_fails_on_wholesale_misalignment_but_not_stray_bars():
    from quant_intelligence.data.data_manager import DataManager

    dm = DataManager()
    _, meta = dm._finalize(_sessions("03:45"), "X", "5min", "test")  # UTC clock, never shifted
    assert meta["quality_status"] == QUALITY_FAIL

    stray = pd.concat([_sessions("09:15"), _bars("2024-06-24 15:30", 2)]).reset_index(drop=True)
    df, meta = dm._finalize(stray, "X", "5min", "test")
    assert meta["n_rows"] == 300
    assert not any("outside" in i for i in meta["quality_report"]["issues"])
