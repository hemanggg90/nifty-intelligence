"""Exchange holidays: built-in fixed dates, a user-editable file, and the calendar functions honouring them."""
import datetime as dt

from quant_intelligence.config import holidays
from quant_intelligence.config.holidays import is_holiday, is_trading_day
from quant_intelligence.utils.market_calendar import expected_last_closed_bar_start, most_recent_expected_bar_time, session_status
from quant_intelligence.utils.market_profile import MCX, NSE
from quant_intelligence.utils.timeutil import is_market_open


def test_fixed_date_holidays_are_built_in_and_1_may_is_nse_only():
    assert is_holiday(dt.date(2026, 10, 2), "NSE") and is_holiday(dt.date(2026, 10, 2), "MCX")
    for month, day in ((1, 26), (8, 15), (12, 25)):
        assert is_holiday(dt.date(2026, month, day), "NSE")
    assert is_holiday(dt.date(2026, 5, 1), "NSE") and not is_holiday(dt.date(2026, 5, 1), "MCX")  # MCX evening session runs
    assert not is_holiday(dt.date(2026, 10, 1), "NSE")


def test_weekends_are_not_trading_days_and_ordinary_weekdays_are():
    assert not is_trading_day(dt.date(2026, 10, 3))  # Saturday
    assert is_trading_day(dt.date(2026, 10, 1))
    assert not is_trading_day(dt.date(2026, 10, 2))  # holiday on a Friday


def test_the_user_file_adds_holidays_per_market_and_ignores_bad_lines(tmp_path, monkeypatch):
    f = tmp_path / "h.txt"
    f.write_text("# comment\n\n2026-03-03 NSE Holi\n2026-11-08 Diwali Laxmi Pujan\nnot-a-date oops\n2026-04-03 MCX Good Friday  # trailing\n",
                 encoding="utf-8")
    monkeypatch.setenv("HOLIDAYS_FILE", str(f))
    assert is_holiday(dt.date(2026, 3, 3), "NSE") and not is_holiday(dt.date(2026, 3, 3), "MCX")
    assert is_holiday(dt.date(2026, 11, 8), "NSE") and is_holiday(dt.date(2026, 11, 8), "MCX")  # no market = both
    assert is_holiday(dt.date(2026, 4, 3), "MCX") and not is_holiday(dt.date(2026, 4, 3), "NSE")


def test_edits_to_the_file_are_picked_up_without_a_restart(tmp_path, monkeypatch):
    import os
    f = tmp_path / "h.txt"
    f.write_text("2026-06-10 NSE test\n", encoding="utf-8")
    monkeypatch.setenv("HOLIDAYS_FILE", str(f))
    assert is_holiday(dt.date(2026, 6, 10), "NSE") and not is_holiday(dt.date(2026, 6, 11), "NSE")
    f.write_text("2026-06-11 NSE test\n", encoding="utf-8")
    os.utime(f, (f.stat().st_atime + 5, f.stat().st_mtime + 5))
    assert is_holiday(dt.date(2026, 6, 11), "NSE") and not is_holiday(dt.date(2026, 6, 10), "NSE")


def test_a_missing_file_means_only_the_built_ins(monkeypatch, tmp_path):
    monkeypatch.setenv("HOLIDAYS_FILE", str(tmp_path / "nope.txt"))
    assert is_holiday(dt.date(2026, 1, 26), "NSE") and not is_holiday(dt.date(2026, 3, 3), "NSE")


def test_staleness_reference_skips_a_holiday():
    # Friday 2 Oct 2026 is a holiday: at 11:00 the newest bar a feed could have is Thursday's last one,
    # so data ending 15:25 Thursday is current, not "stale since yesterday".
    now = dt.datetime(2026, 10, 2, 11, 0)
    assert most_recent_expected_bar_time(now, NSE).replace(tzinfo=None) == dt.datetime(2026, 10, 1, 15, 30)
    assert expected_last_closed_bar_start(now, NSE, 5) == dt.datetime(2026, 10, 1, 15, 25)
    assert most_recent_expected_bar_time(dt.datetime(2026, 10, 5, 8, 0), NSE).replace(tzinfo=None) == dt.datetime(2026, 10, 1, 15, 30)


def test_session_status_and_market_open_respect_holidays():
    holiday_morning = dt.datetime(2026, 10, 2, 10, 0)
    assert not session_status(holiday_morning, NSE)["open"] and not is_market_open(holiday_morning, NSE)
    assert not session_status(holiday_morning, MCX)["open"]
    # next open is Monday 5 Oct 09:15
    assert session_status(holiday_morning, NSE)["change_in"] == dt.datetime(2026, 10, 5, 9, 15) - holiday_morning
    assert is_market_open(dt.datetime(2026, 10, 1, 10, 0), NSE)


def test_mcx_still_trades_on_1_may():
    assert is_trading_day(dt.date(2026, 5, 1), "MCX") == (dt.date(2026, 5, 1).weekday() < 5)
    assert holidays.is_holiday(dt.date(2026, 5, 1), "NSE")
