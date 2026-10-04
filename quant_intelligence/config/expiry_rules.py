"""Option expiry rules and trading-day arithmetic for the model-based backtest premium.

The backtest has no historical expiry lists, so the nearest expiry is derived from a rule per underlying.
THESE RULES ARE APPROXIMATIONS - exchanges change expiry days (NSE moved index expiries to Tuesday in
2025) and MCX option expiry dates are set per contract. Verify them and override per underlying with an
environment variable:

    EXPIRY_RULE_NIFTY=weekly:1        # weekly, weekday 0=Mon .. 4=Fri
    EXPIRY_RULE_BANKNIFTY=monthly:1   # last <weekday> of the month
    EXPIRY_RULE_CRUDEOIL=eom:5        # 5 trading days before the last trading day of the month

An expiry that falls on a holiday moves to the previous trading day. Time to expiry is counted in TRADING
days (weekends and holidays from `config/holidays.py` excluded), never calendar days.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

from quant_intelligence.config.holidays import is_trading_day
from quant_intelligence.utils.market_profile import MarketProfile, profile_for


@dataclass(frozen=True)
class ExpiryRule:
    kind: str  # "weekly" | "monthly" | "eom"
    param: int  # weekday (0=Mon) for weekly/monthly; trading days before month end for eom


_TUE_MONTHLY = ExpiryRule("monthly", 1)
DEFAULT_RULES: dict[str, ExpiryRule] = {
    "NIFTY": ExpiryRule("weekly", 1),
    "BANKNIFTY": _TUE_MONTHLY,
    "FINNIFTY": _TUE_MONTHLY,
    "MIDCPNIFTY": _TUE_MONTHLY,
    "SENSEX": ExpiryRule("weekly", 3),  # BSE: Thursday
}
DEFAULT_STOCK_RULE = _TUE_MONTHLY
DEFAULT_MCX_RULE = ExpiryRule("eom", 5)  # approximation; MCX sets option expiry per contract


def rule_for(symbol: str) -> ExpiryRule:
    sym = symbol.strip().upper()
    raw = os.getenv(f"EXPIRY_RULE_{sym}", "").strip().lower()
    if raw:
        try:
            kind, _, param = raw.partition(":")
            if kind in ("weekly", "monthly", "eom"):
                return ExpiryRule(kind, int(param))
        except ValueError:
            pass
    if sym in DEFAULT_RULES:
        return DEFAULT_RULES[sym]
    return DEFAULT_MCX_RULE if profile_for(sym).name == "MCX" else DEFAULT_STOCK_RULE


def _roll_back_to_trading_day(day: dt.date, market: str) -> dt.date:
    while not is_trading_day(day, market):
        day -= dt.timedelta(days=1)
    return day


def _last_weekday_of_month(year: int, month: int, weekday: int) -> dt.date:
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    day = nxt - dt.timedelta(days=1)
    return day - dt.timedelta(days=(day.weekday() - weekday) % 7)


def _last_trading_day_of_month(year: int, month: int, market: str) -> dt.date:
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return _roll_back_to_trading_day(nxt - dt.timedelta(days=1), market)


def _nth_trading_day_before(day: dt.date, n: int, market: str) -> dt.date:
    while n > 0:
        day -= dt.timedelta(days=1)
        if is_trading_day(day, market):
            n -= 1
    return day


def next_expiry(on: dt.date, symbol: str) -> dt.date:
    """Nearest expiry on or after `on` (an expiry today still counts) for `symbol`."""
    market = profile_for(symbol).name
    rule = rule_for(symbol)
    if rule.kind == "weekly":
        candidate = on + dt.timedelta(days=(rule.param - on.weekday()) % 7)
        while True:
            adjusted = _roll_back_to_trading_day(candidate, market)
            if adjusted >= on:
                return adjusted
            candidate += dt.timedelta(days=7)
    year, month = on.year, on.month
    for _ in range(3):
        if rule.kind == "monthly":
            adjusted = _roll_back_to_trading_day(_last_weekday_of_month(year, month, rule.param), market)
        else:
            adjusted = _nth_trading_day_before(_last_trading_day_of_month(year, month, market), rule.param, market)
        if adjusted >= on:
            return adjusted
        year, month = year + (month == 12), month % 12 + 1
    raise RuntimeError("could not find an expiry")  # unreachable for sane rules


def trading_days_between(start: dt.date, end: dt.date, market: str) -> int:
    """Trading days in (start, end]: sessions strictly after `start`, up to and including `end`."""
    count, day = 0, start
    while day < end:
        day += dt.timedelta(days=1)
        if is_trading_day(day, market):
            count += 1
    return count


def time_to_expiry_years(now: dt.datetime, expiry: dt.date, profile: MarketProfile, trading_days_per_year: int) -> float:
    """Remaining TRADING time to the expiry-day close, in years of trading time (naive IST `now`).

    = (rest of today's session + whole sessions after today up to and including the expiry day) / trading
    days per year. Overnight and weekend/holiday clock time is not counted - the volatility being used is
    annualised per trading day and already includes the overnight move. 0 once the expiry close has passed."""
    market = profile.name
    today = now.date()
    if expiry < today:
        return 0.0
    open_dt = dt.datetime.combine(today, profile.open)
    close_dt = dt.datetime.combine(today, profile.close)
    if not is_trading_day(today, market) or now >= close_dt:
        frac_today = 0.0
    elif now <= open_dt:
        frac_today = 1.0
    else:
        frac_today = (close_dt - now).total_seconds() / (close_dt - open_dt).total_seconds()
    days = frac_today + trading_days_between(today, expiry, market)
    return max(days, 0.0) / trading_days_per_year
