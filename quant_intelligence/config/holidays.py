"""Exchange holidays, so a closed day is not mistaken for "the feed is stale" or "trade now".

Built in: the holidays that fall on the same calendar date every year - 26 Jan (Republic Day), 15 Aug
(Independence Day), 2 Oct (Gandhi Jayanti) and 25 Dec (Christmas) for NSE and MCX, plus 1 May
(Maharashtra Day) for NSE only (MCX still runs its evening session). Weekends are handled elsewhere.

Variable holidays (Holi, Diwali, Eid, Good Friday, ...) move every year and the exchanges publish them
annually. They are NOT guessed here: list them in `config/market_holidays.txt` (or the file named by
HOLIDAYS_FILE), one per line:

    2026-03-03 NSE Holi
    2026-11-08 Diwali Laxmi Pujan      <- no market name = both NSE and MCX
    2026-04-03 MCX Good Friday

Lines starting with # and blank lines are ignored; a malformed line is skipped, never fatal. A day missing
from the list is simply treated as a normal trading day, which only makes the staleness check lenient.
"""
from __future__ import annotations

import datetime as dt
import os
from functools import lru_cache
from pathlib import Path

_DEFAULT_FILE = Path(__file__).with_name("market_holidays.txt")

# (month, day) -> markets closed
_FIXED = {
    (1, 26): ("NSE", "MCX"),
    (5, 1): ("NSE",),
    (8, 15): ("NSE", "MCX"),
    (10, 2): ("NSE", "MCX"),
    (12, 25): ("NSE", "MCX"),
}
_MARKETS = ("NSE", "MCX")


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict[dt.date, tuple[str, ...]]:
    """Parse the holiday file. `mtime` is part of the cache key so edits are picked up without a restart."""
    out: dict[dt.date, tuple[str, ...]] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            day = dt.date.fromisoformat(parts[0])
        except ValueError:
            continue
        markets = (parts[1].upper(),) if len(parts) > 1 and parts[1].upper() in _MARKETS else _MARKETS
        out[day] = tuple(sorted(set(out.get(day, ())) | set(markets)))
    return out


def _file_holidays() -> dict[dt.date, tuple[str, ...]]:
    path = Path(os.getenv("HOLIDAYS_FILE") or _DEFAULT_FILE)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    return _load(str(path), mtime)


def is_holiday(day: dt.date, market: str = "NSE") -> bool:
    market = market.upper()
    return market in _FIXED.get((day.month, day.day), ()) or market in _file_holidays().get(day, ())


def is_trading_day(day: dt.date, market: str = "NSE") -> bool:
    """Weekday and not an exchange holiday for `market` ("NSE" or "MCX")."""
    return day.weekday() < 5 and not is_holiday(day, market)
