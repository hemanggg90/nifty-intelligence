"""Trading-session definitions per exchange.

Session hours used to be NSE constants scattered across data alignment, quality checks,
staleness and the feature engine. They live here now so commodities (MCX) can use a
different session. Every consumer takes an optional `profile` and defaults to NSE, so
existing NSE behaviour is unchanged.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketProfile:
    name: str
    open: dt.time
    close: dt.time  # exclusive
    option_segment: str  # Dhan exchange segment for option orders / LTP
    open_label: str = ""

    @property
    def open_min(self) -> int:
        return self.open.hour * 60 + self.open.minute

    @property
    def close_min(self) -> int:
        return self.close.hour * 60 + self.close.minute

    @property
    def session_minutes(self) -> int:
        return self.close_min - self.open_min

    @property
    def label(self) -> str:
        return f"{self.open:%H:%M}-{self.close:%H:%M} IST"


NSE = MarketProfile("NSE", dt.time(9, 15), dt.time(15, 30), option_segment="NSE_FNO")
# MCX trades 09:00-23:30 IST, extended to 23:55 while US daylight saving is in force. The wider
# window is used so those extended-hours bars are not discarded as strays.
MCX = MarketProfile("MCX", dt.time(9, 0), dt.time(23, 55), option_segment="MCX_COMM")


def profile_for(symbol: str | None) -> MarketProfile:
    """MCX for the supported commodity underlyings, NSE for everything else."""
    from quant_intelligence.config.watchlist import COMMODITY_SYMBOLS

    if symbol and symbol.strip().upper() in COMMODITY_SYMBOLS:
        return MCX
    return NSE
