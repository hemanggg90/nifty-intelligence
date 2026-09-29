"""End-of-session square-off for paper positions.

The paper broker has no notion of a session: an INTRADAY tag is just a label, stops/targets are
only checked while a runner is scanning, and after the close the last price is frozen. Without
this a position opened today would sit open overnight. Each runner therefore closes the open
positions of ITS market (NSE or MCX) shortly before that market's close, at the latest traded
price, and again on the first cycle after the next open if it was not running at the time.
"""
from __future__ import annotations

import datetime as dt
from typing import Callable

from quant_intelligence.execution.position_monitor import _fetch_quotes
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_profile import MarketProfile, profile_for

EOD_REASON = "EOD_SQUARE_OFF"


def in_close_window(now: dt.datetime, profile: MarketProfile, minutes_before_close: int) -> bool:
    """True during the last `minutes_before_close` minutes of a trading day's session. Positions are
    squared off in this window, so no new entries should be taken in it either, or they would be
    closed straight away. (Outside the session altogether, is_market_open is False instead.)"""
    if now.weekday() >= 5:
        return False
    cutoff = (dt.datetime.combine(now.date(), profile.close) - dt.timedelta(minutes=minutes_before_close)).time()
    return cutoff <= now.time() < profile.close


def positions_for_profile(broker, profile: MarketProfile) -> list[dict]:
    """Open option positions belonging to this market. Positions with no security id cannot be
    priced, so they are left alone (they are not created by the option auto-traders)."""
    return [
        p for p in broker.get_open_positions()
        if p.get("option_type") and p.get("security_id")
        and profile_for(p.get("underlying") or p.get("instrument")).name == profile.name
    ]


def square_off_positions(
    broker, client, profile: MarketProfile, lock, on_pnl: Callable[[float], None], reason: str = EOD_REASON,
) -> list[dict]:
    """Close every open position of `profile`'s market at its latest traded price.

    A position whose live price cannot be fetched is left open (and logged) rather than closed at
    a made-up price; the next cycle retries. `lock` is the broker lock, held only around reading
    and closing positions - never during the network call.
    """
    with lock:
        targets = positions_for_profile(broker, profile)
    if not targets:
        return []
    if not client.is_configured():
        log_event("square_off", "Cannot square off: Dhan credentials are not set", level="WARNING")
        return []
    try:
        quotes = _fetch_quotes(client, targets)
    except Exception as e:
        log_event("square_off", f"Square-off price fetch failed, will retry: {e}", level="WARNING")
        return []

    closed: list[dict] = []
    with lock:
        for pos in targets:
            price = (quotes.get(str(pos["security_id"])) or {}).get("last_price")
            if price is None:
                log_event("square_off", f"No live price for {pos['position_id']}; left open", level="WARNING")
                continue
            done = broker.close_position(pos["position_id"], price, reason)
            if done is not None:
                on_pnl(done["net_pnl"])
                closed.append(done)
                log_event(
                    "square_off",
                    f"{reason}: closed {done['instrument']} {done.get('option_type')} {done.get('strike')} "
                    f"@ {price} (P&L {done['net_pnl']:.2f})",
                    level="INFO",
                )
    return closed
