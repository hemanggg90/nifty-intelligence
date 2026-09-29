"""Streamlit panel: scan index + watchlist stocks together and paper-trade them, all as options.

Auto trading runs in a background thread owned by `ENGINE` (execution/engine.py), so it keeps
running when you switch pages or close the tab, until you press Stop.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.config.watchlist import WATCHLIST_STOCKS
from quant_intelligence.execution.capital import capital_summary
from quant_intelligence.execution.engine import ENGINE


def _show_capital() -> None:
    cap = capital_summary(ENGINE.broker)
    required_now = sum(
        r.get("capital_required") or 0 for r in ENGINE.last_rows if r.get("status") == "SETUP_TRIGGERED"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Account value", f"₹{cap['account_value']:,.0f}", help="Starting capital + realised P&L")
    c2.metric("Capital used (open positions)", f"₹{cap['capital_used']:,.0f}", help="Premium paid for open BUY positions")
    c3.metric("Available", f"₹{cap['available']:,.0f}")
    c4.metric(
        "Required by last scan's signals",
        f"₹{required_now:,.0f}",
        help="Premium x quantity for setups that triggered in the last cycle",
    )
    if cap["sell_premium_value"]:
        st.caption(
            f"Open SELL positions carry ₹{cap['sell_premium_value']:,.0f} of premium value; "
            "exchange margin is not modelled."
        )


@st.fragment(run_every="5s")
def _live_status() -> None:
    """Display only. Refreshing (or leaving) this page never affects the background worker."""
    status = ENGINE.status()
    if status["running"]:
        st.success(f"RUNNING since {status['started_at']:%Y-%m-%d %H:%M:%S} IST - {status['cycles']} cycle(s) completed")
    else:
        st.info("STOPPED")
    if status["last_cycle_at"]:
        st.caption(f"Last cycle: {status['last_cycle_at']:%Y-%m-%d %H:%M:%S} IST")
    st.caption(f"Status: {status['last_status']}")
    if status["last_error"]:
        st.warning(f"Last error: {status['last_error']}")

    _show_capital()
    if ENGINE.last_rows:
        st.dataframe(pd.DataFrame(ENGINE.last_rows), width="stretch", hide_index=True)
    else:
        st.caption("No cycle run yet.")


def render_multi_instrument_panel() -> None:
    index_symbols = list(SETTINGS.option_underlyings)  # e.g. NIFTY, BANKNIFTY
    stock_symbols = [r["symbol"] for r in WATCHLIST_STOCKS]

    universe_rows = [{"symbol": s, "name": s, "instrument_type": "INDEX options"} for s in index_symbols]
    universe_rows += [
        {"symbol": r["symbol"], "name": r["name"], "instrument_type": f"STOCK options - {r['sector']}"}
        for r in WATCHLIST_STOCKS
    ]
    st.caption(
        f"Watchlist: {len(index_symbols)} index instrument(s) + {len(stock_symbols)} stock(s) "
        f"= {len(universe_rows)} total (edit in config/watchlist.py)."
    )
    with st.expander("Show watchlist"):
        st.dataframe(pd.DataFrame(universe_rows), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Automation")
    st.caption(
        "Every trade is an OPTION (index and stock options alike). Each cycle re-runs the strategy pipeline "
        "and fetches a live option chain per instrument, which is compute-heavy - keep the watchlist short. "
        "Auto trading runs on the server in the background: it keeps going when you change pages or close "
        "the tab, until you press Stop (or the server restarts / the cloud app goes to sleep)."
    )

    timeframe = st.session_state.get("timeframe", SETTINGS.default_timeframe)
    lookback_days = st.session_state.get("lookback_days", 60)
    status = ENGINE.status()

    market_only = st.checkbox(
        "Only trade during market hours (Mon-Fri 09:15-15:30 IST)",
        value=ENGINE.market_hours_only,
        disabled=status["running"],
    )
    b1, b2, _ = st.columns([1, 1, 2])
    if not status["running"]:
        if b1.button("Start auto trading", type="primary"):
            ENGINE.start(index_symbols, stock_symbols, timeframe, lookback_days, market_hours_only=market_only)
            st.rerun()
    else:
        if b1.button("Stop auto trading", type="primary"):
            ENGINE.stop()
            st.rerun()
    if b2.button("Run one cycle now", disabled=status["running"]):
        with st.spinner("Scanning all instruments..."):
            ENGINE.run_cycle(index_symbols, stock_symbols, timeframe, lookback_days)

    _live_status()

    st.divider()
    st.subheader("Open positions (all pages share the same paper broker)")
    open_positions = ENGINE.broker.get_open_positions()
    if open_positions:
        st.dataframe(open_positions, width="stretch", hide_index=True)
    else:
        st.caption("No open paper positions.")
