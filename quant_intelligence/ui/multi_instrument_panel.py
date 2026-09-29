"""Streamlit panel: scan index + all stocks together and paper-trade them in one pass."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.config.watchlist import WATCHLIST_STOCKS
from quant_intelligence.execution.multi_cycle import run_multi_instrument_cycle
from quant_intelligence.ui.state import get_account_state


def render_multi_instrument_panel() -> None:
    index_symbols = list(SETTINGS.option_underlyings)  # e.g. NIFTY, BANKNIFTY - options, not cash-market
    equity_symbols = [r["symbol"] for r in WATCHLIST_STOCKS]

    universe_rows = [{"symbol": s, "name": s, "instrument_type": "INDEX (options)"} for s in index_symbols]
    universe_rows += [
        {"symbol": r["symbol"], "name": r["name"], "instrument_type": f"EQUITY (cash) - {r['sector']}"}
        for r in WATCHLIST_STOCKS
    ]
    st.caption(
        f"Watchlist: {len(index_symbols)} index instrument(s) + {len(equity_symbols)} stock(s) "
        f"= {len(universe_rows)} total (edit in config/watchlist.py)."
    )
    st.dataframe(pd.DataFrame(universe_rows), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Automation")
    st.caption(
        "Off by default. Each cycle re-runs the full strategy pipeline per instrument (index + stocks "
        "together), which is compute-heavy and makes one Dhan data request per instrument - keep the "
        "watchlist short for a responsive refresh cadence."
    )
    automate = st.checkbox("Automate (auto-select strategy + auto-place PAPER orders on setup trigger)", value=False)

    timeframe = st.session_state.get("timeframe", SETTINGS.default_timeframe)
    lookback_days = st.session_state.get("lookback_days", 60)
    broker = st.session_state["paper_broker"]
    client = st.session_state["dhan_api_client"]

    def _run_cycle() -> None:
        rows, pnl_delta = run_multi_instrument_cycle(
            index_symbols, equity_symbols, timeframe, lookback_days, broker, client, get_account_state
        )
        st.session_state["daily_pnl"] += pnl_delta
        st.session_state["auto_multi_stock_last_run"] = rows

    if automate:

        @st.fragment(run_every=f"{SETTINGS.auto_trade_refresh_seconds}s")
        def _auto_fragment():
            _run_cycle()
            st.caption(f"Last automated cycle: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
            rows = st.session_state.get("auto_multi_stock_last_run", [])
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        _auto_fragment()
    else:
        if st.button("Run one cycle now"):
            with st.spinner("Scanning all instruments..."):
                _run_cycle()
        rows = st.session_state.get("auto_multi_stock_last_run", [])
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("No cycle run yet.")

    st.divider()
    st.subheader("Open positions (all pages share the same paper broker)")
    open_positions = broker.get_open_positions()
    if open_positions:
        st.dataframe(open_positions, width="stretch", hide_index=True)
    else:
        st.caption("No open paper positions.")
