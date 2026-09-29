"""Live open-positions block: current premium and unrealised P&L, refreshed every few seconds.

Shared by the Positions page and the NSE / commodity auto-trading panels. Reads the process-wide
paper broker (ENGINE.broker), so it shows the same book whichever runner opened the position.
Display only - it never places or closes anything.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.engine import ENGINE
from quant_intelligence.execution.position_monitor import fetch_option_ltp_map, positions_with_live_ltp

_COLUMNS = [
    "instrument", "strategy_name", "option_type", "strike", "expiry", "transaction", "quantity",
    "entry_price", "current_ltp", "unrealized_pnl", "stop_price", "target_price",
]


@st.fragment(run_every=f"{SETTINGS.ltp_refresh_seconds}s")
def render_live_positions() -> None:
    broker = ENGINE.broker
    positions = broker.get_open_positions()

    ltp_map: dict = {}
    ltp_error = None
    client = DhanApiClient()
    if positions and client.is_configured():
        try:
            ltp_map = fetch_option_ltp_map(broker, client)
        except Exception as e:  # a failed quote fetch must not blank the page
            ltp_error = str(e)
    rows = positions_with_live_ltp(broker, ltp_map)

    unrealized = sum(r["unrealized_pnl"] for r in rows if r["unrealized_pnl"] is not None)
    realised_today = ENGINE.daily_pnl  # net of costs, closed positions since IST midnight
    c1, c2, c3 = st.columns(3)
    c1.metric("Unrealised P&L (live)", f"₹{unrealized:,.0f}", help="Open positions marked to the live premium, before exit costs")
    c2.metric("Realised today (net)", f"₹{realised_today:,.0f}", help="Closed positions today, after costs")
    c3.metric("Net P&L today", f"₹{realised_today + unrealized:,.0f}", help="Realised (net) + unrealised (live)")

    if not rows:
        st.caption("No open paper positions.")
        return
    if ltp_error:
        st.warning(f"Live prices unavailable: {ltp_error}")
    elif not client.is_configured():
        st.caption("Set your Dhan credentials to see live P&L.")
    elif any(r["current_ltp"] is None for r in rows):
        st.caption("Some positions have no live quote yet; their P&L shows as empty.")

    df = pd.DataFrame(rows)
    st.dataframe(df[[c for c in _COLUMNS if c in df.columns]], width="stretch", hide_index=True)
