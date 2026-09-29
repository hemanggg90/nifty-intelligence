"""Commodity Auto Trading (MCX).

Crude oil, natural gas, gold, silver and copper are pipelined, ranked and automated together,
and every trade is an option on the front-month future. Runs on its own background runner with
its own start/stop and MCX session hours (09:00-23:55 IST), independently of - and at the same
time as - the NSE runner on the Auto Multi-Instrument page. Both share ONE PaperBroker, so
capital, open positions and the kill switch are common.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.execution.engine import COMMODITY_RUNNER
from quant_intelligence.ui.multi_instrument_panel import render_multi_instrument_panel
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Commodity Auto Trading (MCX)")
st.caption(
    "MCX COMMODITIES, ALL traded as options -> PER-INSTRUMENT PIPELINE -> RANKING -> RISK ENGINE -> "
    "PAPER BROKER (simulated). Independent of the NSE auto trader; shares its paper capital."
)
st.warning(
    "Lot sizes are MCX contract multipliers from config/watchlist.py (Dhan's scrip master reports 1 for "
    "every MCX contract). Verify them against the exchange before trusting P&L figures."
)
render_multi_instrument_panel(COMMODITY_RUNNER, commodities=True)
