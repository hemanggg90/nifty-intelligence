"""Auto Multi-Instrument Trading.

Single unified scanner: NIFTY/BANKNIFTY (traded via options, since indices have
no cash-market instrument) and a fixed ~200-stock NSE equity universe (traded
direct, cash-market) are pipelined, ranked, and automated together in one pass -
instead of index options (07_Paper_Trading/12_Live_Options_Trading) and stocks
(this page, formerly stocks-only) being separate, disconnected flows. Every
instrument shares one combined ranked table and one PaperBroker/RiskEngine.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.ui.multi_instrument_panel import render_multi_instrument_panel
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Auto Multi-Instrument Trading")
st.caption(
    "ONE UNIVERSE: index (NIFTY/BANKNIFTY, via options) + the fixed stock watchlist (cash-market) -> "
    "PER-INSTRUMENT PIPELINE -> RANKING -> RISK ENGINE -> PAPER BROKER (simulated)."
)
render_multi_instrument_panel()
