import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.strategies.registry import get_all_strategies
from quant_intelligence.ui.theme import apply_theme

apply_theme()
st.title("Strategy Library")
st.caption("Every strategy implements the same standard interface: generate_historical_setups, "
           "check_setup, generate_signal, calculate_entry_stop_target, simulate_trade.")

strategies = get_all_strategies()

for strategy in strategies:
    with st.expander(f"{strategy.name}", expanded=False):
        st.write(strategy.description)
        st.markdown("**Required features**")
        st.code(", ".join(strategy.required_features))
        st.markdown("**Parameters**")
        st.json(strategy.parameters)

st.divider()
st.info(
    "This MVP ships 3 of the planned 10 strategies (Opening Range Breakout, Momentum, VWAP Mean "
    "Reversion) per the phased development plan. Bollinger Mean Reversion, Volatility Breakout, "
    "Liquidity Sweep Reversal, Opening Range Continuation, Breakout + Volume Confirmation, and "
    "Intraday Mean Reversion can be added by subclassing BaseStrategy and registering in "
    "strategies/registry.py - no changes to the ranking engine, backtester, or UI are required."
)
