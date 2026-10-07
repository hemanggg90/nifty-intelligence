import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.strategies.registry import get_all_strategies
from quant_intelligence.ui.theme import apply_theme

apply_theme()
st.title("Strategy Library")
st.caption("Every strategy implements the same standard interface: generate_historical_setups, "
           "check_setup, generate_signal, calculate_entry_stop_target, simulate_trade.")

strategies = get_all_strategies()

if SETTINGS.stop_atr_scale != 1.0:
    st.warning(
        f"STOP_ATR_SCALE = {SETTINGS.stop_atr_scale:g}: every ATR-based strategy's default stop and target distance is "
        f"multiplied by {SETTINGS.stop_atr_scale:g} (the parameters below already include it). Backtests, the ranking and live "
        "setups all use these scaled values."
    )
else:
    st.caption("Stop and target distances are as designed (STOP_ATR_SCALE = 1.0). Set STOP_ATR_SCALE below 1 to tighten them all.")

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
