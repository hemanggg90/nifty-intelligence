import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import RiskEvent
from quant_intelligence.execution.engine import ENGINE
from quant_intelligence.risk.risk_engine import drawdown_pct, loss_streak, risk_multiplier
from quant_intelligence.ui.state import get_account_state, init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Risk Control")
st.caption("Deterministic risk engine - independent of the research/AI layer, with VETO authority.")

account = get_account_state()
limits = SETTINGS.risk
multiplier, why = risk_multiplier(account)
dd = drawdown_pct(account)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Equity (marked to market)", f"₹{account.equity:,.0f}",
            help=f"Cash + unrealised P&L of open positions (unrealised: ₹{account.unrealised_pnl:,.0f})")
col2.metric("Daily P&L", f"₹{account.daily_pnl:,.0f}", help="Realised today + unrealised")
col3.metric("Open Positions", f"{account.open_positions_count} / {limits.max_open_positions or '∞'}")
col4.metric("Trades Today", f"{account.trades_today} / {limits.max_trades_per_day}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Drawdown", f"{dd:.2f}%", help=f"From peak equity, across days. Hard stop at {limits.max_drawdown_pct:g}%.")
col6.metric("Drawdown peak", f"₹{account.peak_equity:,.0f}")
col7.metric("Position size now", f"{multiplier:.0%}", help=why)
col8.metric("Losing streak today", loss_streak(account))
if multiplier < 1:
    st.warning(f"New trades are sized at {multiplier:.0%} of normal: {why}.")

st.subheader("Configured limits")
st.json(dataclasses.asdict(limits))

with st.expander("Reset drawdown peak"):
    st.caption(
        "Starts a fresh drawdown peak at the current equity, so the drawdown throttle and the "
        f"{limits.max_drawdown_pct:g}% hard stop measure from here. Use it only on purpose, e.g. after "
        "reviewing a losing stretch or adding capital."
    )
    confirm_reset = st.checkbox("I understand this clears the current drawdown", key="confirm_peak_reset")
    if st.button("Reset peak to current equity", disabled=not confirm_reset):
        new_peak = ENGINE.reset_drawdown_peak()
        st.success(f"Drawdown peak reset to ₹{new_peak:,.0f}")
        st.rerun()

st.divider()
st.subheader("Emergency kill switch")
kill = st.toggle("Engage kill switch (blocks ALL new trades immediately)", value=ENGINE.kill_switch)
ENGINE.kill_switch = kill
if kill:
    st.error("KILL SWITCH ENGAGED. No new trades will be approved by the risk engine.")

st.divider()
st.subheader("Recent risk events (audit trail)")
with get_session() as session:
    events = session.query(RiskEvent).order_by(RiskEvent.id.desc()).limit(100).all()
rows = [
    {"Timestamp": e.timestamp, "Decision ID": e.decision_id, "Type": e.event_type, "Reason": e.reason}
    for e in events
]
st.dataframe(rows, width="stretch", hide_index=True)
