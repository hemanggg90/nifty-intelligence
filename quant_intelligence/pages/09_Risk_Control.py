import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import RiskEvent
from quant_intelligence.ui.state import get_account_state, init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Risk Control")
st.caption("Deterministic risk engine - independent of the research/AI layer, with VETO authority.")

account = get_account_state()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Equity", f"₹{account.equity:,.0f}")
col2.metric("Daily P&L", f"₹{account.daily_pnl:,.0f}")
col3.metric("Open Positions", account.open_positions_count)
col4.metric("Trades Today", account.trades_today)

st.subheader("Configured limits")
limits = SETTINGS.risk
st.json(
    {
        "max_risk_per_trade_pct": limits.max_risk_per_trade_pct,
        "max_daily_loss_pct": limits.max_daily_loss_pct,
        "max_strategy_exposure_pct": limits.max_strategy_exposure_pct,
        "max_portfolio_exposure_pct": limits.max_portfolio_exposure_pct,
        "max_trades_per_day": limits.max_trades_per_day,
        "max_drawdown_pct": limits.max_drawdown_pct,
    }
)

st.divider()
st.subheader("Emergency kill switch")
kill = st.toggle("Engage kill switch (blocks ALL new trades immediately)", value=st.session_state["kill_switch_engaged"])
st.session_state["kill_switch_engaged"] = kill
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
