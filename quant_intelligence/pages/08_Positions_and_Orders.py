import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import Fill, Order, Position
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Positions & Orders")
st.caption("Full audit trail: every order, fill, and position with a unique ID.")

with get_session() as session:
    orders = session.query(Order).order_by(Order.id.desc()).limit(200).all()
    fills = session.query(Fill).order_by(Fill.id.desc()).limit(200).all()
    positions = session.query(Position).order_by(Position.id.desc()).limit(200).all()

tab1, tab2, tab3 = st.tabs(["Positions", "Orders", "Fills"])

with tab1:
    rows = [
        {
            "Position ID": p.position_id,
            "Instrument": p.instrument,
            "Strategy": p.strategy_name,
            "Direction": p.direction,
            "Option": p.option_type,
            "Strike": p.strike,
            "Expiry": p.expiry,
            "Transaction": p.transaction,
            "Qty": p.quantity,
            "Entry": p.entry_price,
            "Amount Used": round(p.entry_price * p.quantity, 2) if p.entry_price and p.quantity else None,
            "Stop": p.stop_price,
            "Target": p.target_price,
            "Status": p.status,
            "Exit": p.exit_price,
            "Net P&L": p.net_pnl,
            "Opened": p.opened_at,
            "Closed": p.closed_at,
        }
        for p in positions
    ]
    open_used = sum(r["Amount Used"] or 0 for r, p in zip(rows, positions) if p.status == "OPEN")
    st.metric("Amount currently deployed (open positions)", f"₹{open_used:,.0f}")
    st.dataframe(rows, width="stretch", hide_index=True)
    if rows:
        st.download_button(
            "Download positions (CSV)",
            data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
            file_name=f"positions_{dt.date.today():%Y%m%d}.csv",
            mime="text/csv",
            key="download_positions_csv",
        )

with tab2:
    rows = [
        {
            "Order ID": o.order_id,
            "Decision ID": o.decision_id,
            "Timestamp": o.timestamp,
            "Instrument": o.instrument,
            "Strategy": o.strategy_name,
            "Direction": o.direction,
            "Qty": o.quantity,
            "Type": o.order_type,
            "Status": o.status,
            "Reject Reason": o.reject_reason,
            "Mode": o.mode,
            "Broker": o.broker,
            "Option": o.option_type,
            "Strike": o.strike,
            "Expiry": o.expiry,
        }
        for o in orders
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    if rows:
        st.download_button(
            "Download orders (CSV)",
            data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
            file_name=f"orders_{dt.date.today():%Y%m%d}.csv",
            mime="text/csv",
            key="download_orders_csv",
        )

with tab3:
    rows = [
        {"Order ID": f.order_id, "Timestamp": f.timestamp, "Fill Price": f.fill_price, "Qty": f.quantity, "Slippage": f.slippage}
        for f in fills
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    if rows:
        st.download_button(
            "Download fills (CSV)",
            data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
            file_name=f"fills_{dt.date.today():%Y%m%d}.csv",
            mime="text/csv",
            key="download_fills_csv",
        )
