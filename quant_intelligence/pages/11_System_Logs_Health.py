import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.brokers.dhan_rate_limit import LIMITER
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data.data_keeper import DATA_KEEPER
from quant_intelligence.data_adapters.dhan_adapter import DhanAdapter
from quant_intelligence.database.db import get_engine, get_session
from quant_intelligence.database.models import MarketDataMetadata, SystemEvent
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("System Logs & Health")

col1, col2, col3, col4 = st.columns(4)

db_ok = True
try:
    get_engine().connect().close()
except Exception:
    db_ok = False
col1.metric("Database", "OK" if db_ok else "FAIL")

dhan = DhanAdapter()
col2.metric("Dhan Data API", "CONFIGURED" if dhan.is_available() else "NOT CONFIGURED (no market data without CSV)")

broker = st.session_state["paper_broker"]
col3.metric("Broker (Paper)", "CONNECTED" if broker.is_connected() else "DISCONNECTED")

col4.metric("Trading Mode", SETTINGS.trading_mode)

if SETTINGS.is_live_mode and not SETTINGS.live_mode_fully_authorized:
    st.warning(
        "TRADING_MODE=LIVE is set but TRADING_LIVE_CONFIRM is missing/incorrect. "
        "LIVE order placement remains blocked by design."
    )

st.divider()
st.subheader("Dhan API usage")
snap = LIMITER.snapshot()
if snap["auth_block_remaining"] > 0:
    st.error("Dhan rejected the access token (401): requests are paused until you enter a new one.")
elif snap["cooldown_remaining"] > 0:
    st.warning(f"Rate limit reached: data requests paused for another {snap['cooldown_remaining']:.0f}s (auto-resumes).")
else:
    st.success("Within Dhan's limits - no pause active.")
st.caption(
    f"Calls in the last {snap['window_sec']:.0f}s by Dhan API category, against the minimum spacing this app enforces "
    "(set below Dhan's published account-wide limits). The limit applies to your whole Dhan account - another app "
    "copy, browser tab or script using it counts too."
)
st.dataframe(
    [
        {
            "Category": name,
            "Calls (last 60 s)": c["calls_last_window"],
            "Min spacing (s)": c["min_interval_sec"],
            "HTTP 429 since start": c["total_429"],
        }
        for name, c in snap["categories"].items()
    ],
    width="stretch",
    hide_index=True,
)

st.divider()
st.subheader("Data feed (background refresh)")
feed = DATA_KEEPER.status()
if not feed["running"]:
    st.info("The data keeper is not running (it starts with the first page load).")
else:
    st.caption(
        f"Refreshes every {DATA_KEEPER.interval}s \u00b7 {feed['ok']} of {feed['tracked']} instruments current \u00b7 "
        f"{feed['rounds']} rounds \u00b7 last round {feed['last_round_at']:%H:%M:%S} IST"
        if feed["last_round_at"] else "Starting - first round in progress."
    )
    if feed["error"]:
        st.warning(f"Data feed problem: {feed['error']}")
    if DATA_KEEPER.instruments:
        st.dataframe(
            [{"Instrument": s, "Last bar": v["last_bar"], "Quality": v["quality"], "Detail": "; ".join(v["issues"][:2])}
             for s, v in sorted(DATA_KEEPER.instruments.items())],
            width="stretch", hide_index=True,
        )
if st.button("Refresh candles now", key="keeper_wake"):
    DATA_KEEPER.wake()
    st.toast("Refresh requested")

st.divider()
st.subheader("Latest data fetches")
with get_session() as session:
    meta = session.query(MarketDataMetadata).order_by(MarketDataMetadata.id.desc()).limit(20).all()
rows = [
    {
        "Instrument": m.instrument,
        "Timeframe": m.timeframe,
        "Source": m.source,
        "Rows": m.n_rows,
        "Quality": m.quality_status,
        "Fetched At": m.fetched_at,
    }
    for m in meta
]
st.dataframe(rows, width="stretch", hide_index=True)

st.divider()
st.subheader("Recent system events")
level_filter = st.multiselect("Filter by level", ["INFO", "WARNING", "ERROR"], default=["WARNING", "ERROR"])
with get_session() as session:
    query = session.query(SystemEvent).order_by(SystemEvent.id.desc())
    if level_filter:
        query = query.filter(SystemEvent.level.in_(level_filter))
    events = query.limit(200).all()
rows = [{"Timestamp": e.timestamp, "Component": e.component, "Level": e.level, "Message": e.message} for e in events]
st.dataframe(rows, width="stretch", hide_index=True, height=400)
