"""Shared Streamlit session-state helpers.

Keeps pipeline results, the PaperBroker instance, and account state in
st.session_state so every page sees a consistent snapshot without
recomputing the whole research pipeline on every rerun.
"""
from __future__ import annotations

import datetime as dt
from quant_intelligence.utils.timeutil import now_ist

import streamlit as st

from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.engine import ENGINE
from quant_intelligence.risk.risk_engine import AccountState


def init_session_state() -> None:
    if "instrument" not in st.session_state:
        st.session_state["instrument"] = SETTINGS.default_instrument
    if "timeframe" not in st.session_state:
        st.session_state["timeframe"] = SETTINGS.default_timeframe
    if "lookback_days" not in st.session_state:
        st.session_state["lookback_days"] = 60
    if "pipeline_output" not in st.session_state:
        st.session_state["pipeline_output"] = None
    if "pipeline_error" not in st.session_state:
        st.session_state["pipeline_error"] = None
    # The paper account, risk counters and kill switch are process-wide (ENGINE), not per
    # browser session, so the background auto-trader and every page see the same account.
    st.session_state["paper_broker"] = ENGINE.broker
    if "dhan_api_client" not in st.session_state:
        st.session_state["dhan_api_client"] = DhanApiClient()
    if "option_underlying" not in st.session_state:
        st.session_state["option_underlying"] = SETTINGS.option_underlyings[0] if SETTINGS.option_underlyings else "NIFTY"
    if "option_chain" not in st.session_state:
        st.session_state["option_chain"] = None
    render_engine_sidebar()


def render_engine_sidebar() -> None:
    """Visible on every page: shows the background auto-trader and lets you stop it."""
    status = ENGINE.status()
    with st.sidebar:
        if status["running"]:
            st.success("Auto paper trading is RUNNING (keeps running while you switch pages)")
            st.caption(
                f"Cycles: {status['cycles']} | Last: "
                f"{status['last_cycle_at']:%H:%M:%S} IST" if status["last_cycle_at"] else "No cycle completed yet"
            )
            st.caption(status["last_status"])
            if st.button("Stop auto trading", key="sidebar_stop_engine"):
                ENGINE.stop()
                st.rerun()


def get_account_state() -> AccountState:
    return ENGINE.account_state()


def get_live_account_state(broker) -> AccountState:
    """AccountState for the live Dhan broker, using real fund/position data instead of
    the PaperBroker's in-memory book. `broker` is a DhanBroker."""
    fund = broker.get_fund_limit()
    equity = fund.get("availabelBalance", 0.0)
    ENGINE.peak_equity = max(ENGINE.peak_equity, equity)

    open_positions = [p for p in broker.get_positions() if p.get("netQty", 0) != 0]
    exposure_by_strategy: dict[str, float] = {}
    total_exposure = 0.0
    for p in open_positions:
        risk_amt = abs(p.get("netQty", 0)) * abs(p.get("buyAvg", 0.0) - p.get("sellAvg", 0.0))
        exposure_by_strategy["dhan_live"] = exposure_by_strategy.get("dhan_live", 0.0) + risk_amt
        total_exposure += risk_amt

    return AccountState(
        equity=equity,
        peak_equity=ENGINE.peak_equity,
        daily_pnl=ENGINE.daily_pnl,
        open_positions_count=len(open_positions),
        trades_today=ENGINE.trades_today,
        exposure_by_strategy=exposure_by_strategy,
        total_exposure=total_exposure,
        broker_connected=broker.is_connected(),
        kill_switch_engaged=ENGINE.kill_switch,
    )


def run_pipeline_cached(force: bool = False):
    from quant_intelligence.research.pipeline import run_pipeline

    if st.session_state["pipeline_output"] is not None and not force:
        return st.session_state["pipeline_output"]

    end = now_ist()
    start = end - dt.timedelta(days=st.session_state["lookback_days"])

    try:
        output = run_pipeline(st.session_state["instrument"], st.session_state["timeframe"], start, end)
        st.session_state["pipeline_output"] = output
        st.session_state["pipeline_error"] = None
        return output
    except Exception as e:
        st.session_state["pipeline_error"] = str(e)
        st.session_state["pipeline_output"] = None
        return None
