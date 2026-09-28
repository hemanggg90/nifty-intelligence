"""Shared Streamlit session-state helpers.

Keeps pipeline results, the PaperBroker instance, and account state in
st.session_state so every page sees a consistent snapshot without
recomputing the whole research pipeline on every rerun.
"""
from __future__ import annotations

import datetime as dt

import streamlit as st

from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.config.settings import SETTINGS
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
    if "paper_broker" not in st.session_state:
        st.session_state["paper_broker"] = PaperBroker()
    if "kill_switch_engaged" not in st.session_state:
        st.session_state["kill_switch_engaged"] = False
    if "trades_today" not in st.session_state:
        st.session_state["trades_today"] = 0
    if "daily_pnl" not in st.session_state:
        st.session_state["daily_pnl"] = 0.0
    if "peak_equity" not in st.session_state:
        st.session_state["peak_equity"] = SETTINGS.paper_starting_capital
    if "dhan_api_client" not in st.session_state:
        st.session_state["dhan_api_client"] = DhanApiClient()
    if "option_underlying" not in st.session_state:
        st.session_state["option_underlying"] = SETTINGS.option_underlyings[0] if SETTINGS.option_underlyings else "NIFTY"
    if "option_chain" not in st.session_state:
        st.session_state["option_chain"] = None


def get_account_state() -> AccountState:
    broker: PaperBroker = st.session_state["paper_broker"]
    open_positions = broker.get_open_positions()
    exposure_by_strategy: dict[str, float] = {}
    total_exposure = 0.0
    for p in open_positions:
        risk_amt = abs(p["entry_price"] - (p["stop_price"] or p["entry_price"])) * p["quantity"]
        exposure_by_strategy[p["strategy_name"]] = exposure_by_strategy.get(p["strategy_name"], 0.0) + risk_amt
        total_exposure += risk_amt

    equity = broker.cash
    st.session_state["peak_equity"] = max(st.session_state["peak_equity"], equity)

    return AccountState(
        equity=equity,
        peak_equity=st.session_state["peak_equity"],
        daily_pnl=st.session_state["daily_pnl"],
        open_positions_count=len(open_positions),
        trades_today=st.session_state["trades_today"],
        exposure_by_strategy=exposure_by_strategy,
        total_exposure=total_exposure,
        broker_connected=broker.is_connected(),
        kill_switch_engaged=st.session_state["kill_switch_engaged"],
    )


def get_live_account_state(broker) -> AccountState:
    """AccountState for the live Dhan broker, using real fund/position data instead of
    the PaperBroker's in-memory book. `broker` is a DhanBroker."""
    fund = broker.get_fund_limit()
    equity = fund.get("availabelBalance", 0.0)
    st.session_state["peak_equity"] = max(st.session_state.get("peak_equity", equity), equity)

    open_positions = [p for p in broker.get_positions() if p.get("netQty", 0) != 0]
    exposure_by_strategy: dict[str, float] = {}
    total_exposure = 0.0
    for p in open_positions:
        risk_amt = abs(p.get("netQty", 0)) * abs(p.get("buyAvg", 0.0) - p.get("sellAvg", 0.0))
        exposure_by_strategy["dhan_live"] = exposure_by_strategy.get("dhan_live", 0.0) + risk_amt
        total_exposure += risk_amt

    return AccountState(
        equity=equity,
        peak_equity=st.session_state["peak_equity"],
        daily_pnl=st.session_state.get("daily_pnl", 0.0),
        open_positions_count=len(open_positions),
        trades_today=st.session_state.get("trades_today", 0),
        exposure_by_strategy=exposure_by_strategy,
        total_exposure=total_exposure,
        broker_connected=broker.is_connected(),
        kill_switch_engaged=st.session_state.get("kill_switch_engaged", False),
    )


def run_pipeline_cached(force: bool = False):
    from quant_intelligence.research.pipeline import run_pipeline

    if st.session_state["pipeline_output"] is not None and not force:
        return st.session_state["pipeline_output"]

    end = dt.datetime.now()
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
