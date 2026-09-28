"""Auto Multi-Instrument Trading.

Single unified scanner: NIFTY/BANKNIFTY (traded via options, since indices have
no cash-market instrument) and a fixed ~200-stock NSE equity universe (traded
direct, cash-market) are pipelined, ranked, and automated together in one pass -
instead of index options (07_Paper_Trading/12_Live_Options_Trading) and stocks
(this page, formerly stocks-only) being separate, disconnected flows. Every
instrument shares one combined ranked table and one PaperBroker/RiskEngine.
"""
import sys
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data_adapters.nse_heatmap import load_static_universe
from quant_intelligence.execution.auto_trader import run_auto_equity_cycle, run_auto_option_cycle
from quant_intelligence.execution.position_monitor import monitor_option_positions, monitor_positions
from quant_intelligence.options.option_selector import fetch_chain
from quant_intelligence.research.pipeline import run_pipeline
from quant_intelligence.ui.state import get_account_state, init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Auto Multi-Instrument Trading")
st.caption(
    "ONE UNIVERSE: index (NIFTY/BANKNIFTY, via options) + a fixed NIFTY-50 stock list (cash-market) -> "
    "PER-INSTRUMENT PIPELINE -> RANKING -> RISK ENGINE -> PAPER BROKER (simulated)."
)

# --- Universe -----------------------------------------------------------
top_n = st.number_input(
    "Equity universe size", min_value=1, max_value=250, value=min(SETTINGS.heatmap_universe_top_n, 250), step=1
)

static_universe = load_static_universe(top_n=int(top_n))
st.caption("Equity universe source: **fixed ~200-stock NSE list** (no live NSE fetch, no mover-based ranking).")

index_symbols = list(SETTINGS.option_underlyings)  # e.g. NIFTY, BANKNIFTY - options, not cash-market
equity_constituents = static_universe["constituents"]

# One combined universe table: indices first, then the equity list.
universe_rows = [{"symbol": s, "instrument_type": "INDEX (options)"} for s in index_symbols]
universe_rows += [{"symbol": r["symbol"], "instrument_type": "EQUITY (cash)"} for r in equity_constituents]
df_universe = pd.DataFrame(universe_rows)

st.caption(f"Combined universe: {len(index_symbols)} index instrument(s) + {len(equity_constituents)} stock(s) = {len(df_universe)} total.")
st.dataframe(df_universe, width="stretch", hide_index=True)

st.divider()

# --- Automation -----------------------------------------------------------
st.subheader("Automation")
st.caption(
    "Off by default. Each cycle re-runs the full 17-strategy pipeline per instrument (index + stocks "
    "together), which is compute-heavy - keep the universe size modest for a responsive refresh cadence."
)
automate = st.checkbox("Automate (auto-select strategy + auto-place PAPER orders on setup trigger)", value=False)
timeframe = st.session_state.get("timeframe", SETTINGS.default_timeframe)
lookback_days = st.session_state.get("lookback_days", 60)

broker = st.session_state["paper_broker"]
client = st.session_state["dhan_api_client"]


def _run_cycle():
    account = get_account_state()
    end = dt.datetime.now()
    start = end - dt.timedelta(days=lookback_days)
    rows = []

    all_symbols = [(s, "index") for s in index_symbols] + [(r["symbol"], "equity") for r in equity_constituents]

    for symbol, kind in all_symbols:
        try:
            output = run_pipeline(symbol, timeframe, start, end)
        except Exception as e:
            rows.append({"symbol": symbol, "type": kind, "strategy": "-", "status": "DATA_ERROR", "detail": str(e)})
            continue

        if kind == "index":
            chain = None
            if client.is_configured():
                try:
                    chain = fetch_chain(client, symbol, SETTINGS.option_expiry_preference)
                except Exception as e:
                    rows.append(
                        {
                            "symbol": symbol,
                            "type": kind,
                            "strategy": output.ranking.selected_strategy or "-",
                            "status": "CHAIN_ERROR",
                            "detail": str(e),
                        }
                    )
                    continue
            cycle = run_auto_option_cycle(output, symbol, broker, client, chain, account)
        else:
            cycle = run_auto_equity_cycle(output, broker, account)

        if cycle.order is not None:
            account = get_account_state()  # re-sync exposure/trade-count after a fill
        rows.append(
            {
                "symbol": symbol,
                "type": kind,
                "strategy": cycle.strategy_name or "-",
                "status": cycle.setup_status or ("NO_TRADE" if output.ranking.is_no_trade else "-"),
                "detail": cycle.reason,
            }
        )
        if cycle.order is not None and cycle.order.status == "FILLED":
            latest_bar = output.ohlcv.iloc[-1]
            events = monitor_positions(broker, latest_bar)
            for e in events:
                st.session_state["daily_pnl"] += e["position"]["net_pnl"]
            if kind == "index" and client.is_configured():
                option_events = monitor_option_positions(broker, client)
                for e in option_events:
                    st.session_state["daily_pnl"] += e["position"]["net_pnl"]
    st.session_state["auto_multi_stock_last_run"] = rows


if automate:

    @st.fragment(run_every=f"{SETTINGS.auto_trade_refresh_seconds}s")
    def _auto_fragment():
        _run_cycle()
        st.caption(f"Last automated cycle: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
        rows = st.session_state.get("auto_multi_stock_last_run", [])
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    _auto_fragment()
else:
    if st.button("Run one cycle now"):
        _run_cycle()
    rows = st.session_state.get("auto_multi_stock_last_run", [])
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("No cycle run yet.")

st.divider()
st.subheader("Open positions (all pages share the same paper broker)")
open_positions = broker.get_open_positions()
if open_positions:
    st.dataframe(open_positions, width="stretch", hide_index=True)
else:
    st.caption("No open paper positions.")
