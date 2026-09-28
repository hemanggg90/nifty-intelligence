import sys
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.auto_trader import run_auto_option_cycle
from quant_intelligence.execution.position_monitor import (
    fetch_option_ltp_map,
    monitor_option_positions,
    monitor_positions,
    positions_with_live_ltp,
)
from quant_intelligence.execution.setup_detector import SETUP_TRIGGERED, WAITING_FOR_SETUP, detect_chain_setup, detect_setup
from quant_intelligence.data_adapters.dhan_instrument_master import list_fno_stock_symbols
from quant_intelligence.options.option_selector import (
    OptionSelectionError,
    fetch_chain,
    get_underlying_info,
    select_contract,
)
from quant_intelligence.options.premium_model import PremiumSizingError, translate_setup
from quant_intelligence.risk.risk_engine import ProposedTrade, evaluate_trade
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES, get_strategy
from quant_intelligence.ui.state import get_account_state, init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Paper Trading")
st.caption("SELECT STRATEGY -> WAIT FOR VALID SETUP -> RESOLVE OPTION CONTRACT -> RISK APPROVAL -> EXECUTE (simulated).")

automate = st.checkbox(
    "Automate (auto-fetch chain, auto-select strategy, auto-place PAPER order on setup trigger - no manual click)",
    value=False,
    help="Off by default. All the same safety checks apply (risk engine veto, mandatory stop on SELL) - "
    "this only removes the manual button click.",
)

output = run_pipeline_cached()
if output is None:
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()

broker = st.session_state["paper_broker"]
client = st.session_state["dhan_api_client"]

st.subheader("Option chain")
c1, c2, c3 = st.columns([2, 1, 1])
underlying_choices = list(SETTINGS.option_underlyings) + list_fno_stock_symbols()
underlying = c1.selectbox("Underlying", underlying_choices, key="option_underlying")
if c2.button("Fetch live option chain", disabled=not client.is_configured()):
    try:
        st.session_state["option_chain"] = fetch_chain(client, underlying, SETTINGS.option_expiry_preference)
    except Exception as e:
        st.session_state["option_chain"] = None
        st.error(f"Could not fetch option chain: {e}")
live_ltp = c3.checkbox(
    "Live refresh",
    value=True,
    disabled=not client.is_configured(),
    help=f"Re-fetch spot/chain and open positions' premium every {SETTINGS.ltp_refresh_seconds}s (read-only, no auto-trading).",
)

if not client.is_configured():
    st.info("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env to fetch a live option chain (data access only - "
            "this is separate from the LIVE trading authorization gate).")

if automate:

    @st.fragment(run_every=f"{SETTINGS.auto_trade_refresh_seconds}s")
    def _auto_fragment():
        if client.is_configured():
            try:
                st.session_state["option_chain"] = fetch_chain(client, underlying, SETTINGS.option_expiry_preference)
            except Exception as e:
                st.warning(f"Auto chain refresh failed: {e}")
        chain_now = st.session_state.get("option_chain")
        fresh_output = run_pipeline_cached(force=True)
        if fresh_output is not None:
            account = get_account_state()
            cycle = run_auto_option_cycle(fresh_output, underlying, broker, client, chain_now, account)
            st.caption(f"Last automated cycle ({dt.datetime.now():%H:%M:%S}): {cycle.reason}")
            if cycle.order is not None:
                st.session_state["trades_today"] += 1
                latest_bar = fresh_output.ohlcv.iloc[-1]
                events = monitor_positions(broker, latest_bar)
                for e in events:
                    st.session_state["daily_pnl"] += e["position"]["net_pnl"]
            if client.is_configured():
                option_events = monitor_option_positions(broker, client)
                for e in option_events:
                    st.session_state["daily_pnl"] += e["position"]["net_pnl"]

    _auto_fragment()
    st.divider()

if client.is_configured() and live_ltp and not automate:

    @st.fragment(run_every=f"{SETTINGS.ltp_refresh_seconds}s")
    def _live_chain_fragment():
        try:
            st.session_state["option_chain"] = fetch_chain(client, underlying, SETTINGS.option_expiry_preference)
        except Exception as e:
            st.warning(f"Live LTP refresh failed: {e}")
        chain_now = st.session_state.get("option_chain")
        if chain_now is not None and chain_now.underlying == underlying:
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Spot", round(chain_now.spot_price, 2))
            cc2.metric("ATM strike", chain_now.atm_strike)
            cc3.metric("Total PCR", round(chain_now.total_pcr, 2) if chain_now.total_pcr else "-")
            cc4.metric("ATM IV", round(chain_now.atm_iv, 2) if chain_now.atm_iv else "-")
        st.caption(f"Live prices updated {dt.datetime.now():%H:%M:%S} (every {SETTINGS.ltp_refresh_seconds}s)")

    _live_chain_fragment()
else:
    chain_now = st.session_state.get("option_chain")
    if chain_now is not None and chain_now.underlying == underlying:
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Spot", round(chain_now.spot_price, 2))
        cc2.metric("ATM strike", chain_now.atm_strike)
        cc3.metric("Total PCR", round(chain_now.total_pcr, 2) if chain_now.total_pcr else "-")
        cc4.metric("ATM IV", round(chain_now.atm_iv, 2) if chain_now.atm_iv else "-")

chain = st.session_state.get("option_chain")
st.divider()

if output.ranking.is_no_trade:
    st.warning(f"NO TRADE: {output.ranking.reason}")
else:
    strategy_name = output.ranking.selected_strategy
    st.success(f"Selected strategy: {strategy_name}")

    strategy = get_strategy(strategy_name)
    recent_ohlcv = output.ohlcv.tail(5)
    is_chain_strategy = strategy_name in CHAIN_AWARE_STRATEGY_NAMES

    if is_chain_strategy and chain is None:
        st.warning(f"{strategy_name} needs a live option chain - fetch one above first.")
        setup_status = None
    elif is_chain_strategy:
        setup_status = detect_chain_setup(strategy, output.market_state, recent_ohlcv, chain)
    else:
        setup_status = detect_setup(strategy, output.market_state, recent_ohlcv)

    if setup_status is not None and setup_status.status == WAITING_FOR_SETUP:
        st.info(f"Status: WAITING FOR SETUP ({strategy_name} entry conditions not yet met on the latest bar)")
    elif setup_status is not None and setup_status.status == SETUP_TRIGGERED:
        setup = setup_status.setup
        st.success("Status: SETUP TRIGGERED (underlying terms)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Direction", setup.direction)
        c2.metric("Entry", round(setup.entry_price, 2))
        c3.metric("Stop", round(setup.stop_price, 2))
        c4.metric("Target", round(setup.target_price, 2))

        transaction = setup.meta.get("transaction", "BUY")
        st.caption(f"Strategy-requested transaction: **{transaction}**")

        if chain is None or chain.underlying != underlying:
            st.warning("Fetch a live option chain for this underlying to resolve a tradeable contract.")
        else:
            moneyness_offset = st.slider("Moneyness offset (0 = ATM, + = further OTM)", 0, 5, SETTINGS.option_moneyness_offset)
            try:
                lot_size = get_underlying_info(underlying)["lot_size"]
                contract = select_contract(chain, setup.direction, transaction, moneyness_offset, lot_size)
                premium_setup = translate_setup(setup, contract)
            except (OptionSelectionError, PremiumSizingError) as e:
                st.error(f"Could not resolve a tradeable option: {e}")
            else:
                st.success(f"Resolved contract: {contract.trading_symbol}")
                oc1, oc2, oc3, oc4 = st.columns(4)
                oc1.metric("Premium (entry)", round(premium_setup.entry_price, 2))
                oc2.metric("Premium stop", round(premium_setup.stop_price, 2))
                oc3.metric("Premium target", round(premium_setup.target_price, 2))
                oc4.metric("Delta", round(contract.delta, 3) if contract.delta is not None else "-")

                lots = st.number_input("Lots", min_value=1, value=1, step=1)
                quantity = int(lots) * contract.lot_size
                st.caption(f"Order quantity: {lots} lot(s) x {contract.lot_size} = {quantity}")

                if st.button("Submit to Risk Engine -> Paper Broker", type="primary"):
                    account = get_account_state()
                    proposed = ProposedTrade(
                        strategy_name=strategy_name,
                        direction=setup.direction,
                        entry_price=premium_setup.entry_price,
                        stop_price=premium_setup.stop_price,
                        target_price=premium_setup.target_price,
                        quantity=quantity,
                        relative_volume=output.market_state.get("relative_volume"),
                        data_quality_status=output.data_quality_status,
                    )
                    decision = evaluate_trade(account, proposed)

                    if not decision.approved:
                        st.error(f"RISK ENGINE VETO ({decision.decision_id}): {decision.reason}")
                    else:
                        st.success(f"Risk approved ({decision.decision_id})")
                        order = OrderRequest(
                            strategy_name=strategy_name,
                            instrument=underlying,
                            direction=setup.direction,
                            quantity=quantity,
                            order_type="MARKET",
                            price=premium_setup.entry_price,
                            stop_price=premium_setup.stop_price,
                            target_price=premium_setup.target_price,
                            decision_id=decision.decision_id,
                            security_id=contract.security_id,
                            exchange_segment="NSE_FNO",
                            product_type=SETTINGS.option_product_type,
                            transaction_type=contract.transaction,
                            option_type=contract.option_type,
                            strike=contract.strike,
                            expiry=contract.expiry,
                            lot_size=contract.lot_size,
                        )
                        ack = broker.place_order(order, market_price=premium_setup.entry_price)
                        st.session_state["trades_today"] += 1
                        if ack.status == "FILLED":
                            st.success(f"Paper order FILLED: {contract.transaction} {order.quantity} {contract.trading_symbol} @ {ack.fill_price:.2f} (order {ack.order_id})")
                        else:
                            st.error(f"Order {ack.status}: {ack.reject_reason}")

st.divider()
st.subheader("Position monitor")
mc1, mc2 = st.columns(2)
if mc1.button("Check underlying stop/target hits"):
    latest_bar = output.ohlcv.iloc[-1]
    events = monitor_positions(broker, latest_bar)
    if events:
        for e in events:
            st.write(f"{e['type']}: {e['position']['position_id']} closed @ {e['position']['exit_price']}, net P&L {e['position']['net_pnl']:.2f}")
            st.session_state["daily_pnl"] += e["position"]["net_pnl"]
    else:
        st.info("No stop/target hits on the latest bar.")

if mc2.button("Check option premium stop/target hits (live LTP)", disabled=not client.is_configured()):
    events = monitor_option_positions(broker, client)
    if events:
        for e in events:
            st.write(f"{e['type']}: {e['position']['position_id']} closed @ {e['position']['exit_price']}, net P&L {e['position']['net_pnl']:.2f}")
            st.session_state["daily_pnl"] += e["position"]["net_pnl"]
    else:
        st.info("No option stop/target hits at current LTP.")

if client.is_configured() and live_ltp and not automate:

    @st.fragment(run_every=f"{SETTINGS.ltp_refresh_seconds}s")
    def _positions_ltp_fragment():
        events = monitor_option_positions(broker, client)
        for e in events:
            st.session_state["daily_pnl"] += e["position"]["net_pnl"]
        ltp_map = fetch_option_ltp_map(broker, client)
        rows = positions_with_live_ltp(broker, ltp_map)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.caption("No open paper positions.")
        st.caption(f"Live LTP updated {dt.datetime.now():%H:%M:%S} (every {SETTINGS.ltp_refresh_seconds}s)")

    _positions_ltp_fragment()
else:
    open_positions = broker.get_open_positions()
    if open_positions:
        st.dataframe(open_positions, width="stretch", hide_index=True)
    else:
        st.caption("No open paper positions.")

st.divider()
st.subheader("Trade history")
all_positions = broker.get_positions()
if all_positions:
    trades_df = pd.DataFrame(all_positions)
    closed_count = sum(1 for p in all_positions if p["status"] == "CLOSED")
    st.download_button(
        "Download paper trades (CSV)",
        data=trades_df.to_csv(index=False).encode("utf-8"),
        file_name=f"paper_trades_{dt.date.today():%Y%m%d}.csv",
        mime="text/csv",
    )
    st.caption(f"{len(all_positions)} trade(s) total ({closed_count} closed, {len(all_positions) - closed_count} open).")
else:
    st.caption("No paper trades yet.")
