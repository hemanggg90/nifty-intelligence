"""Fully-automatic paper-trade execution.

Extracts the manual "fetch chain -> resolve setup -> size -> risk-check ->
place order" flow that `pages/07_Paper_Trading.py` walks a human through one
button click at a time, into functions any caller (a Streamlit auto-refresh
fragment, a multi-stock loop) can invoke on a timer. No new order-placement
path is introduced: this calls exactly the same `detect_setup`/
`detect_chain_setup`, `select_contract`, `translate_setup`, `evaluate_trade`,
and `broker.place_order` functions the manual page uses, so every safety
check (risk engine veto, mandatory-stop-on-SELL) applies identically.
"""
from __future__ import annotations

from dataclasses import dataclass

from quant_intelligence.brokers.base_broker import OrderAck, OrderRequest
from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.setup_detector import (
    SETUP_TRIGGERED,
    WAITING_FOR_SETUP,
    detect_chain_setup,
    detect_setup,
)
from quant_intelligence.options.chain_analytics import ChainSnapshot
from quant_intelligence.options.option_selector import OptionSelectionError, get_underlying_info, select_contract
from quant_intelligence.options.premium_model import PremiumSizingError, translate_setup
from quant_intelligence.research.pipeline import PipelineOutput
from quant_intelligence.execution.capital import capital_required, capital_summary
from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, evaluate_trade
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES, get_strategy
from quant_intelligence.utils.market_profile import profile_for


@dataclass
class AutoCycleResult:
    instrument: str
    strategy_name: str | None = None
    setup_status: str | None = None  # WAITING_FOR_SETUP / SETUP_TRIGGERED / None
    order: OrderAck | None = None
    risk_decision_id: str | None = None
    reason: str = ""
    capital_required: float | None = None  # premium x quantity to BUY the resolved contract
    capital_used: float = 0.0  # capital actually committed by a filled BUY order


def size_position(account: AccountState, entry_price: float, stop_price: float, lot_size: int = 1) -> int:
    """Quantity such that the risk-engine's per-trade risk check passes by construction.

    Uses 90% of `max_risk_per_trade_pct` as headroom against rounding, since
    the final `quantity` is rounded down to a whole number of lots.
    """
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0 or account.equity <= 0:
        return 0
    budget = account.equity * (SETTINGS.risk.max_risk_per_trade_pct * 0.9) / 100.0
    raw_qty = budget / stop_distance
    lots = max(int(raw_qty // lot_size), 0)
    return lots * lot_size


def run_auto_option_cycle(
    output: PipelineOutput,
    underlying: str,
    broker: PaperBroker,
    client: DhanApiClient,
    chain: ChainSnapshot | None,
    account: AccountState,
) -> AutoCycleResult:
    """One automatic cycle for the index/options paper-trading flow (07_Paper_Trading)."""
    result = AutoCycleResult(instrument=underlying)

    if output.ranking.is_no_trade:
        result.reason = f"NO TRADE: {output.ranking.reason}"
        return result

    if any(pos["instrument"] == underlying for pos in broker.get_open_positions()):
        result.reason = f"Skipped: already have an open position on {underlying}."
        return result

    strategy_name = output.ranking.selected_strategy
    result.strategy_name = strategy_name
    strategy = get_strategy(strategy_name)
    recent_ohlcv = output.ohlcv.tail(5)
    is_chain_strategy = strategy_name in CHAIN_AWARE_STRATEGY_NAMES

    if is_chain_strategy and chain is None:
        result.reason = f"{strategy_name} needs a live option chain - none fetched."
        return result

    setup_status = (
        detect_chain_setup(strategy, output.market_state, recent_ohlcv, chain)
        if is_chain_strategy
        else detect_setup(strategy, output.market_state, recent_ohlcv)
    )
    result.setup_status = setup_status.status

    if setup_status.status != SETUP_TRIGGERED:
        result.reason = "Waiting for setup." if setup_status.status == WAITING_FOR_SETUP else "No setup."
        return result

    setup = setup_status.setup
    transaction = setup.meta.get("transaction", "BUY")

    if chain is None or chain.underlying != underlying:
        result.reason = "Setup triggered but no live option chain available to resolve a contract."
        return result

    try:
        lot_size = get_underlying_info(underlying)["lot_size"]
        contract = select_contract(chain, setup.direction, transaction, SETTINGS.option_moneyness_offset, lot_size)
        premium_setup = translate_setup(setup, contract)
    except (OptionSelectionError, PremiumSizingError) as e:
        result.reason = f"Could not resolve a tradeable option: {e}"
        return result

    quantity = size_position(account, premium_setup.entry_price, premium_setup.stop_price, contract.lot_size)
    if quantity <= 0:
        result.reason = "Position sizing produced zero quantity (equity too small or stop too wide)."
        return result

    result.capital_required = capital_required(premium_setup.entry_price, quantity)
    if transaction == "BUY":
        available = capital_summary(broker)["available"]
        if result.capital_required > available:
            result.reason = (
                f"Insufficient funds: needs Rs {result.capital_required:,.0f}, "
                f"Rs {available:,.0f} available."
            )
            return result

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
    result.risk_decision_id = decision.decision_id

    if not decision.approved:
        result.reason = f"RISK ENGINE VETO ({decision.decision_id}): {decision.reason}"
        return result

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
        exchange_segment=profile_for(underlying).option_segment,
        product_type=SETTINGS.option_product_type,
        transaction_type=contract.transaction,
        option_type=contract.option_type,
        strike=contract.strike,
        expiry=contract.expiry,
        lot_size=contract.lot_size,
    )
    result.order = broker.place_order(order, market_price=premium_setup.entry_price)
    result.reason = f"Order {result.order.status}"
    if result.order.status == "FILLED" and transaction == "BUY":
        result.capital_used = result.capital_required
    return result


def run_auto_equity_cycle(output: PipelineOutput, broker: PaperBroker, account: AccountState) -> AutoCycleResult:
    """One automatic cycle for a direct equity (cash-market) paper trade - no options.

    Used for per-stock trading where there is no live option chain: only the
    13 price-action strategies apply (chain-aware strategies are skipped, see
    `CHAIN_AWARE_STRATEGY_NAMES`).
    """
    instrument = output.instrument
    result = AutoCycleResult(instrument=instrument)

    if output.ranking.is_no_trade:
        result.reason = f"NO TRADE: {output.ranking.reason}"
        return result

    if any(pos["instrument"] == instrument for pos in broker.get_open_positions()):
        result.reason = f"Skipped: already have an open position on {instrument}."
        return result

    strategy_name = output.ranking.selected_strategy
    result.strategy_name = strategy_name

    if strategy_name in CHAIN_AWARE_STRATEGY_NAMES:
        result.reason = f"{strategy_name} needs an option chain - not supported for per-stock equity trading."
        return result

    strategy = get_strategy(strategy_name)
    recent_ohlcv = output.ohlcv.tail(5)
    setup_status = detect_setup(strategy, output.market_state, recent_ohlcv)
    result.setup_status = setup_status.status

    if setup_status.status != SETUP_TRIGGERED:
        result.reason = "Waiting for setup." if setup_status.status == WAITING_FOR_SETUP else "No setup."
        return result

    setup = setup_status.setup
    quantity = size_position(account, setup.entry_price, setup.stop_price)
    if quantity <= 0:
        result.reason = "Position sizing produced zero quantity (equity too small or stop too wide)."
        return result

    proposed = ProposedTrade(
        strategy_name=strategy_name,
        direction=setup.direction,
        entry_price=setup.entry_price,
        stop_price=setup.stop_price,
        target_price=setup.target_price,
        quantity=quantity,
        relative_volume=output.market_state.get("relative_volume"),
        data_quality_status=output.data_quality_status,
    )
    decision = evaluate_trade(account, proposed)
    result.risk_decision_id = decision.decision_id

    if not decision.approved:
        result.reason = f"RISK ENGINE VETO ({decision.decision_id}): {decision.reason}"
        return result

    order = OrderRequest(
        strategy_name=strategy_name,
        instrument=instrument,
        direction=setup.direction,
        quantity=quantity,
        order_type="MARKET",
        price=setup.entry_price,
        stop_price=setup.stop_price,
        target_price=setup.target_price,
        decision_id=decision.decision_id,
        exchange_segment="NSE_EQ",
        product_type="CNC",
    )
    result.order = broker.place_order(order, market_price=setup.entry_price)
    result.reason = f"Order {result.order.status}"
    return result
