from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.brokers.paper_broker import PaperBroker


def test_paper_broker_fills_market_order():
    broker = PaperBroker(starting_capital=100_000)
    order = OrderRequest(strategy_name="Momentum", instrument="NIFTY", direction="LONG", quantity=50, price=100.0)
    ack = broker.place_order(order, market_price=100.0)
    assert ack.status == "FILLED"
    assert ack.order_id.startswith("PAPER-")
    assert len(broker.get_open_positions()) == 1


def test_paper_broker_rejects_zero_quantity():
    broker = PaperBroker()
    order = OrderRequest(strategy_name="Momentum", instrument="NIFTY", direction="LONG", quantity=0, price=100.0)
    ack = broker.place_order(order, market_price=100.0)
    assert ack.status == "REJECTED"


def test_paper_broker_never_calls_network():
    """PaperBroker must have no network/broker dependency - it's a pure simulation."""
    import inspect

    source = inspect.getsource(PaperBroker)
    for forbidden in ("requests.", "urllib", "socket.", "httpx"):
        assert forbidden not in source


def test_close_position_computes_pnl():
    broker = PaperBroker(starting_capital=100_000)
    order = OrderRequest(strategy_name="Momentum", instrument="NIFTY", direction="LONG", quantity=10, price=100.0, stop_price=98, target_price=105)
    ack = broker.place_order(order, market_price=100.0)
    position_id = broker.get_open_positions()[0]["position_id"]
    closed = broker.close_position(position_id, exit_price=105.0)
    assert closed["status"] == "CLOSED"
    assert closed["net_pnl"] > 0


def test_slippage_applied_in_direction_of_trade():
    broker = PaperBroker()
    long_order = OrderRequest(strategy_name="Momentum", instrument="NIFTY", direction="LONG", quantity=1, price=100.0)
    ack_long = broker.place_order(long_order, market_price=100.0)
    assert ack_long.fill_price >= 100.0  # slippage should be adverse (worse) for a buy

    short_order = OrderRequest(strategy_name="Momentum", instrument="NIFTY", direction="SHORT", quantity=1, price=100.0)
    ack_short = broker.place_order(short_order, market_price=100.0)
    assert ack_short.fill_price <= 100.0  # adverse (worse) for a sell
