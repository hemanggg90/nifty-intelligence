from types import SimpleNamespace

from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.execution import auto_trader
from quant_intelligence.execution.capital import capital_required, capital_summary


def _open(broker, transaction, price=100.0, qty=50):
    order = OrderRequest(
        strategy_name="S", instrument="NIFTY", direction="LONG", quantity=qty, price=price,
        stop_price=price - 10, target_price=price + 10, security_id="1", option_type="CE",
        strike=100, expiry="2030-01-01", lot_size=50, transaction_type=transaction,
    )
    broker.place_order(order, market_price=price)


def test_capital_required_is_premium_times_quantity():
    assert capital_required(120.5, 75) == 120.5 * 75


def test_buy_positions_consume_capital_but_sell_positions_do_not():
    broker = PaperBroker(starting_capital=1_000_000)
    assert capital_summary(broker)["capital_used"] == 0

    _open(broker, "BUY", price=100.0, qty=50)  # fills at 100 + slippage
    used = capital_summary(broker)["capital_used"]
    assert 5000 <= used <= 5100  # ~ 100 x 50

    _open(broker, "SELL", price=80.0, qty=50)
    cap = capital_summary(broker)
    assert cap["capital_used"] == used  # SELL premium is not counted as capital used
    assert cap["sell_premium_value"] > 0
    assert cap["available"] == cap["account_value"] - used


def test_auto_cycle_refuses_a_buy_it_cannot_afford(monkeypatch):
    broker = PaperBroker(starting_capital=1_000)  # far too small for the trade below
    setup = SimpleNamespace(direction="LONG", meta={"transaction": "BUY"})
    contract = SimpleNamespace(lot_size=100, transaction="BUY")
    premium = SimpleNamespace(entry_price=50.0, stop_price=40.0, target_price=70.0)

    monkeypatch.setattr(auto_trader, "get_strategy", lambda name: object())
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status=auto_trader.SETUP_TRIGGERED, setup=setup))
    monkeypatch.setattr(auto_trader, "get_underlying_info", lambda u: {"lot_size": 100})
    monkeypatch.setattr(auto_trader, "select_contract", lambda *a, **k: contract)
    monkeypatch.setattr(auto_trader, "translate_setup", lambda *a, **k: premium)
    monkeypatch.setattr(auto_trader, "size_position", lambda *a, **k: 100)  # 100 x Rs 50 = Rs 5,000 > Rs 1,000

    output = SimpleNamespace(
        ranking=SimpleNamespace(is_no_trade=False, selected_strategy="ORB"),
        ohlcv=SimpleNamespace(tail=lambda n: None),
        market_state=SimpleNamespace(get=lambda k: None),
        data_quality_status="OK",
    )
    chain = SimpleNamespace(underlying="NIFTY")
    result = auto_trader.run_auto_option_cycle(output, "NIFTY", broker, client=None, chain=chain, account=None)

    assert result.order is None
    assert result.capital_required == 5000.0
    assert "Insufficient funds" in result.reason
    assert broker.get_open_positions() == []
