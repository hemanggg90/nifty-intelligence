import pytest

from quant_intelligence.options.option_selector import OptionContract
from quant_intelligence.options.premium_model import (
    MIN_PREMIUM_TICK,
    PremiumSizingError,
    max_lots_within_risk_budget,
    max_loss_per_lot,
    translate_setup,
)
from quant_intelligence.strategies.base_strategy import Setup


def _contract(transaction, delta=0.5, premium=20.0, lot_size=50):
    return OptionContract(
        security_id="123",
        trading_symbol="NIFTY 100 CE",
        underlying="NIFTY",
        strike=100.0,
        option_type="CE",
        expiry="2024-10-31",
        lot_size=lot_size,
        transaction=transaction,
        premium=premium,
        delta=delta,
        iv=15.0,
    )


def _setup(entry=100.0, stop=98.0, target=104.0):
    return Setup(timestamp=None, direction="LONG", entry_price=entry, stop_price=stop, target_price=target)


def test_buy_translation_scales_by_delta_and_floors_at_min_tick():
    contract = _contract("BUY", delta=0.5, premium=20.0)
    result = translate_setup(_setup(entry=100.0, stop=98.0, target=104.0), contract)
    assert result.entry_price == 20.0
    assert result.stop_price == pytest.approx(20.0 - 0.5 * 2.0)
    assert result.target_price == pytest.approx(20.0 + 0.5 * 4.0)
    assert result.stop_price > 0


def test_buy_translation_never_goes_below_min_tick():
    contract = _contract("BUY", delta=0.9, premium=1.0)
    result = translate_setup(_setup(entry=100.0, stop=50.0, target=104.0), contract)
    assert result.stop_price == MIN_PREMIUM_TICK


def test_sell_translation_requires_positive_stop_distance():
    contract = _contract("SELL", delta=0.5, premium=20.0)
    result = translate_setup(_setup(entry=100.0, stop=98.0, target=104.0), contract)
    assert result.stop_price > result.entry_price
    assert result.target_price < result.entry_price


def test_sell_translation_floors_distance_even_with_zero_index_move():
    """The MIN_PREMIUM_TICK floor guarantees a SELL stop is always above entry, even if
    the underlying setup itself has a degenerate (zero-distance) stop - this is what
    keeps every SELL trade defined-risk by construction."""
    contract = _contract("SELL", delta=0.5, premium=20.0)
    result = translate_setup(_setup(entry=100.0, stop=100.0, target=104.0), contract)
    assert result.stop_price > result.entry_price


def test_sell_translation_raises_if_stop_not_above_entry(monkeypatch):
    """Guards the SELL invariant directly: if the floor logic were ever weakened, this
    branch is what stops a zero/negative-risk-distance write from reaching the risk engine."""
    import quant_intelligence.options.premium_model as premium_model

    monkeypatch.setattr(premium_model, "MIN_PREMIUM_TICK", 0.0)
    contract = _contract("SELL", delta=0.5, premium=20.0)
    with pytest.raises(PremiumSizingError):
        translate_setup(_setup(entry=100.0, stop=100.0, target=104.0), contract)


def test_max_loss_per_lot_and_lot_budget():
    from quant_intelligence.config.settings import SETTINGS

    contract = _contract("BUY", delta=0.5, premium=20.0, lot_size=50)
    result = translate_setup(_setup(entry=100.0, stop=98.0, target=104.0), contract)
    per_lot = max_loss_per_lot(result)
    assert per_lot == pytest.approx((20.0 - result.stop_price) * 50)

    equity = 1_000_000
    budget = equity * SETTINGS.risk.max_risk_per_trade_pct / 100.0
    lots = max_lots_within_risk_budget(result, equity=equity)
    assert lots >= 0
    assert lots * per_lot <= budget + 1e-6
    assert (lots + 1) * per_lot > budget
