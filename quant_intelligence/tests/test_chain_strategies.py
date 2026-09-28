import pandas as pd
import pytest

from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow
from quant_intelligence.strategies.iv_crush_writer_strategy import IVCrushWriterStrategy
from quant_intelligence.strategies.oi_buildup_strategy import OIBuildupStrategy
from quant_intelligence.strategies.pcr_contrarian_strategy import PCRContrarianStrategy
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES, STRATEGY_CLASSES
from quant_intelligence.strategies.unusual_volume_fade_strategy import UnusualVolumeFadeStrategy

RECENT_OHLCV = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01 09:20", periods=3, freq="5min"),
        "open": [100, 100, 100],
        "high": [100.5, 100.5, 100.5],
        "low": [99.5, 99.5, 99.5],
        "close": [100, 100, 100],
        "volume": [1000, 1000, 1000],
    }
)


def _row(strike, **overrides):
    base = dict(
        strike=strike,
        ce_ltp=5.0,
        ce_oi=1000,
        ce_oi_change=50,
        ce_volume=100,
        ce_iv=15.0,
        ce_delta=0.5,
        ce_security_id=f"{int(strike)}CE",
        pe_ltp=5.0,
        pe_oi=1000,
        pe_oi_change=50,
        pe_volume=100,
        pe_iv=15.0,
        pe_delta=-0.5,
        pe_security_id=f"{int(strike)}PE",
    )
    base.update(overrides)
    return StrikeRow(**base)


def test_chain_aware_strategies_are_registered():
    assert CHAIN_AWARE_STRATEGY_NAMES.issubset(set(STRATEGY_CLASSES))
    assert len(CHAIN_AWARE_STRATEGY_NAMES) == 4


def test_pcr_contrarian_triggers_long_on_high_pcr_and_oversold_rsi():
    chain = ChainSnapshot(underlying="NIFTY", expiry="2024-10-31", spot_price=100.0, strikes=[_row(100)])
    strategy = PCRContrarianStrategy()
    market_state = {"rsi_2": 5.0, "atr_14": 2.0}
    # Force total_pcr high by overriding via monkeypatch-style: rebuild chain with lopsided OI.
    chain.strikes[0].pe_oi = 3000
    chain.strikes[0].ce_oi = 1000
    setup = strategy.check_chain_setup(market_state, RECENT_OHLCV, chain)
    assert setup is not None
    assert setup.direction == "LONG"
    assert setup.meta["transaction"] == "BUY"


def test_pcr_contrarian_no_setup_when_rsi_not_extreme():
    chain = ChainSnapshot(underlying="NIFTY", expiry="2024-10-31", spot_price=100.0, strikes=[_row(100, pe_oi=3000)])
    strategy = PCRContrarianStrategy()
    market_state = {"rsi_2": 50.0, "atr_14": 2.0}
    assert strategy.check_chain_setup(market_state, RECENT_OHLCV, chain) is None


def test_oi_buildup_confirms_breakout_with_rising_call_oi():
    chain = ChainSnapshot(underlying="NIFTY", expiry="2024-10-31", spot_price=101.0, strikes=[_row(100, ce_oi=1000, ce_oi_change=200)])
    strategy = OIBuildupStrategy()
    market_state = {"donchian_high_20": 100.5, "donchian_low_20": 95.0, "atr_14": 1.0}
    setup = strategy.check_chain_setup(market_state, RECENT_OHLCV, chain)
    assert setup is not None
    assert setup.direction == "LONG"


def test_iv_crush_writer_sells_when_iv_rich_and_flat():
    chain = ChainSnapshot(underlying="NIFTY", expiry="2024-10-31", spot_price=100.0, strikes=[_row(100, ce_iv=25.0, pe_iv=25.0)])
    strategy = IVCrushWriterStrategy()
    market_state = {"realized_vol_20": 0.15, "trend_slope": 0.0001, "atr_14": 1.0}
    setup = strategy.check_chain_setup(market_state, RECENT_OHLCV, chain)
    assert setup is not None
    assert setup.meta["transaction"] == "SELL"


def test_iv_crush_writer_skips_when_trending():
    chain = ChainSnapshot(underlying="NIFTY", expiry="2024-10-31", spot_price=100.0, strikes=[_row(100, ce_iv=25.0, pe_iv=25.0)])
    strategy = IVCrushWriterStrategy()
    market_state = {"realized_vol_20": 0.15, "trend_slope": 0.01, "atr_14": 1.0}
    assert strategy.check_chain_setup(market_state, RECENT_OHLCV, chain) is None


def test_unusual_volume_fade_buys_opposite_side():
    strikes = [_row(s, ce_volume=100, ce_oi=1000, pe_volume=100, pe_oi=1000) for s in (95, 100, 105)]
    strikes[1].ce_volume = 50000  # unusual CE spike at ATM
    chain = ChainSnapshot(underlying="NIFTY", expiry="2024-10-31", spot_price=100.0, strikes=strikes)
    strategy = UnusualVolumeFadeStrategy()
    market_state = {"atr_14": 1.0}
    setup = strategy.check_chain_setup(market_state, RECENT_OHLCV, chain)
    assert setup is not None
    assert setup.direction == "SHORT"  # CE spike faded by buying PE
    assert setup.meta["transaction"] == "BUY"


@pytest.mark.parametrize("cls", [PCRContrarianStrategy, OIBuildupStrategy, IVCrushWriterStrategy, UnusualVolumeFadeStrategy])
def test_chain_strategies_report_no_historical_setups(cls):
    strategy = cls()
    assert strategy.generate_historical_setups(pd.DataFrame(), pd.DataFrame()) == []
    assert strategy.check_setup({}, RECENT_OHLCV) is None
