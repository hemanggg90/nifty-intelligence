import pandas as pd
import pytest

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup
from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
from quant_intelligence.strategies.orb_strategy import ORBStrategy
from quant_intelligence.strategies.registry import STRATEGY_CLASSES
from quant_intelligence.strategies.vwap_mean_reversion_strategy import VWAPMeanReversionStrategy


@pytest.mark.parametrize("cls", list(STRATEGY_CLASSES.values()))
def test_generate_historical_setups_runs(cls, synthetic_ohlcv, synthetic_features):
    strategy = cls()
    setups = strategy.generate_historical_setups(synthetic_ohlcv, synthetic_features)
    assert isinstance(setups, list)
    for s in setups:
        assert s.direction in ("LONG", "SHORT")
        assert s.entry_price > 0
        assert s.stop_price != s.entry_price


def test_simulate_trade_long_hits_target():
    ohlcv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 09:20", periods=5, freq="5min"),
            "open": [100, 100, 100, 100, 100],
            "high": [100, 101, 105, 101, 101],
            "low": [99, 99, 100, 99, 99],
            "close": [100, 100.5, 104, 100, 100],
            "volume": [1000] * 5,
        }
    )
    strategy = ORBStrategy()
    setup = Setup(ohlcv.iloc[0]["timestamp"], "LONG", 100.0, 98.0, 103.0)
    result = strategy.simulate_trade(setup, ohlcv, entry_idx=0, max_holding_bars=4)
    assert result.outcome == "WIN"
    assert result.exit_price == 103.0


def test_simulate_trade_long_hits_stop():
    ohlcv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 09:20", periods=5, freq="5min"),
            "open": [100, 100, 100, 100, 100],
            "high": [100, 100.5, 100.5, 100.5, 100.5],
            "low": [99, 99, 95, 99, 99],
            "close": [100, 99.5, 96, 99, 99],
            "volume": [1000] * 5,
        }
    )
    strategy = ORBStrategy()
    setup = Setup(ohlcv.iloc[0]["timestamp"], "LONG", 100.0, 98.0, 103.0)
    result = strategy.simulate_trade(setup, ohlcv, entry_idx=0, max_holding_bars=4)
    assert result.outcome == "LOSS"
    assert result.exit_price == 98.0


def test_simulate_trade_does_not_use_entry_bar_for_stop_check():
    """Regression test: entry bar's own high/low (which produced the setup) must NOT
    be checked against the stop/target, since the entry price is that bar's close.
    """
    ohlcv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 09:20", periods=3, freq="5min"),
            "open": [100, 100, 100],
            "high": [100, 100.2, 100.2],
            # Entry bar's own low is far below where a stop would be placed - this must be ignored.
            "low": [90, 99.9, 99.9],
            "close": [100, 100.1, 100.1],
            "volume": [1000] * 3,
        }
    )
    strategy = ORBStrategy()
    setup = Setup(ohlcv.iloc[0]["timestamp"], "LONG", 100.0, 98.0, 103.0)
    result = strategy.simulate_trade(setup, ohlcv, entry_idx=0, max_holding_bars=2)
    assert result.outcome != "LOSS", "entry bar's own low must not trigger an immediate stop-out"


def test_calculate_entry_stop_target_direction():
    strategy = MomentumStrategy()
    entry, stop, target = strategy.calculate_entry_stop_target("LONG", 100.0, atr=2.0)
    assert stop < entry < target
    entry, stop, target = strategy.calculate_entry_stop_target("SHORT", 100.0, atr=2.0)
    assert target < entry < stop


def test_invalid_parameters_can_be_rejected():
    class StrictStrategy(BaseStrategy):
        name = "strict"

        def validate_parameters(self, parameters):
            if parameters.get("stop_atr_mult", 1) <= 0:
                raise ValueError("stop_atr_mult must be positive")

        def generate_historical_setups(self, ohlcv, features):
            return []

        def check_setup(self, market_state, recent_ohlcv):
            return None

    with pytest.raises(ValueError):
        StrictStrategy({"stop_atr_mult": -1})
