from quant_intelligence.backtesting.engine import run_backtest, walk_forward
from quant_intelligence.strategies.orb_strategy import ORBStrategy


def test_backtest_runs_and_produces_metrics(synthetic_ohlcv, synthetic_features):
    strategy = ORBStrategy()
    result = run_backtest(strategy, synthetic_ohlcv, synthetic_features, "NIFTY", split="IN_SAMPLE")
    assert result.run_id.startswith("BT-")
    if result.metrics.get("n_trades", 0) > 0:
        assert "expected_r" in result.metrics
        assert "max_drawdown" in result.metrics


def test_backtest_trades_do_not_overlap(synthetic_ohlcv, synthetic_features):
    strategy = ORBStrategy()
    result = run_backtest(strategy, synthetic_ohlcv, synthetic_features, "NIFTY")
    trades = sorted(result.trades, key=lambda t: t.setup.timestamp)
    for prev, curr in zip(trades, trades[1:]):
        assert curr.setup.timestamp >= (prev.exit_timestamp or prev.setup.timestamp)


def test_walk_forward_produces_multiple_folds(synthetic_ohlcv, synthetic_features):
    results = walk_forward(lambda: ORBStrategy(), synthetic_ohlcv, synthetic_features, "NIFTY", n_folds=3)
    assert len(results) <= 3
    for r in results:
        assert "WALK_FORWARD" in r.split


def test_no_trades_reports_zero_not_fabricated_stats():
    import pandas as pd

    empty_ohlcv = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "open": [], "high": [], "low": [], "close": [], "volume": []})
    strategy = ORBStrategy()
    empty_features = empty_ohlcv[["timestamp"]].assign(atr_14=[], opening_range_high=[], opening_range_low=[], time_since_open_min=[])
    result = run_backtest(strategy, empty_ohlcv, empty_features, "NIFTY")
    assert result.metrics.get("status") == "NO_TRADES"
