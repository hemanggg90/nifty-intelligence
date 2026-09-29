"""Integration test: market data -> features -> regime -> strategy ranking -> setup -> risk -> paper order."""
import datetime as dt

from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.execution.setup_detector import detect_setup
from quant_intelligence.research.pipeline import run_pipeline
from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, evaluate_trade
from quant_intelligence.strategies.registry import STRATEGY_CLASSES, get_strategy


def test_full_pipeline_end_to_end(monkeypatch, tmp_path):
    # Production has no synthetic fallback; feed the pipeline synthetic candles for this test only.
    from quant_intelligence.data import data_manager as dm_module
    from quant_intelligence.data.data_manager import DataManager
    from quant_intelligence.data_adapters.synthetic import SyntheticAdapter

    monkeypatch.setattr(dm_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        DataManager, "_fetch",
        lambda self, instrument, timeframe, start, end, prefer: (
            "test", SyntheticAdapter().get_ohlcv(instrument, timeframe, start, end)),
    )
    end = dt.datetime(2024, 6, 28, 15, 30)
    start = end - dt.timedelta(days=60)

    output = run_pipeline("NIFTY", "5min", start, end)

    assert output.market_state is not None
    assert output.regime_label is not None
    assert 0.0 <= output.regime_confidence <= 1.0
    assert len(output.strategy_intel) == len(STRATEGY_CLASSES)
    assert output.ranking is not None

    # Whatever the decision, it must be internally consistent.
    if output.ranking.is_no_trade:
        assert output.ranking.selected_strategy is None
    else:
        assert output.ranking.selected_strategy is not None

    # If a strategy was selected, drive it through setup detection -> risk -> paper broker.
    if not output.ranking.is_no_trade:
        strategy = get_strategy(output.ranking.selected_strategy)
        recent = output.ohlcv.tail(5)
        setup_status = detect_setup(strategy, output.market_state, recent)

        account = AccountState(
            equity=1_000_000,
            peak_equity=1_000_000,
            daily_pnl=0.0,
            open_positions_count=0,
            trades_today=0,
            exposure_by_strategy={},
            total_exposure=0.0,
            broker_connected=True,
            kill_switch_engaged=False,
        )

        if setup_status.setup is not None:
            trade = ProposedTrade(
                strategy_name=strategy.name,
                direction=setup_status.setup.direction,
                entry_price=setup_status.setup.entry_price,
                stop_price=setup_status.setup.stop_price,
                target_price=setup_status.setup.target_price,
                quantity=50,
                data_quality_status=output.data_quality_status,
            )
            decision = evaluate_trade(account, trade)
            assert decision.decision_id.startswith("RISK-")

            if decision.approved:
                broker = PaperBroker()
                order = OrderRequest(
                    strategy_name=strategy.name,
                    instrument=output.instrument,
                    direction=setup_status.setup.direction,
                    quantity=50,
                    price=setup_status.setup.entry_price,
                )
                ack = broker.place_order(order, market_price=setup_status.setup.entry_price)
                assert ack.status in ("FILLED", "REJECTED")


def test_bad_data_quality_prevents_trade_end_to_end(monkeypatch):
    """If the pipeline's data quality is not OK, the ranking decision must be NO TRADE,
    regardless of how good any strategy's backtest numbers look."""
    from quant_intelligence.ranking.ranking_engine import rank_and_select, score_strategy

    scores = [score_strategy({"strategy_name": "A", "confidence_label": "HIGH", "sample_size": 500, "expected_r": 2.0, "prob_positive_return": 0.9}, -1)]
    decision = rank_and_select(scores, data_quality_status="FAIL")
    assert decision.is_no_trade


def test_no_data_source_raises_instead_of_fabricating(monkeypatch, tmp_path):
    import pytest
    from quant_intelligence.data import data_manager as dm_module
    from quant_intelligence.data.data_manager import DataManager, DataUnavailableError

    monkeypatch.setattr(dm_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(DataManager, "_fetch", lambda self, *a: ("none", __import__("pandas").DataFrame()))
    end = dt.datetime(2024, 6, 28, 15, 30)
    with pytest.raises(DataUnavailableError):
        DataManager().get_ohlcv("NIFTY", "5min", end - dt.timedelta(days=5), end)
