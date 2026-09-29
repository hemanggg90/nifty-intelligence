"""
Research Pipeline Orchestrator.

Ties together: DataManager -> feature engine -> market state -> regime engine
-> (per strategy) historical backtest -> analogue engine -> ranking engine ->
NO-TRADE decision -> setup detector. This is the single source of truth the
Streamlit app calls into, so the UI never re-implements pipeline logic.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from quant_intelligence.analogues.analogue_engine import (
    COMPARISON_FEATURES,
    conditional_metrics_from_analogues,
    find_analogues,
)
from quant_intelligence.backtesting.engine import run_backtest, BacktestResult
from quant_intelligence.data.data_manager import DataManager
from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.market_state.market_state_engine import MarketState, build_current_state
from quant_intelligence.ranking.ranking_engine import RankingDecision, StrategyScore, rank_and_select, score_strategy
from quant_intelligence.regimes.regime_engine import classify_regime, regime_confidence
from quant_intelligence.strategies.registry import get_all_strategies
from quant_intelligence.utils.logging_utils import log_event


@dataclass
class StrategyIntelligence:
    strategy_name: str
    global_metrics: dict
    conditional_metrics: dict
    analogues: pd.DataFrame
    score: StrategyScore
    backtest: BacktestResult


@dataclass
class PipelineOutput:
    instrument: str
    timestamp: dt.datetime
    ohlcv: pd.DataFrame
    features: pd.DataFrame
    market_state: MarketState
    data_quality_status: str
    regime_label: str
    regime_probabilities: dict
    regime_confidence: float
    strategy_intel: list[StrategyIntelligence]
    ranking: RankingDecision


def run_pipeline(
    instrument: str,
    timeframe: str,
    start: dt.datetime,
    end: dt.datetime,
    quantity: int = 50,
) -> PipelineOutput:
    data_manager = DataManager()
    ohlcv, metadata = data_manager.get_ohlcv(instrument, timeframe, start, end)

    if len(ohlcv) < 120:
        raise ValueError(
            f"Not enough data to run the pipeline for {instrument} {timeframe} "
            f"({len(ohlcv)} rows). Widen the date range."
        )

    features = compute_features(ohlcv)
    data_quality_status = metadata["quality_status"]

    market_state = build_current_state(instrument, features, data_quality_status)
    if market_state is None:
        raise ValueError("Not enough warmup bars to build a current market state.")

    regime_label, regime_probs = classify_regime(market_state.features)
    conf = regime_confidence(regime_probs)

    log_event(
        "pipeline",
        f"Market state built for {instrument} at {market_state.timestamp}: regime={regime_label} (conf={conf:.2f})",
        level="INFO",
    )

    strategies = get_all_strategies()
    if float(ohlcv["volume"].fillna(0).sum()) <= 0:
        # No traded volume (e.g. index candles): volume features are NaN and NaN comparisons
        # silently pass, so volume-dependent strategies would emit unreliable signals.
        skipped = [s.name for s in strategies if _uses_volume(s)]
        strategies = [s for s in strategies if not _uses_volume(s)]
        log_event("pipeline", f"No volume in data; skipped volume-dependent strategies: {skipped}", level="WARNING")
    strategy_intel: list[StrategyIntelligence] = []
    scores: list[StrategyScore] = []

    for strategy in strategies:
        backtest = run_backtest(strategy, ohlcv, features, instrument, split="RESEARCH", quantity=quantity)
        observations_df = _observations_to_df(backtest, features)

        analogues = find_analogues(market_state.features, observations_df, top_k=30)
        conditional = conditional_metrics_from_analogues(analogues)
        conditional["strategy_name"] = strategy.name

        max_dd = backtest.metrics.get("max_drawdown")
        score = score_strategy(conditional, max_dd)

        strategy_intel.append(
            StrategyIntelligence(
                strategy_name=strategy.name,
                global_metrics=backtest.metrics,
                conditional_metrics=conditional,
                analogues=analogues,
                score=score,
                backtest=backtest,
            )
        )
        scores.append(score)

    ranking = rank_and_select(scores, data_quality_status)

    log_event(
        "pipeline",
        f"Ranking decision: selected={ranking.selected_strategy}, no_trade={ranking.is_no_trade}, reason={ranking.reason}",
        level="INFO",
    )

    return PipelineOutput(
        instrument=instrument,
        timestamp=market_state.timestamp,
        ohlcv=ohlcv,
        features=features,
        market_state=market_state,
        data_quality_status=data_quality_status,
        regime_label=regime_label,
        regime_probabilities=regime_probs,
        regime_confidence=conf,
        strategy_intel=strategy_intel,
        ranking=ranking,
    )


_VOLUME_FEATURES = {"relative_volume", "volume_acceleration", "vwap", "vwap_distance_pct"}


def _uses_volume(strategy) -> bool:
    return bool(_VOLUME_FEATURES & set(strategy.required_features)) or "volume" in strategy.name.lower()


def _observations_to_df(backtest: BacktestResult, features: pd.DataFrame) -> pd.DataFrame:
    """Flatten each backtest trade's entry-time features + outcome into a flat DataFrame
    suitable for the analogue engine (one row per historical observation)."""
    feat_by_ts = features.set_index("timestamp")
    rows = []
    for t in backtest.trades:
        if t.setup.timestamp not in feat_by_ts.index:
            continue
        feat_row = feat_by_ts.loc[t.setup.timestamp]
        row = {c: feat_row.get(c) for c in COMPARISON_FEATURES}
        row["r_multiple"] = t.r_multiple
        row["outcome"] = t.outcome
        row["entry_timestamp"] = t.setup.timestamp
        row["strategy_name"] = backtest.strategy_name
        rows.append(row)
    return pd.DataFrame(rows)
