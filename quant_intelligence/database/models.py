"""
SQLAlchemy ORM models for the Quant Intelligence System.

Designed for SQLite in local dev (zero setup) and a straightforward migration
path to PostgreSQL later (no SQLite-specific types are used).
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), unique=True, nullable=False)
    exchange = Column(String(16), default="NSE")
    segment = Column(String(16), default="INDEX")  # INDEX / FUT / OPT / EQ
    lot_size = Column(Integer, default=50)
    tick_size = Column(Float, default=0.05)
    created_at = Column(DateTime, default=utcnow)


class MarketDataMetadata(Base):
    __tablename__ = "market_data_metadata"

    id = Column(Integer, primary_key=True)
    instrument = Column(String(32), nullable=False)
    timeframe = Column(String(16), nullable=False)
    source = Column(String(32), nullable=False)  # csv, synthetic, dhan
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)
    n_rows = Column(Integer)
    quality_status = Column(String(16))  # OK / DEGRADED / FAIL
    quality_report = Column(JSON)
    fetched_at = Column(DateTime, default=utcnow)


class MarketState(Base):
    __tablename__ = "market_states"

    id = Column(Integer, primary_key=True)
    instrument = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    features = Column(JSON, nullable=False)  # full feature vector snapshot
    data_quality_status = Column(String(16), default="OK")
    created_at = Column(DateTime, default=utcnow)


class RegimeState(Base):
    __tablename__ = "regime_states"

    id = Column(Integer, primary_key=True)
    instrument = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    label = Column(String(32), nullable=False)
    probabilities = Column(JSON, nullable=False)  # {regime_name: prob}
    confidence = Column(Float)
    created_at = Column(DateTime, default=utcnow)


class StrategyDefinition(Base):
    __tablename__ = "strategy_definitions"

    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(Text)
    parameters = Column(JSON)
    required_features = Column(JSON)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)


class StrategyObservation(Base):
    """One historical (backtested or live) occurrence of a strategy setup/trade."""

    __tablename__ = "strategy_observations"

    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(64), nullable=False, index=True)
    instrument = Column(String(32), nullable=False)
    entry_timestamp = Column(DateTime, nullable=False, index=True)
    exit_timestamp = Column(DateTime)
    market_state_features = Column(JSON, nullable=False)
    regime_label = Column(String(32))
    direction = Column(String(8))  # LONG / SHORT
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    exit_price = Column(Float)
    r_multiple = Column(Float)
    holding_period_bars = Column(Integer)
    mfe = Column(Float)
    mae = Column(Float)
    gross_pnl = Column(Float)
    commissions = Column(Float)
    fees = Column(Float)
    slippage = Column(Float)
    net_pnl = Column(Float)
    outcome = Column(String(16))  # WIN / LOSS / SCRATCH
    source = Column(String(16), default="backtest")  # backtest / paper / live
    run_id = Column(String(64))  # links back to a backtest_runs.run_id
    created_at = Column(DateTime, default=utcnow)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(String(64), unique=True, nullable=False)
    strategy_name = Column(String(64), nullable=False)
    instrument = Column(String(32), nullable=False)
    timeframe = Column(String(16))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    split = Column(String(16))  # IN_SAMPLE / VALIDATION / OUT_OF_SAMPLE / WALK_FORWARD
    parameters = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=utcnow)


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"

    id = Column(Integer, primary_key=True)
    run_id = Column(String(64), ForeignKey("backtest_runs.run_id"), nullable=False, index=True)
    entry_timestamp = Column(DateTime, nullable=False)
    exit_timestamp = Column(DateTime)
    direction = Column(String(8))
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    exit_price = Column(Float)
    r_multiple = Column(Float)
    net_pnl = Column(Float)
    outcome = Column(String(16))


class StrategyMetrics(Base):
    """Aggregated (global or conditional) performance metrics for a strategy."""

    __tablename__ = "strategy_metrics"

    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(64), nullable=False, index=True)
    instrument = Column(String(32), nullable=False)
    condition_type = Column(String(16))  # GLOBAL / REGIME / ANALOGUE
    condition_key = Column(String(64))  # e.g. regime label, or 'current_state'
    sample_size = Column(Integer)
    win_rate = Column(Float)
    prob_positive_return = Column(Float)
    expected_r = Column(Float)
    median_r = Column(Float)
    avg_r = Column(Float)
    payoff_ratio = Column(Float)
    sharpe = Column(Float)
    sortino = Column(Float)
    max_drawdown = Column(Float)
    avg_mfe = Column(Float)
    avg_mae = Column(Float)
    avg_holding_bars = Column(Float)
    cost_adjusted_expectancy = Column(Float)
    confidence_label = Column(String(24))  # HIGH / MEDIUM / LOW / INSUFFICIENT_DATA
    computed_at = Column(DateTime, default=utcnow)


class AnalogueMatch(Base):
    __tablename__ = "analogue_matches"

    id = Column(Integer, primary_key=True)
    query_timestamp = Column(DateTime, nullable=False, index=True)
    instrument = Column(String(32), nullable=False)
    strategy_name = Column(String(64), nullable=False)
    observation_id = Column(Integer, ForeignKey("strategy_observations.id"))
    similarity_score = Column(Float)
    created_at = Column(DateTime, default=utcnow)


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    decision_id = Column(String(64), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    instrument = Column(String(32), nullable=False)
    strategy_name = Column(String(64), nullable=False)
    direction = Column(String(8))
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    setup_status = Column(String(24))  # WAITING / TRIGGERED / EXPIRED
    created_at = Column(DateTime, default=utcnow)


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    order_id = Column(String(64), unique=True, nullable=False)
    decision_id = Column(String(64))
    timestamp = Column(DateTime, nullable=False)
    instrument = Column(String(32), nullable=False)
    strategy_name = Column(String(64))
    direction = Column(String(8))
    quantity = Column(Integer)
    order_type = Column(String(16))  # MARKET / LIMIT / SL / SL-M
    price = Column(Float)
    status = Column(String(24))  # PENDING/FILLED/REJECTED/CANCELLED
    reject_reason = Column(Text)
    mode = Column(String(8))  # PAPER / LIVE
    broker = Column(String(16))
    security_id = Column(String(32))
    exchange_segment = Column(String(16))
    product_type = Column(String(16))
    option_type = Column(String(4))  # CE / PE
    strike = Column(Float)
    expiry = Column(String(16))
    lot_size = Column(Integer)


class Fill(Base):
    __tablename__ = "fills"

    id = Column(Integer, primary_key=True)
    order_id = Column(String(64), ForeignKey("orders.order_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    fill_price = Column(Float)
    quantity = Column(Integer)
    slippage = Column(Float)


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    position_id = Column(String(64), unique=True, nullable=False)
    instrument = Column(String(32), nullable=False)
    strategy_name = Column(String(64))
    direction = Column(String(8))
    quantity = Column(Integer)
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    status = Column(String(16))  # OPEN / CLOSED / STALE (left open by an earlier day, never closed)
    opened_at = Column(DateTime)
    closed_at = Column(DateTime)
    exit_price = Column(Float)
    net_pnl = Column(Float)
    exit_reason = Column(String(32))  # STOP / TARGET / EOD_SQUARE_OFF / MANUAL / STALE_ON_RESTART
    mode = Column(String(8))
    underlying = Column(String(32))
    security_id = Column(String(32))
    option_type = Column(String(4))  # CE / PE
    strike = Column(Float)
    expiry = Column(String(16))
    transaction = Column(String(8))  # BUY / SELL


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=utcnow)
    decision_id = Column(String(64))
    event_type = Column(String(32))  # VETO / WARNING / KILL_SWITCH / APPROVED
    reason = Column(Text)
    details = Column(JSON)


class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=utcnow)
    component = Column(String(32))
    level = Column(String(16))  # INFO / WARNING / ERROR
    message = Column(Text)
    details = Column(JSON)


class ResearchReport(Base):
    __tablename__ = "research_reports"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=utcnow)
    instrument = Column(String(32))
    report_type = Column(String(32))
    content_json = Column(JSON)
    content_markdown = Column(Text)
