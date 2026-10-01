"""
Event-driven backtesting engine.

Design principles (per project spec):
  - No look-ahead: strategies compute setups using only data up to bar i
    (enforced by the feature engine and strategy implementations).
  - No overlapping trades per strategy: once a trade is open, subsequent
    setups are ignored until it closes (sequential, realistic capital use).
  - Realistic costs: brokerage + STT-style fee + slippage are subtracted from
    every trade's gross P&L to produce net P&L.
  - Train/validation/out-of-sample separation and walk-forward are supported
    via `run_backtest(..., date_range=...)` being called on disjoint windows;
    `walk_forward` automates rolling that across multiple windows.

This is intentionally NOT a vectorized backtest - trades are simulated one
at a time via BaseStrategy.simulate_trade, which walks forward bar-by-bar
from entry, to avoid the classic vectorized-backtest bug of applying a
signal and a fill in the same timestep.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.strategies.base_strategy import BaseStrategy, TradeResult


@dataclass
class BacktestResult:
    run_id: str
    strategy_name: str
    instrument: str
    split: str
    trades: list[TradeResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _apply_costs(trade: TradeResult, entry_price: float, quantity: int) -> tuple[float, float, float, float]:
    """Return (commissions, fees, slippage, net_pnl)."""
    costs = SETTINGS.costs
    commissions = costs.brokerage_per_order_inr * 2  # entry + exit
    turnover = (entry_price + (trade.exit_price or entry_price)) * quantity
    fees = turnover * costs.stt_rate
    slippage_amount = costs.slippage_ticks * costs.tick_size * quantity * 2  # entry + exit
    net_pnl = trade.gross_pnl * quantity - commissions - fees - slippage_amount
    return commissions, fees, slippage_amount, net_pnl


def net_r_multiple(trade: TradeResult, quantity: int) -> float:
    """Net-of-cost R: (gross P&L - brokerage - fees - slippage) / amount risked. This, not the
    gross `TradeResult.r_multiple`, is what a strategy actually earns and what ranking must use."""
    net_pnl = _apply_costs(trade, trade.setup.entry_price, quantity)[3]
    risk = abs(trade.setup.entry_price - trade.setup.stop_price) * quantity
    return net_pnl / risk if risk > 0 else 0.0


def max_drawdown_in_r(net_r: np.ndarray) -> float | None:
    """Worst peak-to-trough fall of cumulative net R (negative number; None with no trades)."""
    if len(net_r) == 0:
        return None
    equity = np.cumsum(net_r)
    return float((equity - np.maximum.accumulate(equity)).min())


def run_backtest(
    strategy: BaseStrategy,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    instrument: str,
    split: str = "IN_SAMPLE",
    quantity: int = 50,
    max_holding_bars: int = 75,
    persist: bool = True,
) -> BacktestResult:
    run_id = f"BT-{uuid.uuid4().hex[:12]}"

    setups = strategy.generate_historical_setups(ohlcv, features)
    ts_to_idx = {ts: i for i, ts in enumerate(ohlcv["timestamp"])}
    feat_by_ts = features.set_index("timestamp")

    trades: list[TradeResult] = []
    next_available_idx = 0

    for setup in setups:
        entry_idx = ts_to_idx.get(setup.timestamp)
        if entry_idx is None or entry_idx < next_available_idx:
            continue  # skip overlapping setups while a trade would still be open

        result = strategy.simulate_trade(setup, ohlcv, entry_idx, max_holding_bars=max_holding_bars)
        trades.append(result)

        exit_idx = ts_to_idx.get(result.exit_timestamp, entry_idx + max_holding_bars)
        next_available_idx = exit_idx + 1

    run = BacktestResult(run_id=run_id, strategy_name=strategy.name, instrument=instrument, split=split, trades=trades)
    run.metrics = _compute_run_metrics(trades, quantity)

    if persist:
        _persist_backtest(run, strategy, ohlcv, features, feat_by_ts, quantity)
    return run


def _compute_run_metrics(trades: list[TradeResult], quantity: int) -> dict:
    if not trades:
        return {"n_trades": 0, "status": "NO_TRADES"}

    r_multiples = np.array([t.r_multiple for t in trades])
    net_pnls = []
    for t in trades:
        _, _, _, net_pnl = _apply_costs(t, t.setup.entry_price, quantity)
        net_pnls.append(net_pnl)
    net_pnls = np.array(net_pnls)

    wins = r_multiples > 0
    win_rate = float(wins.mean())
    avg_win = r_multiples[wins].mean() if wins.any() else 0.0
    avg_loss = r_multiples[~wins].mean() if (~wins).any() else 0.0
    payoff_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else float("nan")

    net_rs = np.array([net_r_multiple(t, quantity) for t in trades])
    equity_curve = np.cumsum(net_pnls)
    running_max = np.maximum.accumulate(equity_curve) if len(equity_curve) else np.array([0])
    drawdown = equity_curve - running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    r_std = r_multiples.std(ddof=1) if len(r_multiples) > 1 else np.nan
    sharpe = float(r_multiples.mean() / r_std * np.sqrt(252)) if r_std and not np.isnan(r_std) and r_std > 0 else float("nan")

    downside = r_multiples[r_multiples < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = float(r_multiples.mean() / downside_std * np.sqrt(252)) if downside_std and not np.isnan(downside_std) and downside_std > 0 else float("nan")

    return {
        "n_trades": len(trades),
        "win_rate": win_rate,
        "prob_positive_return": float((net_pnls > 0).mean()),
        "expected_r": float(r_multiples.mean()),
        "median_r": float(np.median(r_multiples)),
        "avg_r": float(r_multiples.mean()),
        "payoff_ratio": payoff_ratio,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,  # rupees (display)
        "max_drawdown_r": max_drawdown_in_r(net_rs),  # R units (used by the ranker)
        "net_expected_r": float(net_rs.mean()),
        "avg_mfe": float(np.mean([t.mfe for t in trades])),
        "avg_mae": float(np.mean([t.mae for t in trades])),
        "avg_holding_bars": float(np.mean([t.holding_period_bars for t in trades])),
        "net_pnl_total": float(net_pnls.sum()),
        "cost_adjusted_expectancy": float(net_pnls.mean()),
    }


def _persist_backtest(run: BacktestResult, strategy: BaseStrategy, ohlcv, features, feat_by_ts, quantity: int) -> None:
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import BacktestRun, BacktestTrade, StrategyObservation

        with get_session() as session:
            session.add(
                BacktestRun(
                    run_id=run.run_id,
                    strategy_name=run.strategy_name,
                    instrument=run.instrument,
                    timeframe=None,
                    start_date=ohlcv["timestamp"].min() if len(ohlcv) else None,
                    end_date=ohlcv["timestamp"].max() if len(ohlcv) else None,
                    split=run.split,
                    parameters=strategy.parameters,
                    metrics=run.metrics,
                )
            )

            for t in run.trades:
                commissions, fees, slippage, net_pnl = _apply_costs(t, t.setup.entry_price, quantity)
                session.add(
                    BacktestTrade(
                        run_id=run.run_id,
                        entry_timestamp=t.setup.timestamp,
                        exit_timestamp=t.exit_timestamp,
                        direction=t.setup.direction,
                        entry_price=t.setup.entry_price,
                        stop_price=t.setup.stop_price,
                        target_price=t.setup.target_price,
                        exit_price=t.exit_price,
                        r_multiple=t.r_multiple,
                        net_pnl=net_pnl,
                        outcome=t.outcome,
                    )
                )

                feat_row = feat_by_ts.loc[t.setup.timestamp] if t.setup.timestamp in feat_by_ts.index else None
                market_state_features = _row_to_clean_dict(feat_row) if feat_row is not None else {}
                regime_label = None
                if feat_row is not None:
                    from quant_intelligence.regimes.regime_engine import classify_regime

                    regime_label, _ = classify_regime(market_state_features)

                session.add(
                    StrategyObservation(
                        strategy_name=strategy.name,
                        instrument=run.instrument,
                        entry_timestamp=t.setup.timestamp,
                        exit_timestamp=t.exit_timestamp,
                        market_state_features=market_state_features,
                        regime_label=regime_label,
                        direction=t.setup.direction,
                        entry_price=t.setup.entry_price,
                        stop_price=t.setup.stop_price,
                        target_price=t.setup.target_price,
                        exit_price=t.exit_price,
                        r_multiple=t.r_multiple,
                        holding_period_bars=t.holding_period_bars,
                        mfe=t.mfe,
                        mae=t.mae,
                        gross_pnl=t.gross_pnl * quantity,
                        commissions=commissions,
                        fees=fees,
                        slippage=slippage,
                        net_pnl=net_pnl,
                        outcome="WIN" if t.r_multiple > 0 else ("LOSS" if t.r_multiple < 0 else "SCRATCH"),
                        source="backtest",
                        run_id=run.run_id,
                    )
                )
    except Exception:
        # Persistence must never crash a research run; caller still has run.trades/metrics in memory.
        pass


def _row_to_clean_dict(row: pd.Series) -> dict:
    d = row.to_dict()
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def walk_forward(
    strategy_factory,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    instrument: str,
    n_folds: int = 4,
    quantity: int = 50,
) -> list[BacktestResult]:
    """Rolling walk-forward: split the data into n_folds contiguous chronological
    windows; each fold's data is treated as a fresh out-of-sample test window.
    strategy_factory() must return a new strategy instance (parameters fixed in
    advance, i.e. NOT re-optimized per fold, to avoid look-ahead parameter fitting).
    """
    n = len(ohlcv)
    fold_size = n // n_folds
    results = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = n if fold == n_folds - 1 else (fold + 1) * fold_size
        fold_ohlcv = ohlcv.iloc[start:end].reset_index(drop=True)
        fold_features = features.iloc[start:end].reset_index(drop=True)
        if len(fold_ohlcv) < 50:
            continue
        strategy = strategy_factory()
        result = run_backtest(
            strategy, fold_ohlcv, fold_features, instrument, split=f"WALK_FORWARD_FOLD_{fold+1}", quantity=quantity
        )
        results.append(result)
    return results
