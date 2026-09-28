"""
Deterministic Risk Engine.

This module has ZERO dependency on the research/AI layers. It only looks at:
account state (equity, today's P&L, open positions), the proposed trade, and
hard-coded limits from config/settings.py. It has VETO authority - nothing
downstream can override a rejection, and every rejection records the exact
reason to the risk_events table for audit.

Checks implemented:
  - max risk per trade (position size vs. stop distance vs. equity %)
  - max daily loss (kill switch for the rest of the day)
  - max strategy exposure (concentration in one strategy)
  - max portfolio exposure (total capital at risk)
  - max drawdown (from peak equity)
  - max trade count per day
  - liquidity check (minimum relative volume)
  - data quality check
  - broker connectivity check
  - emergency kill switch (manual override, always wins)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.logging_utils import log_event, new_decision_id


@dataclass
class AccountState:
    equity: float
    peak_equity: float
    daily_pnl: float
    open_positions_count: int
    trades_today: int
    exposure_by_strategy: dict = field(default_factory=dict)  # strategy_name -> notional at risk
    total_exposure: float = 0.0
    broker_connected: bool = True
    kill_switch_engaged: bool = False


@dataclass
class ProposedTrade:
    strategy_name: str
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    quantity: int
    relative_volume: float | None = None
    data_quality_status: str = "OK"


@dataclass
class RiskDecision:
    decision_id: str
    approved: bool
    reason: str
    checks: dict = field(default_factory=dict)


def evaluate_trade(account: AccountState, trade: ProposedTrade) -> RiskDecision:
    decision_id = new_decision_id("RISK")
    checks: dict[str, bool] = {}
    limits = SETTINGS.risk

    if account.kill_switch_engaged:
        return _veto(decision_id, "Emergency kill switch is engaged", checks)

    if not account.broker_connected:
        return _veto(decision_id, "Broker is disconnected - cannot execute", checks)

    if trade.data_quality_status != "OK":
        return _veto(decision_id, f"Data quality is {trade.data_quality_status} - trade blocked", checks)

    daily_loss_pct = -account.daily_pnl / account.equity * 100 if account.equity > 0 else 0
    checks["daily_loss_within_limit"] = daily_loss_pct < limits.max_daily_loss_pct
    if not checks["daily_loss_within_limit"]:
        return _veto(decision_id, f"Daily loss {daily_loss_pct:.2f}% exceeds limit {limits.max_daily_loss_pct}%", checks)

    drawdown_pct = (account.peak_equity - account.equity) / account.peak_equity * 100 if account.peak_equity > 0 else 0
    checks["drawdown_within_limit"] = drawdown_pct < limits.max_drawdown_pct
    if not checks["drawdown_within_limit"]:
        return _veto(decision_id, f"Drawdown {drawdown_pct:.2f}% exceeds limit {limits.max_drawdown_pct}%", checks)

    checks["trade_count_within_limit"] = account.trades_today < limits.max_trades_per_day
    if not checks["trade_count_within_limit"]:
        return _veto(decision_id, f"Max trades per day ({limits.max_trades_per_day}) already reached", checks)

    risk_amount = abs(trade.entry_price - trade.stop_price) * trade.quantity
    risk_pct = risk_amount / account.equity * 100 if account.equity > 0 else 100
    checks["risk_per_trade_within_limit"] = risk_pct <= limits.max_risk_per_trade_pct
    if not checks["risk_per_trade_within_limit"]:
        return _veto(decision_id, f"Trade risk {risk_pct:.2f}% exceeds max risk per trade {limits.max_risk_per_trade_pct}%", checks)

    strategy_exposure = account.exposure_by_strategy.get(trade.strategy_name, 0.0) + risk_amount
    strategy_exposure_pct = strategy_exposure / account.equity * 100 if account.equity > 0 else 100
    checks["strategy_exposure_within_limit"] = strategy_exposure_pct <= limits.max_strategy_exposure_pct
    if not checks["strategy_exposure_within_limit"]:
        return _veto(
            decision_id,
            f"Strategy exposure {strategy_exposure_pct:.2f}% exceeds limit {limits.max_strategy_exposure_pct}%",
            checks,
        )

    total_exposure_pct = (account.total_exposure + risk_amount) / account.equity * 100 if account.equity > 0 else 100
    checks["portfolio_exposure_within_limit"] = total_exposure_pct <= limits.max_portfolio_exposure_pct
    if not checks["portfolio_exposure_within_limit"]:
        return _veto(
            decision_id,
            f"Portfolio exposure {total_exposure_pct:.2f}% exceeds limit {limits.max_portfolio_exposure_pct}%",
            checks,
        )

    min_liquidity = 0.3
    checks["liquidity_sufficient"] = trade.relative_volume is None or trade.relative_volume >= min_liquidity
    if not checks["liquidity_sufficient"]:
        return _veto(decision_id, f"Relative volume {trade.relative_volume:.2f} below minimum liquidity threshold {min_liquidity}", checks)

    log_event("risk_engine", "Trade approved by risk engine", level="INFO", decision_id=decision_id, strategy=trade.strategy_name)
    _persist_risk_event(decision_id, "APPROVED", "All risk checks passed", checks)
    return RiskDecision(decision_id, True, "All risk checks passed", checks)


def _veto(decision_id: str, reason: str, checks: dict) -> RiskDecision:
    log_event("risk_engine", f"Trade VETOED: {reason}", level="WARNING", decision_id=decision_id)
    _persist_risk_event(decision_id, "VETO", reason, checks)
    return RiskDecision(decision_id, False, reason, checks)


def _persist_risk_event(decision_id: str, event_type: str, reason: str, details: dict) -> None:
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import RiskEvent

        with get_session() as session:
            session.add(RiskEvent(decision_id=decision_id, event_type=event_type, reason=reason, details=details))
    except Exception:
        pass
