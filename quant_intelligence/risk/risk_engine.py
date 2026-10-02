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
  - max drawdown (from peak equity, carried across days)
  - max trade count per day
  - max open positions
  - correlated-group exposure (same-direction count and open risk per group)
  - losing-streak cooldown, per-strategy daily loss stop, profit giveback lock
  - liquidity check (minimum relative volume)
  - data quality check
  - broker connectivity check
  - emergency kill switch (manual override, always wins)

Size throttle: `risk_multiplier` scales the per-trade risk budget down as drawdown, the day's loss
or a losing streak builds. The per-trade check and `auto_trader.size_position` both use it, so sizes
shrink in a bad stretch instead of staying full until the hard drawdown veto stops all trading.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from quant_intelligence.config.risk_groups import group_for
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.logging_utils import log_event, new_decision_id
from quant_intelligence.utils.timeutil import now_ist


@dataclass
class AccountState:
    equity: float  # marked to market: cash + unrealised P&L of open positions
    peak_equity: float
    daily_pnl: float  # realised today + unrealised
    open_positions_count: int
    trades_today: int
    exposure_by_strategy: dict = field(default_factory=dict)  # strategy_name -> notional at risk
    total_exposure: float = 0.0
    broker_connected: bool = True
    kill_switch_engaged: bool = False
    unrealised_pnl: float = 0.0
    open_positions: list = field(default_factory=list)  # [{"instrument", "direction", "risk"}]
    closed_today: list = field(default_factory=list)  # [{"strategy", "net_pnl", "closed_at"}], oldest first
    day_peak_pnl: float = 0.0  # highest daily_pnl reached today


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
    instrument: str = ""


@dataclass
class RiskDecision:
    decision_id: str
    approved: bool
    reason: str
    checks: dict = field(default_factory=dict)


def _pct(amount: float, equity: float) -> float:
    return amount / equity * 100 if equity > 0 else 0.0


def drawdown_pct(account: AccountState) -> float:
    return _pct(account.peak_equity - account.equity, account.peak_equity) if account.peak_equity > 0 else 0.0


def loss_streak(account: AccountState) -> int:
    """Consecutive losing trades at the end of today's closed trades (a non-loss ends the streak)."""
    n = 0
    for t in reversed(account.closed_today):
        if (t.get("net_pnl") or 0.0) < 0:
            n += 1
        else:
            break
    return n


def risk_multiplier(account: AccountState) -> tuple[float, str]:
    """Fraction (0-1] of the normal per-trade risk budget allowed right now, and why.

    The smallest of: the drawdown throttle (1.0 below DD_THROTTLE_START_PCT, falling linearly to
    DD_THROTTLE_FLOOR at MAX_DRAWDOWN_PCT), 0.5 once half the daily loss limit is used, and 0.5 after
    LOSS_STREAK_THROTTLE_AFTER consecutive losers today."""
    limits = SETTINGS.risk
    factors: list[tuple[float, str]] = [(1.0, "full size")]

    dd = drawdown_pct(account)
    start, end = limits.dd_throttle_start_pct, limits.max_drawdown_pct
    if end > start and dd > start:
        floor = min(max(limits.dd_throttle_floor, 0.0), 1.0)
        frac = min((dd - start) / (end - start), 1.0)
        factors.append((1.0 - (1.0 - floor) * frac, f"drawdown {dd:.1f}% (throttle from {start:g}%)"))

    daily_loss = _pct(-account.daily_pnl, account.equity)
    if limits.max_daily_loss_pct > 0 and daily_loss >= limits.max_daily_loss_pct / 2:
        factors.append((0.5, f"day's loss {daily_loss:.1f}% is over half the {limits.max_daily_loss_pct:g}% limit"))

    streak = loss_streak(account)
    if limits.loss_streak_throttle_after > 0 and streak >= limits.loss_streak_throttle_after:
        factors.append((0.5, f"{streak} losing trades in a row"))

    return min(factors, key=lambda f: f[0])


def evaluate_trade(account: AccountState, trade: ProposedTrade, now: dt.datetime | None = None) -> RiskDecision:
    decision_id = new_decision_id("RISK")
    checks: dict[str, bool] = {}
    limits = SETTINGS.risk
    now = now or now_ist()

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

    dd = drawdown_pct(account)
    checks["drawdown_within_limit"] = dd < limits.max_drawdown_pct
    if not checks["drawdown_within_limit"]:
        return _veto(decision_id, f"Drawdown {dd:.2f}% exceeds limit {limits.max_drawdown_pct}%", checks)

    day_peak_pct = _pct(account.day_peak_pnl, account.equity)
    checks["profit_lock_not_triggered"] = not (
        limits.profit_lock_trigger_pct > 0
        and day_peak_pct >= limits.profit_lock_trigger_pct
        and account.daily_pnl <= account.day_peak_pnl * (1 - limits.profit_lock_giveback)
    )
    if not checks["profit_lock_not_triggered"]:
        return _veto(
            decision_id,
            f"Profit lock: today's P&L peaked at {day_peak_pct:.2f}% and has given back "
            f"{limits.profit_lock_giveback:.0%} or more - no new trades today",
            checks,
        )

    streak = loss_streak(account)
    if limits.loss_streak_pause > 0 and streak >= limits.loss_streak_pause:
        last_closed = account.closed_today[-1].get("closed_at")
        resume_at = last_closed + dt.timedelta(minutes=limits.loss_streak_pause_min) if last_closed else None
        checks["loss_streak_cooldown_over"] = resume_at is not None and now >= resume_at
        if not checks["loss_streak_cooldown_over"]:
            when = f" until {resume_at:%H:%M}" if resume_at else ""
            return _veto(decision_id, f"{streak} losing trades in a row - new entries paused{when}", checks)

    checks["trade_count_within_limit"] = account.trades_today < limits.max_trades_per_day
    if not checks["trade_count_within_limit"]:
        return _veto(decision_id, f"Max trades per day ({limits.max_trades_per_day}) already reached", checks)

    checks["open_positions_within_limit"] = (
        limits.max_open_positions <= 0 or account.open_positions_count < limits.max_open_positions
    )
    if not checks["open_positions_within_limit"]:
        return _veto(decision_id, f"{account.open_positions_count} positions already open (max {limits.max_open_positions})", checks)

    strategy_losses = sum(
        1 for t in account.closed_today if t.get("strategy") == trade.strategy_name and (t.get("net_pnl") or 0.0) < 0
    )
    checks["strategy_losses_within_limit"] = (
        limits.strategy_max_losses_per_day <= 0 or strategy_losses < limits.strategy_max_losses_per_day
    )
    if not checks["strategy_losses_within_limit"]:
        return _veto(
            decision_id,
            f"{trade.strategy_name} has lost {strategy_losses} trades today (max {limits.strategy_max_losses_per_day}) - done for the day",
            checks,
        )

    multiplier, why = risk_multiplier(account)
    checks["risk_multiplier"] = round(multiplier, 3)
    allowed_risk_pct = limits.max_risk_per_trade_pct * multiplier
    risk_amount = abs(trade.entry_price - trade.stop_price) * trade.quantity
    risk_pct = risk_amount / account.equity * 100 if account.equity > 0 else 100
    checks["risk_per_trade_within_limit"] = risk_pct <= allowed_risk_pct
    if not checks["risk_per_trade_within_limit"]:
        scaled = f" (reduced to {multiplier:.0%}: {why})" if multiplier < 1 else ""
        return _veto(decision_id, f"Trade risk {risk_pct:.2f}% exceeds max risk per trade {allowed_risk_pct:.2f}%{scaled}", checks)

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

    if trade.instrument:
        group = group_for(trade.instrument)
        in_group = [p for p in account.open_positions if group_for(p.get("instrument")) == group]
        same_dir = sum(1 for p in in_group if p.get("direction") == trade.direction)
        checks["group_positions_within_limit"] = (
            limits.max_group_positions <= 0 or same_dir + 1 <= limits.max_group_positions
        )
        if not checks["group_positions_within_limit"]:
            return _veto(
                decision_id,
                f"{same_dir} positions already open {trade.direction} in group {group} (max {limits.max_group_positions})",
                checks,
            )
        group_risk_pct = _pct(sum(p.get("risk") or 0.0 for p in in_group) + risk_amount, account.equity)
        checks["group_risk_within_limit"] = limits.max_group_risk_pct <= 0 or group_risk_pct <= limits.max_group_risk_pct
        if not checks["group_risk_within_limit"]:
            return _veto(
                decision_id,
                f"Open risk in group {group} would be {group_risk_pct:.2f}% of equity (max {limits.max_group_risk_pct}%)",
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
