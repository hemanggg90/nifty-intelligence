from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, evaluate_trade


def _account(**overrides):
    base = dict(
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
    base.update(overrides)
    return AccountState(**base)


def _trade(**overrides):
    base = dict(
        strategy_name="Momentum",
        direction="LONG",
        entry_price=100.0,
        stop_price=99.0,
        target_price=103.0,
        quantity=50,
        relative_volume=1.0,
        data_quality_status="OK",
    )
    base.update(overrides)
    return ProposedTrade(**base)


def test_kill_switch_vetoes_everything():
    decision = evaluate_trade(_account(kill_switch_engaged=True), _trade())
    assert not decision.approved
    assert "kill switch" in decision.reason.lower()


def test_disconnected_broker_vetoes():
    decision = evaluate_trade(_account(broker_connected=False), _trade())
    assert not decision.approved


def test_bad_data_quality_vetoes():
    decision = evaluate_trade(_account(), _trade(data_quality_status="FAIL"))
    assert not decision.approved


def test_daily_loss_limit_vetoes():
    decision = evaluate_trade(_account(daily_pnl=-50_000), _trade())  # 5% loss > 3% default limit
    assert not decision.approved
    assert "daily loss" in decision.reason.lower()


def test_max_trades_per_day_vetoes():
    decision = evaluate_trade(_account(trades_today=10), _trade())
    assert not decision.approved


def test_risk_per_trade_too_large_vetoes():
    # Risk = |100-99|*50 = 50 => 0.005% of equity, should pass. Make stop far away to exceed 1% limit.
    decision = evaluate_trade(_account(), _trade(stop_price=50.0, quantity=1000))
    assert not decision.approved
    assert "risk" in decision.reason.lower()


def test_valid_trade_is_approved():
    decision = evaluate_trade(_account(), _trade())
    assert decision.approved
    assert decision.decision_id.startswith("RISK-")


def test_illiquid_market_vetoes():
    decision = evaluate_trade(_account(), _trade(relative_volume=0.1))
    assert not decision.approved
    assert "liquidity" in decision.reason.lower()


def test_every_veto_has_a_reason():
    for acct, trade in [
        (_account(kill_switch_engaged=True), _trade()),
        (_account(), _trade(data_quality_status="DEGRADED")),
        (_account(trades_today=99), _trade()),
    ]:
        decision = evaluate_trade(acct, trade)
        assert decision.reason and len(decision.reason) > 0
