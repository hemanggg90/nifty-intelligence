"""Loss per stopped trade (~Rs 1,800 on Rs 10 lakh), the clear 'one lot is too big' outcome, and STOP_ATR_SCALE."""
import dataclasses
from types import SimpleNamespace

import pytest

from quant_intelligence.config import settings as settings_module
from quant_intelligence.execution import auto_trader
from quant_intelligence.reports import decision_log
from quant_intelligence.risk import risk_engine
from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, evaluate_trade
from quant_intelligence.strategies.registry import STRATEGY_CLASSES, get_strategy

EQUITY = 1_000_000.0


def _account(equity=EQUITY, daily_pnl=0.0, peak=None):
    return AccountState(equity=equity, peak_equity=peak or equity, daily_pnl=daily_pnl, open_positions_count=0, trades_today=0,
                        exposure_by_strategy={}, total_exposure=0.0, broker_connected=True, kill_switch_engaged=False)


@pytest.fixture
def new_limits(monkeypatch):
    """The shipped defaults (0.2% per trade, 1.5% daily) instead of the 1.0/3.0 the other tests are pinned to."""
    limits = dataclasses.replace(settings_module.SETTINGS.risk, max_risk_per_trade_pct=settings_module.DEFAULT_MAX_RISK_PER_TRADE_PCT,
                                 max_daily_loss_pct=settings_module.DEFAULT_MAX_DAILY_LOSS_PCT)
    new = dataclasses.replace(settings_module.SETTINGS, risk=limits)
    for module in (settings_module, auto_trader, risk_engine):
        monkeypatch.setattr(module, "SETTINGS", new)
    return limits


def test_the_shipped_defaults_are_the_agreed_ones():
    assert settings_module.DEFAULT_MAX_RISK_PER_TRADE_PCT == 0.2 and settings_module.DEFAULT_MAX_DAILY_LOSS_PCT == 1.5


def test_a_stopped_trade_never_loses_more_than_the_budget_and_at_most_one_lot_less(new_limits):
    """Whole lots only, so the loss is the budget rounded DOWN to a lot: <= Rs 1,800 and within one lot of it."""
    budget = auto_trader.risk_budget(_account())[0]
    assert budget == pytest.approx(1800.0)
    for stop_distance, lot in ((15.0, 75), (10.0, 75), (20.0, 30), (6.0, 75)):  # NIFTY-like and BANKNIFTY-like lots
        qty = auto_trader.size_position(_account(), 105.0, 105.0 - stop_distance, lot)
        loss_at_stop, one_lot = qty * stop_distance, lot * stop_distance
        assert qty % lot == 0 and loss_at_stop <= budget + 1e-9
        assert budget - loss_at_stop < one_lot  # could not have fitted another whole lot
    # a typical NIFTY trade: one lot, loss about Rs 1,100 at a 15-point premium stop
    assert auto_trader.size_position(_account(), 105.0, 90.0, 75) == 75


def test_the_budget_scales_with_equity_and_shrinks_in_a_bad_stretch(new_limits):
    assert auto_trader.risk_budget(_account())[0] == pytest.approx(1800.0)
    assert auto_trader.risk_budget(_account(equity=500_000.0))[0] == pytest.approx(900.0)
    losing_day = _account(daily_pnl=-8_000.0)  # over half of the 1.5% daily limit
    budget, mult, why = auto_trader.risk_budget(losing_day)
    assert mult == 0.5 and budget == pytest.approx(900.0) and "limit" in why


def test_the_risk_engine_approves_at_the_budget_and_vetoes_above_it(new_limits):
    ok = ProposedTrade("ORB", "LONG", 105.0, 90.0, 135.0, auto_trader.size_position(_account(), 105.0, 90.0, 75), instrument="NIFTY")
    assert evaluate_trade(_account(), ok).approved
    too_big = ProposedTrade("ORB", "LONG", 105.0, 90.0, 135.0, 300, instrument="NIFTY")  # 300 x 15 = Rs 4,500
    d = evaluate_trade(_account(), too_big)
    assert not d.approved and "exceeds max risk per trade 0.20%" in d.reason


def test_the_daily_loss_limit_is_now_one_and_a_half_percent(new_limits):
    trade = ProposedTrade("ORB", "LONG", 105.0, 90.0, 135.0, 75, instrument="NIFTY")
    d = evaluate_trade(_account(daily_pnl=-16_000.0), trade)  # 1.6% down
    assert not d.approved and "Daily loss" in d.reason and "1.5" in d.reason


def test_one_lot_above_the_budget_gives_zero_quantity_and_says_why(new_limits):
    # a stock option: lot 1,500, premium 40, stop 10 below -> one lot risks Rs 15,000 against a Rs 1,800 budget
    assert auto_trader.size_position(_account(), 40.0, 30.0, 1500) == 0
    reason = auto_trader._zero_size_reason(_account(), 40.0, 30.0, 1500)
    assert reason.startswith(auto_trader.NO_SIZE_PREFIX)
    assert "Rs 15,000" in reason and "Rs 1,800" in reason and "tighter stop" in reason


def test_the_capital_cap_is_named_when_it_is_the_cause(new_limits):
    reason = auto_trader._zero_size_reason(_account(), 400.0, 399.9, 1500)  # tiny stop, but 1 lot costs Rs 6 lakh
    assert "capital cap" in reason and reason.startswith("NO SIZE")


def test_a_reduced_budget_is_explained(new_limits):
    reason = auto_trader._zero_size_reason(_account(daily_pnl=-9_000.0), 105.0, 75.0, 75)
    assert "budget cut to 50%" in reason


def test_the_decision_log_files_it_as_a_triggered_setup_that_could_not_be_sized():
    ranking = SimpleNamespace(is_no_trade=False, selected_strategy="Momentum", tie_break=False, reason="r", ranked=[])
    out = SimpleNamespace(ranking=ranking, timestamp=__import__("pandas").Timestamp("2031-01-01 10:00"))
    res = SimpleNamespace(instrument="HDFCBANK", strategy_name="Momentum", setup_status="SETUP_TRIGGERED", order=None,
                          risk_decision_id=None, tag=None, reason=auto_trader._zero_size_reason(_account(), 40.0, 30.0, 1500))
    rec = decision_log.classify_cycle(out, res)
    assert rec["status"] == "TRIGGERED" and rec["reason_class"] == "NO_SIZE"


def test_the_report_counts_and_names_the_instruments_skipped_for_size():
    from quant_intelligence.reports import eod

    decisions = [{"status": "TRIGGERED", "reason_class": "NO_SIZE", "strategy": "Momentum", "tie_break": False, "reason": "", "instrument": i}
                 for i in ("HDFCBANK", "TCS", "TCS")]
    f = eod._funnel(decisions)
    assert f["no_size"] == 3 and f["no_size_instruments"] == ["HDFCBANK", "TCS"] and f["signals"] == 3


# ---------------------------------------------------------------- STOP_ATR_SCALE
def _with_scale(monkeypatch, scale):
    monkeypatch.setattr(settings_module, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, stop_atr_scale=scale))


def test_scale_one_changes_nothing(monkeypatch):
    _with_scale(monkeypatch, 1.0)
    s = get_strategy("Donchian Channel Breakout")
    assert s.parameters["stop_atr_mult"] == 2.0 and s.parameters["target_atr_mult"] == 4.0


def test_scaling_moves_stop_and_target_together_so_reward_to_risk_is_kept(monkeypatch):
    _with_scale(monkeypatch, 0.5)
    s = get_strategy("Donchian Channel Breakout")
    assert s.parameters["stop_atr_mult"] == 1.0 and s.parameters["target_atr_mult"] == 2.0
    assert s.parameters["cooldown_bars"] == 15  # other parameters untouched
    _, stop, target = s.calculate_entry_stop_target("LONG", 100.0, 10.0)
    assert (100.0 - stop) == 10.0 and (target - 100.0) == 20.0


def test_explicit_parameters_are_never_scaled(monkeypatch):
    _with_scale(monkeypatch, 0.5)
    s = get_strategy("Donchian Channel Breakout", {"stop_atr_mult": 3.0})
    assert s.parameters["stop_atr_mult"] == 3.0 and s.parameters["target_atr_mult"] == 2.0  # only the untouched default scaled


def test_every_atr_strategy_scales_and_structure_based_ones_are_left_alone(monkeypatch):
    _with_scale(monkeypatch, 0.5)
    scaled, untouched = [], []
    for name, cls in STRATEGY_CLASSES.items():
        keys = [k for k in cls.SCALED_PARAMETERS if k in cls.default_parameters]
        (scaled if keys else untouched).append(name)
        params = get_strategy(name).parameters
        for k in keys:
            assert params[k] == pytest.approx(cls.default_parameters[k] * 0.5)
        for k, v in cls.default_parameters.items():
            if k not in cls.SCALED_PARAMETERS:
                assert params[k] == v
    assert len(scaled) >= 10  # most strategies are ATR based


def test_the_scale_is_clamped_to_a_sane_range():
    import importlib
    import os

    old = os.environ.get("STOP_ATR_SCALE")
    try:
        for raw, expected in (("0.01", 0.2), ("9", 3.0), ("0.75", 0.75), ("junk", 1.0)):
            os.environ["STOP_ATR_SCALE"] = raw
            fresh = importlib.reload(settings_module)
            assert fresh.SETTINGS.stop_atr_scale == expected
    finally:
        os.environ["STOP_ATR_SCALE"] = old or "1.0"
        importlib.reload(settings_module)
