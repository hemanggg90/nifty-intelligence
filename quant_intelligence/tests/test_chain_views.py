import pytest

from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow
from quant_intelligence.ui.chain_views import chain_stats, ladder_frame, max_pain, ticket_math


def _row(strike, ce_oi, pe_oi, ce_ltp=10.0, pe_ltp=10.0):
    return StrikeRow(strike=strike, ce_ltp=ce_ltp, ce_oi=ce_oi, ce_oi_change=1.0, ce_volume=5.0, ce_iv=14.0,
                     ce_delta=0.5, ce_security_id=f"c{strike}", pe_ltp=pe_ltp, pe_oi=pe_oi, pe_oi_change=-1.0,
                     pe_volume=5.0, pe_iv=15.0, pe_delta=-0.5, pe_security_id=f"p{strike}")


def _chain(spot=100.0):
    rows = [_row(90, 100, 900), _row(95, 300, 600), _row(100, 800, 800), _row(105, 900, 300), _row(110, 700, 100)]
    return ChainSnapshot(underlying="TEST", expiry="2026-10-27", spot_price=spot, strikes=rows)


def test_ladder_is_centred_on_atm_and_marks_it():
    df = ladder_frame(_chain(spot=101.0), window=1)
    assert list(df["strike"]) == [95, 100, 105] and list(df["is_atm"]) == [False, True, False]
    assert list(df.columns)[4] == "strike"  # calls left of the strike, puts right
    assert len(ladder_frame(_chain(), window=0)) == 5  # window 0 = every strike
    assert ladder_frame(ChainSnapshot("T", "x", 1.0, [])).empty


def test_max_pain_is_the_strike_that_hurts_option_buyers_least():
    # hand check at P=100: calls pay 100-90=10 x100 + 5 x300 = 2,500; puts pay 5 x300(105) + 10 x100(110) = 2,500 -> 5,000
    # at P=105: calls 15x100 + 10x300 + 5x800 = 8,500 ; puts 5x100... higher. 100 is the minimum.
    assert max_pain(_chain()) == 100
    assert max_pain(ChainSnapshot("T", "x", 1.0, [])) is None


def test_chain_stats_support_and_resistance_from_heaviest_open_interest():
    s = chain_stats(_chain())
    assert s["resistance"] == 105 and s["support"] == 90  # most call OI at 105, most put OI at 90
    assert s["pcr"] == pytest.approx(2700 / 2800) and s["atm"] == 100 and s["max_pain"] == 100
    assert s["call_oi"] == 2800 and s["put_oi"] == 2700


# BUY 75 @ 100, stop 90, target 130 on NSE
def test_ticket_math_for_a_buy():
    t = ticket_math(100.0, 90.0, 130.0, 75, "BUY", "NSE", available=50000.0)
    assert t["amount_needed"] == 7500.0 and t["max_loss"] == 750.0 and t["max_profit"] == 2250.0
    assert t["risk_reward"] == pytest.approx(3.0)
    assert t["net_loss"] > t["max_loss"] and t["net_profit"] < t["max_profit"]  # charges widen the loss, shrink the win
    assert t["funds_ok"] and t["utilisation_pct"] == pytest.approx(15.0)


def test_ticket_flags_insufficient_funds_only_for_buys():
    assert not ticket_math(100.0, 90.0, 130.0, 75, "BUY", "NSE", available=5000.0)["funds_ok"]
    sell = ticket_math(100.0, 130.0, 70.0, 75, "SELL", "NSE", available=5000.0)
    assert sell["funds_ok"] and sell["utilisation_pct"] is None and sell["max_loss"] == 2250.0


def test_ticket_with_no_stop_distance_has_no_ratio():
    assert ticket_math(100.0, 100.0, 120.0, 75)["risk_reward"] is None
