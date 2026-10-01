"""Option-premium cost model (Dhan Rs 20/order + statutory option charges).

Expected numbers are worked out by hand in the comments, independently of the implementation.
"""
import pandas as pd
import pytest

from quant_intelligence.backtesting.engine import _apply_costs, market_for, net_r_multiple
from quant_intelligence.strategies.base_strategy import Setup, TradeResult


def _trade(entry=22600.0, stop=22580.0, gross=40.0):
    setup = Setup(pd.Timestamp("2026-09-30 10:00"), "LONG", entry, stop, entry + 2 * (entry - stop))
    return TradeResult(setup=setup, exit_timestamp=pd.Timestamp("2026-09-30 10:30"), exit_price=entry + gross,
                       r_multiple=gross / abs(entry - stop), holding_period_bars=6, mfe=gross, mae=0.0,
                       gross_pnl=gross, outcome="WIN")


# NIFTY-like trade: entry 22600, stop 22580, +40 pts, 75 units, delta 0.5, premium 1.5% of price.
#   entry premium 339.0, exit premium 359.0  -> buy turnover 25,425, sell turnover 26,925
#   brokerage 2 x 20 = 40
#   STT   0.15%    x sell 26,925          = 40.3875
#   txn   0.03553% x turnover 52,350      = 18.6000
#   SEBI  0.0001%  x turnover 52,350      =  0.0524
#   stamp 0.003%   x buy 25,425           =  0.7628
#   GST   18% x (40 + 18.60 + 0.0524)     = 10.5574      -> fees 70.36
#   slippage 1 tick (0.05) x 75 x 2       =  7.5
#   option P&L 0.5 x 40 x 75 = 1,500  ->  net 1,382.14 ; risk 0.5 x 20 x 75 = 750  ->  net R 1.84285
def test_nse_option_costs_match_the_hand_calculation():
    commissions, fees, slippage, net = _apply_costs(_trade(), 22600.0, 75, "NSE")
    assert commissions == 40.0
    assert fees == pytest.approx(70.36, abs=0.01)
    assert slippage == pytest.approx(7.5)
    assert net == pytest.approx(1382.14, abs=0.01)
    assert net_r_multiple(_trade(), 75, "NSE") == pytest.approx(1.84285, abs=1e-4)


def test_mcx_uses_its_own_stt_and_exchange_charge():
    # STT 0.05% x 26,925 = 13.4625; txn 0.0418% x 52,350 = 21.8823; GST 18% x (40 + 21.8823 + 0.0524) = 11.1482
    _, fees, _, net = _apply_costs(_trade(), 22600.0, 75, "MCX")
    assert fees == pytest.approx(47.31, abs=0.01)
    assert net == pytest.approx(1405.19, abs=0.01)


def test_costs_are_charged_on_premium_not_on_underlying_notional():
    """The old model charged 0.05% STT on (22,600 + exit) x qty ~ Rs 1,700 for this trade - more than 100%
    of the Rs 750 risked. Costs now are ~Rs 118 (0.16 R)."""
    commissions, fees, slippage, _ = _apply_costs(_trade(gross=-20.0), 22600.0, 75, "NSE")
    total = commissions + fees + slippage
    assert total < 150
    assert total / (0.5 * 20 * 75) < 0.25  # well under a quarter of the amount risked


def test_a_losing_trade_loses_about_one_r_plus_costs():
    r = net_r_multiple(_trade(gross=-20.0), 75, "NSE")  # stopped out: gross -1R
    assert -1.3 < r < -1.0


def test_fixed_brokerage_weighs_more_on_a_tiny_position():
    big = net_r_multiple(_trade(), 75, "NSE")
    small = net_r_multiple(_trade(), 5, "NSE")
    assert small < big  # Rs 40 of brokerage on a small position costs far more R


def test_market_for_routes_commodities_to_mcx():
    assert market_for("CRUDEOIL") == "MCX"
    assert market_for("NIFTY") == "NSE" and market_for("TCS") == "NSE"
