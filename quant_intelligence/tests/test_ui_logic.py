"""The data layer behind the trading UI: Indian number formatting, itemised charges, position / P&L
analytics and market snapshots. Pure functions - no Streamlit."""
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.execution.charges import option_trade_charges
from quant_intelligence.ui import format as F
from quant_intelligence.ui.market_data import snapshot_from_frame
from quant_intelligence.ui.position_views import (
    activity_frame, equity_curve, exit_reason_label, pnl_by, positions_frame, trade_stats, with_live,
)


# ---------------------------------------------------------------- formatting
@pytest.mark.parametrize("value, expected", [
    (0, "0"), (999, "999"), (1000, "1,000"), (12345, "12,345"), (123456, "1,23,456"),
    (1234567, "12,34,567"), (123456789, "12,34,56,789"), (-1234567, "-12,34,567"),
])
def test_indian_digit_grouping(value, expected):
    assert F.num(value, 0) == expected


def test_rupee_and_decimals_and_sign():
    assert F.inr(1234567.5, 2) == "₹12,34,567.50"
    assert F.inr(-500) == "-₹500"
    assert F.inr(2500, signed=True) == "+₹2,500"
    assert F.inr(None) == "–" and F.inr(float("nan")) == "–"


def test_compact_rupees_use_lakh_and_crore():
    assert F.inr_compact(150000) == "₹1.50L"
    assert F.inr_compact(25000000) == "₹2.50Cr"
    assert F.inr_compact(-9500) == "-₹9,500"


def test_tone_arrow_and_pct():
    assert (F.tone(5), F.tone(-5), F.tone(0), F.tone(None)) == ("good", "critical", "neutral", "neutral")
    assert (F.arrow(5), F.arrow(-5), F.arrow(0)) == (F.UP, F.DOWN, F.FLAT)
    assert F.pct(1.236) == "+1.24%" and F.pct(-0.5) == "-0.50%"


def test_duration_and_contract_label():
    assert F.duration(dt.timedelta(minutes=135)) == "2h 15m"
    assert F.duration(dt.timedelta(seconds=40)) == "40s"
    assert F.duration(dt.timedelta(days=2, hours=3)) == "2d 3h"
    assert F.contract_label("NIFTY", 22700.0, "CE", "2026-10-06") == "NIFTY 22700 CE 06 Oct"
    assert F.contract_label("RELIANCE", None, None, None) == "RELIANCE"


# ---------------------------------------------------------------- charges
# BUY 1 lot of 75 at premium 100, sold at 120 on NSE:
#   buy 7,500, sell 9,000, turnover 16,500; brokerage 2 x 20 = 40
#   STT 0.15% x 9,000 = 13.5 ; exchange 0.03553% x 16,500 = 5.8625 ; SEBI 0.0001% x 16,500 = 0.0165
#   stamp 0.003% x 7,500 = 0.225 ; GST 18% x (40 + 5.8625 + 0.0165) = 8.2585
def test_buy_trade_charges_match_the_hand_calculation():
    c = option_trade_charges(100.0, 120.0, 75, "BUY", "NSE")
    assert c.brokerage == 40.0
    assert c.stt == pytest.approx(13.5)
    assert c.exchange == pytest.approx(5.8625, abs=1e-4)
    assert c.sebi == pytest.approx(0.0165, abs=1e-4)
    assert c.stamp == pytest.approx(0.225)
    assert c.gst == pytest.approx(8.2585, abs=1e-3)
    assert c.total == pytest.approx(67.86, abs=0.01)


def test_sell_trade_puts_stt_on_the_entry_leg_and_mcx_uses_its_own_rates():
    sell = option_trade_charges(100.0, 80.0, 75, "SELL", "NSE")  # sold at 100: STT on 7,500
    assert sell.stt == pytest.approx(0.0015 * 7500)
    assert sell.stamp == pytest.approx(0.00003 * 80 * 75)  # bought back at 80
    mcx = option_trade_charges(100.0, 120.0, 75, "BUY", "MCX")
    assert mcx.stt == pytest.approx(0.0005 * 9000) and mcx.exchange == pytest.approx(0.000418 * 16500)


def test_no_quantity_means_no_charges():
    assert option_trade_charges(100.0, 120.0, 0).total == 0.0
    assert option_trade_charges(None, 120.0, 75).total == 0.0


# ---------------------------------------------------------------- positions frame
def _row(pid, instrument="NIFTY", side="BUY", entry=100.0, qty=75, stop=90.0, target=130.0, status="CLOSED",
         exit_price=120.0, pnl=1500.0, opened="2026-10-01 10:00", closed="2026-10-01 10:45",
         strategy="ORB", security_id="111", strike=22700.0, option_type="CE", expiry="2026-10-06", reason="TARGET"):
    return {
        "position_id": pid, "instrument": instrument, "underlying": instrument, "strategy_name": strategy,
        "direction": "LONG", "transaction": side, "quantity": qty, "entry_price": entry, "stop_price": stop,
        "target_price": target, "status": status, "opened_at": pd.Timestamp(opened).to_pydatetime(),
        "closed_at": pd.Timestamp(closed).to_pydatetime() if closed else None, "exit_price": exit_price,
        "net_pnl": pnl if status == "CLOSED" else None, "security_id": security_id, "strike": strike,
        "option_type": option_type, "expiry": expiry, "exit_reason": reason,
    }


def test_positions_frame_normalises_and_prices_charges():
    df = positions_frame([_row("P1")])
    r = df.iloc[0]
    assert r["contract"] == "NIFTY 22700 CE 06 Oct" and r["side"] == "BUY" and r["market"] == "NSE"
    assert r["invested"] == 7500.0 and r["gross_pnl"] == 1500.0
    assert r["charges"] == pytest.approx(67.86, abs=0.01)
    assert r["net_pnl"] == pytest.approx(1500.0 - 67.86, abs=0.01)
    assert r["return_pct"] == pytest.approx(20.0)
    assert r["held"] == dt.timedelta(minutes=45)


def test_commodity_positions_are_tagged_mcx_and_orm_objects_work():
    orm = SimpleNamespace(**_row("P2", instrument="CRUDEOIL"))
    df = positions_frame([orm])
    assert df.iloc[0]["market"] == "MCX"


def test_open_positions_have_no_booked_pnl():
    r = positions_frame([_row("P3", status="OPEN", exit_price=None, closed=None)]).iloc[0]
    assert r["gross_pnl"] is None or pd.isna(r["gross_pnl"])
    assert r["status"] == "OPEN"


# ---------------------------------------------------------------- live pricing
def test_live_columns_for_a_bought_option():
    df = with_live(positions_frame([_row("P1", status="OPEN", exit_price=None, closed=None)]), {"111": 115.0})
    r = df.iloc[0]
    assert r["ltp"] == 115.0
    assert r["unrealised"] == pytest.approx((115 - 100) * 75)
    assert r["unrealised_pct"] == pytest.approx(15.0)
    assert r["progress"] == pytest.approx((115 - 90) / (130 - 90))  # 62.5% of the way from stop to target
    assert r["to_stop_pct"] == pytest.approx((115 - 90) / 115 * 100)
    assert r["to_target_pct"] == pytest.approx((130 - 115) / 115 * 100)
    assert r["unrealised_net"] < r["unrealised"]  # charges would reduce it


def test_live_pnl_for_a_sold_option_profits_when_premium_falls():
    sell = _row("P4", side="SELL", entry=100.0, stop=130.0, target=70.0, status="OPEN", exit_price=None, closed=None)
    r = with_live(positions_frame([sell]), {"111": 85.0}).iloc[0]
    assert r["unrealised"] == pytest.approx((100 - 85) * 75)
    assert r["progress"] == pytest.approx((85 - 130) / (70 - 130))  # 75% of the way to the (lower) target


def test_missing_quote_leaves_live_columns_empty_and_progress_is_clamped():
    open_row = _row("P5", status="OPEN", exit_price=None, closed=None)
    none = with_live(positions_frame([open_row]), {}).iloc[0]
    assert np.isnan(none["ltp"]) and np.isnan(none["unrealised"])
    beyond = with_live(positions_frame([open_row]), {"111": 200.0}).iloc[0]
    assert beyond["progress"] == 1.0


# ---------------------------------------------------------------- statistics
def _closed(pnls):
    rows = []
    for i, p in enumerate(pnls):
        rows.append(_row(f"T{i}", pnl=p, exit_price=100 + p / 75, closed=f"2026-10-01 {10 + i // 4:02d}:{(i % 4) * 10 + 5:02d}",
                         opened=f"2026-10-01 {10 + i // 4:02d}:{(i % 4) * 10:02d}",
                         strategy="ORB" if i % 2 == 0 else "VWAP", instrument="NIFTY" if i < 3 else "TCS"))
    return positions_frame(rows)


def test_trade_stats_headline_numbers():
    df = _closed([1000.0, -400.0, 600.0, -200.0, 500.0])
    s = trade_stats(df)
    nets = df["net_pnl"].to_numpy()
    assert s["trades"] == 5 and s["wins"] == 3 and s["losses"] == 2
    assert s["win_rate"] == pytest.approx(60.0)
    assert s["net"] == pytest.approx(nets.sum()) and s["gross"] == pytest.approx(1500.0)
    assert s["charges"] == pytest.approx(1500.0 - nets.sum())
    assert s["profit_factor"] == pytest.approx(nets[nets > 0].sum() / -nets[nets < 0].sum())
    assert s["payoff"] == pytest.approx(nets[nets > 0].mean() / abs(nets[nets < 0].mean()))
    assert s["expectancy"] == pytest.approx(nets.mean())
    assert s["best"] == pytest.approx(nets.max()) and s["worst"] == pytest.approx(nets.min())
    assert s["max_win_streak"] == 1 and s["max_loss_streak"] == 1


def test_max_drawdown_is_peak_to_trough_of_cumulative_net_pnl():
    df = _closed([1000.0, 500.0, -800.0, -300.0, 200.0])
    s = trade_stats(df)
    eq = np.cumsum(df.sort_values("closed_at")["net_pnl"].to_numpy())
    assert s["max_drawdown"] == pytest.approx((eq - np.maximum.accumulate(eq)).min())
    assert s["max_drawdown"] < -1000


def test_stats_on_no_trades_are_safe():
    s = trade_stats(positions_frame([]))
    assert s["trades"] == 0 and s["win_rate"] is None and s["net"] == 0.0


def test_equity_curve_and_breakdowns():
    df = _closed([1000.0, -400.0, 600.0, -200.0])
    curve = equity_curve(df, start_capital=100000.0)
    assert list(curve.columns) == ["time", "trade", "pnl", "equity"] and len(curve) == 4
    assert curve["equity"].iloc[-1] == pytest.approx(100000.0 + df["net_pnl"].sum())
    by = pnl_by(df, "strategy")
    assert set(by["strategy"]) == {"ORB", "VWAP"} and by["net_pnl"].is_monotonic_decreasing
    assert by["trades"].sum() == 4
    assert pnl_by(positions_frame([]), "strategy").empty


def test_exit_reason_labels():
    assert exit_reason_label("STOP") == "Stop-loss hit" and exit_reason_label("MANUAL") == "Manual exit"
    assert exit_reason_label(None) == "–" and exit_reason_label("WEIRD") == "WEIRD"


# ---------------------------------------------------------------- market snapshot
def test_snapshot_uses_previous_session_close_for_change():
    day1 = pd.date_range("2026-09-30 09:15", periods=75, freq="5min")
    day2 = pd.date_range("2026-10-01 09:15", periods=10, freq="5min")
    df = pd.concat([
        pd.DataFrame({"timestamp": day1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}),
        pd.DataFrame({"timestamp": day2, "open": 102.0, "high": 106.0, "low": 101.0, "close": 105.0, "volume": 1.0}),
    ], ignore_index=True)
    s = snapshot_from_frame(df)
    assert s["last"] == 105.0 and s["prev_close"] == 100.0
    assert s["change"] == 5.0 and s["change_pct"] == pytest.approx(5.0)
    assert s["open"] == 102.0 and s["high"] == 106.0 and s["low"] == 101.0  # today's range only
    assert s["session"] == dt.date(2026, 10, 1) and len(s["spark"]) == 48


def test_snapshot_of_a_single_session_falls_back_to_its_open_and_empty_is_none():
    day = pd.date_range("2026-10-01 09:15", periods=5, freq="5min")
    df = pd.DataFrame({"timestamp": day, "open": 50.0, "high": 52.0, "low": 49.0, "close": 51.0, "volume": 1.0})
    assert snapshot_from_frame(df)["prev_close"] == 50.0
    assert snapshot_from_frame(df.iloc[0:0]) is None and snapshot_from_frame(None) is None


# ---------------------------------------------------------------- activity feed
def test_activity_feed_merges_orders_vetoes_and_exits_newest_first_in_ist():
    from quant_intelligence.ui.position_views import orders_frame

    orders = orders_frame([
        {"timestamp": dt.datetime(2026, 10, 1, 10, 0), "order_id": "O1", "instrument": "NIFTY", "strike": 22700.0,
         "option_type": "CE", "expiry": "2026-10-06", "direction": "LONG", "quantity": 75, "order_type": "MARKET",
         "price": 100.0, "status": "FILLED", "strategy_name": "ORB", "mode": "PAPER"},
        {"timestamp": dt.datetime(2026, 10, 1, 10, 5), "order_id": "O2", "instrument": "TCS", "direction": "LONG",
         "quantity": 225, "status": "REJECTED", "reject_reason": "Insufficient funds", "mode": "PAPER"},
    ])
    # risk events are stored in UTC: 04:35 UTC = 10:05 IST
    risk = [SimpleNamespace(event_type="VETO", reason="Max trades per day reached", timestamp=dt.datetime(2026, 10, 1, 4, 35)),
            SimpleNamespace(event_type="APPROVED", reason="ok", timestamp=dt.datetime(2026, 10, 1, 4, 30))]
    closed = positions_frame([_row("P1", closed="2026-10-01 10:45")])
    feed = activity_frame(orders, risk, closed)
    assert list(feed["kind"]) == ["Exit", "Order", "Risk", "Order"]  # 10:45, 10:05 (order), 10:05 (veto), 10:00 - newest first
    assert feed.iloc[0]["tone"] == "good" and "Target hit" in feed.iloc[0]["text"]
    veto = feed[feed["kind"] == "Risk"].iloc[0]
    assert veto["time"] == pd.Timestamp("2026-10-01 10:05") and veto["tone"] == "warning"  # UTC -> IST
    assert "Insufficient funds" in feed[feed["tone"] == "critical"].iloc[0]["text"]
    assert not any("APPROVED" in t for t in feed["text"])  # the order row already says it was placed


def test_activity_feed_can_be_limited_to_one_markets_instruments():
    from quant_intelligence.ui.position_views import orders_frame

    orders = orders_frame([
        {"timestamp": dt.datetime(2026, 10, 1, 10, 0), "order_id": "O1", "instrument": "NIFTY", "direction": "LONG",
         "quantity": 75, "status": "FILLED", "price": 1.0},
        {"timestamp": dt.datetime(2026, 10, 1, 10, 1), "order_id": "O2", "instrument": "CRUDEOIL", "direction": "LONG",
         "quantity": 100, "status": "FILLED", "price": 1.0},
    ])
    feed = activity_frame(orders, [], positions_frame([]), instruments={"CRUDEOIL"})
    assert len(feed) == 1 and "CRUDEOIL" in feed.iloc[0]["text"]
    assert activity_frame(orders_frame([]), [], positions_frame([])).empty


# ---------------------------------------------------------------- charts must build (they only fail at render time)
def _sample_chain():
    from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow

    rows = [StrikeRow(strike=float(k), ce_ltp=10.0, ce_oi=1000.0 * (k % 7 + 1), ce_oi_change=1.0, ce_volume=1.0, ce_iv=14.0,
                      ce_delta=0.5, ce_security_id="c", pe_ltp=10.0, pe_oi=900.0 * (k % 5 + 1), pe_oi_change=-1.0,
                      pe_volume=1.0, pe_iv=15.0, pe_delta=-0.5, pe_security_id="p") for k in range(22500, 22951, 50)]
    return ChainSnapshot("NIFTY", "2026-10-06", 22710.0, rows)


def test_option_chain_charts_and_tables_build_without_error():
    from quant_intelligence.ui.chain_views import ladder_frame
    from quant_intelligence.ui.charts import oi_chart
    from quant_intelligence.ui.tables import ladder_table

    chain = _sample_chain()
    fig = oi_chart(chain, window=4)  # regression: a labelled vline on a category axis used to raise TypeError
    assert len(fig.data) == 2 and fig.layout.annotations
    assert ladder_table(ladder_frame(chain, 4), chain.spot_price).to_html()  # renders the styler


def test_price_equity_pnl_and_regime_charts_build():
    from quant_intelligence.ui.charts import equity_curve_chart, pnl_bar_chart, price_chart, regime_chart

    ts = pd.date_range("2026-09-30 09:15", periods=150, freq="5min")
    ts = ts[(ts.hour * 60 + ts.minute >= 555) & (ts.hour * 60 + ts.minute < 930)]
    df = pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 100.0})
    fig = price_chart(df, levels=[{"label": "Stop", "price": 99.0, "tone": "critical"}],
                      markers=[{"time": ts[10], "price": 100.5, "text": "x", "direction": "LONG"}])
    assert any(t.type == "candlestick" for t in fig.data)
    assert price_chart(df.assign(volume=0.0)).data  # index-style data with no volume still draws
    closed = _closed([1000.0, -400.0, 600.0])
    assert equity_curve_chart(equity_curve(closed)).data and equity_curve_chart(equity_curve(positions_frame([]))) is not None
    assert pnl_bar_chart(pnl_by(closed, "strategy"), "strategy").data
    assert regime_chart({"RANGE": 0.5, "TREND_UP": 0.3, "HIGH_VOL": 0.2}).data
