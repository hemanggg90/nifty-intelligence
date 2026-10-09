"""Heatmap mover selection: ranking, freshness, the direction gate, the engine picking it up each cycle, the page."""
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_intelligence.execution import auto_trader, engine as engine_module, mover_selection as ms
from quant_intelligence.execution.mover_selection import DOWN, UP, MoverConfig
from quant_intelligence.reports import decision_log, eod
from quant_intelligence.ui import heatmap_data
from quant_intelligence.utils.market_profile import MCX, NSE

NOW = dt.datetime(2031, 3, 3, 11, 0)
FRESH = NOW - dt.timedelta(minutes=3)


def _row(symbol, group, pct, as_of=FRESH):
    return {"symbol": symbol, "group": group, "change_pct": pct, "as_of": as_of}


# ---------------------------------------------------------------- ranking
def test_picks_the_top_n_gainers_and_losers_per_group():
    rows = [_row(f"S{i}", "Stock", pct) for i, pct in enumerate([2.0, 1.5, 1.0, 0.5, -0.5, -1.0, -1.5, -2.0, 0.1])]
    rows += [_row("NIFTY", "Index", 0.8), _row("BANKNIFTY", "Index", -0.9)]
    sel = ms.rank_movers(rows, MoverConfig(n=2, min_abs_pct=0.3), NOW)
    stocks = {s for s, g in sel.group_of.items() if g == "Stock"}
    assert stocks == {"S0", "S1", "S6", "S7"}  # 2 biggest up, 2 biggest down
    assert sel.picks["S0"] == UP and sel.picks["S7"] == DOWN
    assert sel.picks["NIFTY"] == UP and sel.picks["BANKNIFTY"] == DOWN  # the index group has its own quota
    assert "S8" not in sel.picks  # +0.1% is below the minimum move


def test_the_mover_direction_maps_to_the_only_allowed_trade_direction():
    sel = ms.rank_movers([_row("UPCO", "Stock", 1.2), _row("DNCO", "Stock", -1.2), _row("FLAT", "Stock", 0.0)], MoverConfig(), NOW)
    assert sel.allowed_direction("UPCO") == "LONG" and sel.allowed_direction("DNCO") == "SHORT"
    assert sel.allowed_direction("FLAT") is None and sel.allowed_direction("NOT_THERE") is None


def test_missing_and_stale_data_are_never_selected_or_treated_as_zero():
    rows = [_row("OK", "Stock", 1.0), _row("NODATA", "Stock", None, as_of=None),
            _row("OLD", "Stock", 5.0, as_of=NOW - dt.timedelta(minutes=45)),
            _row("YESTERDAY", "Stock", 6.0, as_of=NOW - dt.timedelta(days=1))]
    sel = ms.rank_movers(rows, MoverConfig(), NOW)
    assert list(sel.picks) == ["OK"] and set(sel.excluded_stale) == {"NODATA", "OLD", "YESTERDAY"}
    assert "left out" in sel.reason


def test_nothing_is_selected_on_a_closed_market_and_it_says_why():
    sel = ms.rank_movers([_row("A", "Stock", 3.0, as_of=NOW - dt.timedelta(days=1))], MoverConfig(), NOW)
    assert not sel.usable and "yesterday" in sel.reason


def test_a_quiet_day_selects_nothing_rather_than_noise():
    sel = ms.rank_movers([_row("A", "Stock", 0.1), _row("B", "Stock", -0.2)], MoverConfig(min_abs_pct=0.3), NOW)
    assert not sel.usable and "0.3%" in sel.reason


def test_subset_keeps_only_the_requested_groups():
    sel = ms.rank_movers([_row("NIFTY", "Index", 1.0), _row("TCS", "Stock", -1.0), _row("GOLD", "Commodity", 2.0)], MoverConfig(), NOW)
    nse = sel.subset(("Index", "Stock"))
    assert set(nse.picks) == {"NIFTY", "TCS"} and set(sel.subset(("Commodity",)).picks) == {"GOLD"}


# ---------------------------------------------------------------- direction gate
def test_the_gate_blocks_only_bought_options_against_the_move():
    assert auto_trader.direction_blocked("SHORT", "BUY", "LONG") is True  # a put on an up-mover
    assert auto_trader.direction_blocked("LONG", "BUY", "LONG") is False
    assert auto_trader.direction_blocked("LONG", "BUY", "SHORT") is True
    assert auto_trader.direction_blocked("SHORT", "SELL", "LONG") is False  # written options are not filtered
    assert auto_trader.direction_blocked("SHORT", "BUY", None) is False  # no selection, no restriction


def _cycle(monkeypatch, setup_direction, allowed, transaction="BUY"):
    from quant_intelligence.brokers.paper_broker import PaperBroker
    from quant_intelligence.risk.risk_engine import AccountState

    chain_fetches = []
    broker = PaperBroker(starting_capital=1_000_000)
    setup = SimpleNamespace(direction=setup_direction, meta={"transaction": transaction})
    contract = SimpleNamespace(lot_size=100, transaction=transaction, security_id="777", option_type="CE" if setup_direction == "LONG" else "PE",
                               strike=22700.0, expiry="2030-01-01", trading_symbol="NIFTY-X")
    premium = SimpleNamespace(entry_price=50.0, stop_price=40.0, target_price=70.0)
    monkeypatch.setattr(auto_trader, "get_strategy", lambda n: object())
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status=auto_trader.SETUP_TRIGGERED, setup=setup))
    monkeypatch.setattr(auto_trader, "get_underlying_info", lambda u: {"lot_size": 100})
    monkeypatch.setattr(auto_trader, "select_contract", lambda *a, **k: contract)
    monkeypatch.setattr(auto_trader, "translate_setup", lambda *a, **k: premium)
    ranking = SimpleNamespace(is_no_trade=False, selected_strategy="Momentum", tie_break=False, reason="r", tie_runner_up=None, ranked=[])
    output = SimpleNamespace(ranking=ranking, ohlcv=SimpleNamespace(tail=lambda n: None), timestamp=pd.Timestamp("2031-03-03 11:00"),
                             market_state=SimpleNamespace(get=lambda k, d=None: None), data_quality_status="OK")
    account = AccountState(equity=1_000_000.0, peak_equity=1_000_000.0, daily_pnl=0.0, open_positions_count=0, trades_today=0,
                           exposure_by_strategy={}, total_exposure=0.0, broker_connected=True, kill_switch_engaged=False)

    def provider():
        chain_fetches.append(1)
        return SimpleNamespace(underlying="NIFTY")

    result = auto_trader.run_auto_option_cycle(output, "NIFTY", broker, None, provider, account, allowed, "(down -1.20% today)")
    return result, broker, chain_fetches, output


def test_a_bullish_setup_on_a_down_mover_is_skipped_before_any_chain_fetch(monkeypatch):
    result, broker, fetches, output = _cycle(monkeypatch, "LONG", "SHORT")
    assert result.order is None and result.reason.startswith(auto_trader.DIRECTION_FILTER_PREFIX)
    assert "NIFTY" in result.reason and "down -1.20%" in result.reason and "bearish (put)" in result.reason
    assert fetches == [] and broker.get_open_positions() == []  # no rate-limited chain call, no trade
    rec = decision_log.classify_cycle(output, result)
    assert (rec["status"], rec["reason_class"]) == ("SKIPPED", "DIRECTION_FILTER")


def test_a_setup_that_follows_the_move_trades_normally(monkeypatch):
    result, broker, fetches, _ = _cycle(monkeypatch, "SHORT", "SHORT")
    assert result.order is not None and result.order.status == "FILLED" and fetches == [1]


def test_without_a_selection_nothing_is_filtered(monkeypatch):
    result, *_ = _cycle(monkeypatch, "SHORT", None)
    assert result.order is not None and result.order.status == "FILLED"


def test_a_sell_setup_is_not_filtered(monkeypatch):
    result, *_ = _cycle(monkeypatch, "SHORT", "LONG", transaction="SELL")
    assert not result.reason.startswith(auto_trader.DIRECTION_FILTER_PREFIX)


def test_chain_needed_skips_the_chain_for_a_setup_the_heatmap_will_refuse(monkeypatch):
    setup = SimpleNamespace(direction="LONG", meta={"transaction": "BUY"})
    monkeypatch.setattr(auto_trader, "tradable", lambda *a: (True, None))
    monkeypatch.setattr(auto_trader, "get_strategy", lambda n: object())
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status=auto_trader.SETUP_TRIGGERED, setup=setup))
    out = SimpleNamespace(ranking=SimpleNamespace(selected_strategy="Momentum"), market_state={}, ohlcv=SimpleNamespace(tail=lambda n: None))
    broker = SimpleNamespace(get_open_positions=lambda: [])
    assert auto_trader.chain_needed(out, "NIFTY", broker) is True
    assert auto_trader.chain_needed(out, "NIFTY", broker, "LONG") is True
    assert auto_trader.chain_needed(out, "NIFTY", broker, "SHORT") is False


def test_the_daily_report_counts_filtered_setups():
    decisions = [{"status": "SKIPPED", "reason_class": "DIRECTION_FILTER", "strategy": "S", "tie_break": False, "reason": "", "instrument": "NIFTY"}] * 3
    assert eod._funnel(decisions)["direction_filtered"] == 3


# ---------------------------------------------------------------- the engine
def _runner(monkeypatch, profile=NSE):
    seen = []
    owner = SimpleNamespace(broker=None, account_state=lambda: None, lock=None, add_trade=lambda: None, add_pnl=lambda x: None)

    def fake_cycle(indices, stocks, *a, selection=None, **k):
        seen.append((list(indices), list(stocks), selection))
        return [], 0.0

    monkeypatch.setattr(engine_module, "run_multi_instrument_cycle", fake_cycle)
    monkeypatch.setattr(engine_module, "DhanApiClient", lambda: None)
    return engine_module.ScanRunner("t", profile, owner=owner), seen


def _fixed(monkeypatch, picks):
    """compute_selection returns whatever `picks` holds at the time (so a test can change it between cycles)."""
    monkeypatch.setattr(engine_module, "compute_selection", lambda groups, config, **k: _selection_from(groups, picks, config))


def _selection_from(groups, picks, config):
    return ms.rank_movers([_row(s, g, picks[s]) for g, syms in groups.items() for s in syms if s in picks], config, NOW)


def test_without_a_config_the_runner_scans_everything_as_before(monkeypatch):
    runner, seen = _runner(monkeypatch)
    assert runner.mover_config is None
    runner.run_cycle(["NIFTY"], ["TCS", "INFY"], "5min", 30)
    assert seen == [(["NIFTY"], ["TCS", "INFY"], None)]


def test_the_runner_scans_only_the_selection_and_follows_it_each_cycle(monkeypatch):
    runner, seen = _runner(monkeypatch)
    picks = {"NIFTY": 1.0, "TCS": 2.0, "INFY": -1.5, "HDFC": 0.1}
    _fixed(monkeypatch, picks)
    runner.set_mover_config(MoverConfig(n=5, min_abs_pct=0.3, auto_refresh=True))
    runner.run_cycle(["NIFTY"], ["TCS", "INFY", "HDFC"], "5min", 30)
    assert seen[-1][:2] == (["NIFTY"], ["TCS", "INFY"]) and seen[-1][2].allowed_direction("INFY") == "SHORT"
    picks.pop("NIFTY"); picks["HDFC"] = 3.0  # the market moved before the next cycle
    runner.run_cycle(["NIFTY"], ["TCS", "INFY", "HDFC"], "5min", 30)
    assert seen[-1][:2] == ([], ["TCS", "INFY", "HDFC"])
    assert runner.mover_selection.summary().startswith("3 selected")


def test_a_fixed_selection_is_kept_until_cleared(monkeypatch):
    runner, seen = _runner(monkeypatch)
    picks = {"TCS": 2.0}
    _fixed(monkeypatch, picks)
    runner.set_mover_config(MoverConfig(auto_refresh=False))
    runner.run_cycle([], ["TCS", "INFY"], "5min", 30)
    picks["INFY"] = 4.0
    runner.run_cycle([], ["TCS", "INFY"], "5min", 30)
    assert seen[-1][1] == ["TCS"]  # INFY's later move is not picked up
    runner.clear_mover_config()
    runner.run_cycle([], ["TCS", "INFY"], "5min", 30)
    assert seen[-1][1] == ["TCS", "INFY"] and seen[-1][2] is None


def test_an_empty_selection_scans_nothing_new_and_reports_why(monkeypatch):
    runner, seen = _runner(monkeypatch)
    _fixed(monkeypatch, {})
    runner.set_mover_config(MoverConfig())
    rows = runner.run_cycle([], ["TCS"], "5min", 30)
    assert seen[-1][:2] == ([], [])
    assert rows[0]["status"] == "NO_SELECTION" and rows[0]["detail"]


def test_the_commodity_runner_groups_its_symbols_as_commodities(monkeypatch):
    runner, seen = _runner(monkeypatch, profile=MCX)
    captured = {}

    def compute(groups, config, **k):
        captured.update(groups)
        return _selection_from(groups, {"GOLD": 1.0}, config)

    monkeypatch.setattr(engine_module, "compute_selection", compute)
    runner.set_mover_config(MoverConfig())
    runner.run_cycle([], ["GOLD", "SILVER"], "5min", 30)
    assert captured == {"Commodity": ["GOLD", "SILVER"]} and seen[-1][1] == ["GOLD"]


def test_the_multi_cycle_hands_each_symbol_its_direction(monkeypatch):
    from quant_intelligence.execution import multi_cycle

    passed = {}
    no_trade = SimpleNamespace(order=None, strategy_name=None, setup_status=None, reason="", capital_required=None, capital_used=0.0, tag=None)
    monkeypatch.setattr(multi_cycle, "run_pipeline", lambda *a: SimpleNamespace(ranking=SimpleNamespace(tie_break=False, is_no_trade=True, selected_strategy=None), ohlcv=None))
    monkeypatch.setattr(multi_cycle, "run_auto_option_cycle",
                        lambda output, symbol, broker, client, chain, account, allowed=None, note="": passed.__setitem__(symbol, (allowed, note)) or no_trade)
    sel = ms.rank_movers([_row("TCS", "Stock", 1.5), _row("INFY", "Stock", -1.1)], MoverConfig(), NOW)
    client = SimpleNamespace(is_configured=lambda: False)
    multi_cycle.run_multi_instrument_cycle([], ["TCS", "INFY"], "5min", 30, None, client, lambda: None, selection=sel)
    assert passed["TCS"][0] == "LONG" and "up +1.50%" in passed["TCS"][1]
    assert passed["INFY"][0] == "SHORT" and "down -1.10%" in passed["INFY"][1]


# ---------------------------------------------------------------- heatmap data
def _candles(symbol_moves):
    """5-minute candles: yesterday closes at 100, today's last bar closes at 100 * (1 + pct/100)."""
    def build(symbol):
        pct = symbol_moves.get(symbol)
        if pct is None:
            return None
        base = dt.datetime.now().replace(second=0, microsecond=0)
        from quant_intelligence.utils.timeutil import now_ist

        today = now_ist().replace(second=0, microsecond=0) - dt.timedelta(minutes=2)
        prev = today - dt.timedelta(days=1)
        rows = [(prev, 100.0), (today - dt.timedelta(minutes=5), 100.0 * (1 + pct / 200)), (today, 100.0 * (1 + pct / 100))]
        return pd.DataFrame({"timestamp": [t for t, _ in rows], "open": [c for _, c in rows], "high": [c * 1.01 for _, c in rows],
                             "low": [c * 0.99 for _, c in rows], "close": [c for _, c in rows], "volume": 1000.0})
    return build


def test_the_frame_has_one_row_per_instrument_and_never_invents_a_zero():
    df = heatmap_data.build_heatmap_frame("Today", loader=_candles({"NIFTY": 1.0, "TCS": -2.0}))
    assert set(df["group"]) == {"Index", "Stock", "Commodity"} and len(df) == len(heatmap_data.universe())
    nifty = df[df["symbol"] == "NIFTY"].iloc[0]
    assert nifty["change_pct"] == pytest.approx(1.0) and nifty["fresh"] and nifty["has_data"]
    assert df[df["symbol"] == "TCS"].iloc[0]["change_pct"] == pytest.approx(-2.0)
    gone = df[df["symbol"] == "GOLD"].iloc[0]
    assert pd.isna(gone["change_pct"]) and not gone["has_data"]


def test_a_window_longer_than_the_history_is_missing_not_guessed():
    df = heatmap_data.build_heatmap_frame("5 days", loader=_candles({"NIFTY": 1.0}))  # only 2 sessions of candles
    assert not df[df["symbol"] == "NIFTY"].iloc[0]["has_data"]


def test_window_change_uses_the_close_n_sessions_back():
    days = [dt.datetime(2031, 3, d, 15, 25) for d in (3, 4, 5, 6, 7, 10)]
    candles = pd.DataFrame({"timestamp": days, "close": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]})
    assert heatmap_data.window_change_pct(candles, 1) == pytest.approx((110 - 108) / 108 * 100)
    assert heatmap_data.window_change_pct(candles, 5) == pytest.approx(10.0)
    assert heatmap_data.window_change_pct(candles, 6) is None


def test_groups_cover_the_whole_watchlist():
    groups = heatmap_data.groups_of()
    assert "NIFTY" in groups["Index"] and "GOLD" in groups["Commodity"] and len(groups["Stock"]) >= 10


# ---------------------------------------------------------------- the page
def test_the_heatmap_page_renders_with_and_without_data(monkeypatch):
    from streamlit.testing.v1 import AppTest

    page = Path(__file__).resolve().parents[1] / "pages" / "16_NSE_Heatmap.py"
    for moves in ({}, {"NIFTY": 1.2, "BANKNIFTY": -0.8, "TCS": 2.0, "INFY": -1.5, "GOLD": 0.9}):
        monkeypatch.setattr(heatmap_data, "load_candles", lambda sym, tf="5min", sessions=2, m=moves: _candles(m)(sym))
        monkeypatch.setattr(ms, "snapshot_rows", lambda groups, timeframe="5min", loader=None, m=moves: [
            {"symbol": s, "group": g, "change_pct": m.get(s), "as_of": dt.datetime.now() if s in m else None}
            for g, syms in groups.items() for s in syms])
        at = AppTest.from_file(str(page), default_timeout=120).run()
        assert not at.exception, [e.value for e in at.exception]


def test_the_apply_button_switches_the_runners_on_and_clear_switches_them_off(monkeypatch):
    from streamlit.testing.v1 import AppTest

    moves = {"NIFTY": 1.2, "TCS": -1.5, "GOLD": 0.9}
    monkeypatch.setattr(heatmap_data, "load_candles", lambda sym, tf="5min", sessions=2: _candles(moves)(sym))
    monkeypatch.setattr(ms, "snapshot_rows", lambda groups, timeframe="5min", loader=None: [
        {"symbol": s, "group": g, "change_pct": moves.get(s), "as_of": dt.datetime.now() if s in moves else None}
        for g, syms in groups.items() for s in syms])
    page = Path(__file__).resolve().parents[1] / "pages" / "16_NSE_Heatmap.py"
    at = AppTest.from_file(str(page), default_timeout=120).run()
    try:
        next(b for b in at.button if b.key == "hm_apply_nse").click().run()
        assert engine_module.ENGINE.mover_config is not None and engine_module.COMMODITY_RUNNER.mover_config is None
        sel = engine_module.ENGINE.mover_selection
        assert sel.allowed_direction("NIFTY") == "LONG" and sel.allowed_direction("TCS") == "SHORT"
        next(b for b in at.button if b.key == "hm_apply_mcx").click().run()
        assert engine_module.COMMODITY_RUNNER.mover_config is not None
    finally:
        next(b for b in at.button if b.key == "hm_clear").click().run()
    assert engine_module.ENGINE.mover_config is None and engine_module.COMMODITY_RUNNER.mover_config is None
