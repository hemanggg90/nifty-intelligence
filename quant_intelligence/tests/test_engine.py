import datetime as dt
import time

from quant_intelligence.execution import engine as engine_module
from quant_intelligence.execution.engine import TradingEngine
from quant_intelligence.utils import timeutil


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _patch_cycle(monkeypatch, calls, fail=False):
    def fake(index, stocks, tf, lb, broker, client, get_account, **kwargs):
        calls.append((list(index), list(stocks)))
        if fail:
            raise RuntimeError("boom")
        return [{"symbol": "NIFTY", "status": "NO_TRADE"}], 0.0

    monkeypatch.setattr(engine_module, "run_multi_instrument_cycle", fake)
    monkeypatch.setattr(engine_module, "DhanApiClient", lambda: object())


def test_background_worker_runs_independently_until_stopped(monkeypatch):
    calls = []
    _patch_cycle(monkeypatch, calls)
    eng = TradingEngine()

    assert eng.start(["NIFTY"], ["TCS"], "5min", 30, interval_seconds=0.05, market_hours_only=False)
    assert not eng.start(["NIFTY"], ["TCS"], "5min", 30)  # already running: no second worker
    assert _wait(lambda: len(calls) >= 3)  # keeps cycling with no page/session involved
    assert eng.status()["running"] and eng.cycles >= 3

    eng.stop()
    assert not eng.status()["running"]
    time.sleep(0.2)
    n = len(calls)
    time.sleep(0.3)
    assert len(calls) == n  # really stopped


def test_restart_after_stop_never_revives_the_old_worker(monkeypatch):
    calls = []
    _patch_cycle(monkeypatch, calls)
    eng = TradingEngine()
    eng.start(["NIFTY"], [], "5min", 30, interval_seconds=0.05, market_hours_only=False)
    first = eng._thread
    eng.stop()
    eng.start(["NIFTY"], [], "5min", 30, interval_seconds=0.05, market_hours_only=False)
    assert _wait(lambda: not first.is_alive())  # old thread exits despite the restart
    eng.stop()


def test_a_failing_cycle_does_not_kill_the_worker(monkeypatch):
    calls = []
    _patch_cycle(monkeypatch, calls, fail=True)
    eng = TradingEngine()
    eng.start(["NIFTY"], [], "5min", 30, interval_seconds=0.05, market_hours_only=False)
    assert _wait(lambda: len(calls) >= 3)  # retried after each failure
    assert "boom" in eng.status()["last_error"]
    eng.stop()


def test_waits_outside_market_hours_and_when_kill_switch_engaged(monkeypatch):
    calls = []
    _patch_cycle(monkeypatch, calls)
    monkeypatch.setattr(engine_module, "is_market_open", lambda **kw: False)
    eng = TradingEngine()
    eng.start(["NIFTY"], [], "5min", 30, interval_seconds=0.05, market_hours_only=True)
    assert _wait(lambda: "Market closed" in eng.status()["last_status"])
    assert calls == []
    eng.stop()

    monkeypatch.setattr(engine_module, "is_market_open", lambda **kw: True)
    eng = TradingEngine()
    eng.kill_switch = True
    eng.start(["NIFTY"], [], "5min", 30, interval_seconds=0.05, market_hours_only=True)
    assert _wait(lambda: "Kill switch" in eng.status()["last_status"])
    assert calls == []
    eng.stop()


def test_daily_counters_reset_on_a_new_ist_day():
    eng = TradingEngine()
    eng.add_trade(3)
    eng.add_pnl(-500.0)
    assert eng.account_state().trades_today == 3
    eng._day = dt.date(2000, 1, 1)  # pretend the counters are from an earlier day
    state = eng.account_state()
    assert state.trades_today == 0 and state.daily_pnl == 0.0


def test_ist_helpers():
    assert timeutil.is_market_open(dt.datetime(2026, 9, 29, 10, 0))  # Tuesday 10:00
    assert not timeutil.is_market_open(dt.datetime(2026, 9, 29, 15, 30))
    assert not timeutil.is_market_open(dt.datetime(2026, 9, 26, 10, 0))  # Saturday
    ist_now, utc_now = timeutil.now_ist(), dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    assert abs((ist_now - utc_now) - dt.timedelta(hours=5, minutes=30)) < dt.timedelta(seconds=5)
