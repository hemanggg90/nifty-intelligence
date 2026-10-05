"""Candles stay current without anyone pressing a button: the background data keeper, the self-refreshing
pipeline cache, and the quiet mode that keeps a per-minute refresh from flooding the database and log."""
import dataclasses
import datetime as dt
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_intelligence.config import settings as settings_module
from quant_intelligence.data import data_keeper as keeper_module
from quant_intelligence.data.data_keeper import DataKeeper


class FakeManager:
    def __init__(self, bad=(), raise_for=()):
        self.calls = []
        self.bad, self.raise_for = set(bad), set(raise_for)

    def get_ohlcv(self, sym, tf, start, end, quiet=False, **kw):
        self.calls.append((sym, tf, quiet))
        if sym in self.raise_for:
            raise RuntimeError("No real market data: Dhan rejected your credentials (401)")
        status = "DEGRADED" if sym in self.bad else "OK"
        issues = ["data is stale: last bar 01 Oct 15:50", "latest refresh failed: timeout"] if sym in self.bad else []
        return pd.DataFrame(), {"end_ts": pd.Timestamp("2026-10-01 15:50"), "quality_status": status,
                                "quality_report": {"issues": issues}, "source": "cache+tail" if sym == "TCS" else "cache"}


def _keeper(manager, symbols=("NIFTY", "TCS", "CRUDEOIL")):
    k = DataKeeper(interval_seconds=15)
    k._manager = manager
    return k, list(symbols)


# ---------------------------------------------------------------- one round
def test_a_round_refreshes_every_instrument_quietly():
    mgr = FakeManager()
    k, symbols = _keeper(mgr)
    out = k.run_once(symbols)
    assert [c[0] for c in mgr.calls] == symbols and all(c[2] is True for c in mgr.calls)  # quiet: no DB/log flood
    assert out == {"refreshed": 1, "stale": [], "error": None}  # only TCS needed a tail fetch
    st = k.status()
    assert st["tracked"] == 3 and st["ok"] == 3 and st["not_ok"] == [] and st["rounds"] == 1


def test_stale_and_failed_instruments_are_reported_with_the_reason():
    mgr = FakeManager(bad={"CRUDEOIL"}, raise_for={"NIFTY"})
    k, symbols = _keeper(mgr)
    out = k.run_once(symbols)
    assert sorted(out["stale"]) == ["CRUDEOIL", "NIFTY"]
    assert "rejected your credentials" in out["error"] or "refresh failed" in out["error"]
    st = k.status()
    assert st["not_ok"] == ["CRUDEOIL", "NIFTY"] and st["ok"] == 1
    assert k.instruments["NIFTY"]["quality"] == "NO DATA"


def test_one_failing_instrument_does_not_stop_the_rest():
    mgr = FakeManager(raise_for={"NIFTY"})
    k, symbols = _keeper(mgr)
    k.run_once(symbols)
    assert [c[0] for c in mgr.calls] == symbols  # TCS and CRUDEOIL were still refreshed


def test_recovery_clears_the_error():
    mgr = FakeManager(bad={"TCS"})
    k, symbols = _keeper(mgr)
    k.run_once(symbols)
    assert k.last_error
    mgr.bad.clear()
    k.run_once(symbols)
    assert k.last_error is None and k.status()["not_ok"] == []


# ---------------------------------------------------------------- the thread
@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(keeper_module, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, data_keeper_enabled=True))


def test_the_keeper_runs_in_the_background_start_is_idempotent_and_wake_runs_a_round_now(enabled, monkeypatch):
    mgr = FakeManager()
    k, symbols = _keeper(mgr)
    monkeypatch.setattr(keeper_module, "watchlist_symbols", lambda: symbols)
    assert k.start() is True and k.start() is False  # second call: already running
    deadline = time.time() + 5
    while k.rounds < 1 and time.time() < deadline:
        time.sleep(0.02)
    assert k.rounds >= 1 and k.running

    before = k.rounds
    k.wake()  # e.g. a new token was just saved: do not wait for the 15 s tick
    deadline = time.time() + 5
    while k.rounds <= before and time.time() < deadline:
        time.sleep(0.02)
    assert k.rounds > before
    k.stop()
    time.sleep(0.2)
    assert not k.running


def test_a_crashing_round_does_not_kill_the_thread(enabled, monkeypatch):
    k, _ = _keeper(FakeManager())
    calls = []

    def boom(symbols=None):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(k, "run_once", boom)
    k.interval = 0.05
    k.start()
    deadline = time.time() + 5
    while len(calls) < 3 and time.time() < deadline:
        k.wake()
        time.sleep(0.03)
    assert len(calls) >= 3 and k.running and "boom" in (k.last_error or "")
    k.stop()


def test_disabled_keeper_never_starts():
    k, _ = _keeper(FakeManager())
    assert k.start() is False and not k.running  # conftest sets DATA_KEEPER=false


# ---------------------------------------------------------------- the pipeline cache refreshes itself
def _out(status="OK", ts="2026-10-01 15:20"):
    return SimpleNamespace(data_quality_status=status, timestamp=pd.Timestamp(ts))


def test_pipeline_cache_rules():
    from quant_intelligence.ui.state import credentials_fingerprint, pipeline_needs_refresh

    fp = credentials_fingerprint()
    during = dt.datetime(2026, 10, 1, 15, 22)  # a Thursday, NSE open
    # fresh OK result, same credentials, no new bar yet -> keep
    assert pipeline_needs_refresh(_out(), 100.0, fp, "NIFTY", "5min", now=130.0, now_dt=during) is None
    # new credentials -> recompute immediately
    assert pipeline_needs_refresh(_out(), 100.0, "stale-fp", "NIFTY", "5min", now=110.0, now_dt=during) == "credentials changed"
    # a not-OK result is retried after a minute, not before
    assert pipeline_needs_refresh(_out("DEGRADED"), 100.0, fp, "NIFTY", "5min", now=130.0, now_dt=during) is None
    assert "DEGRADED" in pipeline_needs_refresh(_out("DEGRADED"), 100.0, fp, "NIFTY", "5min", now=165.0, now_dt=during)
    # an OK result becomes stale once a newer bar has closed (and a minute has passed)
    later = dt.datetime(2026, 10, 1, 15, 31 - 6)  # 15:25 -> bar 15:20 closed at 15:25
    assert pipeline_needs_refresh(_out(ts="2026-10-01 15:15"), 100.0, fp, "NIFTY", "5min", now=170.0, now_dt=later) == "a newer bar has closed"
    # ...but not more often than once a minute
    assert pipeline_needs_refresh(_out(ts="2026-10-01 15:15"), 100.0, fp, "NIFTY", "5min", now=130.0, now_dt=later) is None
    # no run time recorded (e.g. restored session) and not OK -> recompute
    assert pipeline_needs_refresh(_out("FAIL"), None, fp, "NIFTY", "5min", now=130.0, now_dt=during) is not None


def test_credentials_fingerprint_changes_with_the_token_and_never_contains_it(monkeypatch):
    from quant_intelligence.ui import state

    a = state.credentials_fingerprint()
    monkeypatch.setattr(state, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, dhan_access_token="NEW-SECRET-TOKEN"))
    b = state.credentials_fingerprint()
    assert a != b and "SECRET" not in b and len(b) == 12


def test_invalidating_clears_cached_analysis_and_retry_guards():
    from streamlit.testing.v1 import AppTest

    script = """
import streamlit as st
from quant_intelligence.ui.state import invalidate_cached_analysis
from quant_intelligence.execution.engine import ENGINE
from quant_intelligence.data import data_manager as dm
for k in ("pipeline_output", "pipeline_error", "option_chain", "pipeline_for", "pipeline_ran_at", "pipeline_token_fp"):
    st.session_state[k] = "stale"
ENGINE.last_rows = [{"symbol": "X"}]
dm._last_attempt[("A", "5min")] = 1.0
dm._last_refresh_error[("A", "5min")] = "old error"
invalidate_cached_analysis()
st.write("cleared" if all(st.session_state[k] is None for k in ("pipeline_output", "pipeline_error", "option_chain", "pipeline_for", "pipeline_ran_at", "pipeline_token_fp"))
         and ENGINE.last_rows == [] and not dm._last_attempt and not dm._last_refresh_error else "NOT cleared")
"""
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception and at.markdown[0].value == "cleared"


# ---------------------------------------------------------------- quiet mode
def test_quiet_mode_writes_no_metadata_row_when_data_is_ok_but_still_reports_problems(tmp_path, monkeypatch):
    from quant_intelligence.data import data_manager as dm_module
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import MarketDataMetadata
    from quant_intelligence.utils.market_calendar import expected_last_closed_bar_start
    from quant_intelligence.utils.market_profile import profile_for
    from quant_intelligence.utils.timeutil import now_ist

    monkeypatch.setattr(dm_module, "CACHE_DIR", tmp_path)
    end = now_ist().replace(tzinfo=None)
    last = expected_last_closed_bar_start(end, profile_for("QUIETTEST"), 5)  # the bar a live feed has just closed
    ts = pd.date_range(last.replace(hour=9, minute=15, second=0, microsecond=0), last, freq="5min")

    def frame(stamps):
        return pd.DataFrame({"timestamp": stamps, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0})

    frame(ts).to_parquet(tmp_path / "QUIETTEST_5min.parquet", index=False)
    frame(ts - pd.Timedelta(days=7)).to_parquet(tmp_path / "QUIETOLD_5min.parquet", index=False)  # a week stale
    manager = dm_module.DataManager()
    start = end - dt.timedelta(days=30)

    def rows(name):
        with get_session() as session:
            return session.query(MarketDataMetadata).filter_by(instrument=name).count()

    before = rows("QUIETTEST")
    _, meta = manager.get_ohlcv("QUIETTEST", "5min", start, end, quiet=True)
    assert meta["quality_status"] == "OK" and rows("QUIETTEST") == before  # nothing written for a healthy refresh

    manager.get_ohlcv("QUIETTEST", "5min", start, end)  # a normal call still records provenance
    assert rows("QUIETTEST") == before + 1

    manager._fetch = lambda *a, **k: ("none", pd.DataFrame())  # Dhan unreachable
    old_before = rows("QUIETOLD")
    _, bad = manager.get_ohlcv("QUIETOLD", "5min", start, end, quiet=True)
    assert bad["quality_status"] != "OK" and rows("QUIETOLD") == old_before + 1  # problems are always recorded


# ---------------------------------------------------------------- credentials entered AFTER start-up
def test_adapter_built_before_the_token_is_saved_picks_it_up_later(monkeypatch):
    """The keeper's DataManager is created at import, before anyone enters a token. It used to snapshot the
    empty credentials and report 'Dhan credentials not set' forever, even after the token was saved."""
    from quant_intelligence.data_adapters import dhan_adapter

    empty = dataclasses.replace(settings_module.SETTINGS, dhan_client_id="", dhan_access_token="")
    monkeypatch.setattr(dhan_adapter, "SETTINGS", empty)
    adapter = dhan_adapter.DhanAdapter()
    assert adapter.is_available() is False

    object.__setattr__(empty, "dhan_client_id", "1000000001")  # what update_dhan_credentials does
    object.__setattr__(empty, "dhan_access_token", "a.b.c")
    assert adapter.is_available() is True and adapter.client_id == "1000000001" and adapter.access_token == "a.b.c"
