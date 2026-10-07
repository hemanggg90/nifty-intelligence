"""The once-a-day volatility job run by the data keeper (never inside a scan cycle).

After an instrument's market has closed, load its daily history, run and score every model, store the
result. A few instruments per keeper round so one slow fit cannot delay the candle refresh for long; a failure
is recorded, isolated to that instrument, and retried after a pause. Gated by VOL_MODELS_ENABLED.
"""
from __future__ import annotations

import datetime as dt
import threading
import time

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_calendar import most_recent_expected_bar_time
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeutil import is_market_open, now_ist

SETTLE_MINUTES = 10  # wait this long after the close so the final daily bar is published


class VolJob:
    def __init__(self, max_per_round: int = 2, retry_after_sec: float = 1800.0) -> None:
        self.max_per_round = max_per_round
        self.retry_after_sec = retry_after_sec
        self.by_symbol: dict[str, dict] = {}  # symbol -> {"key", "ok", "selected", "reason", "at", "error", "next_try"}
        self.rounds = 0
        self.last_run_at: dt.datetime | None = None
        self.last_error: str | None = None

    @staticmethod
    def session_key(now: dt.datetime, symbol: str) -> dt.date | None:
        """Date of the latest finished session for `symbol` (its close was at least SETTLE_MINUTES ago and the
        market is not open now), else None."""
        profile = profile_for(symbol)
        if is_market_open(now, profile):
            return None
        close_dt = most_recent_expected_bar_time(now, profile).replace(tzinfo=None)
        if now < close_dt + dt.timedelta(minutes=SETTLE_MINUTES):
            return None
        return close_dt.date()

    def due(self, symbols: list[str], now: dt.datetime | None = None) -> list[str]:
        now = now or now_ist()
        mono = time.monotonic()
        out = []
        for sym in symbols:
            key = self.session_key(now, sym)
            st = self.by_symbol.get(sym, {})
            if key is not None and st.get("key") != key and mono >= st.get("next_try", 0.0):
                out.append(sym)
        return out

    def run_one(self, symbol: str, now: dt.datetime | None = None) -> bool:
        """Fit, score and store one instrument. Returns True on success. Never raises."""
        from quant_intelligence.data.daily_history import load_daily_history
        from quant_intelligence.volatility import store
        from quant_intelligence.volatility.forecast_engine import run_instrument

        now = now or now_ist()
        key = self.session_key(now, symbol)
        started = time.monotonic()
        try:
            daily, info = load_daily_history(symbol)
            result = run_instrument(
                symbol, daily, refit_every=SETTINGS.vol_refit_every_days, min_train=SETTINGS.vol_min_train_days,
                min_eval=SETTINGS.vol_min_eval_days, alpha=SETTINGS.vol_dm_alpha,
            )
            if result.asof is None:  # nothing to forecast from
                raise RuntimeError(result.reason + (f" ({info['fetch_error']})" if info.get("fetch_error") else ""))
            store.save_result(result)
            self.by_symbol[symbol] = {"key": key, "ok": True, "selected": result.selected, "reason": result.reason,
                                      "at": now, "error": None, "next_try": 0.0, "seconds": time.monotonic() - started,
                                      "n_daily": info.get("n")}
            log_event("vol_job", f"{symbol}: volatility models run - selected {result.selected} ({result.reason})", level="INFO")
            return True
        except Exception as e:
            self.by_symbol[symbol] = {"key": self.by_symbol.get(symbol, {}).get("key"), "ok": False, "selected": None,
                                      "reason": None, "at": now, "error": f"{type(e).__name__}: {e}",
                                      "next_try": time.monotonic() + self.retry_after_sec}
            self.last_error = f"{symbol}: {e}"
            log_event("vol_job", f"{symbol}: volatility job failed: {e}", level="WARNING")
            return False

    def run_round(self, symbols: list[str], stop: threading.Event | None = None, now: dt.datetime | None = None) -> int:
        """Process up to `max_per_round` due instruments. Returns how many were attempted."""
        if not SETTINGS.vol_models_enabled:
            return 0
        now = now or now_ist()
        done = 0
        for sym in self.due(symbols, now)[: self.max_per_round]:
            if stop is not None and stop.is_set():
                break
            self.run_one(sym, now)
            done += 1
        self.rounds += 1
        if done:
            self.last_run_at = now
        return done

    def status(self, symbols: list[str]) -> dict:
        now = now_ist()
        current = [s for s in symbols if self.by_symbol.get(s, {}).get("key") == self.session_key(now, s) and self.by_symbol.get(s, {}).get("ok")]
        failed = sorted(s for s in symbols if self.by_symbol.get(s, {}).get("ok") is False)
        return {"enabled": SETTINGS.vol_models_enabled, "tracked": len(symbols), "done": len(current), "failed": failed,
                "last_run_at": self.last_run_at, "error": self.last_error, "by_symbol": dict(self.by_symbol)}
