"""Shared, single-flight caches for Dhan reads.

Without these, every open browser tab and both background runners each poll Dhan on their own
(live prices every few seconds per tab, the option chain on every refresh), which is how the account's
request limits get exceeded. Here one network call serves every caller inside the TTL, and concurrent
callers asking for the same thing wait for the single in-flight request instead of firing their own.

  * `QuoteCache`  - live prices (LTP) per (segment, security id); a miss fetches ONE batched request
                    covering everything any caller asked for in the last few seconds.
  * `TtlCache`    - generic keyed cache with per-key single-flight (used for option chains).
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from quant_intelligence.config.settings import SETTINGS

# Ids asked for by ANY caller this recently are folded into the next batched price request.
_INTEREST_WINDOW_SEC = 10.0


class TtlCache:
    """Keyed value cache with a per-key lock so concurrent misses trigger a single `loader()` call."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._values: dict = {}
        self._key_locks: dict = {}
        self._guard = threading.Lock()

    def get(self, key, loader: Callable[[], object], max_age: float):
        with self._guard:
            lock = self._key_locks.setdefault(key, threading.Lock())
        with lock:
            entry = self._values.get(key)
            if entry is not None and self.clock() - entry[0] < max_age:  # strict: max_age=0 never hits
                return entry[1]
            value = loader()  # an exception propagates and nothing is cached
            self._values[key] = (self.clock(), value)
            return value

    def peek(self, key, max_age: float):
        entry = self._values.get(key)
        if entry is not None and self.clock() - entry[0] < max_age:
            return entry[1]
        return None

    def clear(self) -> None:
        with self._guard:
            self._values.clear()
            self._key_locks.clear()


class QuoteCache:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._quotes: dict[tuple[str, str], tuple[float, dict]] = {}
        self._interest: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self._fetch_lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._quotes.clear()
            self._interest.clear()

    def _fresh(self, keys, max_age: float, now: float) -> bool:
        return all(k in self._quotes and now - self._quotes[k][0] < max_age for k in keys)  # strict: 0 = always refetch

    def _view(self, keys, max_age: float | None = None) -> dict[str, dict[str, dict]]:
        now = self.clock()
        out: dict[str, dict[str, dict]] = {}
        for seg, sid in keys:
            entry = self._quotes.get((seg, sid))
            if entry is None or (max_age is not None and now - entry[0] > max_age):
                continue
            out.setdefault(seg, {})[sid] = entry[1]
        return out

    def get_quotes(
        self,
        client,
        wanted: dict[str, list],
        max_age: float | None = None,
        stale_ok_for: float = 0.0,
    ) -> dict[str, dict[str, dict]]:
        """Quotes as {segment: {security_id (str): quote}}.

        `max_age` is how old a cached price may be (default: SETTINGS.dhan.quote_cache_ttl_sec).
        If the fetch fails (e.g. rate limited) and `stale_ok_for` > 0, cached prices up to that many
        seconds old are returned instead of raising - fine for display, not for exit decisions.
        """
        ttl = SETTINGS.dhan.quote_cache_ttl_sec if max_age is None else max_age
        keys = {(seg, str(i)) for seg, ids in wanted.items() for i in ids}
        if not keys:
            return {}
        now = self.clock()
        with self._lock:
            for k in keys:
                self._interest[k] = now
            if self._fresh(keys, ttl, now):
                return self._view(keys)

        with self._fetch_lock:
            now = self.clock()
            with self._lock:
                if self._fresh(keys, ttl, now):  # another caller refreshed while we waited
                    return self._view(keys)
                batch = set(keys) | {k for k, t in self._interest.items() if now - t <= _INTEREST_WINDOW_SEC}
            by_segment: dict[str, list[int]] = {}
            for seg, sid in sorted(batch):
                by_segment.setdefault(seg, []).append(int(sid))
            try:
                data = (client.get_ltp(by_segment) or {}).get("data") or {}
            except Exception:
                if stale_ok_for > 0:
                    with self._lock:
                        stale = self._view(keys, max_age=stale_ok_for)
                    if stale:
                        return stale
                raise
            fetched_at = self.clock()
            with self._lock:
                for seg, seg_quotes in data.items():
                    for sid, quote in (seg_quotes or {}).items():
                        self._quotes[(seg, str(sid))] = (fetched_at, quote)
                return self._view(keys)


QUOTE_CACHE = QuoteCache()
CHAIN_CACHE = TtlCache()


def clear_dhan_caches() -> None:
    QUOTE_CACHE.clear()
    CHAIN_CACHE.clear()
