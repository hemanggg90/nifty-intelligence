"""NSE market-heatmap-derived stock universe.

NOTE: `/api/equity-stockIndices` (used by `fetch_heatmap_universe` below) now answers 404 for every index, so
`fetch_heatmap_universe` always falls back. The mover selection uses `fetch_fno_movers` (further down), which
reads the working `live-analysis-variations` endpoint.

NSE publishes no official public API. nseindia.com's own heatmap/gainers-
losers widgets call an undocumented endpoint,
`/api/equity-stockIndices?index=<INDEX>`, which returned each index
constituent's last price, % change, and traded volume - exactly what a
heatmap colors by. Reaching it requires a short-lived session: a plain GET
to the homepage first (to receive anti-bot cookies), then the API call reusing
those cookies with browser-like headers.

This endpoint is undocumented and anti-bot-protected, so it WILL occasionally
fail (network issues, 403, schema drift). Failure never raises to the caller:
it falls back to a small bundled static NIFTY-50 symbol list and says so via
`source` in the returned metadata - the same "never fabricate, degrade
honestly" pattern the rest of this codebase uses for synthetic OHLCV data.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import requests

from quant_intelligence.config.settings import DATA_CACHE_DIR, SETTINGS
from quant_intelligence.utils.logging_utils import log_event

_HOMEPAGE_URL = "https://www.nseindia.com/"
_API_URL = "https://www.nseindia.com/api/equity-stockIndices"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}

_VARIATIONS_URL = "https://www.nseindia.com/api/live-analysis-variations"
_MOVERS_CACHE_FILE = DATA_CACHE_DIR / "nse_fno_movers.json"

_FALLBACK_CSV = Path(__file__).resolve().parent / "nse_universe_fallback.csv"
_CACHE_FILE = DATA_CACHE_DIR / "nse_universe.json"


def _load_fallback() -> list[dict]:
    with open(_FALLBACK_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{"symbol": row["symbol"].strip().upper(), "pChange": None, "lastPrice": None} for row in reader]


def _fetch_live(index: str) -> list[dict]:
    session = requests.Session()
    session.headers.update(_HEADERS)
    session.get(_HOMEPAGE_URL, timeout=10)  # sets anti-bot cookies
    resp = session.get(_API_URL, params={"index": index}, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    rows = body.get("data", [])
    if not rows:
        raise ValueError("NSE heatmap response had no 'data' rows")
    return [
        {
            "symbol": str(r.get("symbol", "")).strip().upper(),
            "pChange": r.get("pChange"),
            "lastPrice": r.get("lastPrice"),
            "totalTradedVolume": r.get("totalTradedVolume"),
            "industry": (r.get("meta") or {}).get("industry") if isinstance(r.get("meta"), dict) else None,
        }
        for r in rows
        if r.get("symbol") and str(r.get("symbol")).strip().upper() != index.strip().upper()
    ]


def load_static_universe(top_n: int | None = None) -> dict:
    """Fixed ~200-stock NSE constituent list, no live NSE call and no mover-based ranking.

    Use this instead of `fetch_heatmap_universe` when the caller wants a stable,
    always-available equity universe (e.g. "trade all stocks together with the
    indices") rather than a live gainers/losers heatmap.
    """
    top_n = top_n or SETTINGS.heatmap_universe_top_n
    constituents = _load_fallback()
    return {
        "source": "static_universe",
        "index": "NIFTY 50",
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S"),
        "constituents": constituents[:top_n],
    }


def fetch_heatmap_universe(index: str | None = None, top_n: int | None = None, force_refresh: bool = False,
                           max_age_minutes: float | None = None) -> dict:
    """Returns {"source": "nse_live" | "fallback_static", "as_of": iso-ts, "constituents": [...]}.

    `constituents` is sorted by |pChange| descending for live data (biggest
    movers first, matching what a heatmap visually emphasizes) and truncated
    to `top_n`. Cached on disk for `HEATMAP_REFRESH_MINUTES` to avoid
    hammering NSE and tripping rate limiting. `max_age_minutes` overrides that refresh window (the mover selection
    wants a minutes-old quote, not a quarter-hour-old one). Every result carries `fetched_at` (epoch seconds), so a
    caller can tell how old the quotes are.
    """
    index = index or SETTINGS.heatmap_index
    max_age = SETTINGS.heatmap_refresh_minutes if max_age_minutes is None else max_age_minutes
    top_n = top_n or SETTINGS.heatmap_universe_top_n

    if not force_refresh and _CACHE_FILE.exists():
        try:
            cached = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            age_minutes = (time.time() - cached.get("fetched_at", 0)) / 60.0
            if age_minutes < max_age and cached.get("index") == index and len(cached.get("constituents", [])) >= min(top_n, 1):
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    try:
        constituents = _fetch_live(index)
        constituents.sort(key=lambda r: abs(r["pChange"] or 0.0), reverse=True)
        result = {"source": "nse_live", "index": index, "as_of": time.strftime("%Y-%m-%d %H:%M:%S"), "constituents": constituents[:top_n]}
        result["fetched_at"] = time.time()
    except Exception as e:
        log_event("nse_heatmap", f"Live NSE heatmap fetch failed, using static fallback: {e}", level="WARNING")
        constituents = _load_fallback()
        result = {"source": "fallback_static", "index": index, "as_of": time.strftime("%Y-%m-%d %H:%M:%S"), "constituents": constituents[:top_n]}
        result["fetched_at"] = time.time()

    try:
        _CACHE_FILE.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass

    return result


def _parse_nse_time(text: str | None) -> str | None:
    """NSE's '09-Oct-2026 12:17:46' (IST wall clock) -> '2026-10-09T12:17:46'."""
    import datetime as dt

    try:
        return dt.datetime.strptime((text or "").strip(), "%d-%b-%Y %H:%M:%S").isoformat()
    except ValueError:
        return None


def _fetch_fno_movers_live() -> tuple[list[dict], str | None]:
    session = requests.Session()
    session.headers.update(_HEADERS)
    session.get(_HOMEPAGE_URL, timeout=10)  # sets anti-bot cookies
    rows: dict[str, dict] = {}
    quote_time = None
    for which in ("gainers", "loosers"):  # NSE's own spelling
        resp = session.get(_VARIATIONS_URL, params={"index": which}, timeout=10)
        resp.raise_for_status()
        block = resp.json().get("FOSec") or {}
        quote_time = quote_time or _parse_nse_time(block.get("timestamp"))
        for r in block.get("data", []):
            sym = str(r.get("symbol", "")).strip().upper()
            if sym and r.get("perChange") is not None:
                rows[sym] = {"symbol": sym, "pChange": float(r["perChange"]), "lastPrice": r.get("ltp"),
                             "totalTradedVolume": r.get("trade_quantity"), "industry": None}
    if not rows:
        raise ValueError("NSE returned no F&O gainers/losers")
    return list(rows.values()), quote_time


def fetch_fno_movers(max_age_minutes: float = 2.0) -> dict:
    """NSE's live top-20 gainers and top-20 losers among all F&O stocks (every stock with listed options).

    Uses nseindia.com's `live-analysis-variations` endpoint (the one behind its gainers/losers widgets), which carries
    the live price, % change versus the previous close and a quote timestamp. It is undocumented, so it can break or
    be blocked; the result then has `source == "fallback_static"` and no quotes - never made-up ones.
    Returns {"source": "nse_live"|"fallback_static", "fno_only": True, "fetched_at": epoch, "as_of_ist": iso|None,
    "constituents": [...]}, cached on disk for `max_age_minutes`."""
    if _MOVERS_CACHE_FILE.exists():
        try:
            cached = json.loads(_MOVERS_CACHE_FILE.read_text(encoding="utf-8"))
            if (time.time() - cached.get("fetched_at", 0)) / 60.0 < max_age_minutes:
                return cached
        except (json.JSONDecodeError, OSError):
            pass
    try:
        rows, quote_time = _fetch_fno_movers_live()
        result = {"source": "nse_live", "fno_only": True, "fetched_at": time.time(), "as_of_ist": quote_time, "constituents": rows}
    except Exception as e:
        log_event("nse_heatmap", f"Live NSE F&O movers fetch failed: {e}", level="WARNING")
        result = {"source": "fallback_static", "fno_only": True, "fetched_at": time.time(), "as_of_ist": None, "constituents": []}
    try:
        _MOVERS_CACHE_FILE.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass
    return result
