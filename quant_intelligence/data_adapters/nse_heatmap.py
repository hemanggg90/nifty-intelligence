"""NSE market-heatmap-derived stock universe.

NSE publishes no official public API. nseindia.com's own heatmap/gainers-
losers widgets call an undocumented endpoint,
`/api/equity-stockIndices?index=<INDEX>`, which returns each index
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


def fetch_heatmap_universe(index: str | None = None, top_n: int | None = None, force_refresh: bool = False) -> dict:
    """Returns {"source": "nse_live" | "fallback_static", "as_of": iso-ts, "constituents": [...]}.

    `constituents` is sorted by |pChange| descending for live data (biggest
    movers first, matching what a heatmap visually emphasizes) and truncated
    to `top_n`. Cached on disk for `HEATMAP_REFRESH_MINUTES` to avoid
    hammering NSE and tripping rate limiting.
    """
    index = index or SETTINGS.heatmap_index
    top_n = top_n or SETTINGS.heatmap_universe_top_n

    if not force_refresh and _CACHE_FILE.exists():
        try:
            cached = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            age_minutes = (time.time() - cached.get("fetched_at", 0)) / 60.0
            if age_minutes < SETTINGS.heatmap_refresh_minutes and cached.get("index") == index:
                return {k: v for k, v in cached.items() if k != "fetched_at"}
        except (json.JSONDecodeError, OSError):
            pass

    try:
        constituents = _fetch_live(index)
        constituents.sort(key=lambda r: abs(r["pChange"] or 0.0), reverse=True)
        result = {"source": "nse_live", "index": index, "as_of": time.strftime("%Y-%m-%d %H:%M:%S"), "constituents": constituents[:top_n]}
    except Exception as e:
        log_event("nse_heatmap", f"Live NSE heatmap fetch failed, using static fallback: {e}", level="WARNING")
        constituents = _load_fallback()
        result = {"source": "fallback_static", "index": index, "as_of": time.strftime("%Y-%m-%d %H:%M:%S"), "constituents": constituents[:top_n]}

    try:
        _CACHE_FILE.write_text(json.dumps({**result, "fetched_at": time.time()}), encoding="utf-8")
    except OSError:
        pass

    return result
