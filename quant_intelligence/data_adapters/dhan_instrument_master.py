"""Resolves an NSE equity trading symbol to Dhan's security_id.

Dhan publishes a scrip-master CSV covering every instrument it supports
(equities, F&O, indices) at a fixed public URL. It's not the NSE heatmap or
any broker credential - just a static reference file - so it's downloaded
and cached once per day rather than per request.
"""
from __future__ import annotations

import csv
import datetime as dt

import requests

from quant_intelligence.config.settings import DATA_CACHE_DIR
from quant_intelligence.utils.logging_utils import log_event

_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
_CACHE_FILE = DATA_CACHE_DIR / "dhan_scrip_master.csv"
_CACHE_MAX_AGE_HOURS = 24

# Column names as published by Dhan's scrip master.
_COL_SYMBOL = "SEM_TRADING_SYMBOL"
_COL_SECURITY_ID = "SEM_SMST_SECURITY_ID"
_COL_SEGMENT = "SEM_EXM_EXCH_ID"
_COL_INSTRUMENT = "SEM_INSTRUMENT_NAME"
_COL_LOT_UNITS = "SEM_LOT_UNITS"
_COL_STRIKE = "SEM_STRIKE_PRICE"
_COL_EXPIRY = "SEM_EXPIRY_DATE"

_cache: dict[str, dict] | None = None
_fno_cache: dict[str, dict] | None = None


def _cache_is_fresh() -> bool:
    if not _CACHE_FILE.exists():
        return False
    age = dt.datetime.now() - dt.datetime.fromtimestamp(_CACHE_FILE.stat().st_mtime)
    return age < dt.timedelta(hours=_CACHE_MAX_AGE_HOURS)


def _download_scrip_master() -> None:
    resp = requests.get(_SCRIP_MASTER_URL, timeout=30)
    resp.raise_for_status()
    _CACHE_FILE.write_bytes(resp.content)


def _ensure_scrip_master() -> bool:
    """Downloads the scrip-master CSV if the cached copy is stale/missing.

    Returns True if a usable CSV is on disk afterwards, False otherwise.
    """
    if not _cache_is_fresh():
        try:
            _download_scrip_master()
        except Exception as e:
            log_event("dhan_instrument_master", f"Scrip-master download failed: {e}", level="WARNING")
            return _CACHE_FILE.exists()
    return True


def _load_index() -> dict[str, dict]:
    global _cache
    if _cache is not None:
        return _cache

    if not _ensure_scrip_master():
        _cache = {}
        return _cache

    index: dict[str, dict] = {}
    with open(_CACHE_FILE, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(_COL_INSTRUMENT, "").strip().upper() != "EQUITY":
                continue
            if row.get(_COL_SEGMENT, "").strip().upper() != "NSE":
                continue
            symbol = row.get(_COL_SYMBOL, "").strip().upper()
            if not symbol:
                continue
            index[symbol] = {
                "security_id": row.get(_COL_SECURITY_ID, "").strip(),
                "exchange_segment": "NSE_EQ",
            }
    _cache = index
    return _cache


def resolve_equity(trading_symbol: str) -> dict | None:
    """Returns {"security_id": str, "exchange_segment": "NSE_EQ"} or None if unresolved."""
    index = _load_index()
    return index.get(trading_symbol.strip().upper())


_DERIVATIVE_INSTRUMENT_TYPES = {"OPTSTK", "OPTIDX"}


def _load_fno_index() -> dict[str, dict]:
    """Indexes NSE option (OPTSTK stock + OPTIDX index) rows by underlying symbol.

    The scrip master has no dedicated "underlying symbol" column for option
    rows - it's the prefix of SEM_TRADING_SYMBOL before the first "-"
    (e.g. "RELIANCE-Sep2026-700-CE" -> "RELIANCE"). Each underlying's
    strike_step is derived from the minimum gap between consecutive strikes
    on its nearest expiry; lot_size is read directly off its rows (revised
    periodically by NSE, so this is always read fresh from the scrip master
    rather than hardcoded - a stale lot size would misprice live risk).
    """
    global _fno_cache
    if _fno_cache is not None:
        return _fno_cache

    if not _ensure_scrip_master():
        _fno_cache = {}
        return _fno_cache

    by_symbol: dict[str, dict] = {}
    with open(_CACHE_FILE, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instrument = row.get(_COL_INSTRUMENT, "").strip().upper()
            if instrument not in _DERIVATIVE_INSTRUMENT_TYPES:
                continue
            if row.get(_COL_SEGMENT, "").strip().upper() != "NSE":
                continue
            trading_symbol = row.get(_COL_SYMBOL, "").strip().upper()
            symbol = trading_symbol.split("-", 1)[0]
            if not symbol:
                continue
            expiry = row.get(_COL_EXPIRY, "").strip()
            try:
                strike = float(row.get(_COL_STRIKE, "") or "nan")
                lot_size = int(float(row.get(_COL_LOT_UNITS, "") or "nan"))
            except ValueError:
                continue

            entry = by_symbol.setdefault(symbol, {"lot_size": lot_size, "is_stock": instrument == "OPTSTK", "expiries": {}})
            entry["lot_size"] = lot_size
            entry["expiries"].setdefault(expiry, set()).add(strike)

    index: dict[str, dict] = {}
    for symbol, entry in by_symbol.items():
        nearest_expiry = min(entry["expiries"]) if entry["expiries"] else None
        strikes = sorted(entry["expiries"].get(nearest_expiry, [])) if nearest_expiry else []
        gaps = [round(b - a, 2) for a, b in zip(strikes, strikes[1:]) if b > a]
        strike_step = min(gaps) if gaps else 1.0
        index[symbol] = {"lot_size": entry["lot_size"], "strike_step": strike_step, "is_stock": entry["is_stock"]}

    _fno_cache = index
    return _fno_cache


def resolve_derivative_lot_specs(symbol: str) -> dict | None:
    """Returns {"lot_size", "strike_step"} for any NSE underlying (stock or index)
    with currently listed options, read fresh from the scrip master. Used both to
    resolve a new stock underlying and to keep a statically registered index's
    lot_size current (exchange-revised periodically)."""
    entry = _load_fno_index().get(symbol.strip().upper())
    if entry is None:
        return None
    return {"lot_size": entry["lot_size"], "strike_step": entry["strike_step"]}


def resolve_fno_stock(symbol: str) -> dict | None:
    """Returns {"security_id", "seg", "strike_step", "lot_size"} for an NSE F&O stock
    underlying, or None if it has no listed stock options. Dhan's option-chain API
    takes the stock underlying's own security_id (as indices use their IDX_I id) but with
    UnderlyingSeg=NSE_FNO (NSE_EQ is rejected with error 814), so this reuses `resolve_equity` for the id.
    """
    symbol = symbol.strip().upper()
    fno_info = _load_fno_index().get(symbol)
    if fno_info is None or not fno_info["is_stock"]:
        return None
    equity_info = resolve_equity(symbol)
    if equity_info is None:
        return None
    return {
        "security_id": equity_info["security_id"],
        "seg": "NSE_FNO",
        "strike_step": fno_info["strike_step"],
        "lot_size": fno_info["lot_size"],
    }


def list_fno_stock_symbols() -> list[str]:
    """Sorted list of all NSE stock symbols that currently have listed options."""
    return sorted(symbol for symbol, entry in _load_fno_index().items() if entry["is_stock"])
