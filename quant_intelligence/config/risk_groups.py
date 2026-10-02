"""Correlation groups for the risk engine's group-exposure check.

Instruments in one group tend to move together, so several same-direction positions in a group are
really one bigger bet. Groups: all index underlyings together, stocks by watchlist sector (the four
banks form one group), MCX commodities by sector. Anything else is its own group.
"""
from __future__ import annotations

from quant_intelligence.config.watchlist import WATCHLIST_COMMODITIES, WATCHLIST_STOCKS

INDEX_GROUP = "INDICES"
_STOCK_SECTOR = {s["symbol"]: s["sector"] for s in WATCHLIST_STOCKS}
_COMMODITY_SECTOR = {c["symbol"]: c["sector"] for c in WATCHLIST_COMMODITIES}


def group_for(symbol: str | None) -> str:
    from quant_intelligence.options.option_selector import UNDERLYING_REGISTRY

    s = (symbol or "").strip().upper()
    if s in UNDERLYING_REGISTRY:
        return INDEX_GROUP
    if s in _STOCK_SECTOR:
        return f"STOCKS: {_STOCK_SECTOR[s]}"
    if s in _COMMODITY_SECTOR:
        return f"MCX: {_COMMODITY_SECTOR[s]}"
    return s
