from quant_intelligence.config.watchlist import WATCHLIST_STOCKS


def test_watchlist_has_15_unique_symbols():
    symbols = [s["symbol"] for s in WATCHLIST_STOCKS]
    assert len(symbols) == 15 and len(set(symbols)) == 15
    assert "HDFCBANK" in symbols and "LICI" in symbols  # real NSE symbols, not HDFCBNK / LIC
