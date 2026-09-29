from types import SimpleNamespace

from quant_intelligence.execution import multi_cycle


class _Client:
    def is_configured(self):
        return False


def test_multi_cycle_scans_every_symbol_and_isolates_failures(monkeypatch):
    seen = []

    def fake_pipeline(symbol, timeframe, start, end):
        seen.append(symbol)
        if symbol == "BAD":
            raise RuntimeError("no data")
        return SimpleNamespace(ranking=SimpleNamespace(is_no_trade=True, selected_strategy=None), ohlcv=None)

    no_trade = SimpleNamespace(order=None, strategy_name=None, setup_status=None, reason="NO TRADE")
    monkeypatch.setattr(multi_cycle, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(multi_cycle, "run_auto_option_cycle", lambda *a, **k: no_trade)
    monkeypatch.setattr(multi_cycle, "run_auto_equity_cycle", lambda *a, **k: no_trade)

    rows, pnl = multi_cycle.run_multi_instrument_cycle(
        ["NIFTY", "BANKNIFTY"], ["TCS", "BAD", "INFY"], "5min", 30, broker=None, client=_Client(), get_account=lambda: None
    )

    assert seen == ["NIFTY", "BANKNIFTY", "TCS", "BAD", "INFY"]  # indices and all stocks, in one pass
    assert [r["status"] for r in rows].count("DATA_ERROR") == 1  # one bad symbol doesn't stop the rest
    assert len(rows) == 5 and pnl == 0.0
