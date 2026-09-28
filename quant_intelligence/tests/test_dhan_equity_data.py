from unittest.mock import patch

from quant_intelligence.data_adapters import dhan_instrument_master
from quant_intelligence.data_adapters.dhan_adapter import _map_response_to_ohlcv


def test_resolve_equity_reads_scrip_master_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "scrip_master.csv"
    csv_path.write_text(
        "SEM_EXM_EXCH_ID,SEM_TRADING_SYMBOL,SEM_SMST_SECURITY_ID,SEM_INSTRUMENT_NAME\n"
        "NSE,RELIANCE,2885,EQUITY\n"
        "NSE,TCS,11536,EQUITY\n"
        "NSE,NIFTY,13,INDEX\n",  # non-equity row should be excluded
        encoding="utf-8",
    )
    monkeypatch.setattr(dhan_instrument_master, "_CACHE_FILE", csv_path)
    monkeypatch.setattr(dhan_instrument_master, "_cache", None)

    with patch.object(dhan_instrument_master, "_cache_is_fresh", return_value=True):
        resolved = dhan_instrument_master.resolve_equity("reliance")
        assert resolved == {"security_id": "2885", "exchange_segment": "NSE_EQ"}

        assert dhan_instrument_master.resolve_equity("NIFTY") is None
        assert dhan_instrument_master.resolve_equity("UNKNOWN_SYMBOL") is None


_SCRIP_MASTER_HEADER = (
    "SEM_EXM_EXCH_ID,SEM_TRADING_SYMBOL,SEM_SMST_SECURITY_ID,SEM_INSTRUMENT_NAME,"
    "SEM_LOT_UNITS,SEM_STRIKE_PRICE,SEM_EXPIRY_DATE\n"
)


def _write_fno_scrip_master(tmp_path):
    csv_path = tmp_path / "scrip_master.csv"
    csv_path.write_text(
        _SCRIP_MASTER_HEADER
        + "NSE,RELIANCE,2885,EQUITY,,,\n"
        + "NSE,RELIANCE-Sep2026-1300-CE,500001,OPTSTK,500,1300,2026-09-29 14:30:00\n"
        + "NSE,RELIANCE-Sep2026-1340-CE,500002,OPTSTK,500,1340,2026-09-29 14:30:00\n"
        + "NSE,RELIANCE-Oct2026-1300-CE,500003,OPTSTK,500,1300,2026-10-27 14:30:00\n"
        + "NSE,NIFTY-Sep2026-25000-CE,600001,OPTIDX,65,25000,2026-09-30 14:30:00\n"
        + "NSE,NIFTY-Sep2026-25050-CE,600002,OPTIDX,65,25050,2026-09-30 14:30:00\n",
        encoding="utf-8",
    )
    return csv_path


def _patch_scrip_master(monkeypatch, csv_path):
    from quant_intelligence.data_adapters import dhan_instrument_master as mod

    monkeypatch.setattr(mod, "_CACHE_FILE", csv_path)
    monkeypatch.setattr(mod, "_cache", None)
    monkeypatch.setattr(mod, "_fno_cache", None)
    return patch.object(mod, "_cache_is_fresh", return_value=True)


def test_resolve_fno_stock_uses_min_strike_gap_on_nearest_expiry(tmp_path, monkeypatch):
    from quant_intelligence.data_adapters import dhan_instrument_master as mod

    csv_path = _write_fno_scrip_master(tmp_path)
    with _patch_scrip_master(monkeypatch, csv_path):
        info = mod.resolve_fno_stock("reliance")
        assert info == {"security_id": "2885", "seg": "NSE_EQ", "strike_step": 40.0, "lot_size": 500}


def test_resolve_fno_stock_returns_none_for_index_symbol(tmp_path, monkeypatch):
    from quant_intelligence.data_adapters import dhan_instrument_master as mod

    csv_path = _write_fno_scrip_master(tmp_path)
    with _patch_scrip_master(monkeypatch, csv_path):
        assert mod.resolve_fno_stock("NIFTY") is None
        assert mod.resolve_fno_stock("UNKNOWN") is None


def test_list_fno_stock_symbols_excludes_indices(tmp_path, monkeypatch):
    from quant_intelligence.data_adapters import dhan_instrument_master as mod

    csv_path = _write_fno_scrip_master(tmp_path)
    with _patch_scrip_master(monkeypatch, csv_path):
        assert mod.list_fno_stock_symbols() == ["RELIANCE"]


def test_resolve_derivative_lot_specs_works_for_index_too(tmp_path, monkeypatch):
    from quant_intelligence.data_adapters import dhan_instrument_master as mod

    csv_path = _write_fno_scrip_master(tmp_path)
    with _patch_scrip_master(monkeypatch, csv_path):
        assert mod.resolve_derivative_lot_specs("NIFTY") == {"lot_size": 65, "strike_step": 50.0}


def test_map_response_to_ohlcv_converts_epoch_arrays_to_dataframe():
    body = {
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.5, 102.5],
        "volume": [1000, 1500],
        "timestamp": [1_700_000_000, 1_700_000_300],
    }
    df = _map_response_to_ohlcv(body)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[0] == 101.5


def test_map_response_to_ohlcv_raises_on_unexpected_shape():
    import pytest

    with pytest.raises(RuntimeError):
        _map_response_to_ohlcv({"data": {}})
