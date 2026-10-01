"""Index underlyings: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY (NSE) and SENSEX (BSE options)."""
from unittest.mock import patch

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data_adapters import dhan_instrument_master as mod
from quant_intelligence.data_adapters.dhan_adapter import _INDEX_UNDERLYINGS
from quant_intelligence.options.option_selector import get_underlying_info, option_segment_for
from quant_intelligence.utils.market_profile import NSE, profile_for

INDICES = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX")

_HEADER = (
    "SEM_EXM_EXCH_ID,SEM_SEGMENT,SEM_SMST_SECURITY_ID,SEM_INSTRUMENT_NAME,SEM_TRADING_SYMBOL,"
    "SEM_LOT_UNITS,SEM_EXPIRY_DATE,SEM_STRIKE_PRICE,SM_SYMBOL_NAME\n"
)


def _patched_master(monkeypatch, tmp_path):
    path = tmp_path / "scrip_master.csv"
    path.write_text(
        _HEADER
        + "NSE,D,1,OPTIDX,FINNIFTY-Oct2099-25000-CE,60.0,2099-10-27 14:30:00,25000,FINNIFTY\n"
        + "NSE,D,2,OPTIDX,FINNIFTY-Oct2099-25050-CE,60.0,2099-10-27 14:30:00,25050,FINNIFTY\n"
        + "NSE,D,3,OPTIDX,MIDCPNIFTY-Oct2099-13000-CE,120.0,2099-10-27 14:30:00,13000,MIDCPNIFTY\n"
        + "NSE,D,4,OPTIDX,MIDCPNIFTY-Oct2099-13025-CE,120.0,2099-10-27 14:30:00,13025,MIDCPNIFTY\n"
        + "BSE,D,5,OPTIDX,SENSEX-Oct2099-84600-CE,20.0,2099-10-01 15:30:00,84600,SENSEX\n"
        + "BSE,D,6,OPTIDX,SENSEX-Oct2099-84700-CE,20.0,2099-10-01 15:30:00,84700,SENSEX\n"
        # BSE stock options are not traded: must be ignored
        + "BSE,D,7,OPTSTK,RELIANCE-Oct2099-3000-CE,500.0,2099-10-27 15:30:00,3000,RELIANCE\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_CACHE_FILE", path)
    monkeypatch.setattr(mod, "_cache", None)
    monkeypatch.setattr(mod, "_fno_cache", None)
    monkeypatch.setattr(mod, "_mcx_cache", None)
    return patch.object(mod, "_cache_is_fresh", return_value=True)


def test_all_five_indices_are_scanned_by_default():
    assert set(INDICES) <= set(SETTINGS.option_underlyings)


def test_every_index_has_dhan_index_candles_and_the_nse_session():
    for symbol in INDICES:
        assert _INDEX_UNDERLYINGS[symbol]["seg"] == "IDX_I"
        assert profile_for(symbol) is NSE  # SENSEX trades the same 09:15-15:30 hours
    assert {s: _INDEX_UNDERLYINGS[s]["security_id"] for s in ("FINNIFTY", "MIDCPNIFTY", "SENSEX")} == {
        "FINNIFTY": 27, "MIDCPNIFTY": 442, "SENSEX": 51,
    }


def test_sensex_options_route_to_bse_and_the_rest_to_their_profile_segment():
    assert option_segment_for("SENSEX") == "BSE_FNO"
    assert option_segment_for("sensex") == "BSE_FNO"
    for symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "RELIANCE"):
        assert option_segment_for(symbol) == "NSE_FNO"
    assert option_segment_for("CRUDEOIL") == "MCX_COMM"


def test_lot_specs_are_read_from_the_scrip_master_including_bse_sensex(tmp_path, monkeypatch):
    with _patched_master(monkeypatch, tmp_path):
        fin = get_underlying_info("FINNIFTY")
        mid = get_underlying_info("MIDCPNIFTY")
        sensex = get_underlying_info("SENSEX")
        bse_stock = mod.resolve_derivative_lot_specs("RELIANCE")
    assert (fin["lot_size"], fin["strike_step"]) == (60, 50.0)
    assert (mid["lot_size"], mid["strike_step"]) == (120, 25.0)
    assert (sensex["lot_size"], sensex["strike_step"], sensex["security_id"]) == (20, 100.0, 51)
    assert sensex["option_segment"] == "BSE_FNO"
    assert bse_stock is None
