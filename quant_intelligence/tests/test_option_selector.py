from unittest.mock import patch

import pytest

from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow
from quant_intelligence.options.option_selector import (
    OptionSelectionError,
    get_underlying_info,
    resolve_expiry,
    select_contract,
)


def _chain(spot=100.0):
    strikes = []
    for strike in (90, 95, 100, 105, 110):
        strikes.append(
            StrikeRow(
                strike=strike,
                ce_ltp=max(spot - strike, 0) + 2.0,
                ce_oi=1000,
                ce_oi_change=10,
                ce_volume=100,
                ce_iv=15.0,
                ce_delta=0.5,
                ce_security_id=f"{strike}CE",
                pe_ltp=max(strike - spot, 0) + 2.0,
                pe_oi=1000,
                pe_oi_change=10,
                pe_volume=100,
                pe_iv=15.0,
                pe_delta=-0.5,
                pe_security_id=f"{strike}PE",
            )
        )
    return ChainSnapshot(underlying="NIFTY", expiry="2024-10-31", spot_price=spot, strikes=strikes)


def test_resolve_expiry_nearest_picks_earliest():
    assert resolve_expiry(["2024-11-07", "2024-10-31", "2024-11-14"], "NEAREST") == "2024-10-31"


def test_resolve_expiry_empty_raises():
    with pytest.raises(OptionSelectionError):
        resolve_expiry([], "NEAREST")


def test_select_contract_long_direction_picks_ce_atm():
    contract = select_contract(_chain(), direction="LONG", transaction="BUY", moneyness_offset=0, lot_size=50)
    assert contract.option_type == "CE"
    assert contract.strike == 100
    assert contract.transaction == "BUY"
    assert contract.security_id == "100CE"
    assert contract.lot_size == 50


def test_select_contract_short_direction_picks_pe():
    contract = select_contract(_chain(), direction="SHORT", transaction="SELL", moneyness_offset=0)
    assert contract.option_type == "PE"
    assert contract.transaction == "SELL"


def test_select_contract_otm_offset_moves_away_from_spot():
    ce_contract = select_contract(_chain(), direction="LONG", transaction="BUY", moneyness_offset=1)
    assert ce_contract.strike == 105  # one strike above spot for a call

    pe_contract = select_contract(_chain(), direction="SHORT", transaction="BUY", moneyness_offset=1)
    assert pe_contract.strike == 95  # one strike below spot for a put


def test_select_contract_invalid_direction_raises():
    with pytest.raises(OptionSelectionError):
        select_contract(_chain(), direction="SIDEWAYS", transaction="BUY")


def test_select_contract_empty_chain_raises():
    empty = ChainSnapshot(underlying="NIFTY", expiry="2024-10-31", spot_price=100.0, strikes=[])
    with pytest.raises(OptionSelectionError):
        select_contract(empty, direction="LONG", transaction="BUY")


def test_select_contract_defaults_lot_size_to_one_when_unspecified():
    # Regression guard: callers must always pass the real lot_size explicitly -
    # this default exists only as a documented fallback, never silently correct.
    contract = select_contract(_chain(), direction="LONG", transaction="BUY")
    assert contract.lot_size == 1


def test_get_underlying_info_static_index_refreshed_with_live_lot_size():
    with patch(
        "quant_intelligence.data_adapters.dhan_instrument_master.resolve_derivative_lot_specs",
        return_value={"lot_size": 65, "strike_step": 50},
    ):
        info = get_underlying_info("NIFTY")
    assert info["security_id"] == 13
    assert info["seg"] == "IDX_I"
    assert info["lot_size"] == 65  # refreshed, not the static fallback


def test_get_underlying_info_static_index_falls_back_when_refresh_unavailable():
    with patch(
        "quant_intelligence.data_adapters.dhan_instrument_master.resolve_derivative_lot_specs",
        return_value=None,
    ):
        info = get_underlying_info("niftY")
    assert info["lot_size"] == 75  # static fallback from UNDERLYING_REGISTRY


def test_get_underlying_info_resolves_stock_dynamically():
    with patch(
        "quant_intelligence.data_adapters.dhan_instrument_master.resolve_fno_stock",
        return_value={"security_id": "2885", "seg": "NSE_EQ", "strike_step": 20.0, "lot_size": 500},
    ):
        info = get_underlying_info("reliance")
    assert info == {"security_id": "2885", "seg": "NSE_EQ", "strike_step": 20.0, "lot_size": 500}


def test_get_underlying_info_unknown_symbol_raises():
    with patch(
        "quant_intelligence.data_adapters.dhan_instrument_master.resolve_fno_stock",
        return_value=None,
    ):
        with pytest.raises(OptionSelectionError):
            get_underlying_info("NOT_A_REAL_SYMBOL")
