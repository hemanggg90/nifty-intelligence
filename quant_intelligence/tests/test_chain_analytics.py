from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow, parse_option_chain


def _row(strike, ce_ltp=10.0, ce_oi=1000, ce_vol=100, ce_iv=15.0, pe_ltp=10.0, pe_oi=1000, pe_vol=100, pe_iv=15.0):
    return StrikeRow(
        strike=strike,
        ce_ltp=ce_ltp,
        ce_oi=ce_oi,
        ce_oi_change=0,
        ce_volume=ce_vol,
        ce_iv=ce_iv,
        ce_delta=0.5,
        ce_security_id=f"{int(strike)}CE",
        pe_ltp=pe_ltp,
        pe_oi=pe_oi,
        pe_oi_change=0,
        pe_volume=pe_vol,
        pe_iv=pe_iv,
        pe_delta=-0.5,
        pe_security_id=f"{int(strike)}PE",
    )


def test_total_pcr_and_atm_strike():
    snap = ChainSnapshot(
        underlying="NIFTY",
        expiry="2024-10-31",
        spot_price=100.0,
        strikes=[_row(90, ce_oi=1000, pe_oi=500), _row(100, ce_oi=2000, pe_oi=2000), _row(110, ce_oi=500, pe_oi=1000)],
    )
    assert snap.atm_strike == 100
    total_ce = 1000 + 2000 + 500
    total_pe = 500 + 2000 + 1000
    assert snap.total_pcr == total_pe / total_ce


def test_max_pain_strike_picks_minimum_payout():
    # Heavy OI concentrated at 100 on both sides -> max pain should be 100.
    snap = ChainSnapshot(
        underlying="NIFTY",
        expiry="2024-10-31",
        spot_price=100.0,
        strikes=[_row(90, ce_oi=10, pe_oi=10), _row(100, ce_oi=10000, pe_oi=10000), _row(110, ce_oi=10, pe_oi=10)],
    )
    assert snap.max_pain_strike == 100


def test_unusual_volume_strikes_flags_outlier():
    strikes = [_row(s, ce_vol=100, ce_oi=1000) for s in (80, 90, 100, 110, 120)]
    strikes.append(_row(130, ce_vol=50000, ce_oi=1000))  # huge vol/OI ratio outlier
    snap = ChainSnapshot(underlying="NIFTY", expiry="2024-10-31", spot_price=100.0, strikes=strikes)
    assert 130 in snap.unusual_volume_strikes


def test_parse_option_chain_from_raw_response():
    raw = {
        "data": {
            "last_price": 100.5,
            "oc": {
                "100.0": {
                    "ce": {"last_price": 12.5, "oi": 1000, "previous_oi": 900, "volume": 500, "implied_volatility": 14.2,
                           "greeks": {"delta": 0.55}, "security_id": "111"},
                    "pe": {"last_price": 11.0, "oi": 800, "previous_oi": 850, "volume": 300, "implied_volatility": 15.1,
                           "greeks": {"delta": -0.45}, "security_id": "222"},
                }
            },
        }
    }
    snap = parse_option_chain("NIFTY", "2024-10-31", raw)
    assert snap.spot_price == 100.5
    assert len(snap.strikes) == 1
    row = snap.strikes[0]
    assert row.strike == 100.0
    assert row.ce_oi_change == 100
    assert row.pe_oi_change == -50
    assert row.ce_security_id == "111"
