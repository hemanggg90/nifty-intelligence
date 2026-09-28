import pandas as pd

from quant_intelligence.features.feature_engine import compute_features


def test_features_have_no_lookahead(synthetic_ohlcv):
    """Computing features on a truncated dataset must produce identical values
    for all bars that exist in both the full and truncated runs (up to the
    truncation point) - proof that no feature uses information from the future.
    """
    full = compute_features(synthetic_ohlcv)
    cutoff = len(synthetic_ohlcv) - 50
    truncated_input = synthetic_ohlcv.iloc[:cutoff].reset_index(drop=True)
    truncated = compute_features(truncated_input)

    compare_cols = ["return_1", "atr_14", "vwap", "trend_slope", "relative_volume"]
    # Only bars deep enough into the truncated series to have full rolling windows.
    check_from = 150
    for col in compare_cols:
        a = full[col].iloc[check_from:cutoff].reset_index(drop=True)
        b = truncated[col].iloc[check_from:].reset_index(drop=True)
        pd.testing.assert_series_equal(a, b, check_names=False, atol=1e-9)


def test_atr_is_non_negative(synthetic_features):
    valid = synthetic_features["atr_14"].dropna()
    assert (valid >= 0).all()


def test_vwap_distance_matches_definition(synthetic_ohlcv, synthetic_features):
    merged = synthetic_ohlcv.merge(synthetic_features, on="timestamp")
    recomputed = (merged["close"] - merged["vwap"]) / merged["vwap"] * 100
    pd.testing.assert_series_equal(recomputed, merged["vwap_distance_pct"], check_names=False, atol=1e-9)


def test_session_phase_values_are_valid(synthetic_features):
    assert set(synthetic_features["session_phase"].unique()) <= {"OPEN", "MID", "CLOSE"}


def test_opening_range_is_consistent_within_session(synthetic_ohlcv, synthetic_features):
    merged = synthetic_ohlcv.merge(synthetic_features, on="timestamp")
    session = merged["timestamp"].dt.date
    for _, group in merged.groupby(session):
        or_highs = group["opening_range_high"].dropna().unique()
        assert len(or_highs) <= 1, "opening range high must be constant within a session"
