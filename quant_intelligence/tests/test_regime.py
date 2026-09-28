from quant_intelligence.regimes.regime_engine import REGIMES, classify_regime, regime_confidence


def test_probabilities_sum_to_one():
    features = {"trend_slope": 0.001, "momentum_20": 0.01, "atr_percentile_100": 0.5, "vol_expansion": 1.0, "relative_volume": 1.0, "market_structure": 1}
    label, probs = classify_regime(features)
    assert label in REGIMES
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert set(probs.keys()) == set(REGIMES)


def test_strong_uptrend_favors_trend_up():
    features = {"trend_slope": 0.01, "momentum_20": 0.05, "atr_percentile_100": 0.5, "vol_expansion": 1.0, "relative_volume": 1.0, "market_structure": 1}
    label, probs = classify_regime(features)
    assert label == "TREND_UP"


def test_high_atr_percentile_favors_high_vol():
    features = {"trend_slope": 0.0, "momentum_20": 0.0, "atr_percentile_100": 0.98, "vol_expansion": 1.0, "relative_volume": 1.0, "market_structure": 0}
    label, probs = classify_regime(features)
    assert probs["HIGH_VOL"] > probs["LOW_VOL"]


def test_missing_features_do_not_crash():
    label, probs = classify_regime({})
    assert label in REGIMES
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_confidence_is_max_probability():
    features = {"trend_slope": 0.01, "momentum_20": 0.05, "atr_percentile_100": 0.5, "vol_expansion": 1.0, "relative_volume": 1.0, "market_structure": 1}
    _, probs = classify_regime(features)
    assert regime_confidence(probs) == max(probs.values())
