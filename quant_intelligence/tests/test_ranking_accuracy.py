"""Regression tests for the ranking-accuracy defects (see plan: net-of-cost R, drawdown units,
evidence-based confidence, NaN comparison features)."""
import numpy as np
import pandas as pd

from quant_intelligence.analogues.analogue_engine import (
    COMPARISON_FEATURES,
    conditional_metrics_from_analogues,
    find_analogues,
)
from quant_intelligence.ranking.ranking_engine import score_strategy


def _obs(n=40, seed=0, nan_cols=()):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in COMPARISON_FEATURES})
    df["r_multiple"] = rng.normal(0.2, 1.0, size=n)
    df["entry_timestamp"] = pd.date_range("2026-08-03 09:30", periods=n, freq="5min")
    for c in nan_cols:
        df[c] = np.nan
    return df


# ---- analogue matching ----------------------------------------------------------------

def test_all_nan_feature_columns_do_not_break_similarity():
    """NIFTY/BANKNIFTY have no volume: relative_volume / vwap_distance_pct are all-NaN."""
    obs = _obs(nan_cols=("relative_volume", "vwap_distance_pct"))
    current = {c: 0.0 for c in COMPARISON_FEATURES}
    current["relative_volume"] = current["vwap_distance_pct"] = float("nan")

    out = find_analogues(current, obs, top_k=10)

    assert len(out) == 10
    assert out["similarity_score"].notna().all()
    assert out["similarity_score"].is_monotonic_decreasing


def test_nearest_analogues_are_actually_the_nearest_when_a_feature_is_nan():
    obs = _obs(nan_cols=("relative_volume",))
    obs.loc[7, ["trend_slope", "momentum_20", "atr_percentile_100", "vol_expansion", "vwap_distance_pct"]] = 0.0
    current = {c: 0.0 for c in COMPARISON_FEATURES}
    current["relative_volume"] = float("nan")

    out = find_analogues(current, obs, top_k=3)

    assert out.index[0] == 7  # the exact match is ranked first, not just the oldest rows


def test_query_feature_unavailable_now_is_ignored_not_poisoning():
    obs = _obs()
    current = {c: 0.0 for c in COMPARISON_FEATURES}
    current["momentum_20"] = None
    assert find_analogues(current, obs, top_k=5)["similarity_score"].notna().all()


# ---- confidence -----------------------------------------------------------------------

def _analogues(days: int, per_day: int, seed=1):
    rng = np.random.default_rng(seed)
    ts = []
    for d in range(days):
        base = pd.Timestamp("2026-08-03 09:30") + pd.Timedelta(days=d)
        ts += [base + pd.Timedelta(minutes=5 * i) for i in range(per_day)]
    return pd.DataFrame({"r_multiple": rng.normal(0.3, 1.0, size=len(ts)), "entry_timestamp": ts})


def test_confidence_reflects_independent_days_not_row_count():
    same_day = conditional_metrics_from_analogues(_analogues(days=1, per_day=30))
    many_days = conditional_metrics_from_analogues(_analogues(days=30, per_day=1))
    assert same_day["n_days"] == 1 and many_days["n_days"] == 30
    assert same_day["confidence_label"] in ("LOW", "INSUFFICIENT_DATA")  # 30 rows, one day: not MEDIUM
    assert many_days["confidence_label"] == "HIGH"


def test_cluster_robust_stderr_is_wider_when_observations_are_clustered():
    rng = np.random.default_rng(3)
    ts, r = [], []
    for d in range(6):  # 6 days x 5 trades sharing a common daily shock
        shock = rng.normal(0, 1.0)
        for i in range(5):
            ts.append(pd.Timestamp("2026-08-03 09:30") + pd.Timedelta(days=d, minutes=5 * i))
            r.append(shock + rng.normal(0, 0.1))
    m = conditional_metrics_from_analogues(pd.DataFrame({"r_multiple": r, "entry_timestamp": ts}))
    naive = float(np.std(r, ddof=1) / np.sqrt(len(r)))
    assert m["expected_r_stderr"] > 2 * naive


# ---- drawdown penalty -----------------------------------------------------------------

def _metrics(name):
    return {"strategy_name": name, "confidence_label": "MEDIUM", "sample_size": 30, "expected_r": 0.4,
            "prob_positive_return": 0.55}


def test_drawdown_penalty_discriminates_between_strategies():
    """max_drawdown is in R; a 2R drawdown must cost less than a 15R one (it used to saturate for both)."""
    shallow = score_strategy(_metrics("A"), max_drawdown=-2.0)
    deep = score_strategy(_metrics("B"), max_drawdown=-15.0)
    assert shallow.score > deep.score
    assert shallow.components["drawdown_penalty"] > deep.components["drawdown_penalty"]


def test_drawdown_penalty_is_capped():
    huge = score_strategy(_metrics("A"), max_drawdown=-500.0)
    assert huge.components["drawdown_penalty"] >= -0.2 - 1e-9
