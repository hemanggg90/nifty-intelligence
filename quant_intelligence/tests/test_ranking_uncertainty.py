"""Step 2: uncertainty-aware scoring, significance-based tie test, time-stability guard."""
import numpy as np
import pandas as pd

from quant_intelligence.ranking.ranking_engine import rank_and_select, score_strategy
from quant_intelligence.research.ranking_core import assess_observations


def _m(name, mean, se=None, n_days=20, global_r=0.0, folds=0, pos=None, conf="HIGH"):
    d = {"strategy_name": name, "confidence_label": conf, "sample_size": 30, "n_days": n_days,
         "expected_r": mean, "prob_positive_return": 0.55, "expected_r_stderr": se,
         "global_expected_r": global_r, "stability_folds": folds, "stability_positive_frac": pos}
    return d


def test_noisy_high_mean_ranks_below_reliable_modest_mean():
    noisy = score_strategy(_m("noisy", mean=0.9, se=0.8), -3.0)
    reliable = score_strategy(_m("reliable", mean=0.5, se=0.05), -3.0)
    assert reliable.score > noisy.score  # raw means would have said the opposite


def test_shrinkage_pulls_toward_the_strategys_own_global_mean():
    far_from_prior = score_strategy(_m("x", mean=0.9, se=0.5, global_r=-0.6), -3.0)
    near_prior = score_strategy(_m("y", mean=0.9, se=0.5, global_r=0.9), -3.0)
    assert far_from_prior.edge_mean < near_prior.edge_mean
    assert far_from_prior.edge_mean < 0.9


def test_without_a_stderr_the_raw_mean_is_used_unchanged():
    s = score_strategy({"strategy_name": "z", "confidence_label": "HIGH", "sample_size": 150,
                        "expected_r": 0.5, "prob_positive_return": 0.6}, -3.0)
    assert s.eligible and s.edge_mean == 0.5 and s.edge_stderr is None


def test_unstable_edge_is_ineligible_even_with_a_high_conditional_mean():
    s = score_strategy(_m("lucky", mean=0.8, se=0.1, folds=4, pos=0.25), -3.0)
    assert not s.eligible and "stable" in s.ineligibility_reason


def test_stability_guard_needs_enough_folds_to_apply():
    s = score_strategy(_m("few", mean=0.8, se=0.1, folds=2, pos=0.0), -3.0)
    assert s.eligible


def test_tie_is_decided_by_standard_errors_not_a_fixed_gap():
    # Far apart in score but both very uncertain -> not distinguishable -> NO TRADE.
    a = score_strategy(_m("A", mean=0.9, se=0.6), -3.0)
    b = score_strategy(_m("B", mean=0.6, se=0.6), -3.0)
    a.score, b.score = 1.0, 0.5  # gap 0.5 would pass the old fixed 0.08 rule
    decision = rank_and_select([a, b])
    assert decision.is_no_trade and "distinguishable" in decision.reason

    # Same means, tiny standard errors -> clear winner.
    a2 = score_strategy(_m("A", mean=0.9, se=0.03), -3.0)
    b2 = score_strategy(_m("B", mean=0.6, se=0.03), -3.0)
    assert rank_and_select([a2, b2]).selected_strategy == "A"


def _obs(r_by_time):
    ts = pd.date_range("2026-08-03 09:30", periods=len(r_by_time), freq="1D")
    df = pd.DataFrame({"r_multiple": r_by_time, "entry_timestamp": ts})
    for c in ("trend_slope", "momentum_20", "atr_percentile_100", "vol_expansion", "relative_volume", "vwap_distance_pct"):
        df[c] = np.random.default_rng(0).normal(size=len(df))
    return df


def test_assess_reports_time_stability_and_global_mean():
    good_then_bad = [1.0] * 12 + [-1.0] * 36  # positive in 1 of 4 chronological folds
    cur = {c: 0.0 for c in ("trend_slope", "momentum_20", "atr_percentile_100", "vol_expansion", "relative_volume", "vwap_distance_pct")}
    a = assess_observations("S", _obs(good_then_bad), cur, {"max_drawdown_r": -5.0})
    assert a.conditional["stability_folds"] == 4
    assert a.conditional["stability_positive_frac"] == 0.25
    assert abs(a.conditional["global_expected_r"] - np.mean(good_then_bad)) < 1e-9
