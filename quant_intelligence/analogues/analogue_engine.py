"""
Historical Analogue Engine.

Given the current market state, find historical strategy_observations whose
market-state features were most similar, and compute conditional performance
metrics from those analogues. This is the empirical backbone that lets the
system answer "how has this strategy performed under conditions like today's"
instead of relying on unconditional (global) statistics.

Method: cosine/Euclidean similarity in a standardized feature space over a
fixed set of comparison features (chosen because they are always available,
numeric, and behaviorally meaningful: trend_slope, momentum_20,
atr_percentile_100, vol_expansion, relative_volume, vwap_distance_pct).
This is a transparent k-NN approach, not a black box - every analogue used
is inspectable (timestamp, features, outcome).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPARISON_FEATURES = [
    "trend_slope",
    "momentum_20",
    "atr_percentile_100",
    "vol_expansion",
    "relative_volume",
    "vwap_distance_pct",
]

MIN_SAMPLE_FOR_MEDIUM_CONFIDENCE = 30
MIN_SAMPLE_FOR_HIGH_CONFIDENCE = 100
MIN_SAMPLE_FOR_ANY_CONFIDENCE = 10


def find_analogues(
    current_features: dict,
    observations: pd.DataFrame,
    top_k: int = 30,
) -> pd.DataFrame:
    """observations: DataFrame with columns for each COMPARISON_FEATURES value
    (already flattened out of market_state_features JSON) plus r_multiple,
    outcome, entry_timestamp, regime_label, strategy_name.
    Returns the top_k most similar observations with a `similarity_score` column
    (0-1, higher = more similar), sorted descending.
    """
    if len(observations) == 0:
        return observations.assign(similarity_score=pd.Series(dtype=float))

    valid_cols = [c for c in COMPARISON_FEATURES if c in observations.columns]
    obs = observations.dropna(subset=valid_cols, how="all").copy()
    if len(obs) == 0:
        return obs.assign(similarity_score=[])

    for c in valid_cols:
        obs[c] = obs[c].fillna(obs[c].mean())

    means = obs[valid_cols].mean()
    stds = obs[valid_cols].std().replace(0, 1.0)

    query_vec = np.array(
        [
            ((current_features.get(f) if current_features.get(f) is not None else means[f]) - means[f]) / stds[f]
            for f in valid_cols
        ]
    )

    obs_matrix = (obs[valid_cols] - means) / stds
    distances = np.sqrt(((obs_matrix.values - query_vec) ** 2).sum(axis=1))
    max_dist = distances.max() if len(distances) and distances.max() > 0 else 1.0
    similarity = 1 - (distances / max_dist)

    obs = obs.assign(similarity_score=similarity)
    obs = obs.sort_values("similarity_score", ascending=False)
    return obs.head(top_k)


def conditional_metrics_from_analogues(analogues: pd.DataFrame) -> dict:
    """Compute conditional expectancy statistics from a set of analogue observations."""
    n = len(analogues)
    if n < MIN_SAMPLE_FOR_ANY_CONFIDENCE:
        return {
            "sample_size": n,
            "confidence_label": "INSUFFICIENT_DATA",
            "expected_r": None,
            "prob_positive_return": None,
            "win_rate": None,
            "median_r": None,
        }

    r = analogues["r_multiple"].dropna()
    wins = r > 0

    if n >= MIN_SAMPLE_FOR_HIGH_CONFIDENCE:
        confidence = "HIGH"
    elif n >= MIN_SAMPLE_FOR_MEDIUM_CONFIDENCE:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Standard error on expected R, to make uncertainty explicit in the UI.
    r_std = r.std(ddof=1) if len(r) > 1 else np.nan
    se = r_std / np.sqrt(len(r)) if len(r) > 1 else np.nan

    return {
        "sample_size": n,
        "confidence_label": confidence,
        "expected_r": float(r.mean()),
        "expected_r_stderr": float(se) if not np.isnan(se) else None,
        "prob_positive_return": float(wins.mean()),
        "win_rate": float(wins.mean()),
        "median_r": float(r.median()),
    }
