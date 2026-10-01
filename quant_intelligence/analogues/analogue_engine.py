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

MIN_SAMPLE_FOR_ANY_CONFIDENCE = 10
# Confidence is about how many INDEPENDENT trading days the analogues span, not how many rows:
# the nearest analogues are mostly adjacent bars of the same session (autocorrelated), so
# 30 rows from 2 days carry the evidence of ~2 observations, not 30.
MIN_DAYS_FOR_MEDIUM_CONFIDENCE = 6
MIN_DAYS_FOR_HIGH_CONFIDENCE = 15


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

    # Only compare on features that carry information on BOTH sides: a column that is all-NaN
    # (e.g. volume features for NIFTY/BANKNIFTY), has no variance, or is unavailable right now
    # would otherwise turn every distance into NaN and silently return the oldest rows.
    valid_cols = []
    for c in COMPARISON_FEATURES:
        if c not in observations.columns:
            continue
        col = pd.to_numeric(observations[c], errors="coerce")
        q = current_features.get(c)
        if q is None or not np.isfinite(q) or col.notna().sum() < 2 or not (col.std() > 0):
            continue
        valid_cols.append(c)
    if not valid_cols:
        return observations.head(0).assign(similarity_score=pd.Series(dtype=float))

    obs = observations.copy()
    for c in valid_cols:
        obs[c] = pd.to_numeric(obs[c], errors="coerce")
        obs[c] = obs[c].fillna(obs[c].mean())

    means = obs[valid_cols].mean()
    stds = obs[valid_cols].std()

    query_vec = np.array([(current_features[f] - means[f]) / stds[f] for f in valid_cols])

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

    days = _trading_days(analogues.loc[r.index])
    n_days = int(days.nunique()) if days is not None else n  # no timestamps: treat rows as independent
    if n_days >= MIN_DAYS_FOR_HIGH_CONFIDENCE:
        confidence = "HIGH"
    elif n_days >= MIN_DAYS_FOR_MEDIUM_CONFIDENCE:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Standard error of the mean, robust to same-day clustering of the analogues.
    se = _cluster_robust_stderr(r.to_numpy(), days.to_numpy() if days is not None else None)

    return {
        "sample_size": n,
        "n_days": n_days,
        "confidence_label": confidence,
        "expected_r": float(r.mean()),
        "expected_r_stderr": float(se) if not np.isnan(se) else None,
        "prob_positive_return": float(wins.mean()),
        "win_rate": float(wins.mean()),
        "median_r": float(r.median()),
    }


def _trading_days(analogues: pd.DataFrame):
    if "entry_timestamp" not in analogues.columns:
        return None
    return pd.to_datetime(analogues["entry_timestamp"]).dt.date


def _cluster_robust_stderr(values: np.ndarray, days) -> float:
    """SE of the mean with observations clustered by trading day (each day's residuals may move
    together). With no cluster info, or one row per day, this is the ordinary SE."""
    n = len(values)
    if n < 2:
        return np.nan
    resid = values - values.mean()
    if days is None:
        return float(values.std(ddof=1) / np.sqrt(n))
    sums = pd.Series(resid).groupby(np.asarray(days)).sum().to_numpy()
    g = len(sums)
    if g < 2:
        return np.nan  # a single day tells us nothing about day-to-day variability
    return float(np.sqrt(g / (g - 1) * np.sum(sums**2)) / n)
