"""Shared ranking core.

One place that turns a strategy's closed trades + the current market state into a
StrategyScore. Used by research/pipeline.py (live decisions) and by
ranking/evaluate_ranker.py (walk-forward evaluation), so what is evaluated is exactly what
runs live - a change here is measured by the evaluator automatically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_intelligence.analogues.analogue_engine import (
    COMPARISON_FEATURES,
    conditional_metrics_from_analogues,
    find_analogues,
)
from quant_intelligence.backtesting.engine import _compute_run_metrics, net_r_multiple
from quant_intelligence.ranking.ranking_engine import StrategyScore, score_strategy


@dataclass
class Assessment:
    analogues: pd.DataFrame
    conditional: dict
    metrics: dict
    score: StrategyScore


def observations_from_trades(trades, feat_by_ts: pd.DataFrame, strategy_name: str, quantity: int) -> pd.DataFrame:
    """Flatten each trade's entry-time features + outcome into one row per historical
    observation (the analogue engine's input). `feat_by_ts` is features.set_index("timestamp").
    `r_multiple` is NET of costs - what the strategy actually earns - with the gross value kept
    in `gross_r_multiple`."""
    rows = []
    for t in trades:
        if t.setup.timestamp not in feat_by_ts.index:
            continue
        feat_row = feat_by_ts.loc[t.setup.timestamp]
        row = {c: feat_row.get(c) for c in COMPARISON_FEATURES}
        row["r_multiple"] = net_r_multiple(t, quantity)
        row["gross_r_multiple"] = t.r_multiple
        row["outcome"] = t.outcome
        row["entry_timestamp"] = t.setup.timestamp
        row["strategy_name"] = strategy_name
        rows.append(row)
    return pd.DataFrame(rows)


def _stability(observations: pd.DataFrame, n_folds: int = 4, min_trades: int = 3) -> dict:
    """Is the edge positive across time? Split the strategy's closed trades into `n_folds`
    chronological folds and count the share with a positive mean net R (folds with too few trades
    are ignored). A strategy that made all its money in one stretch is not a stable edge."""
    if len(observations) < n_folds * min_trades or "entry_timestamp" not in observations.columns:
        return {"stability_folds": 0, "stability_positive_frac": None}
    ordered = observations.sort_values("entry_timestamp")["r_multiple"].to_numpy()
    means = [c.mean() for c in np.array_split(ordered, n_folds) if len(c) >= min_trades]
    if not means:
        return {"stability_folds": 0, "stability_positive_frac": None}
    return {"stability_folds": len(means), "stability_positive_frac": float(np.mean([m > 0 for m in means]))}


def assess_observations(
    strategy_name: str,
    observations: pd.DataFrame,
    current_features: dict,
    metrics: dict,
    top_k: int = 30,
) -> Assessment:
    analogues = find_analogues(current_features, observations, top_k=top_k)
    conditional = conditional_metrics_from_analogues(analogues)
    conditional["strategy_name"] = strategy_name
    conditional.update(_stability(observations))
    if len(observations):
        conditional["global_expected_r"] = float(observations["r_multiple"].mean())
    score = score_strategy(conditional, metrics.get("max_drawdown_r"))
    return Assessment(analogues=analogues, conditional=conditional, metrics=metrics, score=score)


def assess_strategy(
    strategy_name: str,
    trades,
    feat_by_ts: pd.DataFrame,
    current_features: dict,
    quantity: int,
    metrics: dict | None = None,
    top_k: int = 30,
) -> Assessment:
    observations = observations_from_trades(trades, feat_by_ts, strategy_name, quantity)
    metrics = metrics if metrics is not None else _compute_run_metrics(list(trades), quantity)
    return assess_observations(strategy_name, observations, current_features, metrics, top_k=top_k)
