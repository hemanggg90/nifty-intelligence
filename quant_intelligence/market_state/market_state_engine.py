"""
Market State Engine.

Takes a feature DataFrame (from features.feature_engine) and produces a
single "current market state" snapshot: the latest fully-formed feature row,
plus the data-quality status that gates every downstream decision.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarketState:
    instrument: str
    timestamp: dt.datetime
    features: dict
    data_quality_status: str

    def get(self, key: str, default=None):
        val = self.features.get(key, default)
        if val is None:
            return default
        if isinstance(val, float) and np.isnan(val):
            return default
        return val


def build_current_state(
    instrument: str,
    feature_df: pd.DataFrame,
    data_quality_status: str = "OK",
    min_warmup_bars: int = 100,
) -> MarketState | None:
    """Return the latest usable market state, or None if not enough warmup data exists yet."""
    if len(feature_df) < min_warmup_bars:
        return None

    last_row = feature_df.iloc[-1]
    features = last_row.drop(labels=["timestamp"]).to_dict()

    return MarketState(
        instrument=instrument,
        timestamp=last_row["timestamp"],
        features=features,
        data_quality_status=data_quality_status,
    )


def persist_state(state: MarketState) -> None:
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import MarketState as MarketStateRow

    clean_features = {
        k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in state.features.items()
    }
    with get_session() as session:
        session.add(
            MarketStateRow(
                instrument=state.instrument,
                timestamp=state.timestamp,
                features=clean_features,
                data_quality_status=state.data_quality_status,
            )
        )
