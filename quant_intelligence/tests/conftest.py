import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import datetime as dt

import pytest

from quant_intelligence.data_adapters.synthetic import SyntheticAdapter
from quant_intelligence.features.feature_engine import compute_features


@pytest.fixture(scope="session")
def synthetic_ohlcv():
    adapter = SyntheticAdapter(seed=7)
    end = dt.datetime(2024, 6, 28, 15, 30)
    start = end - dt.timedelta(days=45)
    return adapter.get_ohlcv("NIFTY", "5min", start, end)


@pytest.fixture(scope="session")
def synthetic_features(synthetic_ohlcv):
    return compute_features(synthetic_ohlcv)
