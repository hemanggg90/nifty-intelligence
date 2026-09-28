"""
Data adapter interface.

Every data source (CSV/Parquet files, synthetic generator, Dhan API, and any
future provider) implements this same interface so the rest of the research
engine never depends on a specific vendor.
"""
from __future__ import annotations

import abc
import datetime as dt

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class DataAdapter(abc.ABC):
    """Abstract base class for all market-data sources."""

    name: str = "base"

    @abc.abstractmethod
    def get_ohlcv(
        self,
        instrument: str,
        timeframe: str,
        start: dt.datetime,
        end: dt.datetime,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns OHLCV_COLUMNS, sorted by timestamp ascending.

        Implementations must NOT return any row with timestamp > `end`
        (no look-ahead) and should be idempotent/cacheable.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether this adapter is currently usable (e.g. credentials present)."""
        return True
