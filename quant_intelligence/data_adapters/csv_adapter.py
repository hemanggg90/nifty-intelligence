"""CSV/Parquet file adapter for research/reproducibility.

Expected file layout: quant_intelligence/data_cache/csv/<INSTRUMENT>_<TIMEFRAME>.csv
with columns: timestamp, open, high, low, close, volume
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from quant_intelligence.config.settings import DATA_CACHE_DIR
from quant_intelligence.data_adapters.base import DataAdapter, OHLCV_COLUMNS

CSV_DIR = Path(DATA_CACHE_DIR) / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)


class CSVAdapter(DataAdapter):
    name = "csv"

    def __init__(self, directory: Path | None = None):
        self.directory = directory or CSV_DIR

    def _path_for(self, instrument: str, timeframe: str) -> Path | None:
        for ext in ("parquet", "csv"):
            p = self.directory / f"{instrument}_{timeframe}.{ext}"
            if p.exists():
                return p
        return None

    def is_available(self) -> bool:
        return any(self.directory.glob("*.csv")) or any(self.directory.glob("*.parquet"))

    def get_ohlcv(self, instrument: str, timeframe: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        path = self._path_for(instrument, timeframe)
        if path is None:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[OHLCV_COLUMNS]
