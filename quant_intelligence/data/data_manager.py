"""
DataManager: unified entry point for fetching OHLCV data.

Order of preference: CSV/Parquet (if the user has dropped reproducible
research data into data_cache/csv) -> Dhan (if credentials configured) ->
Synthetic (always available, clearly labelled). Results are cached to
Parquet in data_cache/ so repeated requests don't re-fetch/re-generate data.

Every fetch runs through data-quality validation and records a
MarketDataMetadata row so the system has an auditable record of what data
backed every decision.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from quant_intelligence.config.settings import DATA_CACHE_DIR
from quant_intelligence.data_adapters.csv_adapter import CSVAdapter
from quant_intelligence.data_adapters.dhan_adapter import DhanAdapter
from quant_intelligence.data_adapters.synthetic import SyntheticAdapter, _parse_timeframe_minutes
from quant_intelligence.data.quality import validate_ohlcv, clean_ohlcv, QUALITY_FAIL
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_calendar import most_recent_expected_bar_time

CACHE_DIR = Path(DATA_CACHE_DIR) / "parquet_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Must match validate_ohlcv's default staleness_threshold_minutes: a cache that
# validate_ohlcv would flag as stale must not be reused as-is - it has to be
# refetched instead, or every call just re-derives the same DEGRADED verdict.
CACHE_STALENESS_THRESHOLD_MINUTES = 30


class DataManager:
    def __init__(self):
        self.csv_adapter = CSVAdapter()
        self.dhan_adapter = DhanAdapter()
        self.synthetic_adapter = SyntheticAdapter()

    def _cache_path(self, instrument: str, timeframe: str) -> Path:
        return CACHE_DIR / f"{instrument}_{timeframe}.parquet"

    def get_ohlcv(
        self,
        instrument: str,
        timeframe: str,
        start: dt.datetime,
        end: dt.datetime,
        force_refresh: bool = False,
        prefer_source: str | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Return (dataframe, metadata_dict). metadata includes source + quality report."""
        cache_path = self._cache_path(instrument, timeframe)

        if not force_refresh and cache_path.exists():
            cached = pd.read_parquet(cache_path)
            cached["timestamp"] = pd.to_datetime(cached["timestamp"])
            in_range = cached[(cached["timestamp"] >= start) & (cached["timestamp"] <= end)]
            if len(in_range) > 0:
                last_cached = in_range["timestamp"].max().to_pydatetime()
                reference = most_recent_expected_bar_time(end)
                if last_cached.tzinfo is None:
                    last_cached = last_cached.replace(tzinfo=reference.tzinfo)
                cache_is_fresh = (reference - last_cached) <= dt.timedelta(
                    minutes=CACHE_STALENESS_THRESHOLD_MINUTES
                )
                if cache_is_fresh:
                    source = "cache"
                    df = in_range
                    return self._finalize(df, instrument, timeframe, source)

        source, df = self._fetch(instrument, timeframe, start, end, prefer_source)
        if len(df) > 0:
            df.to_parquet(cache_path, index=False)
        return self._finalize(df, instrument, timeframe, source)

    def _fetch(self, instrument, timeframe, start, end, prefer_source) -> tuple[str, pd.DataFrame]:
        order = [prefer_source] if prefer_source else []
        order += ["csv", "dhan", "synthetic"]

        for source in order:
            if source == "csv" and self.csv_adapter.is_available():
                df = self.csv_adapter.get_ohlcv(instrument, timeframe, start, end)
                if len(df) > 0:
                    return "csv", df
            elif source == "dhan" and self.dhan_adapter.is_available():
                try:
                    df = self.dhan_adapter.get_ohlcv(instrument, timeframe, start, end)
                    if len(df) > 0:
                        return "dhan", df
                except Exception as e:
                    log_event("data_manager", f"Dhan fetch failed: {e}", level="WARNING")
            elif source == "synthetic":
                df = self.synthetic_adapter.get_ohlcv(instrument, timeframe, start, end)
                return "synthetic", df
        return "none", pd.DataFrame()

    def _finalize(self, df: pd.DataFrame, instrument: str, timeframe: str, source: str) -> tuple[pd.DataFrame, dict]:
        df = clean_ohlcv(df)
        tf_minutes = _parse_timeframe_minutes(timeframe)
        report = validate_ohlcv(df, tf_minutes)

        metadata = {
            "instrument": instrument,
            "timeframe": timeframe,
            "source": source,
            "start_ts": df["timestamp"].min() if len(df) else None,
            "end_ts": df["timestamp"].max() if len(df) else None,
            "n_rows": len(df),
            "quality_status": report.status,
            "quality_report": report.as_dict(),
        }

        try:
            from quant_intelligence.database.db import get_session
            from quant_intelligence.database.models import MarketDataMetadata

            with get_session() as session:
                session.add(MarketDataMetadata(**metadata))
        except Exception:
            pass

        log_event(
            "data_manager",
            f"Fetched {len(df)} rows for {instrument} {timeframe} from {source}: {report.status}",
            level="WARNING" if report.status != "OK" else "INFO",
        )

        if report.status == QUALITY_FAIL:
            log_event("data_manager", "Data quality FAIL - downstream should force NO TRADE", level="ERROR", issues=report.issues)

        return df, metadata
