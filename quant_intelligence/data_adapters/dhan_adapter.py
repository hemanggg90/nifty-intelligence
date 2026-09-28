"""
Dhan market-data adapter.

STATUS: wired up for NSE equities via the Dhan v2 `/charts/historical` and
`/charts/intraday` endpoints. Populate DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in
.env to activate; until then, `is_available()` returns False and the
DataManager falls back to CSV/synthetic data automatically (see
data/data_manager.py). Only NSE equities are resolved today (via
`dhan_instrument_master`); NIFTY/BANKNIFTY index OHLCV still has no historical
source here (see options/option_selector.py's UNDERLYING_REGISTRY for their
security IDs, used for the live options-chain layer instead).

Response shape: Dhan's historical/intraday endpoints return parallel arrays
keyed by `open`/`high`/`low`/`close`/`volume`/`timestamp` (epoch seconds) -
this is the shape documented by Dhan's own Python SDK and has NOT been
exercised against a real funded account (same caveat as brokers/dhan_broker.py).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from quant_intelligence.brokers.dhan_api_client import DhanApiClient, DhanApiError
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data_adapters import dhan_instrument_master
from quant_intelligence.data_adapters.base import DataAdapter, OHLCV_COLUMNS
from quant_intelligence.data_adapters.synthetic import _parse_timeframe_minutes

_INTRADAY_MAX_DAYS = 90
_INTRADAY_ALLOWED_MINUTES = {1, 5, 15, 25, 60}


class DhanAdapter(DataAdapter):
    name = "dhan"

    def __init__(self):
        self.client_id = SETTINGS.dhan_client_id
        self.access_token = SETTINGS.dhan_access_token
        self._client: DhanApiClient | None = None

    def is_available(self) -> bool:
        return bool(self.client_id and self.access_token)

    def get_ohlcv(self, instrument: str, timeframe: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        if not self.is_available():
            raise RuntimeError(
                "DhanAdapter is not configured. Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN "
                "in .env, or use the CSV/synthetic adapters for research."
            )
        return self._fetch_from_dhan(instrument, timeframe, start, end)

    def _fetch_from_dhan(self, instrument: str, timeframe: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        resolved = dhan_instrument_master.resolve_equity(instrument)
        if resolved is None:
            raise RuntimeError(
                f"'{instrument}' could not be resolved to a Dhan NSE_EQ security_id "
                "(only individual equities are supported by this adapter; NIFTY/BANKNIFTY "
                "index OHLCV has no historical source wired up)."
            )

        client = self._client or DhanApiClient(self.client_id, self.access_token)
        tf_minutes = _parse_timeframe_minutes(timeframe)

        try:
            if tf_minutes in _INTRADAY_ALLOWED_MINUTES and (end - start).days <= _INTRADAY_MAX_DAYS:
                body = client.get_intraday_minute(
                    security_id=resolved["security_id"],
                    exchange_segment=resolved["exchange_segment"],
                    instrument="EQUITY",
                    interval=str(tf_minutes),
                    from_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                    to_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                body = client.get_historical_daily(
                    security_id=resolved["security_id"],
                    exchange_segment=resolved["exchange_segment"],
                    instrument="EQUITY",
                    from_date=start.strftime("%Y-%m-%d"),
                    to_date=end.strftime("%Y-%m-%d"),
                )
        except DhanApiError as e:
            raise RuntimeError(f"Dhan historical-data fetch failed for {instrument}: {e}") from e

        return _map_response_to_ohlcv(body)


def _map_response_to_ohlcv(body: dict) -> pd.DataFrame:
    required = ("open", "high", "low", "close", "volume", "timestamp")
    if not all(k in body for k in required):
        raise RuntimeError(f"Unexpected Dhan chart response shape: keys={list(body.keys())}")

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(body["timestamp"], unit="s"),
            "open": body["open"],
            "high": body["high"],
            "low": body["low"],
            "close": body["close"],
            "volume": body["volume"],
        }
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[OHLCV_COLUMNS]
