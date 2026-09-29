"""
Dhan market-data adapter.

STATUS: wired up for NSE equities and the NIFTY/BANKNIFTY indices via the Dhan v2
`/charts/historical` and `/charts/intraday` endpoints. Populate DHAN_CLIENT_ID /
DHAN_ACCESS_TOKEN (or use the dashboard sidebar) to activate. There is no
synthetic fallback: with no CSV data and no working Dhan credentials, the
DataManager raises DataUnavailableError. Index candles carry no traded volume
from Dhan, so volume-based features are unreliable for NIFTY/BANKNIFTY.

"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from zoneinfo import ZoneInfo

from quant_intelligence.brokers.dhan_api_client import DhanApiClient, DhanApiError
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data_adapters import dhan_instrument_master
from quant_intelligence.data_adapters.base import DataAdapter, OHLCV_COLUMNS
from quant_intelligence.data_adapters.synthetic import _parse_timeframe_minutes
from quant_intelligence.options.option_selector import UNDERLYING_REGISTRY as _INDEX_UNDERLYINGS

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
                "in .env (or from the dashboard sidebar), or drop CSV files in data_cache/csv."
            )
        return self._fetch_from_dhan(instrument, timeframe, start, end)

    def _fetch_from_dhan(self, instrument: str, timeframe: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        index_info = _INDEX_UNDERLYINGS.get(instrument.upper())
        if index_info is not None:
            resolved = {"security_id": str(index_info["security_id"]), "exchange_segment": index_info["seg"]}
            dhan_instrument = "INDEX"
        else:
            resolved = dhan_instrument_master.resolve_equity(instrument)
            dhan_instrument = "EQUITY"
        if resolved is None:
            raise RuntimeError(
                f"'{instrument}' could not be resolved to a Dhan security_id "
                "(supported: NIFTY, BANKNIFTY and NSE equities)."
            )

        client = self._client or DhanApiClient(self.client_id, self.access_token)
        tf_minutes = _parse_timeframe_minutes(timeframe)

        try:
            if tf_minutes in _INTRADAY_ALLOWED_MINUTES and (end - start).days <= _INTRADAY_MAX_DAYS:
                body = client.get_intraday_minute(
                    security_id=resolved["security_id"],
                    exchange_segment=resolved["exchange_segment"],
                    instrument=dhan_instrument,
                    interval=str(tf_minutes),
                    from_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                    to_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                body = client.get_historical_daily(
                    security_id=resolved["security_id"],
                    exchange_segment=resolved["exchange_segment"],
                    instrument=dhan_instrument,
                    from_date=start.strftime("%Y-%m-%d"),
                    to_date=end.strftime("%Y-%m-%d"),
                )
        except DhanApiError as e:
            raise RuntimeError(f"Dhan historical-data fetch failed for {instrument}: {e}") from e

        intraday = tf_minutes in _INTRADAY_ALLOWED_MINUTES and (end - start).days <= _INTRADAY_MAX_DAYS
        df = _map_response_to_ohlcv(body)
        if intraday:
            df = _align_to_ist_session(df)
            df = drop_forming_bar(df, tf_minutes)
        return df


_SESSION_OPEN_MIN = 9 * 60 + 15
_SESSION_CLOSE_MIN = 15 * 60 + 30


def _in_session_mask(ts: pd.Series) -> pd.Series:
    minutes = ts.dt.hour * 60 + ts.dt.minute
    return (minutes >= _SESSION_OPEN_MIN) & (minutes < _SESSION_CLOSE_MIN)


def _in_session_fraction(ts: pd.Series) -> float:
    return float(_in_session_mask(ts).mean())


def _align_to_ist_session(df: pd.DataFrame) -> pd.DataFrame:
    """Return naive-IST timestamps. The epoch's timezone basis is verified from the data
    itself: NSE bars must fall within 09:15-15:30 IST, so try the raw timestamps and the
    UTC->IST shift and keep whichever fits. If neither fits, refuse rather than guess -
    a shifted clock would silently corrupt every session-based feature."""
    if df.empty:
        return df
    for shift in (pd.Timedelta(0), pd.Timedelta(hours=5, minutes=30)):
        shifted = df["timestamp"] + shift
        if _in_session_fraction(shifted) >= 0.98:
            out = df.copy()
            out["timestamp"] = shifted
            # Drop stray pre/post-close prints (e.g. zero-volume 15:30/15:35 bars).
            return out[_in_session_mask(out["timestamp"])].reset_index(drop=True)
    raise RuntimeError(
        "Dhan intraday timestamps do not align with the NSE 09:15-15:30 IST session under "
        "any known timezone basis; refusing to use them."
    )


def drop_forming_bar(df: pd.DataFrame, tf_minutes: int, now: dt.datetime | None = None) -> pd.DataFrame:
    """Drop the last bar if it is still forming (its interval has not closed yet).
    Signals on an unfinished candle repaint and are not reproducible in a backtest."""
    if df.empty:
        return df
    now = now or dt.datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    closes_at = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=tf_minutes)
    if closes_at > pd.Timestamp(now):
        return df.iloc[:-1].reset_index(drop=True)
    return df


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
