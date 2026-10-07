"""Multi-year DAILY candles for the volatility models.

Sources, in order: user CSV/parquet files (`data_cache/csv/<SYMBOL>_1d.csv`), the parquet cache
(`data_cache/parquet_cache/<SYMBOL>_1d.parquet`), then Dhan's daily-candle endpoint, requested in ~2-year
windows through the shared rate limiter (`DhanApiClient` -> LIMITER, so this can never push the account over
its data-API limit). Only the missing tail is fetched once a history exists. Real data only - no synthetic
fallback. Daily bars are not run through the intraday quality gate (its staleness/session rules are for
intraday bars); instead rows with non-positive or NaN prices are dropped and the result says how current it is.

Known limits:
* MCX commodities are only available as the CURRENT front-month future, so their daily history is short and
  has no roll adjustment; treat MCX volatility forecasts as indicative until a longer series is supplied as a CSV.
* Dhan's daily-candle timestamps have not been verified live from this code base (the local token is expired);
  `_normalise_dates` handles both a midnight-IST and a 18:30-UTC epoch convention.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from quant_intelligence.config.holidays import is_trading_day
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data.data_manager import CACHE_DIR, _flight_lock, _read_cache, _write_cache
from quant_intelligence.data_adapters.csv_adapter import CSVAdapter
from quant_intelligence.data_adapters.dhan_adapter import DhanAdapter
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeutil import is_market_open, now_ist

DAILY_TF = "1d"
CHUNK_DAYS = 730
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _normalise_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Daily bars labelled by their (naive IST) calendar date."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    times = out["timestamp"].dt.strftime("%H:%M")
    if len(out) and (times == "18:30").all():  # epoch of IST midnight read as UTC
        out["timestamp"] = out["timestamp"] + pd.Timedelta(hours=5, minutes=30)
    out["timestamp"] = out["timestamp"].dt.normalize()
    return out


def _tidy(df: pd.DataFrame, max_date: dt.date | None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    out = _normalise_dates(df[[c for c in COLUMNS if c in df.columns]])
    if "volume" not in out:
        out["volume"] = float("nan")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1) & (out["high"] >= out["low"])]
    out = out.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    if max_date is not None:
        out = out[out["timestamp"].dt.date <= max_date]  # a bar newer than the last finished session is still forming
    return out.reset_index(drop=True)[COLUMNS]


def last_complete_session(now: dt.datetime, symbol: str) -> dt.date:
    """The most recent trading day whose daily bar has finished as of `now`."""
    profile = profile_for(symbol)
    day = now.date()
    if is_market_open(now, profile) or not is_trading_day(day, profile.name) or now.time() < profile.close:
        day -= dt.timedelta(days=1)
    while not is_trading_day(day, profile.name):
        day -= dt.timedelta(days=1)
    return day


def load_daily_history(symbol: str, years: int | None = None, allow_fetch: bool = True,
                       dhan: DhanAdapter | None = None, csv: CSVAdapter | None = None,
                       now: dt.datetime | None = None) -> tuple[pd.DataFrame, dict]:
    """(daily candles, info). info: {"source", "n", "first", "last", "expected_last", "fetch_error"}.
    Never raises: a failed Dhan call leaves whatever the cache already holds and reports `fetch_error`."""
    years = years or SETTINGS.vol_history_years
    now = now or now_ist()
    path = CACHE_DIR / f"{symbol}_{DAILY_TF}.parquet"
    info: dict = {"source": "cache", "fetch_error": None}
    with _flight_lock(symbol, DAILY_TF):
        cached = _read_cache(path) if path.exists() else None
        base = _tidy(cached, None) if cached is not None else pd.DataFrame(columns=COLUMNS)

        csv_df = pd.DataFrame(columns=COLUMNS)
        try:
            csv_df = (csv or CSVAdapter()).get_ohlcv(symbol, DAILY_TF, dt.datetime(1990, 1, 1), now + dt.timedelta(days=1))
        except Exception as e:
            log_event("daily_history", f"Could not read the {symbol} daily CSV: {e}", level="WARNING")
        if len(csv_df):
            base = _tidy(pd.concat([base, csv_df[COLUMNS]], ignore_index=True), None)
            info["source"] = "csv+cache"

        expected = last_complete_session(now, symbol)
        info["expected_last"] = expected
        have_last = base["timestamp"].iloc[-1].date() if len(base) else None
        wanted_start = (now - dt.timedelta(days=int(years * 365.25))).date()

        if allow_fetch and (have_last is None or have_last < expected):
            adapter = dhan or DhanAdapter()
            if adapter.is_available():
                start = wanted_start if have_last is None else have_last - dt.timedelta(days=5)
                end = now.date()
                fetched = []
                chunk_start = start
                while chunk_start <= end:
                    chunk_end = min(chunk_start + dt.timedelta(days=CHUNK_DAYS), end)
                    try:
                        part = adapter.get_ohlcv(symbol, DAILY_TF, dt.datetime.combine(chunk_start, dt.time.min),
                                                 dt.datetime.combine(chunk_end, dt.time.max))
                        if len(part):
                            fetched.append(part)
                    except Exception as e:
                        info["fetch_error"] = str(e)
                        break
                    chunk_start = chunk_end + dt.timedelta(days=1)
                if fetched:
                    base = _tidy(pd.concat([base] + fetched, ignore_index=True), expected)
                    _write_cache(base, path)
                    info["source"] = "dhan+cache" if len(base) else info["source"]
            else:
                info["fetch_error"] = "Dhan credentials not set"
        elif len(base) and cached is None:
            _write_cache(base, path)  # persist CSV-only history so later runs are cheap

    base = base[base["timestamp"] >= pd.Timestamp(wanted_start)].reset_index(drop=True) if len(base) else base
    info.update(n=len(base), first=base["timestamp"].iloc[0].date() if len(base) else None,
                last=base["timestamp"].iloc[-1].date() if len(base) else None)
    return base, info
