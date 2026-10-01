"""Quote snapshots for the market-watch tables, read from the on-disk candle cache.

No Dhan request is made here: the candles are whatever the pipeline last cached (refreshed once per
closed bar), so a market-watch of 25 instruments costs nothing against Dhan's rate limits. Prices are
therefore "as of the last closed bar", and the snapshot says so.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from quant_intelligence.config.settings import DATA_CACHE_DIR

CACHE_DIR = DATA_CACHE_DIR / "parquet_cache"


def snapshot_from_frame(df: pd.DataFrame, spark_bars: int = 48) -> dict | None:
    """Last price, day change, day range and a sparkline from a candle frame (naive IST timestamps)."""
    if df is None or len(df) == 0:
        return None
    df = df.sort_values("timestamp")
    ts = pd.to_datetime(df["timestamp"])
    last_day = ts.dt.date.iloc[-1]
    today = df[ts.dt.date == last_day]
    prior = df[ts.dt.date < last_day]
    last = float(df["close"].iloc[-1])
    prev_close = float(prior["close"].iloc[-1]) if len(prior) else float(today["open"].iloc[0])
    change = last - prev_close
    return {
        "last": last,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change / prev_close * 100.0 if prev_close else 0.0,
        "open": float(today["open"].iloc[0]),
        "high": float(today["high"].max()),
        "low": float(today["low"].min()),
        "as_of": pd.Timestamp(ts.iloc[-1]).to_pydatetime(),
        "session": last_day,
        "spark": [float(x) for x in df["close"].tail(spark_bars)],
    }


def load_snapshot(symbol: str, timeframe: str = "5min") -> dict | None:
    path = CACHE_DIR / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        return None
    try:
        return snapshot_from_frame(pd.read_parquet(path))
    except Exception:
        return None


def load_candles(symbol: str, timeframe: str = "5min", sessions: int = 2) -> pd.DataFrame | None:
    """The last `sessions` trading sessions of cached candles (for the price chart)."""
    path = CACHE_DIR / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    days = sorted(df["timestamp"].dt.date.unique())[-sessions:]
    return df[df["timestamp"].dt.date.isin(days)].sort_values("timestamp").reset_index(drop=True)


def staleness_minutes(as_of: dt.datetime | None, now: dt.datetime) -> float | None:
    return None if as_of is None else (now - as_of).total_seconds() / 60.0
