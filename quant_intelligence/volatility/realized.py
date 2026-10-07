"""Daily realised variance from intraday bars, and a range-based daily proxy when only daily bars exist.

Realised variance (RV) of a session = sum of squared intraday bar log returns (the first bar of the session
contributes its open-to-close return). The overnight gap (open vs the previous session's close) is kept as a
SEPARATE term, so total close-to-close variance ~ rv + gap^2. Everything here is per-session and causal: a
session's numbers use only that session's bars (plus the previous close for the gap).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant_intelligence.utils.market_profile import NSE, MarketProfile
from quant_intelligence.volatility.estimators import bars_per_day, infer_tf_minutes

LN2 = math.log(2.0)


def daily_from_intraday(
    ohlcv: pd.DataFrame, profile: MarketProfile = NSE, tf_minutes: int | None = None, min_bar_frac: float = 0.9
) -> pd.DataFrame:
    """One row per session: timestamp (date), open, high, low, close, volume, rv, gap, n_bars, complete.

    `complete` is False for a session with fewer than `min_bar_frac` of the expected bars (a half day, or a
    session still in progress): its rv is understated and should not be used as a target."""
    if ohlcv.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "rv", "gap", "n_bars", "complete"])
    tf = tf_minutes or infer_tf_minutes(ohlcv)
    expected = bars_per_day(profile, tf)
    df = ohlcv.reset_index(drop=True).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["session"] = df["timestamp"].dt.normalize()
    prev_close = df["close"].shift(1)
    first = df["session"] != df["session"].shift(1)
    r = np.log(df["close"] / prev_close).where(~first, np.log(df["close"] / df["open"]))
    df["r2"] = r ** 2

    g = df.groupby("session", sort=True)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum() if "volume" in df else np.nan,
        "rv": g["r2"].sum(), "n_bars": g["close"].size(),
    })
    out["gap"] = np.log(out["open"] / out["close"].shift(1))
    out["complete"] = out["n_bars"] >= min_bar_frac * expected
    out = out.reset_index().rename(columns={"session": "timestamp"})
    return out[["timestamp", "open", "high", "low", "close", "volume", "rv", "gap", "n_bars", "complete"]]


def parkinson_daily_variance(daily: pd.DataFrame) -> pd.Series:
    """Range-based daily variance proxy ln(H/L)^2 / (4 ln 2) - far less noisy than a squared daily return."""
    return (np.log(daily["high"] / daily["low"]) ** 2) / (4.0 * LN2)


def total_variance_target(daily: pd.DataFrame) -> pd.Series:
    """The realised variance a forecast is scored against, per session.

    With intraday-derived rv: rv + gap^2 (full close-to-close variance, incomplete sessions NaN). Otherwise
    the Parkinson range proxy. Mixing the two inside one evaluation is avoided by the caller choosing one."""
    if "rv" in daily.columns and daily["rv"].notna().any():
        gap2 = (daily["gap"].fillna(0.0)) ** 2 if "gap" in daily.columns else 0.0
        target = daily["rv"] + gap2
        if "complete" in daily.columns:
            target = target.where(daily["complete"])
        return target
    return parkinson_daily_variance(daily)
