"""Remove the intraday volatility seasonality (the U-shape: wild at the open and close, quiet at lunch)
before fitting any intraday volatility model.

Each bar's absolute return is divided by the average absolute return of the SAME time-of-day slot over
PREVIOUS sessions only (expanding, shifted by one session), normalised by the average over all earlier
bars. A bar therefore never influences its own seasonal factor, nor any earlier bar's: no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_factors(ohlcv: pd.DataFrame, min_sessions: int = 5) -> pd.Series:
    """Per-bar multiplier (1.0 = average-volatility slot) known BEFORE the bar; NaN until `min_sessions`
    earlier sessions exist for that slot. Index matches `ohlcv`."""
    frame = ohlcv.reset_index(drop=True)
    ts = pd.to_datetime(frame["timestamp"])
    session = ts.dt.normalize()
    first = session != session.shift(1)
    prev_close = frame["close"].shift(1)
    r = np.log(frame["close"] / prev_close).where(~first, np.log(frame["close"] / frame["open"]))
    absr = r.abs()
    slot = ts.dt.hour * 60 + ts.dt.minute

    # mean |r| of the slot over earlier sessions: expanding mean within the slot, shifted by one occurrence
    by_slot = absr.groupby(slot)
    slot_mean = by_slot.transform(lambda s: s.shift(1).expanding(min_periods=min_sessions).mean())
    # average |r| over every earlier bar (a causal normaliser)
    overall = absr.shift(1).expanding(min_periods=min_sessions * 10).mean()
    factor = slot_mean / overall
    factor.index = ohlcv.index
    return factor


def deseasonalised_returns(ohlcv: pd.DataFrame, min_sessions: int = 5) -> pd.Series:
    """Bar log returns divided by their seasonal factor (NaN where the factor is not yet known)."""
    frame = ohlcv.reset_index(drop=True)
    ts = pd.to_datetime(frame["timestamp"])
    first = ts.dt.normalize() != ts.dt.normalize().shift(1)
    r = np.log(frame["close"] / frame["close"].shift(1)).where(~first, np.log(frame["close"] / frame["open"]))
    out = r / seasonal_factors(ohlcv, min_sessions).reset_index(drop=True)
    out.index = ohlcv.index
    return out
