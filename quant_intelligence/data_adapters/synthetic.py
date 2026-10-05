"""
Synthetic intraday OHLCV generator.

This exists so the whole research pipeline (features -> regimes -> strategies
-> backtests -> ranking -> UI) can be exercised immediately without requiring
API keys or large historical downloads. It is NOT a substitute for real data
and is clearly labelled as synthetic everywhere it is used (source="synthetic"
in market_data_metadata).

The generator produces a regime-switching random walk (alternating trend /
range / high-vol segments) so that regime and strategy-conditional-edge
research has *some* structure to find, rather than pure noise.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from quant_intelligence.data_adapters.base import DataAdapter, OHLCV_COLUMNS

NSE_OPEN = dt.time(9, 15)
NSE_CLOSE = dt.time(15, 30)


def _session_timestamps(start: dt.date, end: dt.date, freq_minutes: int) -> pd.DatetimeIndex:
    all_ts = []
    day = start
    while day <= end:
        if day.weekday() < 5:  # Mon-Fri, holidays not modelled here
            day_start = dt.datetime.combine(day, NSE_OPEN)
            day_end = dt.datetime.combine(day, NSE_CLOSE)
            ts = pd.date_range(day_start, day_end, freq=f"{freq_minutes}min")
            all_ts.append(ts)
        day += dt.timedelta(days=1)
    if not all_ts:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate(all_ts))


class SyntheticAdapter(DataAdapter):
    name = "synthetic"

    def __init__(self, seed: int = 42, base_price: float = 22000.0):
        self.seed = seed
        self.base_price = base_price

    def is_available(self) -> bool:
        return True

    def get_ohlcv(self, instrument: str, timeframe: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        freq_minutes = _parse_timeframe_minutes(timeframe)
        ts_index = _session_timestamps(start.date(), end.date(), freq_minutes)
        ts_index = ts_index[(ts_index >= start) & (ts_index <= end)]
        if len(ts_index) == 0:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        rng = np.random.default_rng(abs(hash((self.seed, instrument, timeframe))) % (2**32))
        n = len(ts_index)

        # Regime-switching drift/vol: segments of ~150 bars each, randomly trend/range/highvol.
        segment_len = 150
        drift = np.zeros(n)
        vol = np.zeros(n)
        i = 0
        while i < n:
            seg_n = min(segment_len, n - i)
            regime_choice = rng.choice(["trend_up", "trend_down", "range", "high_vol", "low_vol"], p=[0.2, 0.2, 0.3, 0.15, 0.15])
            if regime_choice == "trend_up":
                drift[i:i + seg_n] = 0.00012
                vol[i:i + seg_n] = 0.0009
            elif regime_choice == "trend_down":
                drift[i:i + seg_n] = -0.00012
                vol[i:i + seg_n] = 0.0009
            elif regime_choice == "range":
                drift[i:i + seg_n] = 0.0
                vol[i:i + seg_n] = 0.0006
            elif regime_choice == "high_vol":
                drift[i:i + seg_n] = 0.0
                vol[i:i + seg_n] = 0.0022
            else:  # low_vol
                drift[i:i + seg_n] = 0.0
                vol[i:i + seg_n] = 0.0003
            i += seg_n

        # Reset drift/vol at each session open (gap event) to avoid unrealistic multi-day trends.
        day_change = pd.Series(ts_index).dt.date.values
        session_open_mask = np.concatenate([[True], day_change[1:] != day_change[:-1]])

        log_returns = rng.normal(drift, vol, size=n)
        gap_returns = rng.normal(0, 0.003, size=n)
        log_returns = np.where(session_open_mask, gap_returns, log_returns)

        close = self.base_price * np.exp(np.cumsum(log_returns))

        # Build OHLC around the close path with realistic intrabar noise.
        intrabar_noise = rng.normal(0, vol / 2, size=n)
        open_ = np.roll(close, 1)
        open_[0] = self.base_price
        open_ = np.where(session_open_mask, open_ * (1 + rng.normal(0, 0.0015, size=n)), open_)

        high = np.maximum(open_, close) * (1 + np.abs(intrabar_noise))
        low = np.minimum(open_, close) * (1 - np.abs(intrabar_noise))

        base_volume = 50_000
        volume = (base_volume * (1 + np.abs(log_returns) * 200) * rng.uniform(0.6, 1.4, size=n)).astype(int)

        df = pd.DataFrame(
            {
                "timestamp": ts_index,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        return df.reset_index(drop=True)


from quant_intelligence.utils.timeframe import parse_timeframe_minutes as _parse_timeframe_minutes  # noqa: E402,F401
