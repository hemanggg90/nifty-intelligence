"""Realised-volatility estimators. All are causal (rolling/EWM windows ending at the current row).

The range estimators (Parkinson, Garman-Klass, Yang-Zhang) need only OHLC, so they work for NIFTY/BANKNIFTY
where Dhan gives no volume. Outputs are ANNUALISED volatilities (decimal, 0.15 = 15%) unless a function says
otherwise; `ppy` is the number of bars of the input per year.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.market_profile import NSE, MarketProfile
from quant_intelligence.volatility.base import VolForecast, VolModel

LN2 = math.log(2.0)


def periods_per_year(profile: MarketProfile, tf_minutes: int, trading_days: int | None = None) -> float:
    """Bars per year for a session profile (NSE 5-minute: 75 bars/day x 252 days)."""
    days = trading_days or SETTINGS.trading_days_per_year
    return days * profile.session_minutes / tf_minutes


def bars_per_day(profile: MarketProfile, tf_minutes: int) -> float:
    return profile.session_minutes / tf_minutes


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def close_close_vol(close: pd.Series, window: int, ppy: float) -> pd.Series:
    """Sample standard deviation of log returns over `window` bars, annualised."""
    return log_returns(close).rolling(window, min_periods=window).std(ddof=1) * math.sqrt(ppy)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int, ppy: float) -> pd.Series:
    """Parkinson (1980): uses the high-low range; far more efficient than close-close for driftless prices."""
    hl2 = np.log(high / low) ** 2
    var = hl2.rolling(window, min_periods=window).mean() / (4.0 * LN2)
    return np.sqrt(var * ppy)


def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int, ppy: float) -> pd.Series:
    """Garman-Klass (1980): range plus open-close; assumes no overnight jump within the window's bars."""
    term = 0.5 * np.log(high / low) ** 2 - (2.0 * LN2 - 1.0) * np.log(close / open_) ** 2
    var = term.rolling(window, min_periods=window).mean()
    return np.sqrt(var.clip(lower=0.0) * ppy)


def yang_zhang_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int, ppy: float) -> pd.Series:
    """Yang-Zhang (2000): handles overnight jumps (open vs previous close) and drift; the most robust OHLC
    estimator. Needs window >= 2."""
    if window < 2:
        raise ValueError("Yang-Zhang needs window >= 2")
    overnight = np.log(open_ / close.shift(1))
    open_close = np.log(close / open_)
    rs = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)
    var_o = overnight.rolling(window, min_periods=window).var(ddof=1)
    var_c = open_close.rolling(window, min_periods=window).var(ddof=1)
    var_rs = rs.rolling(window, min_periods=window).mean()
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    var = var_o + k * var_c + (1.0 - k) * var_rs
    return np.sqrt(var.clip(lower=0.0) * ppy)


def ewma_variance(returns: pd.Series, lam: float | None = None, halflife: float | None = None, min_periods: int = 10) -> pd.Series:
    """RiskMetrics-style zero-mean EWMA of squared returns: var_t = lam*var_{t-1} + (1-lam)*r_t^2.
    Give `lam` (daily default 0.94) or `halflife` (in observations), not both."""
    if lam is not None and halflife is not None:
        raise ValueError("give lam or halflife, not both")
    sq = returns.astype(float) ** 2
    if halflife is not None:
        return sq.ewm(halflife=halflife, adjust=False, min_periods=min_periods).mean()
    lam = 0.94 if lam is None else lam
    return sq.ewm(alpha=1.0 - lam, adjust=False, min_periods=min_periods).mean()


def ewma_vol(returns: pd.Series, ppy: float, lam: float | None = 0.94, halflife: float | None = None, min_periods: int = 10) -> pd.Series:
    """Annualised EWMA volatility (daily lambda 0.94 by default; pass `halflife` for intraday bars)."""
    var = ewma_variance(returns, lam=None if halflife is not None else lam, halflife=halflife, min_periods=min_periods)
    return np.sqrt(var * ppy)


def rolling_percentile(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile rank (0..1) of each value within the trailing `window` values (itself included): causal."""
    def rank(values: np.ndarray) -> float:
        last = values[-1]
        if np.isnan(last):
            return np.nan
        valid = values[~np.isnan(values)]
        return float((valid <= last).sum() / len(valid))

    return series.rolling(window, min_periods=min_periods or max(2, window // 4)).apply(rank, raw=True)


def infer_tf_minutes(ohlcv: pd.DataFrame) -> int:
    """Bar length in minutes from the median spacing within sessions (5 for 5-minute data)."""
    ts = pd.to_datetime(ohlcv["timestamp"]).reset_index(drop=True)
    diffs = ts.diff()
    same_day = diffs[(ts.dt.date == ts.dt.date.shift(1)) & diffs.notna()]
    if same_day.empty:
        return 1440
    return max(1, int(round(same_day.median().total_seconds() / 60.0)))


def intraday_ewma_annualised_vol(
    ohlcv: pd.DataFrame,
    profile: MarketProfile = NSE,
    tf_minutes: int | None = None,
    halflife_days: float = 5.0,
    gap_halflife_days: float = 20.0,
    trading_days: int | None = None,
) -> pd.Series:
    """Annualised volatility as of each bar from intraday bars, causal.

    Total daily variance = bars_per_day x EWMA(intra-session bar return^2) + EWMA(overnight gap^2), so the
    overnight jump is counted once per day and not as a spike inside the bar EWMA. The first bar of a session
    contributes its open-to-close return (not the gap). The gap term needs >= 3 observed gaps; before that it
    is omitted (understating total variance slightly). Intraday seasonality is not removed here (Phase 2).
    """
    if ohlcv.empty:
        return pd.Series(dtype=float)
    days = trading_days or SETTINGS.trading_days_per_year
    tf = tf_minutes or infer_tf_minutes(ohlcv)
    bpd = bars_per_day(profile, tf)
    frame = ohlcv.reset_index(drop=True)
    open_, close = frame["open"].astype(float), frame["close"].astype(float)
    session = pd.to_datetime(frame["timestamp"]).dt.date
    first = session != session.shift(1)

    prev_close = close.shift(1)
    r = np.log(close / prev_close)
    r_intra = r.where(~first, np.log(close / open_))
    bar_var = (r_intra ** 2).ewm(halflife=halflife_days * bpd, adjust=False, min_periods=max(int(bpd), 5)).mean()

    gap_sq = (np.log(open_ / prev_close) ** 2).where(first).dropna()
    gap_var_daily = gap_sq.ewm(halflife=gap_halflife_days, adjust=False, min_periods=3).mean()
    gap_var = gap_var_daily.reindex(frame.index).ffill().fillna(0.0)

    out = np.sqrt((bpd * bar_var + gap_var) * days)
    out.index = ohlcv.index
    return out


class EwmaModel(VolModel):
    """Daily EWMA benchmark and universal fallback. `history` needs timestamp and close (daily bars)."""

    name = "EWMA"

    def __init__(self, lam: float = 0.94) -> None:
        self.lam = lam
        self._asof = None
        self._var = float("nan")
        self._n = 0

    def fit(self, history: pd.DataFrame, asof) -> "EwmaModel":
        asof = pd.Timestamp(asof)
        h = history[pd.to_datetime(history["timestamp"]) <= asof].sort_values("timestamp")
        r = log_returns(h["close"].astype(float)).dropna()
        self._n = len(r)
        self._asof = asof
        self._var = float(ewma_variance(r, lam=self.lam, min_periods=1).iloc[-1]) if len(r) else float("nan")
        return self

    def forecast(self, horizon_days: int = 1) -> VolForecast:
        sigma = math.sqrt(self._var) if self._var == self._var else float("nan")
        return VolForecast(self.name, self._asof, sigma, sigma * math.sqrt(horizon_days), horizon_days,
                           {"lambda": self.lam, "n_obs": self._n})
