"""HAR-RV (Corsi 2009): a heterogeneous autoregression on LOG realised variance.

    ln RV_{t+1} = b0 + b1 ln RV_d + b2 ln RV_w + b3 ln RV_m + e,
with RV_d = today's RV, RV_w = mean of the last 5 sessions, RV_m = mean of the last 22. Estimated by OLS with
Newey-West (HAC) standard errors. Converting back to variance applies the log-normal bias correction
exp(mu + s^2/2) - exp(mu) alone would systematically under-forecast. The overnight gap is not part of RV, so
the total 1-day forecast adds the mean squared gap over the last 60 sessions.

`history` needs timestamp and either an `rv` column (intraday-derived, optionally `gap`) or daily
high/low (Parkinson proxy is used). Only rows up to `asof` are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from quant_intelligence.volatility.base import VolForecast, VolModel
from quant_intelligence.volatility.realized import parkinson_daily_variance

MIN_OBS = 60
FLOOR = 1e-12
GAP_WINDOW = 60
HAC_LAGS = 5


def _rv_series(history: pd.DataFrame) -> pd.Series:
    """Realised variance per day: the intraday-derived `rv` if there is enough of it, else the daily-range proxy.
    All-NaN when neither is available (the caller then reports why)."""
    if "rv" in history.columns and history["rv"].notna().sum() >= MIN_OBS:
        rv = history["rv"].astype(float)
        if "complete" in history.columns:
            rv = rv.where(history["complete"])
    elif {"high", "low"} <= set(history.columns):
        rv = parkinson_daily_variance(history)
    else:
        return pd.Series(np.nan, index=history.index)
    return rv.where(rv > 0).clip(lower=FLOOR)


def design(rv: pd.Series) -> pd.DataFrame:
    """Features at t (known at the close of t) and the target ln RV_{t+1} (aligned to row t)."""
    d = np.log(rv)
    w = np.log(rv.rolling(5).mean())
    m = np.log(rv.rolling(22).mean())
    return pd.DataFrame({"d": d, "w": w, "m": m, "target": d.shift(-1)})


def _fit(frame: pd.DataFrame):
    x = sm.add_constant(frame[["d", "w", "m"]])
    return sm.OLS(frame["target"], x).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def _predict_var(coefs, resid_var: float, features: dict) -> float:
    mu = coefs["const"] + coefs["d"] * features["d"] + coefs["w"] * features["w"] + coefs["m"] * features["m"]
    return float(np.exp(mu + 0.5 * resid_var))


class HarRvModel(VolModel):
    name = "HAR-RV"

    def __init__(self) -> None:
        self._asof = None
        self._res = None
        self._rv: pd.Series | None = None
        self._gap_var = 0.0
        self._reason: str | None = None

    def fit(self, history: pd.DataFrame, asof) -> "HarRvModel":
        asof = pd.Timestamp(asof)
        h = history[pd.to_datetime(history["timestamp"]) <= asof].sort_values("timestamp").reset_index(drop=True)
        self._asof, self._res, self._reason = asof, None, None
        self._rv = _rv_series(h)
        if "gap" in h.columns:
            self._gap_var = float((h["gap"].dropna().iloc[-GAP_WINDOW:] ** 2).mean()) if h["gap"].notna().any() else 0.0
        frame = design(self._rv).dropna()
        if len(frame) < MIN_OBS:
            self._reason = f"only {len(frame)} usable observations (need {MIN_OBS})"
            return self
        try:
            self._res = _fit(frame)
            if not np.all(np.isfinite(self._res.params.values)):
                self._res, self._reason = None, "non-finite coefficients"
        except Exception as e:
            self._res, self._reason = None, f"fit raised {type(e).__name__}: {e}"
        return self

    def forecast(self, horizon_days: int = 1) -> VolForecast:
        if self._res is None or self._rv is None:
            raise ValueError(f"HAR-RV is not fitted: {self._reason}")  # the engine catches this and falls back
        coefs, s2 = self._res.params, float(self._res.scale)
        rv = list(self._rv.dropna().values)
        per_day = []
        for _ in range(horizon_days):  # iterate: feed each forecast back in as the next day's RV
            tail = pd.Series(rv)
            feats = {"d": np.log(tail.iloc[-1]), "w": np.log(tail.iloc[-5:].mean()), "m": np.log(tail.iloc[-22:].mean())}
            nxt = _predict_var(coefs, s2, feats)
            per_day.append(nxt + self._gap_var)
            rv.append(nxt)
        var = np.asarray(per_day)
        diag = {"coef": {k: float(v) for k, v in coefs.items()}, "pvalues": {k: float(v) for k, v in self._res.pvalues.items()},
                "r2": float(self._res.rsquared), "nobs": int(self._res.nobs), "gap_var": self._gap_var}
        return VolForecast(self.name, self._asof, float(np.sqrt(var[0])), float(np.sqrt(var.sum())), horizon_days, diag)


def rolling_har_forecasts(history: pd.DataFrame, refit_every: int = 5, min_train: int = 250) -> pd.DataFrame:
    """Out-of-sample 1-day HAR-RV forecasts: row t is made at the close of t for t+1 using data up to t.
    Coefficients are re-estimated every `refit_every` sessions on pairs fully observed by t (the target of the
    last pair is RV_t, so nothing after t is touched). Column: sigma_1d (decimal, total incl. the gap term)."""
    h = history.sort_values("timestamp").reset_index(drop=True)
    rv = _rv_series(h)
    frame = design(rv)
    gap = h["gap"] if "gap" in h.columns else pd.Series(0.0, index=h.index)
    gap2 = (gap.fillna(0.0) ** 2).rolling(GAP_WINDOW, min_periods=1).mean()
    out = np.full(len(h), np.nan)
    coefs, s2, next_refit = None, 0.0, min_train
    for t in range(min_train, len(h)):
        if coefs is None or t >= next_refit:
            known = frame.iloc[: t].dropna()  # pairs (s -> s+1) with s+1 <= t
            if len(known) >= MIN_OBS:
                try:
                    res = _fit(known)
                    coefs, s2 = res.params, float(res.scale)
                except Exception:
                    coefs = None
            next_refit = t + refit_every
        row = frame.iloc[t]
        if coefs is None or row[["d", "w", "m"]].isna().any():
            continue
        out[t] = np.sqrt(_predict_var(coefs, s2, row) + float(gap2.iloc[t]))
    return pd.DataFrame({"sigma_1d": out}, index=pd.to_datetime(h["timestamp"]))
