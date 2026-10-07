"""GARCH(1,1), GJR-GARCH(1,1,1) and EGARCH(1,1) volatility models (Student-t innovations) via the `arch` library.

Daily log returns are scaled by 100 (percent) for numerical stability and scaled back on the way out. A
model that cannot be trusted - too little data, no convergence, non-stationary or infinite-variance fit, or
an exception - NEVER raises into the caller: it falls back to EWMA and says so in `diagnostics["fallback"]`.

Forecasts are made at the close of day t for t+1 .. t+h using only returns up to t. `rolling_vol_forecasts`
re-estimates parameters on a schedule (default every 5 sessions) and in between only FILTERS the conditional
variance with the frozen parameters (arch's `fix`), which is how a live system would run it.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from quant_intelligence.volatility.base import VolForecast, VolModel
from quant_intelligence.volatility.estimators import EwmaModel, ewma_variance, log_returns

KINDS: dict[str, dict] = {
    "GARCH": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "GJR": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    "EGARCH": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
}
MIN_OBS = 250  # about a trading year of daily returns
SCALE = 100.0
SIMULATIONS = 2000


def _build(y_pct: pd.Series, kind: str):
    from arch import arch_model

    spec = KINDS[kind]
    return arch_model(y_pct, mean="Constant", dist="t", rescale=False, **spec)


def _persistence(params: pd.Series, kind: str) -> float:
    """Sum that must stay below 1 for a stationary variance process."""
    p = params.to_dict()
    if kind == "EGARCH":
        return abs(p.get("beta[1]", 0.0))
    return p.get("alpha[1]", 0.0) + p.get("beta[1]", 0.0) + 0.5 * p.get("gamma[1]", 0.0)


def fit_checked(returns: pd.Series, kind: str, min_obs: int = MIN_OBS):
    """(fit result, None) or (None, reason). `returns` are decimal log returns."""
    y = returns.dropna()
    if len(y) < min_obs:
        return None, f"only {len(y)} daily returns (need {min_obs})"
    if float(y.std()) <= 1e-12:
        return None, "returns have no variation"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _build(y * SCALE, kind).fit(disp="off", update_freq=0, options={"maxiter": 500})
    except Exception as e:  # numerical failures inside the optimiser
        return None, f"fit raised {type(e).__name__}: {e}"
    if res.convergence_flag != 0:
        return None, "optimiser did not converge"
    if not np.all(np.isfinite(res.params.values)):
        return None, "non-finite parameters"
    if _persistence(res.params, kind) >= 0.9999:
        return None, "non-stationary variance process (persistence >= 1)"
    if "nu" in res.params and res.params["nu"] <= 2.05:
        return None, "Student-t degrees of freedom <= 2 (infinite variance)"
    return res, None


def _variance_path(result, horizon: int, kind: str) -> np.ndarray:
    """Forecast variances (in percent^2) for days 1..horizon, from a fitted or fixed arch result."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "EGARCH" and horizon > 1:  # no analytic multi-step: simulate with OUR seeded shocks (deterministic)
            nu = float(result.params["nu"]) if "nu" in result.params else None
            gen = np.random.default_rng(0)
            scale = math.sqrt((nu - 2.0) / nu) if nu else 1.0
            draw = (lambda size: gen.standard_t(nu, size) * scale) if nu else (lambda size: gen.standard_normal(size))
            fc = result.forecast(horizon=horizon, method="simulation", simulations=SIMULATIONS, reindex=False, rng=draw)
        else:
            fc = result.forecast(horizon=horizon, reindex=False)
    return np.asarray(fc.variance.values[-1], dtype=float)


class GarchFamilyModel(VolModel):
    """kind: "GARCH" | "GJR" | "EGARCH". `history` needs timestamp and close (daily bars)."""

    def __init__(self, kind: str = "GARCH", max_obs: int | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; choose from {sorted(KINDS)}")
        self.kind, self.name, self.max_obs = kind, kind, max_obs
        self._res = None
        self._fallback = EwmaModel()
        self._reason: str | None = None
        self._asof = None

    def fit(self, history: pd.DataFrame, asof) -> "GarchFamilyModel":
        asof = pd.Timestamp(asof)
        h = history[pd.to_datetime(history["timestamp"]) <= asof].sort_values("timestamp")
        r = log_returns(h["close"].astype(float)).dropna()
        if self.max_obs:
            r = r.iloc[-self.max_obs:]
        self._asof = asof
        self._fallback.fit(h, asof)  # always available as the safety net
        self._res, self._reason = fit_checked(r, self.kind)
        return self

    def forecast(self, horizon_days: int = 1) -> VolForecast:
        if self._res is not None:
            try:
                var = _variance_path(self._res, horizon_days, self.kind) / SCALE ** 2
                if np.all(np.isfinite(var)) and np.all(var > 0):
                    diag = {"params": {k: float(v) for k, v in self._res.params.items()},
                            "loglik": float(self._res.loglikelihood), "nobs": int(self._res.nobs)}
                    return VolForecast(self.name, self._asof, float(np.sqrt(var[0])), float(np.sqrt(var.sum())),
                                       horizon_days, diag)
                self._reason = "forecast was not finite and positive"
            except Exception as e:
                self._reason = f"forecast raised {type(e).__name__}: {e}"
        fb = self._fallback.forecast(horizon_days)
        return VolForecast(self.name, fb.asof, fb.sigma_1d, fb.sigma_horizon, horizon_days,
                           {"fallback": "EWMA", "reason": self._reason or "no fit", **fb.diagnostics})


def rolling_vol_forecasts(
    returns: pd.Series, kind: str, refit_every: int = 5, min_train: int = 500, window: int | None = None,
) -> pd.DataFrame:
    """Out-of-sample 1-day-ahead volatility forecasts, one per date.

    Row t is the forecast MADE AT THE CLOSE OF t for t+1, using returns up to and including t. Parameters are
    re-estimated every `refit_every` sessions on an expanding window (or the last `window` returns); between
    refits the conditional variance is filtered with the frozen parameters. A failed refit is retried at the
    next scheduled refit and EWMA is used meanwhile (`fallback` True).
    Columns: sigma_1d (decimal), fallback (bool).
    """
    y = returns.dropna()
    n = len(y)
    out_sigma = np.full(n, np.nan)
    out_fb = np.zeros(n, dtype=bool)
    ew_var = ewma_variance(y, lam=0.94, min_periods=20).values
    params = None
    next_refit = min_train - 1
    for t in range(min_train - 1, n):
        if t >= next_refit:
            lo = 0 if window is None else max(0, t + 1 - window)
            res, _ = fit_checked(y.iloc[lo:t + 1], kind)
            params = None if res is None else res.params
            next_refit = t + refit_every
        value = np.nan
        if params is not None:
            lo = 0 if window is None else max(0, t + 1 - window)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fixed = _build(y.iloc[lo:t + 1] * SCALE, kind).fix(params)
                    value = float(np.sqrt(_variance_path(fixed, 1, kind)[0])) / SCALE
            except Exception:
                value = np.nan
        if not np.isfinite(value) or value <= 0:
            value, out_fb[t] = float(np.sqrt(ew_var[t])), True
        out_sigma[t] = value
    return pd.DataFrame({"sigma_1d": out_sigma, "fallback": out_fb}, index=y.index)
