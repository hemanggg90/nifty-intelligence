"""Out-of-sample forecast evaluation against realised variance.

All functions take VARIANCE forecasts and VARIANCE targets (not volatilities) on the same index.

* QLIKE (Patton 2011): mean(ln f + r / f). Robust to noise in the target and, unlike MSE, penalises
  under-forecasting more than over-forecasting - the right loss for risk and option pricing. Lower is better.
  It is a loss up to an additive constant, so only differences between models are meaningful.
* MSE on variance (dominated by the largest days; reported for completeness).
* Mincer-Zarnowitz: regress the target on the forecast, r = a + b f + e. An efficient forecast has a = 0,
  b = 1; the joint Wald p-value (HAC) says whether it can be rejected.
* Diebold-Mariano (1995) with Newey-West variance and the Harvey-Leybourne-Newbold small-sample correction:
  is model A's loss significantly lower than the benchmark's?
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

FLOOR = 1e-12


def _clean(forecast: pd.Series, target: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    df = pd.concat([forecast.rename("f"), target.rename("r")], axis=1).dropna()
    df = df[(df["f"] > 0) & (df["r"] >= 0)]
    return df["f"].to_numpy(float).clip(min=FLOOR), df["r"].to_numpy(float)


def qlike_losses(forecast: pd.Series, target: pd.Series) -> pd.Series:
    """Per-observation QLIKE loss (index = the aligned, valid dates)."""
    df = pd.concat([forecast.rename("f"), target.rename("r")], axis=1).dropna()
    df = df[(df["f"] > 0) & (df["r"] >= 0)]
    f = df["f"].clip(lower=FLOOR)
    return np.log(f) + df["r"] / f


def qlike(forecast: pd.Series, target: pd.Series) -> float:
    losses = qlike_losses(forecast, target)
    return float(losses.mean()) if len(losses) else float("nan")


def mse(forecast: pd.Series, target: pd.Series) -> float:
    f, r = _clean(forecast, target)
    return float(np.mean((f - r) ** 2)) if len(f) else float("nan")


def mincer_zarnowitz(forecast: pd.Series, target: pd.Series, lags: int = 5) -> dict:
    """{"alpha","beta","r2","p_joint","n"}; p_joint tests alpha = 0 and beta = 1 together (HAC)."""
    f, r = _clean(forecast, target)
    nan = {"alpha": float("nan"), "beta": float("nan"), "r2": float("nan"), "p_joint": float("nan"), "n": len(f)}
    if len(f) < 30 or np.ptp(f) <= 0:
        return nan
    res = sm.OLS(r, sm.add_constant(f)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    wald = res.wald_test("const = 0, x1 = 1", scalar=True)
    return {"alpha": float(res.params[0]), "beta": float(res.params[1]), "r2": float(res.rsquared),
            "p_joint": float(wald.pvalue), "n": len(f)}


def diebold_mariano(loss_model: pd.Series, loss_benchmark: pd.Series, horizon: int = 1) -> dict:
    """Tests whether the model's loss is lower than the benchmark's.

    d_t = loss_benchmark - loss_model, so a POSITIVE mean favours the model. Returns
    {"stat","p_two_sided","p_model_better","mean_diff","n"}. Newey-West variance with horizon-1 lags,
    Harvey-Leybourne-Newbold correction and a Student-t reference distribution."""
    d = (loss_benchmark - loss_model).dropna()
    n = len(d)
    nan = {"stat": float("nan"), "p_two_sided": float("nan"), "p_model_better": float("nan"),
           "mean_diff": float(d.mean()) if n else float("nan"), "n": n}
    if n < 20:
        return nan
    x = d.to_numpy(float)
    mean = x.mean()
    lag = max(horizon - 1, 0)
    gamma0 = float(np.mean((x - mean) ** 2))
    var = gamma0
    for k in range(1, lag + 1):
        cov = float(np.mean((x[k:] - mean) * (x[:-k] - mean)))
        var += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    if var <= 0:
        return nan
    stat = mean / math.sqrt(var / n)
    hln = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat *= hln
    p_two = float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))
    p_better = float(1 - stats.t.cdf(stat, df=n - 1))
    return {"stat": float(stat), "p_two_sided": p_two, "p_model_better": p_better, "mean_diff": float(mean), "n": n}
