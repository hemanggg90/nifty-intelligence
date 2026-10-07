"""Run every volatility model for one instrument, score them out of sample, and pick one.

Daily history in, `InstrumentResult` out. The expensive part (refitting GARCH-family models over years of
data) belongs in the data keeper's once-a-day job - trading cycles only READ the stored result (store.py).

Selection rule (deliberately conservative): EWMA is the default. Another model is selected only if its QLIKE
loss is significantly lower than EWMA's (Diebold-Mariano, one-sided p < `alpha`, default 0.10) on at least
`min_eval` common out-of-sample days; among those, the lowest QLIKE wins. Nothing beating EWMA is a normal
and expected outcome. This module never raises: problems become `diagnostics`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_intelligence.volatility import evaluation as ev
from quant_intelligence.volatility.base import VolForecast
from quant_intelligence.volatility.estimators import EwmaModel, ewma_variance, log_returns
from quant_intelligence.volatility.garch_family import KINDS, GarchFamilyModel, rolling_vol_forecasts
from quant_intelligence.volatility.har_rv import HarRvModel, rolling_har_forecasts
from quant_intelligence.volatility.realized import total_variance_target

DEFAULT_MODELS = ("EWMA", "GARCH", "GJR", "EGARCH", "HAR-RV")
BENCHMARK = "EWMA"


@dataclass
class InstrumentResult:
    instrument: str
    asof: pd.Timestamp | None
    selected: str
    reason: str
    forecasts: dict[str, VolForecast] = field(default_factory=dict)  # latest forecast per model
    scores: list[dict] = field(default_factory=list)  # one row per model (out-of-sample)
    diagnostics: dict = field(default_factory=dict)

    @property
    def forecast(self) -> VolForecast | None:
        return self.forecasts.get(self.selected)


def _daily_index(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.normalize()
    return d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def oos_variance_forecasts(daily: pd.DataFrame, models=DEFAULT_MODELS, refit_every: int = 5, min_train: int = 500,
                           window: int | None = None) -> tuple[dict[str, pd.Series], dict]:
    """Each model's out-of-sample VARIANCE forecast, indexed by the date it is FOR (made at the previous
    close). Returns (series by model, info) where info holds fallback counts and per-model errors."""
    d = _daily_index(daily)
    idx = pd.DatetimeIndex(d["timestamp"])
    close = pd.Series(d["close"].astype(float).values, index=idx)
    returns = log_returns(close).dropna()
    out: dict[str, pd.Series] = {}
    info: dict = {"fallbacks": {}, "errors": {}}
    for name in models:
        try:
            if name == "EWMA":
                var = ewma_variance(returns, lam=0.94, min_periods=20)
            elif name in KINDS:
                rf = rolling_vol_forecasts(returns, name, refit_every=refit_every, min_train=min_train, window=window)
                var = rf["sigma_1d"] ** 2
                info["fallbacks"][name] = int(rf["fallback"].sum())
            elif name == "HAR-RV":
                var = rolling_har_forecasts(d, refit_every=refit_every, min_train=min(min_train, 250))["sigma_1d"] ** 2
            else:
                info["errors"][name] = "unknown model"
                continue
            out[name] = var.shift(1).dropna()  # forecast made at t is FOR t+1
        except Exception as e:
            info["errors"][name] = f"{type(e).__name__}: {e}"
    return out, info


def score_models(forecasts: dict[str, pd.Series], target: pd.Series, min_eval: int = 250, alpha: float = 0.10) -> tuple[list[dict], str, str]:
    """(score rows, selected model, reason). Scored on the dates ALL models share, so the comparison is fair."""
    names = [n for n in forecasts if forecasts[n].notna().any()]
    if not names:
        return [], BENCHMARK, "no model produced forecasts"
    common = target.dropna().index
    for n in names:
        common = common.intersection(forecasts[n].dropna().index)
    tgt = target.loc[common]
    rows, losses = [], {}
    for n in names:
        f = forecasts[n].loc[common]
        losses[n] = ev.qlike_losses(f, tgt)
        mz = ev.mincer_zarnowitz(f, tgt)
        rows.append({"model": n, "n": int(len(losses[n])), "qlike": float(losses[n].mean()) if len(losses[n]) else float("nan"),
                     "mse": ev.mse(f, tgt), "mz_alpha": mz["alpha"], "mz_beta": mz["beta"], "mz_r2": mz["r2"],
                     "mz_p": mz["p_joint"], "dm_stat": float("nan"), "dm_p": float("nan"), "selected": False})
    if BENCHMARK in losses:
        for row in rows:
            if row["model"] != BENCHMARK:
                dm = ev.diebold_mariano(losses[row["model"]], losses[BENCHMARK])
                row["dm_stat"], row["dm_p"] = dm["stat"], dm["p_model_better"]
    selected, reason = BENCHMARK if BENCHMARK in losses else rows[0]["model"], ""
    if len(common) < min_eval:
        reason = f"only {len(common)} common out-of-sample days (need {min_eval}); keeping {selected}"
    else:
        winners = [r for r in rows if r["model"] != BENCHMARK and r["dm_p"] == r["dm_p"] and r["dm_p"] < alpha and r["dm_stat"] > 0]
        if winners:
            best = min(winners, key=lambda r: r["qlike"])
            selected = best["model"]
            reason = f"{selected} beats {BENCHMARK} on QLIKE (Diebold-Mariano one-sided p={best['dm_p']:.3f}, {len(common)} days)"
        else:
            reason = f"no model significantly beats {BENCHMARK} (alpha={alpha}, {len(common)} days)"
    for row in rows:
        row["selected"] = row["model"] == selected
    return rows, selected, reason


def latest_forecasts(daily: pd.DataFrame, models=DEFAULT_MODELS, horizon_days: int = 1) -> dict[str, VolForecast]:
    """Fit each model on ALL the history and forecast from its last date."""
    d = _daily_index(daily)
    asof = d["timestamp"].iloc[-1]
    out: dict[str, VolForecast] = {}
    ewma = EwmaModel().fit(d, asof).forecast(horizon_days)
    for name in models:
        try:
            if name == "EWMA":
                out[name] = ewma
            elif name in KINDS:
                out[name] = GarchFamilyModel(name).fit(d, asof).forecast(horizon_days)
            elif name == "HAR-RV":
                out[name] = HarRvModel().fit(d, asof).forecast(horizon_days)
        except Exception as e:  # HAR raises when not fitted
            out[name] = VolForecast(name, asof, ewma.sigma_1d, ewma.sigma_horizon, horizon_days,
                                    {"fallback": "EWMA", "reason": str(e)})
    return out


def run_instrument(instrument: str, daily: pd.DataFrame, models=DEFAULT_MODELS, horizon_days: int = 1,
                   refit_every: int = 5, min_train: int = 500, min_eval: int = 250, alpha: float = 0.10,
                   window: int | None = None) -> InstrumentResult:
    """Score and select. With too little history only EWMA is produced, with the reason recorded."""
    try:
        d = _daily_index(daily)
        n = len(d)
        if n < 30:
            return InstrumentResult(instrument, None, BENCHMARK, f"only {n} daily bars - no forecast possible",
                                    diagnostics={"n_daily": n, "error": "insufficient history"})
        asof = d["timestamp"].iloc[-1]
        target = pd.Series(total_variance_target(d).values, index=pd.DatetimeIndex(d["timestamp"]))
        diag: dict = {"n_daily": n, "target": "rv+gap^2" if "rv" in d.columns and d["rv"].notna().any() else "parkinson"}
        if n < min_train + min_eval:
            models_run = ("EWMA",)
            diag["note"] = f"{n} daily bars < {min_train + min_eval}: models other than EWMA are not evaluated"
            forecasts = latest_forecasts(d, models_run, horizon_days)
            return InstrumentResult(instrument, asof, BENCHMARK, diag["note"], forecasts, [], diag)
        series, info = oos_variance_forecasts(d, models, refit_every, min_train, window)
        diag.update(info)
        rows, selected, reason = score_models(series, target, min_eval, alpha)
        forecasts = latest_forecasts(d, models, horizon_days)
        if selected not in forecasts:
            selected, reason = BENCHMARK, reason + f"; {selected} could not forecast, using {BENCHMARK}"
        return InstrumentResult(instrument, asof, selected, reason, forecasts, rows, diag)
    except Exception as e:  # never raise into the keeper / scan loop
        return InstrumentResult(instrument, None, BENCHMARK, f"vol engine failed: {type(e).__name__}: {e}",
                                diagnostics={"error": str(e)})
