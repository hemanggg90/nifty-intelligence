"""Phase 2 of the volatility layer: realised variance, seasonality, GARCH family, HAR-RV, evaluation, the
engine's model selection, daily history, the store and the daily job. No network, no look-ahead."""
import dataclasses
import datetime as dt
import importlib.util
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.config import settings as settings_module
from quant_intelligence.tests.test_vol_phase1 import gbm_ohlc
from quant_intelligence.utils.market_profile import NSE
from quant_intelligence.volatility import evaluation as ev
from quant_intelligence.volatility import forecast_engine as fe
from quant_intelligence.volatility import garch_family as gf
from quant_intelligence.volatility import har_rv, realized, seasonality, store
from quant_intelligence.volatility.estimators import EwmaModel


# ---------------------------------------------------------------- simulation helpers
def sim_garch(n, kind="GARCH", seed=0, omega=0.05, alpha=0.08, beta=0.90, gamma=0.0, nu=8.0, burn=500):
    """Percent returns from a GARCH / GJR / EGARCH(1,1) process with Student-t shocks (unit variance)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n + burn) * math.sqrt((nu - 2) / nu)
    r = np.zeros(n + burn)
    if kind == "EGARCH":
        lv = omega / (1 - beta)
        for t in range(n + burn):
            r[t] = math.exp(0.5 * lv) * z[t]
            lv = omega + beta * lv + alpha * (abs(z[t]) - math.sqrt(2 / math.pi)) + gamma * z[t]
    else:
        v = omega / (1 - alpha - beta - 0.5 * gamma)
        for t in range(n + burn):
            r[t] = math.sqrt(v) * z[t]
            v = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * v
    return pd.Series(r[burn:] / 100.0, index=pd.bdate_range("2012-01-02", periods=n))


def sim_daily(n=700, seed=1):
    """Daily frame with a GARCH volatility, a realised variance proxy (rv), zero gaps and a close path."""
    rng = np.random.default_rng(seed)
    ret = sim_garch(n, seed=seed)
    sigma2 = np.empty(n)
    v = 0.05 / 0.02
    for t in range(n):
        sigma2[t] = v
        v = 0.05 + 0.08 * (ret.iloc[t] * 100) ** 2 + 0.90 * v
    rv = sigma2 / 1e4 * rng.gamma(30, 1 / 30, n)  # a low-noise realised variance
    close = 100 * np.exp(ret.cumsum().values)
    return pd.DataFrame({
        "timestamp": ret.index, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": 1.0, "rv": rv, "gap": 0.0, "n_bars": 75, "complete": True,
    })


# ---------------------------------------------------------------- realised variance
def test_daily_rv_matches_the_known_volatility_and_flags_incomplete_sessions():
    df = gbm_ohlc(n_days=30, sigma_annual=0.20)
    df = df.iloc[:-30]  # the last session is cut short
    daily = realized.daily_from_intraday(df, NSE, 5)
    assert len(daily) == 30 and daily["complete"].iloc[:-1].all() and not daily["complete"].iloc[-1]
    assert daily["n_bars"].iloc[0] == 75
    assert math.sqrt(daily["rv"].iloc[:-1].mean() * 252) == pytest.approx(0.20, rel=0.12)
    assert daily["gap"].iloc[0] != daily["gap"].iloc[0]  # NaN: no previous close for the first session
    target = realized.total_variance_target(daily)
    assert target.iloc[:-1].notna().all() and np.isnan(target.iloc[-1])  # incomplete session is never a target


def test_range_proxy_used_when_there_is_no_intraday_rv():
    daily = pd.DataFrame({"high": [102.0, 103.0], "low": [100.0, 100.0]})
    t = realized.total_variance_target(daily)
    assert t.iloc[0] == pytest.approx(math.log(1.02) ** 2 / (4 * math.log(2)))


def test_daily_rv_is_causal():
    df = gbm_ohlc(n_days=20)
    base = realized.daily_from_intraday(df.iloc[: 75 * 10], NSE, 5)
    mutated = df.copy()
    mutated.loc[75 * 10:, ["open", "high", "low", "close"]] *= 3.0
    after = realized.daily_from_intraday(mutated, NSE, 5).iloc[:10]
    np.testing.assert_allclose(base["rv"].values, after["rv"].values)


# ---------------------------------------------------------------- seasonality
def u_shaped_bars(n_days=40, bars=75, seed=3):
    rng = np.random.default_rng(seed)
    slots = np.arange(bars)
    shape = 1.0 + 1.2 * np.exp(-slots / 6.0) + 0.9 * np.exp(-(bars - 1 - slots) / 6.0)  # loud open and close
    rows, price = [], 20000.0
    for day in pd.bdate_range("2026-06-01", periods=n_days):
        for s, ts in enumerate(pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=bars, freq="5min")):
            o = price
            price = o * math.exp(rng.normal(0, 0.0004 * shape[s]))
            rows.append((ts, o, max(o, price), min(o, price), price, 1.0))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_seasonal_factor_sees_the_u_shape_and_is_causal():
    df = u_shaped_bars()
    f = seasonality.seasonal_factors(df, min_sessions=5)
    assert f.iloc[: 75 * 5].isna().all()  # needs earlier sessions for the slot
    last_day = f.iloc[-75:].reset_index(drop=True)
    midday = last_day.iloc[30:46].mean()  # one slot is noisy; the midday block is not
    assert last_day.iloc[1] > 1.3 and midday < 0.95 and last_day.iloc[-2] > 1.1  # loud open and close, quiet middle

    cut = 75 * 25
    mutated = df.copy()
    mutated.loc[cut:, ["open", "high", "low", "close"]] *= 2.0
    np.testing.assert_allclose(seasonality.seasonal_factors(mutated).iloc[:cut].dropna().values, f.iloc[:cut].dropna().values)


def test_deseasonalising_flattens_volatility_across_the_session():
    df = u_shaped_bars(n_days=60)
    r = np.log(df["close"] / df["open"])
    z = seasonality.deseasonalised_returns(df)
    slot = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    raw = r.groupby(slot).std()
    flat = z.iloc[75 * 10:].groupby(slot.iloc[75 * 10:]).std()
    assert (raw.max() / raw.min()) > 2.0 and (flat.max() / flat.min()) < (raw.max() / raw.min()) / 1.5


# ---------------------------------------------------------------- GARCH family: parameter recovery
def test_garch_recovers_known_parameters():
    ret = sim_garch(4000, "GARCH", seed=11)
    res, why = gf.fit_checked(ret, "GARCH")
    assert why is None
    p = res.params
    assert p["alpha[1]"] == pytest.approx(0.08, abs=0.035) and p["beta[1]"] == pytest.approx(0.90, abs=0.05)
    assert p["alpha[1]"] + p["beta[1]"] == pytest.approx(0.98, abs=0.025)
    assert 5 < p["nu"] < 14


def test_gjr_detects_the_leverage_effect():
    ret = sim_garch(4000, "GJR", seed=12, alpha=0.03, gamma=0.12, beta=0.88, omega=0.04)
    res, why = gf.fit_checked(ret, "GJR")
    assert why is None and res.params["gamma[1]"] > 0.05  # negative shocks add variance
    plain = gf.fit_checked(ret, "GARCH")[0]
    assert res.loglikelihood > plain.loglikelihood  # the asymmetric model fits this data better


def test_egarch_recovers_persistence_and_leverage_sign():
    ret = sim_garch(4000, "EGARCH", seed=13, omega=0.02, alpha=0.12, beta=0.96, gamma=-0.06)
    res, why = gf.fit_checked(ret, "EGARCH")
    assert why is None
    assert res.params["beta[1]"] == pytest.approx(0.96, abs=0.04) and res.params["gamma[1]"] < -0.02


@pytest.mark.parametrize("kind", ["GARCH", "GJR", "EGARCH"])
def test_forecasts_are_positive_and_scale_with_the_horizon(kind):
    ret = sim_garch(1500, seed=21)
    hist = pd.DataFrame({"timestamp": ret.index, "close": 100 * np.exp(ret.cumsum().values)})
    model = gf.GarchFamilyModel(kind).fit(hist, hist["timestamp"].iloc[-1])
    f1, f10 = model.forecast(1), model.forecast(10)
    assert "fallback" not in f1.diagnostics and f1.sigma_1d > 0
    assert f10.sigma_horizon > f10.sigma_1d  # total vol over 10 days exceeds the 1-day figure
    assert f10.sigma_horizon == pytest.approx(f1.sigma_1d * math.sqrt(10), rel=0.6)
    assert 0.001 < f1.sigma_1d < 0.05


def test_egarch_multi_step_is_deterministic():
    ret = sim_garch(1200, seed=22)
    hist = pd.DataFrame({"timestamp": ret.index, "close": 100 * np.exp(ret.cumsum().values)})
    a = gf.GarchFamilyModel("EGARCH").fit(hist, hist["timestamp"].iloc[-1]).forecast(5)
    b = gf.GarchFamilyModel("EGARCH").fit(hist, hist["timestamp"].iloc[-1]).forecast(5)
    assert a.sigma_horizon == b.sigma_horizon  # simulation uses a fixed seed


# ---------------------------------------------------------------- fallbacks: never raise, say why
def test_short_history_falls_back_to_ewma_with_a_reason():
    ret = sim_garch(120, seed=5)
    hist = pd.DataFrame({"timestamp": ret.index, "close": 100 * np.exp(ret.cumsum().values)})
    f = gf.GarchFamilyModel("GJR").fit(hist, hist["timestamp"].iloc[-1]).forecast(3)
    assert f.diagnostics["fallback"] == "EWMA" and "only" in f.diagnostics["reason"] and f.sigma_1d > 0


def test_constant_prices_fall_back_instead_of_crashing():
    hist = pd.DataFrame({"timestamp": pd.bdate_range("2020-01-01", periods=400), "close": 100.0})
    f = gf.GarchFamilyModel("GARCH").fit(hist, hist["timestamp"].iloc[-1]).forecast(1)
    assert f.diagnostics["fallback"] == "EWMA" and "no variation" in f.diagnostics["reason"]


def test_non_convergence_and_exceptions_are_reported_not_raised(monkeypatch):
    ret = sim_garch(800, seed=6)
    hist = pd.DataFrame({"timestamp": ret.index, "close": 100 * np.exp(ret.cumsum().values)})
    monkeypatch.setattr(gf, "fit_checked", lambda *a, **k: (None, "optimiser did not converge"))
    f = gf.GarchFamilyModel("GARCH").fit(hist, hist["timestamp"].iloc[-1]).forecast(1)
    assert f.diagnostics["reason"] == "optimiser did not converge"

    monkeypatch.undo()
    model = gf.GarchFamilyModel("GARCH").fit(hist, hist["timestamp"].iloc[-1])
    monkeypatch.setattr(gf, "_variance_path", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    f = model.forecast(1)
    assert f.diagnostics["fallback"] == "EWMA" and "boom" in f.diagnostics["reason"]


def test_unknown_kind_is_rejected_loudly():
    with pytest.raises(ValueError):
        gf.GarchFamilyModel("ARCH9")


def test_rolling_forecast_uses_ewma_when_every_fit_fails(monkeypatch):
    ret = sim_garch(420, seed=7)
    monkeypatch.setattr(gf, "fit_checked", lambda *a, **k: (None, "nope"))
    out = gf.rolling_vol_forecasts(ret, "GARCH", refit_every=10, min_train=300)
    assert out["fallback"].iloc[300:].all() and out["sigma_1d"].iloc[300:].notna().all()
    assert out["sigma_1d"].iloc[:299].isna().all()


# ---------------------------------------------------------------- no look-ahead in the rolling forecasts
def test_rolling_garch_forecast_at_t_does_not_depend_on_later_data():
    ret = sim_garch(420, seed=8)
    cut = 360
    base = gf.rolling_vol_forecasts(ret.iloc[: cut + 1], "GARCH", refit_every=7, min_train=300)
    future_shock = ret.copy()
    future_shock.iloc[cut + 1:] *= 25.0  # wreck everything after the cut
    after = gf.rolling_vol_forecasts(future_shock, "GARCH", refit_every=7, min_train=300).iloc[: cut + 1]
    np.testing.assert_allclose(base["sigma_1d"].values, after["sigma_1d"].values, rtol=1e-9, equal_nan=True)


def test_rolling_har_forecast_does_not_depend_on_later_data():
    d = sim_daily(420, seed=9)
    cut = 330
    base = har_rv.rolling_har_forecasts(d.iloc[: cut + 1], refit_every=5, min_train=250)
    mutated = d.copy()
    mutated.loc[cut + 1:, "rv"] *= 50.0
    after = har_rv.rolling_har_forecasts(mutated, refit_every=5, min_train=250).iloc[: cut + 1]
    np.testing.assert_allclose(base["sigma_1d"].values, after["sigma_1d"].values, rtol=1e-9, equal_nan=True)


def test_har_and_garch_models_ignore_rows_after_asof():
    d = sim_daily(500, seed=10)
    asof = d["timestamp"].iloc[400]
    spiked = d.copy()
    spiked.loc[spiked["timestamp"] > asof, ["rv", "close"]] *= 40.0
    a = har_rv.HarRvModel().fit(d, asof).forecast(2)
    b = har_rv.HarRvModel().fit(spiked, asof).forecast(2)
    assert a.sigma_1d == pytest.approx(b.sigma_1d)
    g1 = gf.GarchFamilyModel("GARCH").fit(d, asof).forecast(1)
    g2 = gf.GarchFamilyModel("GARCH").fit(spiked, asof).forecast(1)
    assert g1.sigma_1d == pytest.approx(g2.sigma_1d)


# ---------------------------------------------------------------- HAR-RV
def sim_har_frame(n=1500, persistence=0.9, seed=4):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = persistence * x[t - 1] + rng.normal(0, 0.35)
    rv = np.exp(x - 9.0)
    return pd.DataFrame({"timestamp": pd.bdate_range("2015-01-01", periods=n), "rv": rv, "gap": 0.0})


def test_har_recovers_the_persistence_of_log_variance():
    d = sim_har_frame()
    model = har_rv.HarRvModel().fit(d, d["timestamp"].iloc[-1])
    f = model.forecast(1)
    coef = f.diagnostics["coef"]
    assert 0.7 < coef["d"] + coef["w"] + coef["m"] < 1.05 and f.diagnostics["r2"] > 0.5
    assert f.sigma_1d > 0 and all(v == v for v in f.diagnostics["pvalues"].values())


def test_har_applies_the_lognormal_bias_correction_and_adds_the_gap_term():
    d = sim_har_frame()
    d["gap"] = 0.01
    model = har_rv.HarRvModel().fit(d, d["timestamp"].iloc[-1])
    coefs, s2 = model._res.params, float(model._res.scale)
    rv = model._rv
    feats = {"d": np.log(rv.iloc[-1]), "w": np.log(rv.iloc[-5:].mean()), "m": np.log(rv.iloc[-22:].mean())}
    mu = coefs["const"] + coefs["d"] * feats["d"] + coefs["w"] * feats["w"] + coefs["m"] * feats["m"]
    f = model.forecast(1)
    assert f.sigma_1d ** 2 == pytest.approx(math.exp(mu + 0.5 * s2) + 0.01 ** 2, rel=1e-9)
    assert f.sigma_1d ** 2 > math.exp(mu) + 0.01 ** 2  # the correction raises the forecast


def test_har_multi_step_forecast_accumulates():
    d = sim_har_frame()
    f = har_rv.HarRvModel().fit(d, d["timestamp"].iloc[-1]).forecast(5)
    assert f.horizon_days == 5 and f.sigma_horizon > f.sigma_1d


def test_har_with_too_little_data_refuses_and_the_engine_falls_back():
    d = sim_har_frame(40)
    model = har_rv.HarRvModel().fit(d, d["timestamp"].iloc[-1])
    with pytest.raises(ValueError, match="not fitted"):
        model.forecast(1)
    out = fe.latest_forecasts(d.assign(close=100.0), ("EWMA", "HAR-RV"))
    assert out["HAR-RV"].diagnostics["fallback"] == "EWMA"


def test_har_falls_back_to_the_range_proxy_without_intraday_rv():
    d = sim_daily(300, seed=2).drop(columns=["rv", "gap", "complete"])
    f = har_rv.HarRvModel().fit(d, d["timestamp"].iloc[-1]).forecast(1)
    assert f.sigma_1d > 0


# ---------------------------------------------------------------- evaluation metrics
def test_qlike_is_minimised_by_the_true_variance_and_punishes_under_forecasting_more():
    rng = np.random.default_rng(0)
    true = pd.Series(np.exp(rng.normal(-9, 0.5, 4000)))
    target = true * rng.chisquare(1, 4000)  # noisy but unbiased realised variance
    exact, low, high = ev.qlike(true, target), ev.qlike(true * 0.5, target), ev.qlike(true * 2.0, target)
    assert exact < low and exact < high
    assert (low - exact) > (high - exact)  # halving costs more than doubling


def test_mse_and_input_cleaning():
    f = pd.Series([1.0, 2.0, np.nan, -1.0, 4.0])
    r = pd.Series([1.0, 4.0, 3.0, 2.0, 4.0])
    assert ev.mse(f, r) == pytest.approx(((1 - 1) ** 2 + (2 - 4) ** 2 + (4 - 4) ** 2) / 3)  # NaN and non-positive dropped
    assert math.isnan(ev.qlike(pd.Series([np.nan]), pd.Series([1.0])))


def test_mincer_zarnowitz_accepts_an_efficient_forecast_and_rejects_a_biased_one():
    rng = np.random.default_rng(1)
    f = pd.Series(np.exp(rng.normal(-9, 0.6, 3000)))
    r = f * rng.gamma(20, 1 / 20, 3000)
    good = ev.mincer_zarnowitz(f, r)
    assert good["beta"] == pytest.approx(1.0, abs=0.15) and abs(good["alpha"]) < 2e-4 and good["p_joint"] > 0.05
    bad = ev.mincer_zarnowitz(f * 0.6, r)  # systematically too low: beta ~ 1.67
    assert bad["beta"] > 1.4 and bad["p_joint"] < 0.01
    assert math.isnan(ev.mincer_zarnowitz(f.iloc[:10], r.iloc[:10])["alpha"])


def test_diebold_mariano_detects_a_better_forecast_and_is_neutral_for_equals():
    rng = np.random.default_rng(2)
    true = pd.Series(np.exp(rng.normal(-9, 0.5, 1500)))
    target = true * rng.gamma(30, 1 / 30, 1500)
    good = ev.qlike_losses(true, target)
    poor = ev.qlike_losses(true * np.exp(rng.normal(0, 0.5, 1500)), target)
    dm = ev.diebold_mariano(good, poor)
    assert dm["stat"] > 2.5 and dm["p_model_better"] < 0.01 and dm["p_two_sided"] < 0.02 and dm["mean_diff"] > 0
    flipped = ev.diebold_mariano(poor, good)
    assert flipped["stat"] < -2.5 and flipped["p_model_better"] > 0.99
    same = ev.diebold_mariano(good, good + rng.normal(0, 1e-12, len(good)))
    assert abs(same["mean_diff"]) < 1e-9
    assert math.isnan(ev.diebold_mariano(good.iloc[:10], poor.iloc[:10])["stat"])


# ---------------------------------------------------------------- model selection
def _selection_inputs(n=600, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    true = pd.Series(np.exp(rng.normal(-9, 0.5, n)), index=idx)
    target = true * rng.gamma(30, 1 / 30, n)
    noisy = true * np.exp(rng.normal(0, 0.4, n))
    return idx, true, target, noisy, rng


def test_a_clearly_better_model_is_selected_over_ewma():
    idx, true, target, noisy, rng = _selection_inputs()
    rows, selected, reason = fe.score_models({"EWMA": noisy, "GARCH": true * np.exp(rng.normal(0, 0.05, len(idx)))}, target, min_eval=250)
    assert selected == "GARCH" and "beats EWMA" in reason
    assert [r["selected"] for r in rows if r["model"] == "GARCH"] == [True]


def test_ewma_stays_the_default_when_nothing_beats_it_significantly():
    idx, true, target, noisy, rng = _selection_inputs()
    worse = true * np.exp(rng.normal(0, 0.6, len(idx)))
    rows, selected, reason = fe.score_models({"EWMA": noisy, "HAR-RV": worse}, target, min_eval=250)
    assert selected == "EWMA" and "no model significantly beats" in reason


def test_too_few_common_days_keeps_ewma():
    idx, true, target, noisy, rng = _selection_inputs(n=120)
    rows, selected, reason = fe.score_models({"EWMA": noisy, "GARCH": true}, target, min_eval=250)
    assert selected == "EWMA" and "only" in reason


def test_scoring_uses_the_dates_all_models_share():
    idx, true, target, noisy, _ = _selection_inputs()
    rows, _, _ = fe.score_models({"EWMA": noisy, "GARCH": true.iloc[200:]}, target, min_eval=100)
    assert {r["n"] for r in rows} == {len(idx) - 200}  # EWMA scored on the same, shorter, window


def test_engine_scores_every_model_and_records_a_reason(monkeypatch):
    d = sim_daily(700, seed=14)
    result = fe.run_instrument("SIM", d, models=("EWMA", "GARCH", "HAR-RV"), refit_every=10, min_train=300, min_eval=100)
    assert {r["model"] for r in result.scores} == {"EWMA", "GARCH", "HAR-RV"}
    assert result.selected in {"EWMA", "GARCH", "HAR-RV"} and result.reason and result.forecast is not None
    assert result.forecast.sigma_1d > 0 and result.asof == d["timestamp"].iloc[-1]
    assert result.diagnostics["target"] == "rv+gap^2" and sum(r["selected"] for r in result.scores) == 1
    assert all(r["n"] > 300 for r in result.scores)


def test_engine_with_little_history_returns_only_ewma_and_never_raises():
    d = sim_daily(400, seed=15)
    r = fe.run_instrument("SIM", d, min_train=500, min_eval=250)
    assert r.selected == "EWMA" and r.scores == [] and "not evaluated" in r.reason and r.forecast.sigma_1d > 0
    tiny = fe.run_instrument("SIM", d.iloc[:10])
    assert tiny.asof is None and "no forecast possible" in tiny.reason
    broken = fe.run_instrument("SIM", pd.DataFrame({"timestamp": ["x"], "close": ["y"]}))
    assert broken.asof is None and ("failed" in broken.reason or "no forecast" in broken.reason)


# ---------------------------------------------------------------- daily history
class FakeDhan:
    def __init__(self, last_day=None, fail_after=None, available=True):
        self.calls, self.last_day, self.fail_after, self.available = [], last_day, fail_after, available

    def is_available(self):
        return self.available

    def get_ohlcv(self, symbol, tf, start, end):
        self.calls.append((start, end))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("Dhan rate limit")
        stop = min(end, self.last_day) if self.last_day else end
        days = pd.bdate_range(start.date(), stop.date())
        close = 100 + np.arange(len(days)) * 0.1
        return pd.DataFrame({"timestamp": days.astype("datetime64[ns]"), "open": close, "high": close + 1, "low": close - 1,
                             "close": close, "volume": 1000.0})


@pytest.fixture
def daily_env(monkeypatch, tmp_path):
    from quant_intelligence.data import daily_history

    monkeypatch.setattr(daily_history, "CACHE_DIR", tmp_path)
    return daily_history, tmp_path


class NoCsv:
    def get_ohlcv(self, *a, **k):
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


NOW = dt.datetime(2026, 10, 7, 20, 0)  # Wednesday evening, NSE closed, 7 Oct session complete


def test_history_is_fetched_in_chunks_cached_and_then_only_the_tail_is_requested(daily_env):
    dh, tmp = daily_env
    fake = FakeDhan(last_day=dt.datetime(2026, 10, 5))
    df, info = dh.load_daily_history("NIFTY", years=5, dhan=fake, csv=NoCsv(), now=NOW)
    assert len(fake.calls) >= 3 and all((e - s).days <= dh.CHUNK_DAYS + 1 for s, e in fake.calls)  # 2-year windows
    assert info["source"] == "dhan+cache" and info["last"] == dt.date(2026, 10, 5) and info["n"] > 1000
    assert (tmp / "NIFTY_1d.parquet").exists()

    again = FakeDhan(last_day=dt.datetime(2026, 10, 7))
    df2, info2 = dh.load_daily_history("NIFTY", years=5, dhan=again, csv=NoCsv(), now=NOW)
    assert len(again.calls) == 1 and (again.calls[0][0].date() >= dt.date(2026, 9, 28))  # only the missing tail
    assert info2["last"] == dt.date(2026, 10, 7) and len(df2) == info["n"] + 2
    assert df2["timestamp"].is_monotonic_increasing and df2["timestamp"].is_unique

    current = FakeDhan()
    dh.load_daily_history("NIFTY", years=5, dhan=current, csv=NoCsv(), now=NOW)
    assert current.calls == []  # already has the last finished session: no request at all


def test_a_failed_chunk_keeps_what_was_cached_and_reports_it(daily_env):
    dh, _ = daily_env
    df, info = dh.load_daily_history("NIFTY", years=5, dhan=FakeDhan(fail_after=1), csv=NoCsv(), now=NOW)
    assert info["fetch_error"] == "Dhan rate limit" and len(df) > 200  # the first chunk made it in


def test_missing_credentials_and_csv_only_modes(daily_env):
    dh, _ = daily_env
    df, info = dh.load_daily_history("NIFTY", years=2, dhan=FakeDhan(available=False), csv=NoCsv(), now=NOW)
    assert df.empty and info["fetch_error"] == "Dhan credentials not set"
    fake = FakeDhan()
    dh.load_daily_history("NIFTY", years=2, dhan=fake, csv=NoCsv(), now=NOW, allow_fetch=False)
    assert fake.calls == []


def test_a_user_csv_is_used_and_cached(daily_env):
    dh, tmp = daily_env

    class OneCsv:
        def get_ohlcv(self, *a, **k):
            days = pd.bdate_range("2024-01-01", "2026-10-07")
            c = 100.0 + np.arange(len(days)) * 0.05
            return pd.DataFrame({"timestamp": days, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0})

    df, info = dh.load_daily_history("RELIANCE", years=3, dhan=FakeDhan(available=False), csv=OneCsv(), now=NOW)
    assert info["source"] == "csv+cache" and info["last"] == dt.date(2026, 10, 7) and info["fetch_error"] is None
    assert (tmp / "RELIANCE_1d.parquet").exists()


def test_a_still_forming_bar_and_bad_rows_are_dropped(daily_env):
    dh, _ = daily_env
    midday = dt.datetime(2026, 10, 7, 12, 0)  # NSE open: the 7 Oct bar is not final
    assert dh.last_complete_session(midday, "NIFTY") == dt.date(2026, 10, 6)

    class Dirty:
        is_available = staticmethod(lambda: True)

        def get_ohlcv(self, *a, **k):
            d = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-06"]),
                              "open": [100, 101, 102, 101.5], "high": [101, 102, 103, 102.5], "low": [99, 100, 101, 100.5],
                              "close": [100.5, 101.5, 102.5, 101.8], "volume": 1.0})
            d.loc[0, "close"] = np.nan
            return d

    df, _ = dh.load_daily_history("NIFTY", years=1, dhan=Dirty(), csv=NoCsv(), now=midday)
    assert list(df["timestamp"].dt.date) == [dt.date(2026, 10, 6)]  # NaN row, forming 7 Oct bar and the duplicate gone
    assert df["close"].iloc[0] == 101.8  # the later duplicate wins


def test_epoch_timestamps_at_1830_are_read_as_ist_dates(daily_env):
    dh, _ = daily_env
    raw = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-05 18:30", "2026-10-06 18:30"])})
    assert list(dh._normalise_dates(raw)["timestamp"].dt.date) == [dt.date(2026, 10, 6), dt.date(2026, 10, 7)]
    already = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-05", "2026-10-06"])})
    assert list(dh._normalise_dates(already)["timestamp"].dt.date) == [dt.date(2026, 10, 5), dt.date(2026, 10, 6)]


def test_last_complete_session_skips_holidays_and_weekends(daily_env):
    dh, _ = daily_env
    assert dh.last_complete_session(dt.datetime(2026, 10, 6, 9, 0), "NIFTY") == dt.date(2026, 10, 5)  # Tue before the open: Monday
    assert dh.last_complete_session(dt.datetime(2026, 10, 5, 9, 0), "NIFTY") == dt.date(2026, 10, 1)  # Mon: skips weekend + Fri 2 Oct holiday
    assert dh.last_complete_session(dt.datetime(2026, 10, 3, 12, 0), "NIFTY") == dt.date(2026, 10, 1)  # Saturday


# ---------------------------------------------------------------- store (DB)
def _result(instrument="STORETEST"):
    d = sim_daily(700, seed=16)
    return fe.run_instrument(instrument, d, models=("EWMA", "HAR-RV"), min_train=300, min_eval=100)


def test_results_round_trip_through_the_database():
    r = _result()
    assert store.save_result(r) is True
    latest = store.latest_forecast("STORETEST")
    assert latest["model"] == r.selected and latest["sigma_1d"] == pytest.approx(r.forecast.sigma_1d)
    assert latest["sigma_horizon"] == pytest.approx(r.forecast.sigma_horizon) and latest["asof"] == r.asof.to_pydatetime()
    scores = store.latest_scores("STORETEST")
    assert {s["model"] for s in scores} == {"EWMA", "HAR-RV"} and sum(s["selected"] for s in scores) == 1
    assert store.latest_forecast("STORETEST", max_age_days=1) is not None
    assert store.latest_forecast("STORETEST", max_age_days=-1) is None  # too old
    assert store.latest_forecast("NOT_THERE") is None and store.latest_scores("NOT_THERE") == []


def test_nan_and_numpy_values_are_stored_as_valid_json():
    assert store._jsonable({"a": float("nan"), "b": np.float64(1.5), "c": [np.int64(3), float("inf")], "d": pd.Timestamp("2026-01-01")}) == {
        "a": None, "b": 1.5, "c": [3, None], "d": "2026-01-01 00:00:00"}


# ---------------------------------------------------------------- the daily job
@pytest.fixture
def job_env(monkeypatch):
    from quant_intelligence.data import daily_history
    from quant_intelligence.volatility import job as job_module

    monkeypatch.setattr(job_module, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, vol_models_enabled=True))
    ran = []

    def fake_load(symbol, **k):
        return pd.DataFrame({"timestamp": [pd.Timestamp("2026-10-05")]}), {"n": 1, "fetch_error": None}

    def fake_run(symbol, daily, **k):
        ran.append(symbol)
        if symbol == "BAD":
            raise RuntimeError("model blew up")
        f = fe.VolForecast("EWMA", pd.Timestamp("2026-10-07"), 0.01, 0.01, 1, {})
        return fe.InstrumentResult(symbol, pd.Timestamp("2026-10-07"), "EWMA", "default", {"EWMA": f}, [], {})

    monkeypatch.setattr(daily_history, "load_daily_history", fake_load)
    monkeypatch.setattr(fe, "run_instrument", fake_run)
    monkeypatch.setattr(store, "save_result", lambda r: True)
    return job_module, ran


def test_the_job_only_runs_after_the_close_and_once_per_session(job_env):
    jm, ran = job_env
    job = jm.VolJob(max_per_round=2)
    open_now = dt.datetime(2026, 10, 7, 11, 0)
    assert job.session_key(open_now, "NIFTY") is None and job.due(["NIFTY"], open_now) == []
    too_soon = dt.datetime(2026, 10, 7, 15, 35)  # 5 min after close
    assert job.session_key(too_soon, "NIFTY") is None
    after = dt.datetime(2026, 10, 7, 15, 45)
    assert job.session_key(after, "NIFTY") == dt.date(2026, 10, 7)

    symbols = ["NIFTY", "BANKNIFTY", "TCS"]
    assert job.run_round(symbols, now=after) == 2 and ran == ["NIFTY", "BANKNIFTY"]  # capped per round
    assert job.run_round(symbols, now=after) == 1 and ran[-1] == "TCS"
    assert job.run_round(symbols, now=after) == 0  # all done for this session
    next_day = dt.datetime(2026, 10, 8, 16, 0)
    assert job.run_round(symbols, now=next_day) == 2  # a new session is due again
    st = job.status(symbols)
    assert st["enabled"] and st["tracked"] == 3


def test_a_failing_instrument_is_isolated_and_retried_later(job_env):
    jm, ran = job_env
    job = jm.VolJob(max_per_round=5, retry_after_sec=0.05)
    after = dt.datetime(2026, 10, 7, 15, 45)
    job.run_round(["BAD", "NIFTY"], now=after)
    assert ran == ["BAD", "NIFTY"] and job.by_symbol["NIFTY"]["ok"] and job.by_symbol["BAD"]["ok"] is False
    assert "blew up" in job.last_error and job.status(["BAD", "NIFTY"])["failed"] == ["BAD"]
    assert job.due(["BAD"], after) == []  # paused for a moment
    time.sleep(0.08)
    assert job.due(["BAD"], after) == ["BAD"]


def test_the_job_is_off_unless_the_flag_is_set(monkeypatch):
    from quant_intelligence.volatility import job as job_module

    job = job_module.VolJob()
    assert job.run_round(["NIFTY"], now=dt.datetime(2026, 10, 7, 15, 45)) == 0  # conftest pins the flag off
    assert job.status(["NIFTY"])["enabled"] is False


def test_mcx_runs_after_its_late_close(job_env):
    jm, _ = job_env
    job = jm.VolJob()
    assert job.session_key(dt.datetime(2026, 10, 7, 23, 0), "CRUDEOIL") is None  # still trading
    assert job.session_key(dt.datetime(2026, 10, 8, 0, 20), "CRUDEOIL") == dt.date(2026, 10, 7)


def test_a_crashing_vol_job_does_not_stop_the_data_keeper(monkeypatch):
    from quant_intelligence.data import data_keeper as keeper_module
    from quant_intelligence.tests.test_data_keeper import FakeManager

    monkeypatch.setattr(keeper_module, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, data_keeper_enabled=True))
    monkeypatch.setattr(keeper_module, "watchlist_symbols", lambda: ["NIFTY"])
    k = keeper_module.DataKeeper(interval_seconds=15)
    k._manager = FakeManager()
    k.interval = 0.05
    calls = []
    k.vol_job.run_round = lambda *a, **kw: calls.append(1) or (_ for _ in ()).throw(RuntimeError("vol exploded"))
    k.start()
    deadline = time.time() + 5
    while len(calls) < 2 and time.time() < deadline:
        k.wake()
        time.sleep(0.03)
    assert len(calls) >= 2 and k.running and k.rounds >= 2
    k.stop()


# ---------------------------------------------------------------- CLI and packaging
def test_cli_prints_the_comparison_table(monkeypatch, capsys):
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_vol_models.py"
    spec = importlib.util.spec_from_file_location("evaluate_vol_models", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_daily_history", lambda sym, **k: (sim_daily(700, seed=17), {"n": 700, "source": "test", "first": None, "last": None, "fetch_error": None}))
    monkeypatch.setattr(sys, "argv", ["evaluate_vol_models", "SIM", "--models", "EWMA", "HAR-RV", "--min-train", "300", "--min-eval", "100", "--csv-only"])
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "QLIKE" in out and "HAR-RV" in out and "selected" in out and "latest" in out


def test_cli_explains_when_there_is_not_enough_history(monkeypatch, capsys):
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_vol_models.py"
    spec = importlib.util.spec_from_file_location("evaluate_vol_models2", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_daily_history", lambda sym, **k: (sim_daily(60, seed=18), {"n": 60, "source": "cache", "first": None, "last": None, "fetch_error": "Dhan credentials not set"}))
    monkeypatch.setattr(sys, "argv", ["evaluate_vol_models", "SIM", "--csv-only"])
    assert mod.main() == 2
    out = capsys.readouterr().out
    assert "not evaluated" in out and "Dhan credentials not set" in out


def test_arch_is_a_declared_dependency():
    req = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text(encoding="utf-8")
    assert "arch" in req and "statsmodels" in req and "scipy" in req


def test_flags_stay_off_by_default():
    s = settings_module.SETTINGS
    assert not (s.vol_models_enabled or s.vol_features_in_analogues or s.vol_premium_model_in_backtest
                or s.vol_target_sizing or s.vol_risk_check or s.vol_iv_gate)
