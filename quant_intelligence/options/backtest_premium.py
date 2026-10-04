"""Model-based option premium for the backtest (behind VOL_PREMIUM_MODEL_IN_BACKTEST).

Replaces the fixed "premium = X% of spot, moves by delta" assumption with Black-Scholes premiums, so theta,
gamma and vega affect trade P&L and the costs scale with the real premium.

For a trade the model prices a BOUGHT ATM option (call for LONG, put for SHORT) at:
  * entry - spot = entry price, time = close of the signal bar, volatility = the causal EWMA estimate as of
    that bar (nothing after the entry bar is used),
  * exit  - spot = the underlying's exit price, time = close of the exit bar, SAME volatility (we do not let
    the forecast move with the outcome, which would leak the future),
  * risk  - the premium lost if the underlying reached the stop at entry time (what the R denominator is).
Expiry comes from `config/expiry_rules.py` (nearest expiry rule, trading-day clock).

Simplifications (documented limits): flat volatility, no skew, strike = spot (ATM, no strike rounding or
moneyness offset), one vol for the whole holding period, overnight holds priced on the trading-time clock.
A trade the model cannot price (no vol yet, missing exit) returns None and the caller keeps the fixed
assumption for that trade.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from quant_intelligence.config.expiry_rules import next_expiry, time_to_expiry_years
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.options import black_scholes as bs
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.volatility.estimators import infer_tf_minutes, intraday_ewma_annualised_vol

MIN_PREMIUM = 0.05  # one tick, as in the backtest engine


class BacktestPremiumModel:
    def __init__(self, ohlcv: pd.DataFrame, instrument: str, tf_minutes: int | None = None) -> None:
        self.instrument = instrument
        self.profile = profile_for(instrument)
        self.tf = tf_minutes or infer_tf_minutes(ohlcv)
        self.vol = intraday_ewma_annualised_vol(ohlcv, self.profile, self.tf)
        self._idx = {ts: i for i, ts in enumerate(ohlcv["timestamp"])}
        self.r = SETTINGS.risk_free_rate
        # Commodity options are on futures: the forward equals spot, so carry q = r.
        self.q = self.r if self.profile.name == "MCX" else SETTINGS.dividend_yield
        self.ty = SETTINGS.trading_days_per_year
        self.priced = 0
        self.skipped = 0

    def _bar_close(self, ts) -> dt.datetime:
        return pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None) + dt.timedelta(minutes=self.tf)

    def price(self, trade) -> dict | None:
        """{"entry","exit","risk","sigma","expiry","kind","T_entry"} per unit, or None if unpriceable."""
        setup = trade.setup
        i = self._idx.get(setup.timestamp)
        if i is None or trade.exit_price is None or trade.exit_timestamp is None:
            self.skipped += 1
            return None
        sigma = float(self.vol.iloc[i])
        if not (sigma == sigma) or sigma <= 0:
            self.skipped += 1
            return None

        kind = bs.CALL if setup.direction == "LONG" else bs.PUT
        t_entry = self._bar_close(setup.timestamp)
        t_exit = max(self._bar_close(trade.exit_timestamp), t_entry)
        expiry = next_expiry(t_entry.date(), self.instrument)
        T_entry = time_to_expiry_years(t_entry, expiry, self.profile, self.ty)
        T_exit = time_to_expiry_years(t_exit, expiry, self.profile, self.ty)
        spot, strike = float(setup.entry_price), float(setup.entry_price)

        entry = max(bs.price(spot, strike, T_entry, self.r, self.q, sigma, kind), MIN_PREMIUM)
        exit_ = max(bs.price(float(trade.exit_price), strike, T_exit, self.r, self.q, sigma, kind), MIN_PREMIUM)
        at_stop = max(bs.price(float(setup.stop_price), strike, T_entry, self.r, self.q, sigma, kind), MIN_PREMIUM)
        risk = entry - at_stop
        if risk <= 0:
            self.skipped += 1
            return None
        self.priced += 1
        return {"entry": entry, "exit": exit_, "risk": risk, "sigma": sigma, "expiry": expiry.isoformat(),
                "kind": kind, "T_entry": T_entry}
