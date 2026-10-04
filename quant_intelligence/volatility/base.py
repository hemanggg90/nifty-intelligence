"""Shared types for volatility forecasting.

Units: `sigma_1d` and `sigma_horizon` are standard deviations of log returns as decimals (0.01 = 1%),
NOT annualised. Multiply by sqrt(trading_days_per_year) to annualise `sigma_1d`.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class VolForecast:
    model: str
    asof: pd.Timestamp  # the forecast uses data up to and including this time, nothing later
    sigma_1d: float  # forecast daily volatility
    sigma_horizon: float  # forecast volatility over `horizon_days` trading days (total, not per day)
    horizon_days: int = 1
    diagnostics: dict = field(default_factory=dict)  # e.g. {"fallback": "EWMA", "reason": "..."}


class VolModel(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def fit(self, history: pd.DataFrame, asof) -> "VolModel":
        """Fit using only rows of `history` with timestamp <= asof."""

    @abc.abstractmethod
    def forecast(self, horizon_days: int = 1) -> VolForecast:
        """Forecast made at the `asof` of the last fit."""
