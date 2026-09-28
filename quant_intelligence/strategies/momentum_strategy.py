"""Momentum strategy.

Setup: strong recent momentum (return_20 beyond a threshold) combined with
trend agreement (trend_slope same sign) and above-average volume confirming
participation. Enters in the direction of momentum.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class MomentumStrategy(BaseStrategy):
    name = "Momentum"
    description = "Enters in the direction of strong recent momentum confirmed by trend and volume."
    required_features = ["momentum_20", "trend_slope", "relative_volume", "atr_14"]
    default_parameters = {
        "momentum_threshold": 0.003,
        "min_relative_volume": 1.1,
        "stop_atr_mult": 1.0,
        "target_atr_mult": 2.0,
        "cooldown_bars": 10,
    }

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        setups: list[Setup] = []
        merged = ohlcv.merge(features, on="timestamp", suffixes=("", "_f"))
        last_setup_idx = -10**9
        cooldown = self.parameters["cooldown_bars"]

        for i in range(len(merged)):
            if i - last_setup_idx < cooldown:
                continue
            row = merged.iloc[i]
            mom = row["momentum_20"]
            slope = row["trend_slope"]
            rel_vol = row["relative_volume"]
            atr = row["atr_14"]
            if pd.isna(mom) or pd.isna(slope) or pd.isna(rel_vol) or pd.isna(atr):
                continue
            if rel_vol < self.parameters["min_relative_volume"]:
                continue

            entry = row["close"]
            if mom > self.parameters["momentum_threshold"] and slope > 0:
                _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, target, {"momentum_20": mom}))
                last_setup_idx = i
            elif mom < -self.parameters["momentum_threshold"] and slope < 0:
                _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, target, {"momentum_20": mom}))
                last_setup_idx = i

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        mom = market_state.get("momentum_20")
        slope = market_state.get("trend_slope")
        rel_vol = market_state.get("relative_volume")
        atr = market_state.get("atr_14")
        if mom is None or slope is None or rel_vol is None or atr is None:
            return None
        if rel_vol < self.parameters["min_relative_volume"]:
            return None

        entry = recent_ohlcv.iloc[-1]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]

        if mom > self.parameters["momentum_threshold"] and slope > 0:
            _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
            return Setup(ts, "LONG", entry, stop, target, {"momentum_20": mom})
        if mom < -self.parameters["momentum_threshold"] and slope < 0:
            _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
            return Setup(ts, "SHORT", entry, stop, target, {"momentum_20": mom})
        return None
