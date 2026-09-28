"""Supertrend Trend Following.

Setup: Supertrend (ATR(10), multiplier 3) flips direction, and the strategy
enters in the new trend direction with the Supertrend line itself as the
trailing stop. Supertrend is one of the most widely used open-source
indicators among Indian retail/algo traders (NSE F&O), popular precisely
because its ATR-anchored trailing stop tends to keep drawdowns contained.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class SupertrendStrategy(BaseStrategy):
    name = "Supertrend Trend Following"
    description = (
        "Enters on a Supertrend(10, 3) direction flip and trails the stop at the "
        "Supertrend line, a widely used low-drawdown trend-following approach on NSE."
    )
    required_features = ["supertrend", "supertrend_direction", "atr_14"]
    default_parameters = {"target_atr_mult": 3.0}

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        setups: list[Setup] = []
        merged = ohlcv.merge(features, on="timestamp", suffixes=("", "_f"))

        for i in range(1, len(merged)):
            row = merged.iloc[i]
            prev = merged.iloc[i - 1]
            direction = row["supertrend_direction"]
            prev_direction = prev["supertrend_direction"]
            st = row["supertrend"]
            atr = row["atr_14"]
            if pd.isna(direction) or pd.isna(prev_direction) or pd.isna(st) or pd.isna(atr):
                continue
            if direction == prev_direction:
                continue

            entry = row["close"]
            target_mult = self.parameters["target_atr_mult"]
            if direction == 1:
                target = entry + target_mult * atr
                setups.append(Setup(row["timestamp"], "LONG", entry, st, target, {"supertrend": st}))
            elif direction == -1:
                target = entry - target_mult * atr
                setups.append(Setup(row["timestamp"], "SHORT", entry, st, target, {"supertrend": st}))

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        direction = market_state.get("supertrend_direction")
        prev_direction = market_state.get("prev_supertrend_direction")
        st = market_state.get("supertrend")
        atr = market_state.get("atr_14")
        if direction is None or st is None or atr is None:
            return None
        if prev_direction is not None and direction == prev_direction:
            return None

        entry = recent_ohlcv.iloc[-1]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]
        target_mult = self.parameters["target_atr_mult"]

        if direction == 1:
            return Setup(ts, "LONG", entry, st, entry + target_mult * atr, {"supertrend": st})
        if direction == -1:
            return Setup(ts, "SHORT", entry, st, entry - target_mult * atr, {"supertrend": st})
        return None
