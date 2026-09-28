"""Inside Bar Breakout.

Setup: an "inside bar" (high/low fully contained within the prior bar's
range, signalling a coil/consolidation) is followed by a breakout beyond
that inside bar's high or low. This is a classic, widely published
price-action pattern; because the stop sits at the opposite end of a
tight inside bar, risk per trade is small and the pattern is commonly
cited for a favorable win-rate-to-drawdown profile in trending markets
like NSE index futures.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class InsideBarBreakoutStrategy(BaseStrategy):
    name = "Inside Bar Breakout"
    description = (
        "Trades breakouts of an inside bar's high/low following a consolidation "
        "coil, a classic tight-risk price-action setup."
    )
    required_features = ["is_inside_bar", "atr_14"]
    default_parameters = {"target_atr_mult": 2.0, "cooldown_bars": 3}

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        setups: list[Setup] = []
        merged = ohlcv.merge(features, on="timestamp", suffixes=("", "_f"))
        last_idx = -10**9
        cooldown = self.parameters["cooldown_bars"]

        for i in range(1, len(merged)):
            if i - last_idx < cooldown:
                continue
            row = merged.iloc[i]
            prev = merged.iloc[i - 1]
            atr = row["atr_14"]
            if pd.isna(prev["is_inside_bar"]) or pd.isna(atr):
                continue
            if not bool(prev["is_inside_bar"]):
                continue

            entry = row["close"]
            inside_high, inside_low = prev["high"], prev["low"]
            target_mult = self.parameters["target_atr_mult"]

            if entry > inside_high:
                setups.append(
                    Setup(row["timestamp"], "LONG", entry, inside_low, entry + target_mult * atr, {"inside_bar_high": inside_high})
                )
                last_idx = i
            elif entry < inside_low:
                setups.append(
                    Setup(row["timestamp"], "SHORT", entry, inside_high, entry - target_mult * atr, {"inside_bar_low": inside_low})
                )
                last_idx = i

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        if len(recent_ohlcv) < 2:
            return None
        atr = market_state.get("atr_14")
        if atr is None:
            return None

        prev_bar = recent_ohlcv.iloc[-2]
        prev_prev_bar = recent_ohlcv.iloc[-3] if len(recent_ohlcv) >= 3 else None
        if prev_prev_bar is None:
            return None
        is_inside = prev_bar["high"] < prev_prev_bar["high"] and prev_bar["low"] > prev_prev_bar["low"]
        if not is_inside:
            return None

        entry = recent_ohlcv.iloc[-1]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]
        target_mult = self.parameters["target_atr_mult"]
        inside_high, inside_low = prev_bar["high"], prev_bar["low"]

        if entry > inside_high:
            return Setup(ts, "LONG", entry, inside_low, entry + target_mult * atr, {"inside_bar_high": inside_high})
        if entry < inside_low:
            return Setup(ts, "SHORT", entry, inside_high, entry - target_mult * atr, {"inside_bar_low": inside_low})
        return None
