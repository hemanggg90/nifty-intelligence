"""Bollinger Band Mean Reversion.

Setup: close pierces outside the 20-bar, 2-std Bollinger Band and then closes
back inside it (the piercing-and-reclaim pattern), while the market is not in
a strong trend. This is one of the most widely published mean-reversion
setups and tends to produce a high win rate with small, well-defined targets
back at the band midline.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class BollingerBandReversionStrategy(BaseStrategy):
    name = "Bollinger Band Mean Reversion"
    description = (
        "Fades closes that pierce outside the 20-period Bollinger Bands and reclaim "
        "them, targeting the band midline, filtered to range-bound conditions."
    )
    required_features = ["bb_upper_20", "bb_lower_20", "bb_mid_20", "trend_slope", "atr_14"]
    default_parameters = {
        "max_trend_slope": 0.0006,
        "stop_atr_mult": 1.0,
        "target_atr_mult": 1.3,
        "cooldown_bars": 8,
    }

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
            upper, lower, mid = row["bb_upper_20"], row["bb_lower_20"], row["bb_mid_20"]
            slope = row["trend_slope"]
            atr = row["atr_14"]
            if pd.isna(upper) or pd.isna(lower) or pd.isna(slope) or pd.isna(atr):
                continue
            if abs(slope) > self.parameters["max_trend_slope"]:
                continue

            entry = row["close"]
            # Pierced outside the band on the prior bar, reclaimed (closed back inside) now.
            prev_upper = prev["bb_upper_20"]
            prev_lower = prev["bb_lower_20"]
            if pd.isna(prev_upper) or pd.isna(prev_lower):
                continue

            if prev["close"] > prev_upper and entry <= upper:
                stop = entry + self.parameters["stop_atr_mult"] * atr
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, mid, {"bb_upper_20": upper}))
                last_idx = i
            elif prev["close"] < prev_lower and entry >= lower:
                stop = entry - self.parameters["stop_atr_mult"] * atr
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, mid, {"bb_lower_20": lower}))
                last_idx = i

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        if len(recent_ohlcv) < 2:
            return None
        upper = market_state.get("bb_upper_20")
        lower = market_state.get("bb_lower_20")
        mid = market_state.get("bb_mid_20")
        slope = market_state.get("trend_slope")
        atr = market_state.get("atr_14")
        if None in (upper, lower, mid, slope, atr):
            return None
        if abs(slope) > self.parameters["max_trend_slope"]:
            return None

        entry = recent_ohlcv.iloc[-1]["close"]
        prev_close = recent_ohlcv.iloc[-2]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]

        # Bands move slowly bar-to-bar, so the current band is used as the
        # reference for whether the prior close had pierced outside it.
        if prev_close > upper and entry <= upper:
            stop = entry + self.parameters["stop_atr_mult"] * atr
            return Setup(ts, "SHORT", entry, stop, mid, {"bb_upper_20": upper})
        if prev_close < lower and entry >= lower:
            stop = entry - self.parameters["stop_atr_mult"] * atr
            return Setup(ts, "LONG", entry, stop, mid, {"bb_lower_20": lower})
        return None
