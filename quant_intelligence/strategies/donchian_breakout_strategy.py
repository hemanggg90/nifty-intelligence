"""Donchian Channel Breakout (Turtle-style).

Setup: close breaks above the 20-bar high (excluding the current bar) or
below the 20-bar low. This is the entry rule from the publicly documented
"Turtle Trading" system, one of the most well known open-source trend
strategies; the wide channel and ATR-based stop keep it selective, which is
associated with its historically low drawdown relative to shorter breakout
systems.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class DonchianBreakoutStrategy(BaseStrategy):
    name = "Donchian Channel Breakout"
    description = (
        "Turtle-style 20-bar Donchian channel breakout: enters on a new 20-bar "
        "high/low with an ATR-based stop and cooldown to avoid re-triggering."
    )
    required_features = ["donchian_high_20", "donchian_low_20", "atr_14"]
    default_parameters = {"stop_atr_mult": 2.0, "target_atr_mult": 4.0, "cooldown_bars": 15}

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        setups: list[Setup] = []
        merged = ohlcv.merge(features, on="timestamp", suffixes=("", "_f"))
        last_idx = -10**9
        cooldown = self.parameters["cooldown_bars"]

        for i in range(len(merged)):
            if i - last_idx < cooldown:
                continue
            row = merged.iloc[i]
            high_ch, low_ch, atr = row["donchian_high_20"], row["donchian_low_20"], row["atr_14"]
            if pd.isna(high_ch) or pd.isna(low_ch) or pd.isna(atr):
                continue

            entry = row["close"]
            if entry > high_ch:
                _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, target, {"donchian_high_20": high_ch}))
                last_idx = i
            elif entry < low_ch:
                _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, target, {"donchian_low_20": low_ch}))
                last_idx = i

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        high_ch = market_state.get("donchian_high_20")
        low_ch = market_state.get("donchian_low_20")
        atr = market_state.get("atr_14")
        if high_ch is None or low_ch is None or atr is None:
            return None

        entry = recent_ohlcv.iloc[-1]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]

        if entry > high_ch:
            _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
            return Setup(ts, "LONG", entry, stop, target, {"donchian_high_20": high_ch})
        if entry < low_ch:
            _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
            return Setup(ts, "SHORT", entry, stop, target, {"donchian_low_20": low_ch})
        return None
