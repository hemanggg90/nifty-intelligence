"""EMA 50/200 Crossover (Golden Cross / Death Cross).

Setup: the fast EMA(50) crosses the slow EMA(200). One of the oldest,
most widely published open-source trend-following systems; the long lookback
filters out most whipsaw, which is why it is typically cited for low
drawdown relative to shorter-term trend systems, at the cost of fewer trades.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class EMACrossoverStrategy(BaseStrategy):
    name = "EMA 50/200 Crossover"
    description = (
        "Classic golden-cross/death-cross trend following: enters when EMA(50) "
        "crosses EMA(200), exits are ATR-based since crossovers are infrequent."
    )
    required_features = ["ema_50", "ema_200", "atr_14"]
    default_parameters = {"stop_atr_mult": 2.0, "target_atr_mult": 6.0}

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        setups: list[Setup] = []
        merged = ohlcv.merge(features, on="timestamp", suffixes=("", "_f"))

        for i in range(1, len(merged)):
            row = merged.iloc[i]
            prev = merged.iloc[i - 1]
            fast, slow = row["ema_50"], row["ema_200"]
            prev_fast, prev_slow = prev["ema_50"], prev["ema_200"]
            atr = row["atr_14"]
            if pd.isna(fast) or pd.isna(slow) or pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(atr):
                continue

            entry = row["close"]
            crossed_up = prev_fast <= prev_slow and fast > slow
            crossed_down = prev_fast >= prev_slow and fast < slow
            if crossed_up:
                _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, target, {"ema_50": fast, "ema_200": slow}))
            elif crossed_down:
                _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, target, {"ema_50": fast, "ema_200": slow}))

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        fast = market_state.get("ema_50")
        slow = market_state.get("ema_200")
        prev_fast = market_state.get("prev_ema_50")
        prev_slow = market_state.get("prev_ema_200")
        atr = market_state.get("atr_14")
        if None in (fast, slow, prev_fast, prev_slow, atr):
            return None

        entry = recent_ohlcv.iloc[-1]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]

        if prev_fast <= prev_slow and fast > slow:
            _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
            return Setup(ts, "LONG", entry, stop, target, {"ema_50": fast, "ema_200": slow})
        if prev_fast >= prev_slow and fast < slow:
            _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
            return Setup(ts, "SHORT", entry, stop, target, {"ema_50": fast, "ema_200": slow})
        return None
