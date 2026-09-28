"""RSI(2) Mean Reversion (Larry Connors style).

Setup: a short-term extreme reading on the 2-period RSI, taken only in the
direction of the prevailing longer-term trend (price vs. EMA(200)), which is
the classic Connors filter that keeps this widely-published mean-reversion
strategy's win rate high by avoiding counter-trend entries.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class RSI2MeanReversionStrategy(BaseStrategy):
    name = "RSI(2) Mean Reversion"
    description = (
        "Connors-style RSI(2) pullback strategy: buys oversold dips in uptrends "
        "(price above EMA200) and sells overbought rallies in downtrends, targeting "
        "a reversion to the short-term mean."
    )
    required_features = ["rsi_2", "ema_200", "atr_14"]
    default_parameters = {
        "oversold": 10.0,
        "overbought": 90.0,
        "stop_atr_mult": 1.5,
        "target_atr_mult": 1.5,
        "cooldown_bars": 5,
    }

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        setups: list[Setup] = []
        merged = ohlcv.merge(features, on="timestamp", suffixes=("", "_f"))
        last_idx = -10**9
        cooldown = self.parameters["cooldown_bars"]

        for i in range(len(merged)):
            if i - last_idx < cooldown:
                continue
            row = merged.iloc[i]
            rsi2 = row["rsi_2"]
            ema200 = row["ema_200"]
            atr = row["atr_14"]
            if pd.isna(rsi2) or pd.isna(ema200) or pd.isna(atr):
                continue

            entry = row["close"]
            if entry > ema200 and rsi2 <= self.parameters["oversold"]:
                _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, target, {"rsi_2": rsi2}))
                last_idx = i
            elif entry < ema200 and rsi2 >= self.parameters["overbought"]:
                _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, target, {"rsi_2": rsi2}))
                last_idx = i

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        rsi2 = market_state.get("rsi_2")
        ema200 = market_state.get("ema_200")
        atr = market_state.get("atr_14")
        if rsi2 is None or ema200 is None or atr is None:
            return None

        entry = recent_ohlcv.iloc[-1]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]

        if entry > ema200 and rsi2 <= self.parameters["oversold"]:
            _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
            return Setup(ts, "LONG", entry, stop, target, {"rsi_2": rsi2})
        if entry < ema200 and rsi2 >= self.parameters["overbought"]:
            _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
            return Setup(ts, "SHORT", entry, stop, target, {"rsi_2": rsi2})
        return None
