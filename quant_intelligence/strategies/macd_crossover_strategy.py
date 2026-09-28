"""MACD Signal-Line Crossover.

Setup: the MACD line (EMA12 - EMA26) crosses its 9-bar signal line, taken
only in the direction of the broader trend (EMA200), a standard public-domain
momentum-confirmation filter that reduces whipsaw crossovers against the
prevailing trend and helps keep drawdown lower than an unfiltered crossover.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class MACDCrossoverStrategy(BaseStrategy):
    name = "MACD Signal Crossover"
    description = (
        "Enters on a MACD/signal-line crossover filtered by the EMA(200) trend "
        "direction, a standard momentum-confirmation setup."
    )
    required_features = ["macd_line", "macd_signal", "ema_200", "atr_14"]
    default_parameters = {"stop_atr_mult": 1.5, "target_atr_mult": 2.5, "cooldown_bars": 10}

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
            macd, sig = row["macd_line"], row["macd_signal"]
            prev_macd, prev_sig = prev["macd_line"], prev["macd_signal"]
            ema200 = row["ema_200"]
            atr = row["atr_14"]
            if pd.isna(macd) or pd.isna(sig) or pd.isna(prev_macd) or pd.isna(prev_sig) or pd.isna(ema200) or pd.isna(atr):
                continue

            entry = row["close"]
            crossed_up = prev_macd <= prev_sig and macd > sig
            crossed_down = prev_macd >= prev_sig and macd < sig

            if crossed_up and entry > ema200:
                _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, target, {"macd_line": macd}))
                last_idx = i
            elif crossed_down and entry < ema200:
                _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, target, {"macd_line": macd}))
                last_idx = i

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        macd = market_state.get("macd_line")
        sig = market_state.get("macd_signal")
        prev_macd = market_state.get("prev_macd_line")
        prev_sig = market_state.get("prev_macd_signal")
        ema200 = market_state.get("ema_200")
        atr = market_state.get("atr_14")
        if None in (macd, sig, prev_macd, prev_sig, ema200, atr):
            return None

        entry = recent_ohlcv.iloc[-1]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]

        if prev_macd <= prev_sig and macd > sig and entry > ema200:
            _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
            return Setup(ts, "LONG", entry, stop, target, {"macd_line": macd})
        if prev_macd >= prev_sig and macd < sig and entry < ema200:
            _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
            return Setup(ts, "SHORT", entry, stop, target, {"macd_line": macd})
        return None
