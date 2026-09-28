"""Camarilla Pivot Reversal.

Setup: price touches the prior session's Camarilla R3/R4 resistance or
S3/S4 support and reverses. Camarilla pivots are a widely published,
open-domain intraday level system popular with Indian option/futures
day traders; R3/S3 are traded as the primary reversal levels (R4/S4 as
the extreme breakout levels), so this strategy fades R3/S3 touches back
toward the pivot.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class CamarillaReversalStrategy(BaseStrategy):
    name = "Camarilla Pivot Reversal"
    description = (
        "Fades touches of the prior session's Camarilla R3/S3 levels back toward "
        "the pivot, a widely used Indian intraday mean-reversion setup."
    )
    required_features = ["camarilla_r3", "camarilla_s3", "cpr_pivot", "atr_14", "time_since_open_min"]
    default_parameters = {
        "stop_atr_mult": 0.75,
        "min_minutes_since_open": 16,
        "cooldown_bars": 10,
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
            if row["time_since_open_min"] < self.parameters["min_minutes_since_open"]:
                continue
            r3, s3, pivot, atr = row["camarilla_r3"], row["camarilla_s3"], row["cpr_pivot"], row["atr_14"]
            if pd.isna(r3) or pd.isna(s3) or pd.isna(pivot) or pd.isna(atr):
                continue

            entry = row["close"]
            stop_mult = self.parameters["stop_atr_mult"]

            # Touched/exceeded R3 then reversed down -> SHORT back to pivot.
            if prev["high"] >= r3 and entry < prev["close"]:
                stop = max(prev["high"], entry + stop_mult * atr)
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, pivot, {"camarilla_r3": r3}))
                last_idx = i
            # Touched/exceeded S3 then reversed up -> LONG back to pivot.
            elif prev["low"] <= s3 and entry > prev["close"]:
                stop = min(prev["low"], entry - stop_mult * atr)
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, pivot, {"camarilla_s3": s3}))
                last_idx = i

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        if len(recent_ohlcv) < 2:
            return None
        minutes_since_open = market_state.get("time_since_open_min")
        r3 = market_state.get("camarilla_r3")
        s3 = market_state.get("camarilla_s3")
        pivot = market_state.get("cpr_pivot")
        atr = market_state.get("atr_14")
        if minutes_since_open is None or minutes_since_open < self.parameters["min_minutes_since_open"]:
            return None
        if r3 is None or s3 is None or pivot is None or atr is None:
            return None

        prev_bar = recent_ohlcv.iloc[-2]
        entry = recent_ohlcv.iloc[-1]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]
        stop_mult = self.parameters["stop_atr_mult"]

        if prev_bar["high"] >= r3 and entry < prev_bar["close"]:
            stop = max(prev_bar["high"], entry + stop_mult * atr)
            return Setup(ts, "SHORT", entry, stop, pivot, {"camarilla_r3": r3})
        if prev_bar["low"] <= s3 and entry > prev_bar["close"]:
            stop = min(prev_bar["low"], entry - stop_mult * atr)
            return Setup(ts, "LONG", entry, stop, pivot, {"camarilla_s3": s3})
        return None
