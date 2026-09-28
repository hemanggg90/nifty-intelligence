"""VWAP Mean Reversion.

Setup: price stretches meaningfully away from session VWAP (beyond a
percentage threshold) while NOT in a strong trend (mean reversion tends to
fail in strong trends), then a reversal confirmation bar occurs. Enters
counter to the stretch, targeting VWAP.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class VWAPMeanReversionStrategy(BaseStrategy):
    name = "VWAP Mean Reversion"
    description = "Fades stretched moves away from session VWAP when trend strength is low, targeting VWAP."
    required_features = ["vwap", "vwap_distance_pct", "trend_slope", "atr_14"]
    default_parameters = {
        "stretch_threshold_pct": 0.35,
        "max_trend_slope": 0.0006,
        "stop_atr_mult": 1.0,
        "target_atr_mult": 1.2,
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
            dist = row["vwap_distance_pct"]
            slope = row["trend_slope"]
            atr = row["atr_14"]
            vwap = row["vwap"]
            if pd.isna(dist) or pd.isna(slope) or pd.isna(atr):
                continue
            if abs(slope) > self.parameters["max_trend_slope"]:
                continue

            entry = row["close"]
            thresh = self.parameters["stretch_threshold_pct"]

            # Overstretched above VWAP + reversal down bar -> SHORT, target = VWAP.
            if dist > thresh and row["close"] < prev["close"]:
                stop = entry + self.parameters["stop_atr_mult"] * atr
                target = vwap
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, target, {"vwap_distance_pct": dist}))
                last_idx = i
            # Overstretched below VWAP + reversal up bar -> LONG, target = VWAP.
            elif dist < -thresh and row["close"] > prev["close"]:
                stop = entry - self.parameters["stop_atr_mult"] * atr
                target = vwap
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, target, {"vwap_distance_pct": dist}))
                last_idx = i

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        if len(recent_ohlcv) < 2:
            return None
        dist = market_state.get("vwap_distance_pct")
        slope = market_state.get("trend_slope")
        atr = market_state.get("atr_14")
        vwap = market_state.get("vwap")
        if dist is None or slope is None or atr is None or vwap is None:
            return None
        if abs(slope) > self.parameters["max_trend_slope"]:
            return None

        entry = recent_ohlcv.iloc[-1]["close"]
        prev_close = recent_ohlcv.iloc[-2]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]
        thresh = self.parameters["stretch_threshold_pct"]

        if dist > thresh and entry < prev_close:
            stop = entry + self.parameters["stop_atr_mult"] * atr
            return Setup(ts, "SHORT", entry, stop, vwap, {"vwap_distance_pct": dist})
        if dist < -thresh and entry > prev_close:
            stop = entry - self.parameters["stop_atr_mult"] * atr
            return Setup(ts, "LONG", entry, stop, vwap, {"vwap_distance_pct": dist})
        return None
