"""Opening Range Breakout (ORB).

Setup: price breaks above the opening-range high (first 30 min of the
session) -> LONG, or below the opening-range low -> SHORT. Only one setup
per session per direction is taken (first breakout only) to avoid overlapping
signals from the same range.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class ORBStrategy(BaseStrategy):
    name = "Opening Range Breakout"
    description = "Trades breakouts beyond the first-30-minute opening range of the session."
    required_features = ["opening_range_high", "opening_range_low", "atr_14", "time_since_open_min"]
    default_parameters = {"stop_atr_mult": 0.75, "target_atr_mult": 1.5, "min_minutes_since_open": 31}

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        setups: list[Setup] = []
        merged = ohlcv.merge(features, on="timestamp", suffixes=("", "_f"))
        session = merged["timestamp"].dt.date
        taken_long: set = set()
        taken_short: set = set()

        for i in range(1, len(merged)):
            row = merged.iloc[i]
            prev = merged.iloc[i - 1]
            sess = session.iloc[i]

            if row["time_since_open_min"] < self.parameters["min_minutes_since_open"]:
                continue
            if pd.isna(row["opening_range_high"]) or pd.isna(row["atr_14"]):
                continue

            or_high = row["opening_range_high"]
            or_low = row["opening_range_low"]
            atr = row["atr_14"]

            if sess not in taken_long and prev["close"] <= or_high < row["close"]:
                entry = row["close"]
                _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, target, {"opening_range_high": or_high}))
                taken_long.add(sess)

            if sess not in taken_short and prev["close"] >= or_low > row["close"]:
                entry = row["close"]
                _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, target, {"opening_range_low": or_low}))
                taken_short.add(sess)

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        if len(recent_ohlcv) < 2:
            return None
        minutes_since_open = market_state.get("time_since_open_min")
        or_high = market_state.get("opening_range_high")
        or_low = market_state.get("opening_range_low")
        atr = market_state.get("atr_14")
        if minutes_since_open is None or minutes_since_open < self.parameters["min_minutes_since_open"]:
            return None
        if or_high is None or atr is None:
            return None

        prev_close = recent_ohlcv.iloc[-2]["close"]
        close = recent_ohlcv.iloc[-1]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]

        if prev_close <= or_high < close:
            _, stop, target = self.calculate_entry_stop_target("LONG", close, atr)
            return Setup(ts, "LONG", close, stop, target, {"opening_range_high": or_high})
        if prev_close >= or_low > close:
            _, stop, target = self.calculate_entry_stop_target("SHORT", close, atr)
            return Setup(ts, "SHORT", close, stop, target, {"opening_range_low": or_low})
        return None
