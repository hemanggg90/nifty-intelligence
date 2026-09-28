"""Central Pivot Range (CPR) Breakout.

Setup: price breaks above the prior session's CPR top-central (TC) or below
its bottom-central (BC). CPR is one of the most widely used public-domain
intraday levels among Indian (NSE) traders, derived purely from the prior
session's high/low/close; a narrow CPR range historically precedes trending
moves, which this strategy trades directionally with an ATR stop.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class CPRBreakoutStrategy(BaseStrategy):
    name = "CPR Breakout"
    description = (
        "Trades breakouts of the prior session's Central Pivot Range (TC/BC), a "
        "widely used intraday level system among Indian NSE traders."
    )
    required_features = ["cpr_tc", "cpr_bc", "atr_14", "time_since_open_min"]
    default_parameters = {
        "stop_atr_mult": 1.0,
        "target_atr_mult": 2.0,
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
            tc, bc, atr = row["cpr_tc"], row["cpr_bc"], row["atr_14"]
            if pd.isna(tc) or pd.isna(bc) or pd.isna(atr):
                continue

            hi = max(tc, bc)
            lo = min(tc, bc)
            entry = row["close"]

            if prev["close"] <= hi < entry:
                _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, target, {"cpr_tc": tc}))
                last_idx = i
            elif prev["close"] >= lo > entry:
                _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, target, {"cpr_bc": bc}))
                last_idx = i

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        if len(recent_ohlcv) < 2:
            return None
        minutes_since_open = market_state.get("time_since_open_min")
        tc = market_state.get("cpr_tc")
        bc = market_state.get("cpr_bc")
        atr = market_state.get("atr_14")
        if minutes_since_open is None or minutes_since_open < self.parameters["min_minutes_since_open"]:
            return None
        if tc is None or bc is None or atr is None:
            return None

        hi = max(tc, bc)
        lo = min(tc, bc)
        prev_close = recent_ohlcv.iloc[-2]["close"]
        entry = recent_ohlcv.iloc[-1]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]

        if prev_close <= hi < entry:
            _, stop, target = self.calculate_entry_stop_target("LONG", entry, atr)
            return Setup(ts, "LONG", entry, stop, target, {"cpr_tc": tc})
        if prev_close >= lo > entry:
            _, stop, target = self.calculate_entry_stop_target("SHORT", entry, atr)
            return Setup(ts, "SHORT", entry, stop, target, {"cpr_bc": bc})
        return None
