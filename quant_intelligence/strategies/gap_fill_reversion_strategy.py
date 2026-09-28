"""Gap Fill Reversion.

Setup: the session opens with a gap beyond a minimum threshold versus the
prior close, and price shows an early reversal back toward that prior close
within the first part of the session. Index gap-fill is a widely documented,
publicly known behavioral pattern on NSE (Nifty/Bank Nifty gaps fill a large
majority of the time on moderate-sized gaps), targeting the prior session's
close as the fill level.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class GapFillReversionStrategy(BaseStrategy):
    name = "Gap Fill Reversion"
    description = (
        "Fades a moderate opening gap back toward the prior session's close, a "
        "well-documented behavioral pattern on NSE index products."
    )
    required_features = ["gap_pct", "atr_14", "time_since_open_min"]
    default_parameters = {
        "min_gap_pct": 0.3,
        "max_gap_pct": 1.5,
        "max_minutes_since_open": 20,
        "stop_atr_mult": 1.0,
        "cooldown_bars": 100,
    }

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        setups: list[Setup] = []
        merged = ohlcv.merge(features, on="timestamp", suffixes=("", "_f"))
        session = merged["timestamp"].dt.date
        taken_sessions: set = set()

        for i in range(1, len(merged)):
            row = merged.iloc[i]
            prev = merged.iloc[i - 1]
            sess = session.iloc[i]
            if sess in taken_sessions:
                continue

            minutes = row["time_since_open_min"]
            gap = row["gap_pct"]
            atr = row["atr_14"]
            if minutes is None or minutes > self.parameters["max_minutes_since_open"]:
                continue
            if pd.isna(gap) or pd.isna(atr):
                continue
            if abs(gap) < self.parameters["min_gap_pct"] or abs(gap) > self.parameters["max_gap_pct"]:
                continue

            # Prior session's close, reconstructed from the gap: session_open - gap.
            session_open = merged[session == sess].iloc[0]["open"]
            prior_close = session_open / (1 + gap / 100.0)
            entry = row["close"]

            # Gapped up and reversing down -> SHORT toward prior close.
            if gap > 0 and entry < prev["close"]:
                stop = entry + self.parameters["stop_atr_mult"] * atr
                setups.append(Setup(row["timestamp"], "SHORT", entry, stop, prior_close, {"gap_pct": gap}))
                taken_sessions.add(sess)
            # Gapped down and reversing up -> LONG toward prior close.
            elif gap < 0 and entry > prev["close"]:
                stop = entry - self.parameters["stop_atr_mult"] * atr
                setups.append(Setup(row["timestamp"], "LONG", entry, stop, prior_close, {"gap_pct": gap}))
                taken_sessions.add(sess)

        return setups

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        if len(recent_ohlcv) < 2:
            return None
        minutes = market_state.get("time_since_open_min")
        gap = market_state.get("gap_pct")
        atr = market_state.get("atr_14")
        if minutes is None or minutes > self.parameters["max_minutes_since_open"]:
            return None
        if gap is None or atr is None:
            return None
        if abs(gap) < self.parameters["min_gap_pct"] or abs(gap) > self.parameters["max_gap_pct"]:
            return None

        session_open = recent_ohlcv.iloc[0]["open"]
        prior_close = session_open / (1 + gap / 100.0)
        entry = recent_ohlcv.iloc[-1]["close"]
        prev_close = recent_ohlcv.iloc[-2]["close"]
        ts = recent_ohlcv.iloc[-1]["timestamp"]

        if gap > 0 and entry < prev_close:
            stop = entry + self.parameters["stop_atr_mult"] * atr
            return Setup(ts, "SHORT", entry, stop, prior_close, {"gap_pct": gap})
        if gap < 0 and entry > prev_close:
            stop = entry - self.parameters["stop_atr_mult"] * atr
            return Setup(ts, "LONG", entry, stop, prior_close, {"gap_pct": gap})
        return None
