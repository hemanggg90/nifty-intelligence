"""Timeframe strings ("5min", "1d") to bar length in minutes."""
from __future__ import annotations


def parse_timeframe_minutes(timeframe: str) -> int:
    timeframe = timeframe.strip().lower()
    if timeframe.endswith("min"):
        return int(timeframe.replace("min", ""))
    if timeframe.endswith("d"):
        return 375  # full session as one bar
    raise ValueError(f"Unsupported timeframe: {timeframe}")
