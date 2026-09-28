"""
Setup Detector.

Deliberately separate from the ranking engine: selecting a strategy does NOT
mean entering a trade. A selected strategy sits in WAITING_FOR_SETUP until
its own entry conditions (implemented per-strategy in check_setup) actually
occur on the live/paper bar stream.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_intelligence.strategies.base_strategy import Setup

WAITING_FOR_SETUP = "WAITING_FOR_SETUP"
SETUP_TRIGGERED = "SETUP_TRIGGERED"
NO_STRATEGY_SELECTED = "NO_STRATEGY_SELECTED"


@dataclass
class SetupStatus:
    status: str
    setup: Setup | None = None
    strategy_name: str | None = None


def detect_setup(selected_strategy, market_state, recent_ohlcv: pd.DataFrame) -> SetupStatus:
    if selected_strategy is None:
        return SetupStatus(status=NO_STRATEGY_SELECTED)

    setup = selected_strategy.check_setup(market_state, recent_ohlcv)
    if setup is None:
        return SetupStatus(status=WAITING_FOR_SETUP, strategy_name=selected_strategy.name)

    return SetupStatus(status=SETUP_TRIGGERED, setup=setup, strategy_name=selected_strategy.name)


def detect_chain_setup(selected_strategy, market_state, recent_ohlcv: pd.DataFrame, chain) -> SetupStatus:
    """Same as detect_setup, but for options-chain-aware strategies (see
    BaseStrategy.check_chain_setup / quant_intelligence/options/)."""
    if selected_strategy is None:
        return SetupStatus(status=NO_STRATEGY_SELECTED)

    setup = selected_strategy.check_chain_setup(market_state, recent_ohlcv, chain)
    if setup is None:
        return SetupStatus(status=WAITING_FOR_SETUP, strategy_name=selected_strategy.name)

    return SetupStatus(status=SETUP_TRIGGERED, setup=setup, strategy_name=selected_strategy.name)
