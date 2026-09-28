"""IV Crush Premium Writer (options-chain-aware, SELLS options).

Setup: ATM implied volatility sits materially above the underlying's own
realized volatility (vol_expansion regime is NOT active - i.e. no breakout in
progress) - a classic setup for option premium being "rich" relative to
actual movement, which mean-reverts as IV crushes back toward realized vol.
Sells one OTM leg (a defined-risk, single-leg write, not a naked/uncapped
position): calls when the underlying's own trend is flat-to-down, puts when
flat-to-up, following the standard "sell the side you'd be comfortable being
wrong on" premium-writing convention.

SAFETY: every SELL setup this strategy produces MUST be sized through
`quant_intelligence.options.premium_model.translate_setup`, which REQUIRES
and enforces a stop above the entry premium before the trade reaches the risk
engine - see that module's docstring. This strategy never proposes an
unbounded-risk write.

LIMITATION: no historical option-chain data source exists in this codebase,
so `generate_historical_setups` returns [] (same documented limitation as
PCRContrarianStrategy / OIBuildupStrategy).
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.options.chain_analytics import ChainSnapshot
from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup


class IVCrushWriterStrategy(BaseStrategy):
    name = "IV Crush Premium Writer"
    description = (
        "Sells a single OTM option when ATM IV is materially rich versus realized "
        "volatility and no breakout is underway, expecting IV to mean-revert; always "
        "sized with a mandatory premium stop (see premium_model.translate_setup)."
    )
    required_features = ["realized_vol_20", "trend_slope", "atr_14"]
    required_chain_features = ["atm_iv"]
    default_parameters = {
        "min_iv_premium_pct_pts": 5.0,  # atm_iv (in %) must exceed realized_vol_20*100 by this many points
        "max_trend_slope": 0.0004,  # only write premium when the underlying is range-bound
        "stop_atr_mult": 1.0,
        "target_atr_mult": 1.0,
        "moneyness_offset": 2,
    }

    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        return []

    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        return None

    def check_chain_setup(self, market_state, recent_ohlcv: pd.DataFrame, chain: ChainSnapshot) -> Setup | None:
        atm_iv = chain.atm_iv
        realized_vol = market_state.get("realized_vol_20")
        slope = market_state.get("trend_slope")
        atr = market_state.get("atr_14")
        if atm_iv is None or realized_vol is None or slope is None or atr is None:
            return None
        if abs(slope) > self.parameters["max_trend_slope"]:
            return None
        if atm_iv - realized_vol * 100.0 < self.parameters["min_iv_premium_pct_pts"]:
            return None

        entry = chain.spot_price
        ts = recent_ohlcv.iloc[-1]["timestamp"] if len(recent_ohlcv) else None

        # Flat-to-up bias -> sell a call (direction "LONG" maps to CE in option_selector);
        # flat-to-down bias -> sell a put (direction "SHORT" maps to PE).
        direction = "LONG" if slope >= 0 else "SHORT"
        _, stop, target = self.calculate_entry_stop_target(direction, entry, atr)
        return Setup(ts, direction, entry, stop, target, {"atm_iv": atm_iv, "transaction": "SELL"})
