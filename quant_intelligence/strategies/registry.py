"""Strategy library registry.

Adding a new strategy = write a BaseStrategy subclass + add it here. Nothing
in the ranking engine, backtester, or UI needs to change.
"""
from __future__ import annotations

from quant_intelligence.strategies.bollinger_band_reversion_strategy import BollingerBandReversionStrategy
from quant_intelligence.strategies.camarilla_reversal_strategy import CamarillaReversalStrategy
from quant_intelligence.strategies.cpr_breakout_strategy import CPRBreakoutStrategy
from quant_intelligence.strategies.donchian_breakout_strategy import DonchianBreakoutStrategy
from quant_intelligence.strategies.ema_crossover_strategy import EMACrossoverStrategy
from quant_intelligence.strategies.gap_fill_reversion_strategy import GapFillReversionStrategy
from quant_intelligence.strategies.inside_bar_breakout_strategy import InsideBarBreakoutStrategy
from quant_intelligence.strategies.iv_crush_writer_strategy import IVCrushWriterStrategy
from quant_intelligence.strategies.macd_crossover_strategy import MACDCrossoverStrategy
from quant_intelligence.strategies.momentum_strategy import MomentumStrategy
from quant_intelligence.strategies.oi_buildup_strategy import OIBuildupStrategy
from quant_intelligence.strategies.orb_strategy import ORBStrategy
from quant_intelligence.strategies.pcr_contrarian_strategy import PCRContrarianStrategy
from quant_intelligence.strategies.rsi2_mean_reversion_strategy import RSI2MeanReversionStrategy
from quant_intelligence.strategies.supertrend_strategy import SupertrendStrategy
from quant_intelligence.strategies.unusual_volume_fade_strategy import UnusualVolumeFadeStrategy
from quant_intelligence.strategies.vwap_mean_reversion_strategy import VWAPMeanReversionStrategy

STRATEGY_CLASSES = {
    ORBStrategy.name: ORBStrategy,
    MomentumStrategy.name: MomentumStrategy,
    VWAPMeanReversionStrategy.name: VWAPMeanReversionStrategy,
    RSI2MeanReversionStrategy.name: RSI2MeanReversionStrategy,
    BollingerBandReversionStrategy.name: BollingerBandReversionStrategy,
    SupertrendStrategy.name: SupertrendStrategy,
    EMACrossoverStrategy.name: EMACrossoverStrategy,
    DonchianBreakoutStrategy.name: DonchianBreakoutStrategy,
    MACDCrossoverStrategy.name: MACDCrossoverStrategy,
    CPRBreakoutStrategy.name: CPRBreakoutStrategy,
    CamarillaReversalStrategy.name: CamarillaReversalStrategy,
    InsideBarBreakoutStrategy.name: InsideBarBreakoutStrategy,
    GapFillReversionStrategy.name: GapFillReversionStrategy,
    PCRContrarianStrategy.name: PCRContrarianStrategy,
    OIBuildupStrategy.name: OIBuildupStrategy,
    IVCrushWriterStrategy.name: IVCrushWriterStrategy,
    UnusualVolumeFadeStrategy.name: UnusualVolumeFadeStrategy,
}

CHAIN_AWARE_STRATEGY_NAMES = {
    PCRContrarianStrategy.name,
    OIBuildupStrategy.name,
    IVCrushWriterStrategy.name,
    UnusualVolumeFadeStrategy.name,
}


def get_all_strategies(parameters_by_name: dict | None = None) -> list:
    parameters_by_name = parameters_by_name or {}
    return [cls(parameters_by_name.get(name)) for name, cls in STRATEGY_CLASSES.items()]


def get_strategy(name: str, parameters: dict | None = None):
    cls = STRATEGY_CLASSES.get(name)
    if cls is None:
        raise KeyError(f"Unknown strategy: {name}")
    return cls(parameters)
