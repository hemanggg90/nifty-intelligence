"""
Central configuration for the Quant Intelligence System.

Everything reads from environment variables (loaded from .env via python-dotenv).
Sensible, conservative defaults are provided so the app runs safely out of the box
in PAPER mode with SQLite, even if .env is missing entirely.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "y")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RiskLimits:
    max_risk_per_trade_pct: float = _float_env("MAX_RISK_PER_TRADE_PCT", 1.0)
    # Ceiling on the capital (premium x quantity) committed to ONE trade, as % of equity. Risk-based
    # sizing alone lets a tight stop produce a huge quantity; 0 disables the cap.
    max_capital_per_trade_pct: float = _float_env("MAX_CAPITAL_PER_TRADE_PCT", 5.0)
    max_daily_loss_pct: float = _float_env("MAX_DAILY_LOSS_PCT", 3.0)
    max_strategy_exposure_pct: float = _float_env("MAX_STRATEGY_EXPOSURE_PCT", 10.0)
    max_portfolio_exposure_pct: float = _float_env("MAX_PORTFOLIO_EXPOSURE_PCT", 20.0)
    max_trades_per_day: int = _int_env("MAX_TRADES_PER_DAY", 6)
    max_drawdown_pct: float = _float_env("MAX_DRAWDOWN_PCT", 8.0)


@dataclass(frozen=True)
class CostAssumptions:
    brokerage_per_order_inr: float = _float_env("BROKERAGE_PER_ORDER_INR", 20.0)
    stt_rate: float = _float_env("STT_RATE", 0.0005)
    slippage_ticks: int = _int_env("SLIPPAGE_TICKS", 1)
    tick_size: float = _float_env("TICK_SIZE", 0.05)


@dataclass(frozen=True)
class Settings:
    trading_mode: str = os.getenv("TRADING_MODE", "PAPER").strip().upper()
    trading_live_confirm: str = os.getenv("TRADING_LIVE_CONFIRM", "").strip()

    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{(DATA_CACHE_DIR / 'quant_intelligence.db').as_posix()}"
    )

    dhan_client_id: str = os.getenv("DHAN_CLIENT_ID", "")
    dhan_access_token: str = os.getenv("DHAN_ACCESS_TOKEN", "")
    dhan_base_url: str = os.getenv("DHAN_BASE_URL", "https://api.dhan.co/v2")

    # Options trading (see quant_intelligence/options/)
    option_underlyings: tuple[str, ...] = tuple(
        s.strip().upper() for s in os.getenv("OPTION_UNDERLYINGS", "NIFTY,BANKNIFTY").split(",") if s.strip()
    )
    option_moneyness_offset: int = _int_env("OPTION_MONEYNESS_OFFSET", 0)  # 0=ATM, +N = N strikes OTM
    option_expiry_preference: str = os.getenv("OPTION_EXPIRY_PREFERENCE", "NEAREST").strip().upper()
    option_product_type: str = os.getenv("OPTION_PRODUCT_TYPE", "INTRADAY").strip().upper()

    default_instrument: str = os.getenv("DEFAULT_INSTRUMENT", "NIFTY")
    default_timeframe: str = os.getenv("DEFAULT_TIMEFRAME", "5min")

    paper_starting_capital: float = _float_env("PAPER_STARTING_CAPITAL", 1_000_000.0)

    # Automation
    auto_trade_refresh_seconds: int = _int_env("AUTO_TRADE_REFRESH_SECONDS", 60)
    ltp_refresh_seconds: int = _int_env("LTP_REFRESH_SECONDS", 5)

    # NSE heatmap-sourced stock universe (see data_adapters/nse_heatmap.py)
    heatmap_index: str = os.getenv("HEATMAP_INDEX", "NIFTY 50")
    heatmap_refresh_minutes: int = _int_env("HEATMAP_REFRESH_MINUTES", 15)
    heatmap_universe_top_n: int = _int_env("HEATMAP_UNIVERSE_TOP_N", 200)

    risk: RiskLimits = field(default_factory=RiskLimits)
    costs: CostAssumptions = field(default_factory=CostAssumptions)

    @property
    def is_live_mode(self) -> bool:
        return self.trading_mode == "LIVE"

    @property
    def live_mode_fully_authorized(self) -> bool:
        """LIVE mode requires BOTH the mode flag AND an explicit confirmation phrase.

        This is a deliberate double-gate: a single environment variable
        (which could be flipped accidentally) is never enough to enable
        real order flow.
        """
        return (
            self.is_live_mode
            and self.trading_live_confirm == "YES_I_UNDERSTAND_THE_RISK"
        )


SETTINGS = Settings()
