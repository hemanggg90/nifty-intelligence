"""Process-wide paper-trading engine.

Owns the shared PaperBroker, the daily risk counters and the kill switch, and runs the
auto-trading scan in a background thread. Because it lives at module level (not in a
Streamlit session or page), switching pages - or closing the browser tab - does not stop it;
only `stop()` (or the server restarting / the cloud app sleeping) does.

Streamlit-free on purpose: a background thread has no ScriptRunContext, so nothing here
may call `st.*`.
"""
from __future__ import annotations

import threading

from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.multi_cycle import run_multi_instrument_cycle
from quant_intelligence.risk.risk_engine import AccountState
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.timeutil import is_market_open, now_ist, today_ist


class TradingEngine:
    def __init__(self) -> None:
        self.broker = PaperBroker()
        self.lock = threading.RLock()
        self.kill_switch = False
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.peak_equity = SETTINGS.paper_starting_capital
        self._day = today_ist()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.running = False
        self.started_at = None
        self.cycles = 0
        self.last_cycle_at = None
        self.last_rows: list[dict] = []
        self.last_status = "Not started"
        self.last_error: str | None = None
        self.market_hours_only = True

    # ---- shared risk counters ------------------------------------------------
    def _roll_day(self) -> None:
        today = today_ist()
        if today != self._day:
            self._day = today
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.peak_equity = max(self.broker.cash, SETTINGS.paper_starting_capital)

    def add_trade(self, n: int = 1) -> None:
        with self.lock:
            self._roll_day()
            self.trades_today += n

    def add_pnl(self, amount: float) -> None:
        with self.lock:
            self._roll_day()
            self.daily_pnl += amount

    def account_state(self) -> AccountState:
        with self.lock:
            self._roll_day()
            open_positions = self.broker.get_open_positions()
            exposure_by_strategy: dict[str, float] = {}
            total_exposure = 0.0
            for p in open_positions:
                risk_amt = abs(p["entry_price"] - (p["stop_price"] or p["entry_price"])) * p["quantity"]
                exposure_by_strategy[p["strategy_name"]] = exposure_by_strategy.get(p["strategy_name"], 0.0) + risk_amt
                total_exposure += risk_amt

            equity = self.broker.cash
            self.peak_equity = max(self.peak_equity, equity)
            return AccountState(
                equity=equity,
                peak_equity=self.peak_equity,
                daily_pnl=self.daily_pnl,
                open_positions_count=len(open_positions),
                trades_today=self.trades_today,
                exposure_by_strategy=exposure_by_strategy,
                total_exposure=total_exposure,
                broker_connected=self.broker.is_connected(),
                kill_switch_engaged=self.kill_switch,
            )

    # ---- scanning ------------------------------------------------------------
    def run_cycle(self, index_symbols, stock_symbols, timeframe: str, lookback_days: int) -> list[dict]:
        """One full scan. Safe to call from the UI thread ('Run one cycle now') or the worker."""
        with self.lock:
            before = len(self.broker.positions)
            # Fresh client each cycle so a token updated in the sidebar/secrets applies immediately.
            rows, pnl_delta = run_multi_instrument_cycle(
                list(index_symbols), list(stock_symbols), timeframe, lookback_days,
                self.broker, DhanApiClient(), self.account_state,
            )
            self.add_trade(len(self.broker.positions) - before)
            self.add_pnl(pnl_delta)
            self.last_rows = rows
            self.last_cycle_at = now_ist()
            self.cycles += 1
            return rows

    def start(self, index_symbols, stock_symbols, timeframe: str, lookback_days: int,
              interval_seconds: int | None = None, market_hours_only: bool = True) -> bool:
        with self.lock:
            if self.running:
                return False
            self.market_hours_only = market_hours_only
            self._stop = threading.Event()  # fresh event per run: an old worker can never be revived
            self.running = True
            self.started_at = now_ist()
            self.last_error = None
            self.last_status = "Starting"
            interval = interval_seconds or SETTINGS.auto_trade_refresh_seconds
            self._thread = threading.Thread(
                target=self._loop,
                args=(self._stop, list(index_symbols), list(stock_symbols), timeframe, lookback_days, interval),
                name="auto-paper-trader",
                daemon=True,
            )
            self._thread.start()
        log_event("engine", "Auto paper trading started", level="INFO")
        return True

    def stop(self) -> None:
        # No lock: must return immediately even while a long scan cycle holds it.
        self._stop.set()
        self.running = False
        self.last_status = "Stopped"
        log_event("engine", "Auto paper trading stopped", level="INFO")

    def _loop(self, stop, index_symbols, stock_symbols, timeframe, lookback_days, interval) -> None:
        while not stop.is_set():
            try:
                if self.market_hours_only and not is_market_open():
                    self.last_status = "Market closed - waiting for 09:15 IST"
                elif self.kill_switch:
                    self.last_status = "Kill switch engaged - not trading"
                else:
                    self.last_status = "Scanning"
                    self.run_cycle(index_symbols, stock_symbols, timeframe, lookback_days)
                    self.last_status = "Idle until next cycle"
                    self.last_error = None
            except Exception as e:  # a bad cycle must never kill the worker
                self.last_error = f"{type(e).__name__}: {e}"
                self.last_status = "Last cycle failed - will retry"
                log_event("engine", f"Auto cycle failed: {e}", level="ERROR")
            stop.wait(interval)

    def status(self) -> dict:
        alive = bool(self._thread and self._thread.is_alive())
        return {
            "running": self.running and alive,
            "started_at": self.started_at,
            "cycles": self.cycles,
            "last_cycle_at": self.last_cycle_at,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }


ENGINE = TradingEngine()
