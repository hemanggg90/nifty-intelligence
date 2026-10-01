"""Process-wide paper-trading engine.

Owns the shared PaperBroker, the daily risk counters and the kill switch, and runs the
auto-trading scan in a background thread. Because it lives at module level (not in a
Streamlit session or page), switching pages - or closing the browser tab - does not stop it;
only `stop()` (or the server restarting / the cloud app sleeping) does.

A second runner (`COMMODITY_RUNNER`) scans MCX commodities on its own thread and session
hours while borrowing the NSE engine's broker, counters and kill switch, so both draw on one
shared cash pool.

Streamlit-free on purpose: a background thread has no ScriptRunContext, so nothing here
may call `st.*`.
"""
from __future__ import annotations

import datetime as dt
import threading

from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.multi_cycle import run_multi_instrument_cycle
from quant_intelligence.execution.square_off import in_close_window, square_off_positions
from quant_intelligence.risk.risk_engine import AccountState
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_profile import MCX, NSE, MarketProfile
from quant_intelligence.utils.timeutil import is_market_open, now_ist, today_ist


class ScanRunner:
    """A background scan loop over one watchlist in one market session.

    Several runners can run at once (NSE indices+stocks, MCX commodities). They share one
    `owner` TradingEngine - and therefore one broker, one cash pool, one set of daily risk
    counters and one kill switch - but each has its own thread, stop event, session profile
    and status. A runner holds only its own `scan_lock` for the whole scan; the owner's short
    `lock` is taken just around broker mutations (see multi_cycle), so a long NSE scan never
    blocks the commodity runner.
    """

    def __init__(self, name: str, profile: MarketProfile, owner: "TradingEngine | None" = None) -> None:
        self.name = name
        self.profile = profile
        self.owner = owner if owner is not None else self  # TradingEngine owns its own state
        self.scan_lock = threading.RLock()

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

    # ---- scanning ------------------------------------------------------------
    def run_cycle(self, index_symbols, stock_symbols, timeframe: str, lookback_days: int) -> list[dict]:
        """One full scan. Safe to call from the UI thread ('Run one cycle now') or the worker."""
        owner = self.owner
        with self.scan_lock:
            # Fresh client each cycle so a token updated in the sidebar/secrets applies immediately.
            rows, pnl_delta = run_multi_instrument_cycle(
                list(index_symbols), list(stock_symbols), timeframe, lookback_days,
                owner.broker, DhanApiClient(), owner.account_state,
                broker_lock=owner.lock, on_fill=owner.add_trade,
            )
            owner.add_pnl(pnl_delta)
            self.last_rows = rows
            self.last_cycle_at = now_ist()
            self.cycles += 1
            return rows

    def square_off_now(self) -> list[dict]:
        """Close this market's open positions at their latest price (end of session, or on demand).
        Safe to call from the UI thread; prices are fetched outside the broker lock."""
        owner = self.owner
        return square_off_positions(owner.broker, DhanApiClient(), self.profile, owner.lock, owner.add_pnl)

    def start(self, index_symbols, stock_symbols, timeframe: str, lookback_days: int,
              interval_seconds: int | None = None, market_hours_only: bool = True) -> bool:
        from quant_intelligence.data.data_keeper import DATA_KEEPER

        DATA_KEEPER.start()  # candles stay current even between this runner's own scans
        with self.owner.lock:
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
                name=f"auto-paper-trader-{self.name}",
                daemon=True,
            )
            self._thread.start()
        log_event("engine", f"Auto paper trading started ({self.name})", level="INFO")
        return True

    def stop(self) -> None:
        # No lock: must return immediately even while a long scan cycle holds it.
        self._stop.set()
        self.running = False
        self.last_status = "Stopped"
        log_event("engine", f"Auto paper trading stopped ({self.name})", level="INFO")

    def _loop(self, stop, index_symbols, stock_symbols, timeframe, lookback_days, interval) -> None:
        while not stop.is_set():
            try:
                market_open = is_market_open(profile=self.profile)
                closing = SETTINGS.eod_square_off and in_close_window(
                    now_ist(), self.profile, SETTINGS.eod_square_off_minutes
                )
                if self.market_hours_only and SETTINGS.eod_square_off and (closing or not market_open):
                    # Session ending or over: flatten this market's positions (also catches ones left
                    # from before a restart) instead of leaving them open with a frozen price.
                    self.square_off_now()
                if self.market_hours_only and not market_open:
                    self.last_status = f"Market closed - waiting for {self.profile.open:%H:%M} IST"
                elif self.market_hours_only and closing:
                    # No new trades in the last minutes: they would be squared off again immediately.
                    self.last_status = "Closing window - positions squared off, no new trades"
                elif self.owner.kill_switch:
                    self.last_status = "Kill switch engaged - not trading"
                else:
                    self.last_status = "Scanning"
                    self.run_cycle(index_symbols, stock_symbols, timeframe, lookback_days)
                    self.last_status = "Idle until next cycle"
                    self.last_error = None
            except Exception as e:  # a bad cycle must never kill the worker
                self.last_error = f"{type(e).__name__}: {e}"
                self.last_status = "Last cycle failed - will retry"
                log_event("engine", f"Auto cycle failed ({self.name}): {e}", level="ERROR")
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


class TradingEngine(ScanRunner):
    """The NSE runner, which also owns the shared broker, risk counters and kill switch."""

    def __init__(self) -> None:
        super().__init__("nse", NSE)
        self.broker = PaperBroker()
        self.lock = threading.RLock()
        self.kill_switch = False
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.peak_equity = SETTINGS.paper_starting_capital
        self._day = today_ist()
        self._restored = False

    # ---- restart recovery ------------------------------------------------------
    def restore_from_db(self) -> None:
        """Rebuild today's book from the database after a process restart (idempotent).

        The broker lives in memory, so a restart used to empty it while the database kept the rows
        as OPEN - positions nobody monitored, and daily loss / trade-count limits that silently reset.
        Today's positions are reloaded; cash is starting capital + today's realised P&L, and the
        daily counters are restored. OPEN rows from an earlier day can never be priced or closed
        properly, so they are marked STALE (kept, never deleted) rather than left looking live.
        """
        with self.lock:
            if self._restored:
                return
            self._restored = True
            try:
                from quant_intelligence.database.db import get_session, init_db
                from quant_intelligence.database.models import Position

                init_db()  # a page can be the first thing loaded; make sure new columns exist
                day_start = dt.datetime.combine(today_ist(), dt.time.min)
                cols = [c.name for c in Position.__table__.columns if c.name != "id"]
                with get_session() as session:
                    stale = session.query(Position).filter(
                        Position.status == "OPEN",
                        (Position.opened_at < day_start) | (Position.opened_at.is_(None)),
                    ).all()
                    for row in stale:
                        row.status = "STALE"
                        row.exit_reason = "STALE_ON_RESTART"
                    today_rows = [
                        {c: getattr(r, c) for c in cols}
                        for r in session.query(Position).filter(Position.opened_at >= day_start).all()
                    ]
                realised = 0.0
                for d in today_rows:
                    self.broker.positions[d["position_id"]] = d
                    if d["status"] == "CLOSED" and d.get("net_pnl") is not None:
                        realised += d["net_pnl"]
                self.trades_today = len(today_rows)
                self.daily_pnl = realised
                self.broker.cash = self.broker.capital + realised
                self.peak_equity = max(self.peak_equity, self.broker.cash)
                log_event(
                    "engine",
                    f"Restored {sum(1 for d in today_rows if d['status'] == 'OPEN')} open position(s) and "
                    f"{len(today_rows)} trade(s) from today; marked {len(stale)} older OPEN row(s) STALE",
                    level="INFO",
                )
            except Exception as e:  # never block startup on recovery
                log_event("engine", f"Could not restore paper book from the database: {e}", level="ERROR")

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


ENGINE = TradingEngine()
COMMODITY_RUNNER = ScanRunner("commodities", MCX, owner=ENGINE)
