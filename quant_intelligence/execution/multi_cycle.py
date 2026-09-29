"""One paper-trading scan cycle across many instruments, ALL traded as options.

Indices (NIFTY/BANKNIFTY) and stocks are each run through the research pipeline and,
when a setup triggers, resolved to an option contract from the live chain and paper-traded
via `run_auto_option_cycle`. Shared by 07_Paper_Trading ("All instruments together" mode) and
13_Auto_Multi_Stock_Trading. Streamlit-free: callers pass the account-state getter and
apply the returned P&L to their own session state.
"""
from __future__ import annotations

import datetime as dt
from typing import Callable

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.auto_trader import run_auto_option_cycle
from quant_intelligence.execution.position_monitor import monitor_option_positions
from quant_intelligence.options.option_selector import fetch_chain
from quant_intelligence.research.pipeline import run_pipeline


def run_multi_instrument_cycle(
    index_symbols: list[str],
    stock_symbols: list[str],
    timeframe: str,
    lookback_days: int,
    broker,
    client,
    get_account: Callable,
) -> tuple[list[dict], float]:
    """Returns (result rows, realised P&L from positions closed during the cycle)."""
    account = get_account()
    end = dt.datetime.now()
    start = end - dt.timedelta(days=lookback_days)
    rows: list[dict] = []
    pnl_delta = 0.0

    all_symbols = [(s, "index") for s in index_symbols] + [(s, "stock") for s in stock_symbols]

    for symbol, kind in all_symbols:
        try:
            output = run_pipeline(symbol, timeframe, start, end)
        except Exception as e:
            rows.append({"symbol": symbol, "type": kind, "strategy": "-", "status": "DATA_ERROR", "detail": str(e)})
            continue

        chain = None
        if client.is_configured():
            try:
                chain = fetch_chain(client, symbol, SETTINGS.option_expiry_preference)
            except Exception as e:
                rows.append(
                    {
                        "symbol": symbol,
                        "type": kind,
                        "strategy": output.ranking.selected_strategy or "-",
                        "status": "CHAIN_ERROR",
                        "detail": str(e),
                    }
                )
                continue

        cycle = run_auto_option_cycle(output, symbol, broker, client, chain, account)
        if cycle.order is not None:
            account = get_account()  # re-sync exposure/trade-count after a fill
        rows.append(
            {
                "symbol": symbol,
                "type": kind,
                "strategy": cycle.strategy_name or "-",
                "status": cycle.setup_status or ("NO_TRADE" if output.ranking.is_no_trade else "-"),
                "capital_required": round(cycle.capital_required, 2) if cycle.capital_required is not None else None,
                "capital_used": round(cycle.capital_used, 2),
                "detail": cycle.reason,
            }
        )

    # Check every open option position's live premium against its stop/target once per cycle
    # (not only right after a new fill), so exits are never skipped on quiet cycles.
    if client.is_configured():
        try:
            for e in monitor_option_positions(broker, client):
                pnl_delta += e["position"]["net_pnl"]
        except Exception as e:
            rows.append({"symbol": "-", "type": "monitor", "strategy": "-", "status": "MONITOR_ERROR", "detail": str(e)})
    return rows, pnl_delta
