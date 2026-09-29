"""One paper-trading scan cycle across many instruments (indices via options + cash equities).

Shared by 07_Paper_Trading (\"All instruments together\" mode) and
13_Auto_Multi_Stock_Trading. Streamlit-free: callers pass the account-state getter
and apply the returned P&L to their own session state.
"""
from __future__ import annotations

import datetime as dt
from typing import Callable

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.auto_trader import run_auto_equity_cycle, run_auto_option_cycle
from quant_intelligence.execution.position_monitor import monitor_option_positions, monitor_positions
from quant_intelligence.options.option_selector import fetch_chain
from quant_intelligence.research.pipeline import run_pipeline


def run_multi_instrument_cycle(
    index_symbols: list[str],
    equity_symbols: list[str],
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

    all_symbols = [(s, "index") for s in index_symbols] + [(s, "equity") for s in equity_symbols]

    for symbol, kind in all_symbols:
        try:
            output = run_pipeline(symbol, timeframe, start, end)
        except Exception as e:
            rows.append({"symbol": symbol, "type": kind, "strategy": "-", "status": "DATA_ERROR", "detail": str(e)})
            continue

        if kind == "index":
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
        else:
            cycle = run_auto_equity_cycle(output, broker, account)

        if cycle.order is not None:
            account = get_account()  # re-sync exposure/trade-count after a fill
        rows.append(
            {
                "symbol": symbol,
                "type": kind,
                "strategy": cycle.strategy_name or "-",
                "status": cycle.setup_status or ("NO_TRADE" if output.ranking.is_no_trade else "-"),
                "detail": cycle.reason,
            }
        )
        if cycle.order is not None and cycle.order.status == "FILLED":
            latest_bar = output.ohlcv.iloc[-1]
            for e in monitor_positions(broker, latest_bar):
                pnl_delta += e["position"]["net_pnl"]
            if kind == "index" and client.is_configured():
                for e in monitor_option_positions(broker, client):
                    pnl_delta += e["position"]["net_pnl"]
    return rows, pnl_delta
