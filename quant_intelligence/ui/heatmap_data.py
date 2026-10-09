"""The table behind the NSE heatmap: one row per watchlist instrument, % change over a chosen window.

Everything comes from the on-disk 5-minute candle cache (kept current by the data keeper), so building the
heatmap makes no Dhan request. An instrument with no cached candles is returned with `has_data=False` and a
missing change - it is never drawn as 0%. Streamlit-free so it can be tested without a page.
"""
from __future__ import annotations

import datetime as dt
from typing import Callable

import pandas as pd

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.config.watchlist import WATCHLIST_COMMODITIES, WATCHLIST_STOCKS
from quant_intelligence.execution.mover_selection import is_fresh, nse_stock_rows
from quant_intelligence.ui.market_data import load_candles, snapshot_from_frame
from quant_intelligence.utils.timeutil import now_ist

WINDOWS = {"Today": 1, "5 days": 5, "1 month": 21}  # sessions back; "Today" = vs the previous session's close
GROUPS = ("Index", "Stock", "Commodity")
MAX_SESSIONS = max(WINDOWS.values()) + 2


def _short_sector(text: str) -> str:
    """'Financial Services (Banking)' -> 'Financial Services'."""
    return text.split(" (")[0].strip() if text else "Other"


def universe() -> list[dict]:
    rows = [{"symbol": s, "name": s, "group": "Index", "sector": "Index"} for s in SETTINGS.option_underlyings]
    rows += [{"symbol": r["symbol"], "name": r["name"], "group": "Stock", "sector": _short_sector(r["sector"])} for r in WATCHLIST_STOCKS]
    rows += [{"symbol": r["symbol"], "name": r["name"], "group": "Commodity", "sector": _short_sector(r["sector"])} for r in WATCHLIST_COMMODITIES]
    return rows


def groups_of(rows: list[dict] | None = None) -> dict[str, list[str]]:
    """{'Index': [...], 'Stock': [...], 'Commodity': [...]} for the movers selector."""
    out: dict[str, list[str]] = {g: [] for g in GROUPS}
    for r in rows or universe():
        out[r["group"]].append(r["symbol"])
    return out


def window_change_pct(candles: pd.DataFrame, sessions: int) -> float | None:
    """% change of the last close against the last close `sessions` sessions earlier (1 = previous session)."""
    if candles is None or len(candles) == 0:
        return None
    ts = pd.to_datetime(candles["timestamp"])
    closes = candles.assign(_day=ts.dt.date).sort_values("timestamp").groupby("_day")["close"].last()
    if len(closes) <= sessions:
        return None
    ref = float(closes.iloc[-(sessions + 1)])
    return (float(candles.sort_values("timestamp")["close"].iloc[-1]) - ref) / ref * 100.0 if ref else None


def build_heatmap_frame(window: str = "Today", now: dt.datetime | None = None, max_age_minutes: float | None = None,
                        loader: Callable[[str], pd.DataFrame | None] | None = None, nse_stocks: bool = False,
                        nse_loader: Callable[[], dict] | None = None, fno_symbols: set[str] | None = None) -> pd.DataFrame:
    """Columns: symbol, name, group, sector, change_pct, last, day_range_pct, as_of, fresh, has_data.

    `nse_stocks=True` adds NSE's live top gainers and losers among ALL F&O stocks to the watchlist stock tiles (NSE's
    quote wins where both exist), when that feed is usable (NSE open, reachable) and the window is "Today" (NSE gives
    only today's change). Indices and commodities always come from the candle cache. NSE publishes only the top 20 per
    side, not all ~200 stocks. `df.attrs["universe_note"]` says what was used, or why not."""
    now = now or now_ist()
    max_age = max_age_minutes if max_age_minutes is not None else SETTINGS.mover_max_age_minutes
    load = loader or (lambda sym: load_candles(sym, "5min", sessions=MAX_SESSIONS))
    sessions = WINDOWS[window]
    out = []
    for u in universe():
        candles = load(u["symbol"])
        snap = snapshot_from_frame(candles) if candles is not None and len(candles) else None
        change = window_change_pct(candles, sessions) if snap else None
        span = (snap["high"] - snap["low"]) / snap["prev_close"] * 100.0 if snap and snap["prev_close"] else None
        out.append({**u, "change_pct": change, "last": snap["last"] if snap else None, "day_range_pct": span,
                    "as_of": snap["as_of"] if snap else None,
                    "fresh": bool(snap and is_fresh(snap["as_of"], now, max_age)), "has_data": snap is not None and change is not None})
    df = pd.DataFrame(out)
    note = ""
    if nse_stocks and window != "Today":
        note = "NSE's live heatmap only gives today's change, so the 5-day and 1-month views use the watchlist from the candle cache."
    elif nse_stocks:
        nse = nse_stock_rows(now, nse_loader, fno_symbols)
        note = nse["note"] if nse["rows"] is not None else nse["note"] + " Showing the watchlist from the candle cache instead."
        if nse["rows"] is not None:
            rows = [{"symbol": r["symbol"], "name": r["symbol"], "group": "Stock", "sector": "NSE F&O movers",
                     "change_pct": r["change_pct"], "last": r.get("last"), "day_range_pct": None, "as_of": r["as_of"],
                     "fresh": r["as_of"] is not None and is_fresh(r["as_of"], now, max_age), "has_data": r["change_pct"] is not None}
                    for r in nse["rows"]]
            live = {r["symbol"] for r in rows}
            df = pd.concat([df[~((df["group"] == "Stock") & df["symbol"].isin(live))], pd.DataFrame(rows)], ignore_index=True)
    df.attrs["universe_note"] = note
    return df
