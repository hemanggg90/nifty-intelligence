"""Pick the day's biggest movers, for the auto paper trader to scan.

Per group (NSE indices, NSE stocks, MCX commodities) the N biggest gainers and N biggest losers by % change
since the previous session's close are selected. A gainer may only take bullish setups (a bought call), a
loser only bearish ones (a bought put): see `Selection.allowed_direction`.

Where the moves come from:
* NSE stocks, `universe="nse"` (the default setting): NSE's own live top-20 gainers and top-20 losers among ALL stocks
  that have listed options (`data_adapters/nse_heatmap.fetch_fno_movers`), so the pick is the whole F&O market, not
  just the 15-stock watchlist.
  That endpoint is undocumented and sometimes blocked; when it is unreachable (or NSE is closed) the selection
  says so and falls back to the watchlist read from the candle cache - it never invents a quote.
* Indices and MCX commodities (NSE publishes nothing for MCX): the candle cache, no Dhan request.

Rules that keep this honest:
* An instrument with no current quote is never treated as 0% and never selected.
* Nothing is selected on yesterday's move: the quote/candle must be from today and recent.
* A move smaller than `min_abs_pct` is noise, so a quiet day selects fewer than N (possibly none).

Streamlit-free (the scan thread uses it).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.timeutil import now_ist

UP, DOWN = "UP", "DOWN"
ALLOWED = {UP: "LONG", DOWN: "SHORT"}


@dataclass(frozen=True)
class MoverConfig:
    n: int = 5  # gainers and losers selected per group
    min_abs_pct: float = 0.3
    max_age_minutes: float = 20.0  # newest candle must be this recent
    auto_refresh: bool = True  # recompute every scan cycle (False = keep the selection made when applied)
    universe: str = "watchlist"  # "nse" = every NSE F&O stock from NSE's live heatmap; "watchlist" = the 15 stocks only

    @classmethod
    def from_settings(cls) -> "MoverConfig":
        return cls(n=SETTINGS.mover_top_n, min_abs_pct=SETTINGS.mover_min_abs_pct, max_age_minutes=SETTINGS.mover_max_age_minutes,
                   universe=SETTINGS.mover_universe)


@dataclass
class Selection:
    picks: dict[str, str] = field(default_factory=dict)  # symbol -> UP / DOWN
    change_pct: dict[str, float] = field(default_factory=dict)
    group_of: dict[str, str] = field(default_factory=dict)
    made_at: dt.datetime | None = None
    reason: str = ""  # why it is empty, or a note about excluded instruments
    excluded_stale: list[str] = field(default_factory=list)
    source: str = "cache"  # where the NSE stock moves came from: "nse_live" or "cache" (watchlist candles)
    universe_note: str = ""  # what the universe was, or why it fell back to the watchlist

    @property
    def usable(self) -> bool:
        return bool(self.picks)

    def allowed_direction(self, symbol: str) -> str | None:
        """'LONG' for an up-mover, 'SHORT' for a down-mover, None when the symbol is not part of the selection."""
        side = self.picks.get(symbol)
        return ALLOWED.get(side) if side else None

    def subset(self, groups) -> "Selection":
        """The same selection restricted to some groups (the NSE runner takes Index+Stock, the MCX runner Commodity)."""
        keep = {s for s, g in self.group_of.items() if g in set(groups)}
        return Selection(picks={s: v for s, v in self.picks.items() if s in keep},
                         change_pct={s: v for s, v in self.change_pct.items() if s in keep},
                         group_of={s: g for s, g in self.group_of.items() if s in keep},
                         made_at=self.made_at, reason=self.reason, excluded_stale=list(self.excluded_stale),
                         source=self.source, universe_note=self.universe_note)

    def merge(self, other: "Selection") -> "Selection":
        """Both selections' picks together (the NSE and commodity halves of one page preview)."""
        return Selection(picks={**self.picks, **other.picks}, change_pct={**self.change_pct, **other.change_pct},
                         group_of={**self.group_of, **other.group_of}, made_at=self.made_at or other.made_at,
                         reason=" ".join(x for x in (self.reason, other.reason) if x),
                         excluded_stale=self.excluded_stale + other.excluded_stale,
                         source=self.source if self.source == "nse_live" else other.source,
                         universe_note=" ".join(x for x in (self.universe_note, other.universe_note) if x))

    def only(self, symbols) -> list[str]:
        return [s for s in symbols if s in self.picks]

    def in_group(self, group: str) -> list[str]:
        return [s for s, g in self.group_of.items() if g == group]

    def summary(self) -> str:
        ups = sum(1 for v in self.picks.values() if v == UP)
        return f"{len(self.picks)} selected ({ups} up, {len(self.picks) - ups} down)"


def is_fresh(as_of: dt.datetime | None, now: dt.datetime, max_age_minutes: float) -> bool:
    if as_of is None or as_of.date() != now.date():
        return False
    return (now - as_of).total_seconds() / 60.0 <= max_age_minutes


def rank_movers(rows: list[dict], config: MoverConfig, now: dt.datetime | None = None) -> Selection:
    """rows: {"symbol", "group", "change_pct" (None when unknown), "as_of"}. Pure function."""
    now = now or now_ist()
    sel = Selection(made_at=now)
    fresh, stale = [], []
    for r in rows:
        if r.get("change_pct") is None or not is_fresh(r.get("as_of"), now, config.max_age_minutes):
            stale.append(r["symbol"])
        else:
            fresh.append(r)
    sel.excluded_stale = stale
    if not fresh:
        sel.reason = (f"No instrument has a candle from the last {config.max_age_minutes:g} minutes of today, so nothing is "
                      "selected (yesterday's move is not used). The market may be closed, or the data keeper is not refreshing.")
        return sel
    for group in sorted({r["group"] for r in fresh}):
        members = [r for r in fresh if r["group"] == group]
        ups = sorted((r for r in members if r["change_pct"] >= config.min_abs_pct), key=lambda r: -r["change_pct"])[: config.n]
        downs = sorted((r for r in members if r["change_pct"] <= -config.min_abs_pct), key=lambda r: r["change_pct"])[: config.n]
        for side, chosen in ((UP, ups), (DOWN, downs)):
            for r in chosen:
                sel.picks[r["symbol"]] = side
                sel.change_pct[r["symbol"]] = r["change_pct"]
                sel.group_of[r["symbol"]] = group
    if not sel.picks:
        sel.reason = f"No instrument has moved at least {config.min_abs_pct:g}% yet, so nothing is selected."
    elif stale:
        sel.reason = f"{len(stale)} instrument(s) without a current candle were left out: {', '.join(stale[:6])}{'...' if len(stale) > 6 else ''}."
    return sel


def snapshot_rows(groups: dict[str, list[str]], timeframe: str = "5min", loader: Callable | None = None) -> list[dict]:
    """One row per symbol from the candle cache (`groups` maps a group name to its symbols)."""
    if loader is None:
        from quant_intelligence.ui.market_data import load_snapshot as loader
    rows = []
    for group, symbols in groups.items():
        for sym in symbols:
            snap = loader(sym, timeframe)
            rows.append({"symbol": sym, "group": group, "change_pct": snap["change_pct"] if snap else None,
                         "as_of": snap["as_of"] if snap else None})
    return rows


def compute_selection(groups: dict[str, list[str]], config: MoverConfig, timeframe: str = "5min",
                      now: dt.datetime | None = None, loader: Callable | None = None) -> Selection:
    return rank_movers(snapshot_rows(groups, timeframe, loader), config, now)


# ---------------------------------------------------------------- the full NSE market (live heatmap data)
def _default_nse_loader() -> dict:
    from quant_intelligence.data_adapters.nse_heatmap import fetch_fno_movers

    return fetch_fno_movers(max_age_minutes=SETTINGS.mover_nse_refresh_minutes)


def nse_stock_rows(now: dt.datetime | None = None, nse_loader: Callable[[], dict] | None = None,
                   fno_symbols: set[str] | None = None) -> dict:
    """{"rows": [...] | None, "note": str}. rows are NSE stocks that have listed options, one dict per stock
    (symbol, group "Stock", change_pct, as_of, last, industry, volume). rows is None - never an empty guess - when
    NSE is closed or its feed cannot be used, and the note says why so the caller can fall back to the watchlist."""
    from quant_intelligence.config.watchlist import WATCHLIST_STOCKS
    from quant_intelligence.utils.market_profile import NSE
    from quant_intelligence.utils.timeutil import IST, is_market_open

    now = now or now_ist()
    if not is_market_open(now, NSE):
        return {"rows": None, "note": "NSE is closed, so there are no live NSE moves."}
    try:
        data = (nse_loader or _default_nse_loader)()
    except Exception as e:
        return {"rows": None, "note": f"NSE heatmap feed failed ({type(e).__name__}: {e})."}
    if data.get("source") != "nse_live":
        return {"rows": None, "note": "NSE's live heatmap feed is unreachable from here (NSE blocks some servers)."}
    as_of = None
    if data.get("as_of_ist"):  # NSE's own quote time (IST wall clock)
        try:
            as_of = dt.datetime.fromisoformat(data["as_of_ist"])
        except ValueError:
            as_of = None
    if as_of is None and data.get("fetched_at"):
        as_of = dt.datetime.fromtimestamp(float(data["fetched_at"]), tz=IST).replace(tzinfo=None)

    if fno_symbols is None:
        try:
            from quant_intelligence.data_adapters.dhan_instrument_master import list_fno_stock_symbols

            fno_symbols = set(list_fno_stock_symbols())
        except Exception:
            fno_symbols = set()
    note, check = "", True
    if not fno_symbols:
        if data.get("fno_only"):  # NSE's own F&O list: every row already has listed options
            check = False
        else:  # scrip master not available and the feed is not F&O-only: trust only the known watchlist
            fno_symbols = {r["symbol"] for r in WATCHLIST_STOCKS}
            note = "Dhan's scrip master is unavailable, so only the watchlist stocks are treated as tradeable. "
    rows = []
    for c in data.get("constituents", []):
        sym = c.get("symbol")
        if not sym or (check and sym not in fno_symbols):
            continue
        pct = c.get("pChange")
        rows.append({"symbol": sym, "group": "Stock", "change_pct": float(pct) if pct is not None else None, "as_of": as_of,
                     "last": c.get("lastPrice"), "industry": c.get("industry"), "volume": c.get("totalTradedVolume")})
    if not rows:
        return {"rows": None, "note": "NSE returned no F&O stocks."}
    return {"rows": rows, "note": f"{note}NSE live heatmap: top gainers and losers of {'all F&O stocks' if data.get('fno_only') else 'its F&O stocks'} "
                                  f"({len(rows)} stocks, quotes as of {as_of:%H:%M} IST)." if as_of else
                                  f"{note}NSE live heatmap: {len(rows)} F&O stocks."}


def select(index_symbols, other_symbols, other_group: str, config: MoverConfig, now: dt.datetime | None = None,
           loader: Callable | None = None, nse_loader: Callable[[], dict] | None = None,
           fno_symbols: set[str] | None = None, timeframe: str = "5min") -> Selection:
    """The selection for one runner: its indices plus `other_symbols` (stocks for NSE, commodities for MCX).
    With `config.universe == "nse"` the NSE stocks are every F&O stock on NSE's live heatmap, else the watchlist."""
    now = now or now_ist()
    cache_groups = {g: list(s) for g, s in (("Index", index_symbols), (other_group, other_symbols)) if s}
    rows: list[dict] = []
    source, note = "cache", ""
    if config.universe == "nse" and other_group == "Stock":
        nse = nse_stock_rows(now, nse_loader, fno_symbols)
        if nse["rows"] is not None:
            rows += nse["rows"]
            cache_groups.pop("Stock", None)
            source, note = "nse_live", nse["note"]
        else:
            note = nse["note"] + " Scanning the watchlist from the candle cache instead."
    rows += snapshot_rows(cache_groups, timeframe, loader)
    sel = rank_movers(rows, config, now)
    sel.source, sel.universe_note = source, note
    return sel
