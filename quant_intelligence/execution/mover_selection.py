"""Pick the day's biggest movers from the candle cache, for the auto paper trader to scan.

Per group (NSE indices, NSE stocks, MCX commodities) the N biggest gainers and N biggest losers by % change
since the previous session's close are selected. A gainer may only take bullish setups (a bought call), a
loser only bearish ones (a bought put): see `Selection.allowed_direction`.

Rules that keep this honest:
* An instrument with no current candle is never treated as 0% and never selected.
* Nothing is selected on yesterday's move: the newest bar must be from today and recent.
* A move smaller than `min_abs_pct` is noise, so a quiet day selects fewer than N (possibly none).

Streamlit-free (the scan thread uses it). Reads only the on-disk candle cache: no Dhan request.
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

    @classmethod
    def from_settings(cls) -> "MoverConfig":
        return cls(n=SETTINGS.mover_top_n, min_abs_pct=SETTINGS.mover_min_abs_pct, max_age_minutes=SETTINGS.mover_max_age_minutes)


@dataclass
class Selection:
    picks: dict[str, str] = field(default_factory=dict)  # symbol -> UP / DOWN
    change_pct: dict[str, float] = field(default_factory=dict)
    group_of: dict[str, str] = field(default_factory=dict)
    made_at: dt.datetime | None = None
    reason: str = ""  # why it is empty, or a note about excluded instruments
    excluded_stale: list[str] = field(default_factory=list)

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
                         made_at=self.made_at, reason=self.reason, excluded_stale=list(self.excluded_stale))

    def only(self, symbols) -> list[str]:
        return [s for s in symbols if s in self.picks]

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
