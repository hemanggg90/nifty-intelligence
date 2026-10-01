"""Pure data helpers behind the Positions & Orders / trading pages (no Streamlit): normalise broker or
database position rows into one frame, add live P&L and charges, and compute trade statistics.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Iterable

import numpy as np
import pandas as pd

from quant_intelligence.execution.charges import option_trade_charges
from quant_intelligence.ui.format import contract_label
from quant_intelligence.utils.market_profile import profile_for

_COLUMNS = [
    "position_id", "instrument", "strategy", "contract", "side", "option_type", "strike", "expiry", "qty",
    "entry", "stop", "target", "status", "opened_at", "closed_at", "exit_price", "exit_reason", "security_id",
    "market", "invested", "gross_pnl", "charges", "net_pnl", "return_pct", "held",
]


def _get(row, key, default=None):
    value = row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)
    if isinstance(value, float) and math.isnan(value):
        return default
    return value


def positions_frame(rows: Iterable, now: dt.datetime | None = None) -> pd.DataFrame:
    """One tidy frame from paper-broker dicts or ORM Position rows.

    `gross_pnl` is the broker's booked premium P&L; `charges` are itemised at Dhan's rates on the actual
    entry/exit premiums (display only); `net_pnl` = gross - charges. Open rows have no exit yet, so
    these three are None until `with_live` prices them.
    """
    records = []
    for r in rows:
        instrument = _get(r, "underlying") or _get(r, "instrument")
        qty = _get(r, "quantity", 0) or 0
        entry = _get(r, "entry_price")
        exit_price = _get(r, "exit_price")
        opened, closed = _get(r, "opened_at"), _get(r, "closed_at")
        status = _get(r, "status", "OPEN")
        side = _get(r, "transaction") or ("BUY" if _get(r, "direction") in ("LONG", "BUY") else "SELL")
        market = profile_for(instrument).name
        gross = _get(r, "net_pnl") if status == "CLOSED" else None
        charges = None
        if status == "CLOSED" and entry and exit_price is not None:
            charges = option_trade_charges(entry, exit_price, qty, side, market).total
        end = closed if closed is not None else (now or dt.datetime.now())
        records.append(
            {
                "position_id": _get(r, "position_id"),
                "instrument": instrument,
                "strategy": _get(r, "strategy_name"),
                "contract": contract_label(instrument, _get(r, "strike"), _get(r, "option_type"), _get(r, "expiry")),
                "side": side,
                "option_type": _get(r, "option_type"),
                "strike": _get(r, "strike"),
                "expiry": _get(r, "expiry"),
                "qty": qty,
                "entry": entry,
                "stop": _get(r, "stop_price"),
                "target": _get(r, "target_price"),
                "status": status,
                "opened_at": opened,
                "closed_at": closed,
                "exit_price": exit_price,
                "exit_reason": _get(r, "exit_reason"),
                "security_id": _get(r, "security_id"),
                "market": market,
                "invested": (entry or 0.0) * qty,
                "gross_pnl": gross,
                "charges": charges,
                "net_pnl": None if gross is None else gross - (charges or 0.0),
                "return_pct": (gross / ((entry or 0.0) * qty) * 100.0) if gross is not None and entry and qty else None,
                "held": (end - opened) if opened is not None else None,
            }
        )
    return pd.DataFrame.from_records(records, columns=_COLUMNS)


def with_live(df: pd.DataFrame, ltp_map: dict) -> pd.DataFrame:
    """Add live columns for OPEN rows: ltp, unrealised gross/charges/net, return %, progress along the
    stop->target range (0 = at stop, 1 = at target) and the % distance to each."""
    df = df.copy()
    n = len(df)
    ltp = np.full(n, np.nan)
    for i, row in enumerate(df.itertuples(index=False)):
        if row.status == "OPEN" and row.security_id is not None:
            v = ltp_map.get(row.security_id)
            if v is None:
                v = ltp_map.get(str(row.security_id))
            if v is not None:
                ltp[i] = float(v)
    df["ltp"] = ltp
    sign = np.where(df["side"] == "SELL", -1.0, 1.0)
    entry = df["entry"].astype(float)
    qty = df["qty"].astype(float)
    open_live = (df["status"] == "OPEN") & ~np.isnan(ltp)
    unreal = np.where(open_live, sign * (ltp - entry) * qty, np.nan)
    df["unrealised"] = unreal
    df["unrealised_charges"] = [
        option_trade_charges(e, l, int(q), s, m).total if ok else np.nan
        for e, l, q, s, m, ok in zip(entry, ltp, qty, df["side"], df["market"], open_live)
    ]
    df["unrealised_net"] = df["unrealised"] - df["unrealised_charges"]
    base = entry * qty
    df["unrealised_pct"] = np.where(open_live & (base > 0), unreal / base * 100.0, np.nan)

    stop, target = df["stop"].astype(float), df["target"].astype(float)
    span = target - stop
    with np.errstate(divide="ignore", invalid="ignore"):
        progress = (ltp - stop) / span
    df["progress"] = np.where(open_live & (span != 0), np.clip(progress, 0.0, 1.0), np.nan)
    df["to_stop_pct"] = np.where(open_live & (ltp > 0), (ltp - stop) / ltp * 100.0, np.nan)
    df["to_target_pct"] = np.where(open_live & (ltp > 0), (target - ltp) / ltp * 100.0, np.nan)
    return df


def trade_stats(closed: pd.DataFrame) -> dict:
    """Headline statistics over CLOSED trades (uses net P&L after charges)."""
    closed = closed[closed["status"] == "CLOSED"].dropna(subset=["net_pnl"]) if len(closed) else closed
    n = len(closed)
    empty = {"trades": 0, "wins": 0, "losses": 0, "win_rate": None, "avg_win": None, "avg_loss": None,
             "payoff": None, "profit_factor": None, "expectancy": None, "gross": 0.0, "charges": 0.0, "net": 0.0,
             "max_drawdown": 0.0, "best": None, "worst": None, "avg_held": None,
             "max_win_streak": 0, "max_loss_streak": 0}
    if n == 0:
        return empty
    ordered = closed.sort_values("closed_at")
    net = ordered["net_pnl"].to_numpy(dtype=float)
    wins, losses = net[net > 0], net[net < 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    equity = np.cumsum(net)
    drawdown = float((equity - np.maximum.accumulate(equity)).min())
    avg_win = float(wins.mean()) if len(wins) else None
    avg_loss = float(losses.mean()) if len(losses) else None

    def longest(mask):
        best = cur = 0
        for m in mask:
            cur = cur + 1 if m else 0
            best = max(best, cur)
        return best

    held = ordered["held"].dropna()
    return {
        "trades": n,
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float(len(wins) / n * 100.0),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": (avg_win / abs(avg_loss)) if avg_win is not None and avg_loss else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (None if gross_win == 0 else float("inf")),
        "expectancy": float(net.mean()),
        "gross": float(ordered["gross_pnl"].sum()),
        "charges": float(ordered["charges"].sum()),
        "net": float(net.sum()),
        "max_drawdown": drawdown,
        "best": float(net.max()),
        "worst": float(net.min()),
        "avg_held": held.mean() if len(held) else None,
        "max_win_streak": longest(net > 0),
        "max_loss_streak": longest(net < 0),
    }


def equity_curve(closed: pd.DataFrame, start_capital: float = 0.0) -> pd.DataFrame:
    """Cumulative net P&L after each closed trade (columns: time, trade, pnl, equity)."""
    c = closed[closed["status"] == "CLOSED"].dropna(subset=["net_pnl", "closed_at"]).sort_values("closed_at")
    if c.empty:
        return pd.DataFrame(columns=["time", "trade", "pnl", "equity"])
    return pd.DataFrame(
        {
            "time": c["closed_at"].to_numpy(),
            "trade": c["contract"].to_numpy(),
            "pnl": c["net_pnl"].to_numpy(dtype=float),
            "equity": start_capital + np.cumsum(c["net_pnl"].to_numpy(dtype=float)),
        }
    )


def pnl_by(closed: pd.DataFrame, key: str) -> pd.DataFrame:
    """Net P&L, trade count and win rate per `key` (e.g. 'strategy', 'instrument'), best first."""
    c = closed[closed["status"] == "CLOSED"].dropna(subset=["net_pnl"])
    if c.empty:
        return pd.DataFrame(columns=[key, "net_pnl", "trades", "win_rate"])
    g = c.groupby(key)["net_pnl"]
    out = pd.DataFrame({"net_pnl": g.sum(), "trades": g.size(), "win_rate": g.apply(lambda s: (s > 0).mean() * 100.0)})
    return out.reset_index().sort_values("net_pnl", ascending=False).reset_index(drop=True)


def exit_reason_label(reason: str | None) -> str:
    return {
        "STOP": "Stop-loss hit", "TARGET": "Target hit", "EOD_SQUARE_OFF": "End-of-day square-off",
        "MANUAL": "Manual exit", "STALE_ON_RESTART": "Left open (restart)",
    }.get(reason or "", reason or "–")


def charges_breakdown(closed: pd.DataFrame) -> dict:
    """Total of each charge line over CLOSED trades (what a contract note would sum to)."""
    totals = {"Brokerage": 0.0, "STT": 0.0, "Exchange charges": 0.0, "SEBI fee": 0.0, "Stamp duty": 0.0, "GST": 0.0}
    for r in closed[closed["status"] == "CLOSED"].itertuples(index=False):
        if r.entry and r.exit_price is not None and not pd.isna(r.exit_price):
            c = option_trade_charges(r.entry, r.exit_price, int(r.qty), r.side, r.market)
            for name, value in c.as_rows():
                totals[name] += value
    return totals


def orders_frame(rows: Iterable) -> pd.DataFrame:
    """Order-book rows (ORM or dicts) with a readable contract label."""
    records = []
    for o in rows:
        instrument = _get(o, "instrument")
        records.append(
            {
                "timestamp": _get(o, "timestamp"),
                "order_id": _get(o, "order_id"),
                "instrument": instrument,
                "contract": contract_label(instrument, _get(o, "strike"), _get(o, "option_type"), _get(o, "expiry")),
                "side": _get(o, "direction"),
                "quantity": _get(o, "quantity", 0),
                "order_type": _get(o, "order_type"),
                "price": _get(o, "price"),
                "status": _get(o, "status"),
                "reject_reason": _get(o, "reject_reason"),
                "strategy_name": _get(o, "strategy_name"),
                "mode": _get(o, "mode"),
            }
        )
    cols = ["timestamp", "order_id", "instrument", "contract", "side", "quantity", "order_type", "price",
            "status", "reject_reason", "strategy_name", "mode"]
    return pd.DataFrame.from_records(records, columns=cols)


def fills_frame(fills: Iterable, orders: pd.DataFrame) -> pd.DataFrame:
    """Fills joined to their orders so each shows the contract traded."""
    by_order = dict(zip(orders["order_id"], orders["contract"])) if len(orders) else {}
    records = [
        {
            "timestamp": _get(f, "timestamp"),
            "order_id": _get(f, "order_id"),
            "contract": by_order.get(_get(f, "order_id"), "–"),
            "quantity": _get(f, "quantity", 0) or 0,
            "fill_price": _get(f, "fill_price"),
            "slippage": _get(f, "slippage", 0.0) or 0.0,
        }
        for f in fills
    ]
    return pd.DataFrame.from_records(
        records, columns=["timestamp", "order_id", "contract", "quantity", "fill_price", "slippage"]
    )


_UTC_TO_IST = dt.timedelta(hours=5, minutes=30)


def activity_frame(orders: pd.DataFrame, risk_events: Iterable, closed: pd.DataFrame, limit: int = 40,
                   instruments: set | None = None) -> pd.DataFrame:
    """A newest-first timeline of what the trader did: orders placed/rejected, risk-engine vetoes and exits.

    Orders and exits are limited to `instruments` when given. Risk events carry no instrument, so vetoes
    are account-wide; their timestamps are stored in UTC and shown in IST like everything else.
    Columns: time, kind, tone ('good'/'warning'/'critical'/'neutral'), text.
    """
    rows = []
    if len(orders):
        for o in orders.itertuples(index=False):
            if instruments is not None and o.instrument not in instruments:
                continue
            ok = o.status == "FILLED"
            text = f"{o.status}: {o.contract} {o.side} x{int(o.quantity or 0)}" + (
                f" @ {o.price:.2f}" if o.price is not None and not pd.isna(o.price) else "")
            if not ok and o.reject_reason:
                text += f" - {o.reject_reason}"
            rows.append({"time": o.timestamp, "kind": "Order", "tone": "good" if ok else "critical", "text": text})
    for e in risk_events:
        kind = _get(e, "event_type") or ""
        if kind == "APPROVED":
            continue  # the matching order row already says it was placed
        ts = _get(e, "timestamp")
        rows.append({"time": ts + _UTC_TO_IST if ts is not None else None, "kind": "Risk",
                     "tone": "warning" if kind == "VETO" else "critical", "text": f"{kind}: {_get(e, 'reason') or ''}"})
    if len(closed):
        for c in closed[closed["status"] == "CLOSED"].itertuples(index=False):
            if instruments is not None and c.instrument not in instruments:
                continue
            net = c.net_pnl if c.net_pnl is not None and not pd.isna(c.net_pnl) else 0.0
            rows.append({"time": c.closed_at, "kind": "Exit", "tone": "good" if net > 0 else ("critical" if net < 0 else "neutral"),
                         "text": f"{c.contract} closed - {exit_reason_label(c.exit_reason)}, net {net:+,.0f}"})
    out = pd.DataFrame(rows, columns=["time", "kind", "tone", "text"]).dropna(subset=["time"])
    return out.sort_values("time", ascending=False).head(limit).reset_index(drop=True)
