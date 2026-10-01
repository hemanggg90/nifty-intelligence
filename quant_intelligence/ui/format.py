"""Number/time formatting for the trading UI (Indian conventions: rupees, lakh/crore grouping)."""
from __future__ import annotations

import datetime as dt
import math

UP, DOWN, FLAT = "▲", "▼", "–"  # triangle up / triangle down / en dash
RUPEE = "₹"


def _group_indian(integer_digits: str) -> str:
    """'1234567' -> '12,34,567' (last three digits, then pairs)."""
    if len(integer_digits) <= 3:
        return integer_digits
    head, tail = integer_digits[:-3], integer_digits[-3:]
    pairs = []
    while len(head) > 2:
        pairs.insert(0, head[-2:])
        head = head[:-2]
    if head:
        pairs.insert(0, head)
    return ",".join(pairs + [tail])


def num(x: float | int | None, decimals: int = 2, signed: bool = False, dash: str = "–") -> str:
    """Indian-grouped number: 1234567.891 -> '12,34,567.89'."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return dash
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    body = f"{abs(x):.{decimals}f}"
    whole, _, frac = body.partition(".")
    out = _group_indian(whole) + (f".{frac}" if frac else "")
    return f"{sign}{out}"


def inr(x: float | int | None, decimals: int = 0, signed: bool = False, dash: str = "–") -> str:
    """Rupee amount with Indian grouping: 1234567 -> '₹12,34,567'; -500 -> '-₹500'."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return dash
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}{RUPEE}{num(abs(x), decimals)}"


def inr_compact(x: float | None) -> str:
    """Short form for tiles: 1,50,000 -> '₹1.50L'; 2,50,00,000 -> '₹2.50Cr'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e7:
        return f"{sign}{RUPEE}{a / 1e7:.2f}Cr"
    if a >= 1e5:
        return f"{sign}{RUPEE}{a / 1e5:.2f}L"
    return f"{sign}{RUPEE}{num(a, 0)}"


def pct(x: float | None, decimals: int = 2, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    sign = "+" if signed and x > 0 else ""
    return f"{sign}{x:.{decimals}f}%"


def tone(x: float | None, eps: float = 1e-9) -> str:
    """'good' / 'critical' / 'neutral' for the sign of a P&L-like number."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or abs(x) <= eps:
        return "neutral"
    return "good" if x > 0 else "critical"


def arrow(x: float | None, eps: float = 1e-9) -> str:
    t = tone(x, eps)
    return UP if t == "good" else (DOWN if t == "critical" else FLAT)


def duration(delta: dt.timedelta | None) -> str:
    """'2h 15m', '45m', '30s'."""
    if delta is None:
        return "–"
    total = int(max(delta.total_seconds(), 0))
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m"
    return f"{s}s"


def short_date(value) -> str:
    """'2026-10-06' or a datetime -> '06 Oct'."""
    if value is None or value == "":
        return "–"
    try:
        d = value if isinstance(value, (dt.date, dt.datetime)) else dt.datetime.strptime(str(value)[:10], "%Y-%m-%d")
        return f"{d:%d %b}"
    except (ValueError, TypeError):
        return str(value)


def contract_label(instrument: str, strike, option_type, expiry) -> str:
    """'NIFTY 22700 CE 06 Oct'. Falls back to the bare instrument for non-option rows."""
    if not option_type or strike is None:
        return instrument
    return f"{instrument} {strike:g} {option_type} {short_date(expiry)}"
