"""Downloadable forms of a daily report: Excel (one sheet per section), Markdown and JSON."""
from __future__ import annotations

import io
import json

import pandas as pd


def _sheet(rows) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame({"note": ["(none)"]})


def sheets(report: dict) -> dict[str, pd.DataFrame]:
    """Name -> table for every section of a report's JSON (names are Excel-safe, 31 characters max)."""
    s, f = report["summary"], report["funnel"]
    out = {
        "Summary": _sheet([
            {"period": "Today", **{k: v for k, v in s["today"].items() if not isinstance(v, (dict, list))}},
            {"period": "All time", **{k: v for k, v in s["cumulative"].items() if not isinstance(v, (dict, list))}},
        ]),
        "Narrative": pd.DataFrame({"how things went": report["narrative"]}),
        "Strategies (all time)": _sheet(report["strategies"]["all_time"]),
        "Strategies (30 days)": _sheet(next((v for k, v in report["strategies"].items() if k.startswith("last_")), [])),
        "Strategies (today)": _sheet(report["strategies"]["today"]),
        "Signals by strategy": _sheet(f["by_strategy"]),
        "No-trade reasons": _sheet(f["no_trade_reasons"]),
        "Veto reasons": _sheet(f["veto_reasons"]),
        "Expected vs realised": _sheet(report["expected_vs_realised"]),
        "Daily": _sheet(report["daily"]),
        "Order rejects": _sheet(report["orders"]["reject_reasons"]),
        "Risk vetoes": _sheet(report["risk"]["vetoes"]),
    }
    for key, rows in report["breakdowns"].items():
        out[f"By {key}"[:31]] = _sheet(rows)
    return out


def to_excel(report: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets(report).items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def to_json(report: dict) -> bytes:
    return json.dumps(report, indent=2, default=str).encode("utf-8")
