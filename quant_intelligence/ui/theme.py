"""Dark, information-dense trading theme shared across pages.

Design tokens follow the data-viz palette (validated in dark mode): categorical blue / orange / aqua /
violet for series identity, and the reserved status colours good / warning / serious / critical for
state and P&L sign. Status colours are never the only signal - they always travel with an icon
(a triangle, a dot or a label) so they survive colour-blindness and print.
"""
from __future__ import annotations

import streamlit as st

# ---- tokens (kept in sync with the CSS custom properties below and with ui/charts.py) --------------
SURFACE = "#0e1117"
CARD = "#151a23"
CARD_2 = "#1b2130"
BORDER = "#262d3b"
TEXT = "#e8eaed"
TEXT_2 = "#aab0bc"
MUTED = "#7d8594"

BLUE, ORANGE, AQUA, VIOLET = "#3987e5", "#d95926", "#199e70", "#9085e9"  # categorical slots 1-3, 7
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"  # status (reserved)

TONE_COLOR = {"good": GOOD, "warning": WARNING, "serious": SERIOUS, "critical": CRITICAL,
              "info": BLUE, "neutral": MUTED, "muted": MUTED}

DARK_CSS = f"""
<style>
:root {{
  --qi-surface: {SURFACE}; --qi-card: {CARD}; --qi-card-2: {CARD_2}; --qi-border: {BORDER};
  --qi-text: {TEXT}; --qi-text-2: {TEXT_2}; --qi-muted: {MUTED};
  --qi-blue: {BLUE}; --qi-orange: {ORANGE}; --qi-aqua: {AQUA}; --qi-violet: {VIOLET};
  --qi-good: {GOOD}; --qi-warning: {WARNING}; --qi-serious: {SERIOUS}; --qi-critical: {CRITICAL};
}}
.stApp {{ background-color: var(--qi-surface); }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
h1 {{ font-size: 1.7rem !important; font-weight: 650 !important; letter-spacing: -0.01em; margin-bottom: 0.1rem; }}
h2, h3 {{ letter-spacing: -0.005em; }}
[data-testid="stMetricValue"] {{ font-size: 1.45rem; font-weight: 600; }}
[data-testid="stMetricLabel"] p {{ color: var(--qi-text-2); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
[data-testid="stMetric"] {{ background: var(--qi-card); border-radius: 10px; }}

/* chips / badges (icon + label, never colour alone) */
.qi-chip {{ display: inline-flex; align-items: center; gap: 5px; padding: 2px 9px; border-radius: 999px;
  font-size: 0.74rem; font-weight: 600; line-height: 1.5; white-space: nowrap;
  border: 1px solid var(--qi-border); background: var(--qi-card-2); color: var(--qi-text-2); }}
.qi-chip.good {{ color: #7ee27e; border-color: rgba(12,163,12,.45); background: rgba(12,163,12,.12); }}
.qi-chip.warning {{ color: var(--qi-warning); border-color: rgba(250,178,25,.45); background: rgba(250,178,25,.10); }}
.qi-chip.serious {{ color: var(--qi-serious); border-color: rgba(236,131,90,.45); background: rgba(236,131,90,.10); }}
.qi-chip.critical {{ color: #f08a8a; border-color: rgba(208,59,59,.55); background: rgba(208,59,59,.14); }}
.qi-chip.info {{ color: #8dbcf5; border-color: rgba(57,135,229,.5); background: rgba(57,135,229,.12); }}
.qi-chip.muted {{ color: var(--qi-text-2); }}
.qi-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; background: currentColor; }}

/* top status bar */
.qi-statusbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 8px 12px; margin: 4px 0 14px 0;
  background: var(--qi-card); border: 1px solid var(--qi-border); border-radius: 10px; }}
.qi-statusbar .qi-clock {{ font-variant-numeric: tabular-nums; font-weight: 650; color: var(--qi-text); margin-right: 6px; }}
.qi-statusbar .qi-sep {{ width: 1px; height: 18px; background: var(--qi-border); margin: 0 4px; }}

/* P&L text: colour + triangle + sign */
.qi-pnl {{ font-variant-numeric: tabular-nums; font-weight: 650; white-space: nowrap; }}
.qi-pnl.good {{ color: #2bc02b; }} .qi-pnl.critical {{ color: #ee6b6b; }} .qi-pnl.neutral {{ color: var(--qi-text-2); }}
.qi-pnl small {{ font-weight: 500; opacity: .85; margin-left: 4px; }}

/* position card */
.qi-pos {{ display: block; }}
.qi-pos-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }}
.qi-sym {{ font-size: 1.05rem; font-weight: 650; color: var(--qi-text); }}
.qi-pos-pnl {{ margin-left: auto; font-size: 1.25rem; text-align: right; }}
.qi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(105px, 1fr)); gap: 6px 14px; margin: 10px 0 8px 0; }}
.qi-grid div span {{ display: block; font-size: 0.68rem; color: var(--qi-muted); text-transform: uppercase; letter-spacing: .05em; }}
.qi-grid div b {{ font-size: 0.95rem; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--qi-text); }}

/* stop - entry - target range bar */
.qi-range {{ position: relative; height: 26px; margin: 4px 2px 2px 2px; }}
.qi-range .track {{ position: absolute; left: 0; right: 0; top: 11px; height: 4px; border-radius: 4px; background: var(--qi-border); }}
.qi-range .fill {{ position: absolute; top: 11px; height: 4px; border-radius: 4px; }}
.qi-range .tick {{ position: absolute; top: 6px; width: 2px; height: 14px; background: var(--qi-text-2); border-radius: 2px; }}
.qi-range .ltp {{ position: absolute; top: 3px; width: 12px; height: 12px; margin-left: -6px; border-radius: 50%;
  border: 2px solid var(--qi-card); box-shadow: 0 0 0 1px var(--qi-text); }}
.qi-range-labels {{ display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--qi-text-2); font-variant-numeric: tabular-nums; }}

/* quote header */
.qi-quote {{ padding: 4px 2px; }}
.qi-quote-main {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-top: 2px; }}
.qi-quote-main .big {{ font-size: 2.0rem; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--qi-text); }}
.qi-quote-sub {{ font-size: 0.78rem; color: var(--qi-text-2); font-variant-numeric: tabular-nums; margin-top: 2px; }}
.qi-dayrange {{ position: relative; height: 18px; margin-top: 6px; }}
.qi-dayrange .track {{ position: absolute; left: 0; right: 0; top: 7px; height: 4px; border-radius: 4px; background: var(--qi-border); }}
.qi-dayrange .dot {{ position: absolute; top: 3px; width: 12px; height: 12px; margin-left: -6px; border-radius: 50%;
  background: var(--qi-blue); border: 2px solid var(--qi-card); box-shadow: 0 0 0 1px var(--qi-text); }}
.qi-dayrange-labels {{ display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--qi-muted); }}

/* decision / ticket cards */
.qi-decision-title {{ font-size: 1.25rem; font-weight: 700; margin: 4px 0 2px 0; }}
.qi-kv {{ display: grid; grid-template-columns: 1fr auto; gap: 3px 12px; font-size: 0.86rem; margin: 6px 0; }}
.qi-kv span:nth-child(odd) {{ color: var(--qi-text-2); }}
.qi-kv span:nth-child(even) {{ text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }}
.qi-meter {{ height: 6px; border-radius: 6px; background: var(--qi-border); overflow: hidden; margin: 4px 0 8px 0; }}
.qi-meter > div {{ height: 100%; border-radius: 6px; }}

/* empty states */
.qi-empty {{ text-align: center; padding: 28px 12px; color: var(--qi-text-2); border: 1px dashed var(--qi-border); border-radius: 10px; }}
.qi-empty b {{ display: block; color: var(--qi-text); font-size: 1.0rem; margin-bottom: 4px; }}

/* legacy badges */
.qi-status-ok {{ color: #2ecc71; font-weight: 600; }} .qi-status-warn {{ color: #f39c12; font-weight: 600; }}
.qi-status-fail {{ color: #e74c3c; font-weight: 600; }}
.qi-badge {{ display: inline-block; padding: 2px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; margin-right: 6px; }}
.qi-badge-trade {{ background-color: #1e5631; color: #d4f7dc; }}
.qi-badge-notrade {{ background-color: #5c1e1e; color: #f7d4d4; }}
.qi-badge-paper {{ background-color: #1e3a5c; color: #d4e7f7; }}
.qi-badge-live {{ background-color: #7a1f1f; color: #ffe0e0; }}
</style>
"""


def apply_theme() -> None:
    st.set_page_config(page_title="Quant Intelligence Terminal", layout="wide", page_icon="\U0001F4C8")
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    cls = {"OK": "qi-status-ok", "DEGRADED": "qi-status-warn", "FAIL": "qi-status-fail"}.get(status, "qi-status-warn")
    return f'<span class="{cls}">{status}</span>'
