"""Dark, information-dense theme helpers shared across pages."""
from __future__ import annotations

import streamlit as st

DARK_CSS = """
<style>
.stApp { background-color: #0e1117; }
[data-testid="stMetricValue"] { font-size: 1.4rem; }
.qi-status-ok { color: #2ecc71; font-weight: 600; }
.qi-status-warn { color: #f39c12; font-weight: 600; }
.qi-status-fail { color: #e74c3c; font-weight: 600; }
.qi-badge {
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-size: 0.8rem; font-weight: 600; margin-right: 6px;
}
.qi-badge-trade { background-color: #1e5631; color: #d4f7dc; }
.qi-badge-notrade { background-color: #5c1e1e; color: #f7d4d4; }
.qi-badge-paper { background-color: #1e3a5c; color: #d4e7f7; }
.qi-badge-live { background-color: #7a1f1f; color: #ffe0e0; }
</style>
"""


def apply_theme() -> None:
    st.set_page_config(page_title="Quant Intelligence Terminal", layout="wide", page_icon="\U0001F4C8")
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    cls = {"OK": "qi-status-ok", "DEGRADED": "qi-status-warn", "FAIL": "qi-status-fail"}.get(status, "qi-status-warn")
    return f'<span class="{cls}">{status}</span>'
