import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import ResearchReport
from quant_intelligence.reports.report_generator import generate_report, persist_report
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Research Reports")
st.caption("Structured reports generated entirely from stored system data - no invented numbers.")

output = run_pipeline_cached()
if output is None:
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()

if st.button("Generate report from current pipeline state", type="primary"):
    report = generate_report(output)
    persist_report(output.instrument, report)
    st.session_state["last_report"] = report

if "last_report" in st.session_state:
    st.markdown(st.session_state["last_report"]["content_markdown"])
    st.download_button(
        "Download as Markdown",
        st.session_state["last_report"]["content_markdown"],
        file_name=f"research_report_{output.instrument}.md",
    )

st.divider()
st.subheader("Past reports")
with get_session() as session:
    reports = session.query(ResearchReport).order_by(ResearchReport.id.desc()).limit(20).all()
for r in reports:
    with st.expander(f"{r.instrument} - {r.report_type} - {r.created_at}"):
        st.markdown(r.content_markdown or "(no content)")
