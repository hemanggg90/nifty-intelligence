"""Streamlit widget for changing Dhan API credentials from the dashboard."""
from __future__ import annotations

import streamlit as st

from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.config.credentials import mask_secret, update_dhan_credentials
from quant_intelligence.config.settings import SETTINGS


def render_credentials_panel() -> None:
    with st.expander("Dhan API credentials", expanded=not SETTINGS.dhan_access_token):
        st.caption(
            f"Client ID: `{SETTINGS.dhan_client_id or '(not set)'}`  \n"
            f"Access token: `{mask_secret(SETTINGS.dhan_access_token)}`"
        )
        with st.form("dhan_credentials_form", clear_on_submit=True):
            client_id = st.text_input("Client ID", value=SETTINGS.dhan_client_id)
            token = st.text_input("Access token", type="password", placeholder="Paste new token")
            submitted = st.form_submit_button("Save credentials")
        if submitted:
            try:
                update_dhan_credentials(client_id, token)
            except ValueError as e:
                st.error(str(e))
                return
            # Drop clients/brokers built with the old credentials.
            st.session_state["dhan_api_client"] = DhanApiClient()
            st.session_state.pop("dhan_broker", None)
            st.success("Credentials saved to .env and applied.")
            st.rerun()
