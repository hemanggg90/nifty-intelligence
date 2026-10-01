"""Streamlit widget for viewing, changing and TESTING the Dhan API credentials from the dashboard.

Shows when the saved access token expires (read from the token itself), validates a newly saved token by
making one cheap Dhan call, and offers a Test connection button - so "saved" never silently means "saved a
token that does not work".
"""
from __future__ import annotations

import streamlit as st

from quant_intelligence.brokers.dhan_api_client import DhanApiClient, DhanApiError
from quant_intelligence.config.credentials import mask_secret, token_status, update_dhan_credentials
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.ui import format as F


def test_connection() -> tuple[bool, str]:
    """One non-trading call (fund limit) to prove the saved Client ID + token are accepted by Dhan."""
    client = DhanApiClient()
    if not client.is_configured():
        return False, "No credentials are set."
    try:
        data = client.get_fund_limit()
    except DhanApiError as e:
        return False, str(e)
    except Exception as e:  # network
        return False, f"Could not reach Dhan: {e}"
    balance = data.get("availabelBalance") if isinstance(data, dict) else None
    return True, "Dhan accepted the credentials" + (f" (available balance {F.inr(balance)})." if balance is not None else ".")


def _token_line() -> str:
    ts = token_status(SETTINGS.dhan_access_token)
    state = ts["state"]
    if state == "missing":
        return "Access token: **not set**"
    if state == "expired":
        return (f"Access token: `{mask_secret(SETTINGS.dhan_access_token)}` ✖ **EXPIRED** {ts['expires_at']:%d %b %H:%M} IST "
                f"({F.duration(-ts['delta'])} ago) - generate a new one in your Dhan account")
    if state == "expiring":
        return (f"Access token: `{mask_secret(SETTINGS.dhan_access_token)}` ▲ expires {ts['expires_at']:%d %b %H:%M} IST "
                f"(in {F.duration(ts['delta'])})")
    if state == "valid":
        return (f"Access token: `{mask_secret(SETTINGS.dhan_access_token)}` ● valid until {ts['expires_at']:%d %b %H:%M} IST "
                f"(in {F.duration(ts['delta'])})")
    return f"Access token: `{mask_secret(SETTINGS.dhan_access_token)}` (expiry unknown)"


def render_credentials_panel() -> None:
    expired = token_status(SETTINGS.dhan_access_token)["state"] in ("missing", "expired")
    with st.expander("Dhan API credentials", expanded=expired):
        st.caption(f"Client ID: `{SETTINGS.dhan_client_id or '(not set)'}`")
        st.markdown(_token_line())
        if st.button("Test connection", key="dhan_test_conn"):
            ok, msg = test_connection()
            (st.success if ok else st.error)(msg)
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
            ok, msg = test_connection()
            if ok:
                st.success(f"Saved to .env and applied. {msg}")
            else:
                st.error(f"Saved, but Dhan did not accept it: {msg}")
            st.caption("On Streamlit Cloud this panel only changes the running app (it resets on reboot and is shared by "
                       "everyone who opens it). Put the token in the app's Secrets for a lasting change.")
