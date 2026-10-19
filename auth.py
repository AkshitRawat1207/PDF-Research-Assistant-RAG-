"""Optional single-password gate.

If APP_PASSWORD is set (env / secrets), the app requires it before use — this
protects your API bill when the app is deployed publicly. If it is unset, the
gate is a no-op (convenient for local use).
"""

import hmac

import streamlit as st

import config


def require_login() -> bool:
    """Return True if access is allowed; otherwise render a login form and stop."""
    if not config.APP_PASSWORD:
        return True  # no password configured -> open access

    if st.session_state.get("_authenticated"):
        return True

    st.title("🔒 PDF Research Assistant")
    st.caption("Enter the access password to continue.")
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        if hmac.compare_digest(entered, config.APP_PASSWORD):
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()
    return False
