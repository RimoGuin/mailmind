import streamlit as st
import streamlit.components.v1 as components
import requests
import time
import os
import sys

sys.path.append(os.path.dirname(__file__))
from utils import validate_form

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MailMind",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── hide Streamlit chrome for cleaner look ──
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; }
  section[data-testid="stSidebar"] { width: 280px !important; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✉️ MailMind")
    st.caption("Email Intelligence Dashboard")
    st.divider()

    # API health
    try:
        res = requests.get(f"{API_URL}/health", timeout=2)
        st.success("API online", icon="●")
    except:
        st.error("API offline — start uvicorn", icon="🚨")

    st.divider()
    st.markdown("#### Send Email")

    with st.form("send_form", clear_on_submit=True):
        to      = st.text_input("To", placeholder="recipient@example.com")
        subject = st.text_input("Subject")
        body    = st.text_area("Message", height=120)
        sent    = st.form_submit_button("Send", use_container_width=True)

    if sent:
        errors = validate_form(to, subject, body)
        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Sending..."):
                try:
                    r = requests.post(f"{API_URL}/send-email",
                        json={"to": to, "subject": subject, "body": body},
                        timeout=10)
                    if r.status_code == 200:
                        st.success(f"Sent to {to}")
                    else:
                        st.error(r.json().get("detail", "Failed"))
                except Exception as ex:
                    st.error(str(ex))

    st.divider()

    # Auto-refresh toggle
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    if st.button("Refresh now", use_container_width=True):
        st.rerun()

# ── MAIN: embed the HTML dashboard ───────────────────
dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard.html")

with open(dashboard_path, "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=950, scrolling=False)

# ── Auto-refresh loop ─────────────────────────────────
if auto_refresh:
    time.sleep(60)
    st.rerun()