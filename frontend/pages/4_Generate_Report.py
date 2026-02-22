"""Page 4: Generate Report — executive risk briefing."""

import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="Generate Report", page_icon="📄", layout="wide")
st.title("📄 Generate Report")
st.markdown("Generate an executive risk briefing from the analyzed risk flags. "
            "Run risk analysis first on the Risk Dashboard page.")

st.markdown("---")

# --- Filing Selector ---
filing_id = st.text_input(
    "Filing ID",
    value=st.session_state.get("current_filing_id", ""),
    placeholder="Paste a filing ID from the Ingest page",
)

if not filing_id:
    st.info("Enter a filing ID to generate a report. Ingest and analyze a filing first.")
    st.stop()

# --- Generate Button ---
if st.button("Generate Executive Report", type="primary"):
    with st.spinner("Generating executive risk briefing... (10-15 seconds)"):
        try:
            resp = requests.post(
                f"{API_URL}/api/report/{filing_id}",
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                report_md = data["report_markdown"]

                # Store in session state so it persists
                st.session_state["current_report"] = report_md

            elif resp.status_code == 404:
                st.error("Filing not found. Check the filing ID.")
            else:
                st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")

        except requests.exceptions.Timeout:
            st.error("Report generation timed out. Try again.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Is the API running?")

# --- Display Report ---
report_md = st.session_state.get("current_report", "")

if report_md:
    st.markdown("---")
    st.markdown(report_md)

    # Download button
    st.markdown("---")
    st.download_button(
        label="📥 Download Report (.md)",
        data=report_md,
        file_name="compliance_report.md",
        mime="text/markdown",
    )