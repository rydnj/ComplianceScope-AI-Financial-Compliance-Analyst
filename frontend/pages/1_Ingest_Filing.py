"""Page 1: Ingest a SEC filing by ticker and filing type."""

import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="Ingest Filing", page_icon="📥", layout="wide")
st.title("📥 Ingest Filing")
st.markdown("Fetch and process a SEC filing from EDGAR. This will parse the filing, "
            "chunk the text, generate embeddings, and store everything for analysis.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    ticker = st.text_input(
        "Company Ticker",
        placeholder="e.g., AAPL, MSFT, TSLA",
        max_chars=10,
    ).strip().upper()

with col2:
    filing_type = st.selectbox("Filing Type", ["10-K", "10-Q"])

if st.button("Fetch & Process", type="primary", disabled=not ticker):
    with st.status("Processing filing...", expanded=True) as status:
        st.write("🔍 Looking up company on SEC EDGAR...")

        try:
            response = requests.post(
                f"{API_URL}/api/ingest",
                json={"ticker": ticker, "filing_type": filing_type},
                timeout=120,  # Ingestion can take a while
            )

            if response.status_code == 200:
                data = response.json()
                filing = data["filing"]
                company = data["company"]

                status.update(label="Filing processed successfully!", state="complete")

                st.success(data["message"])

                # Display filing details
                st.markdown("### Filing Details")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Company", company["name"])
                col2.metric("Filing Type", filing["filing_type"])
                col3.metric("Filing Date", filing["filing_date"])
                col4.metric("Total Chunks", filing["total_chunks"])

                # Store filing info in session state for other pages
                st.session_state["current_filing_id"] = filing["id"]
                st.session_state["current_company"] = company["name"]

                st.info(
                    f"💡 Filing ID: `{filing['id']}`\n\n"
                    f"Head to **Risk Dashboard** to run analysis, or "
                    f"**Ask Questions** to query this filing."
                )

            elif response.status_code == 404:
                status.update(label="Filing not found", state="error")
                st.error(response.json().get("detail", "Filing not found"))
            else:
                status.update(label="Error", state="error")
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.Timeout:
            status.update(label="Timeout", state="error")
            st.error("Request timed out. The filing may be very large — try again.")
        except requests.exceptions.ConnectionError:
            status.update(label="Connection error", state="error")
            st.error("Could not connect to the backend. Is the API running?")

# Show previously ingested companies
st.markdown("---")
st.markdown("### Ingested Companies")

try:
    resp = requests.get(f"{API_URL}/api/companies", timeout=10)
    if resp.status_code == 200:
        companies = resp.json()
        if companies:
            for company in companies:
                st.write(f"**{company['ticker']}** — {company['name']}")
        else:
            st.caption("No companies ingested yet. Use the form above to get started.")
except requests.exceptions.ConnectionError:
    st.caption("Could not connect to the backend.")