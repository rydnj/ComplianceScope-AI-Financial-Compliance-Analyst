"""Page 3: Ask Questions — RAG-powered Q&A on ingested filings."""

import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="Ask Questions", page_icon="💬", layout="wide")
st.title("💬 Ask Questions")
st.markdown("Ask natural language questions about any ingested filing. "
            "Answers are grounded in the actual filing text with source citations.")

st.markdown("---")

# --- Filing Selector ---
filing_id = st.text_input(
    "Filing ID",
    value=st.session_state.get("current_filing_id", ""),
    placeholder="Paste a filing ID from the Ingest page",
)

if not filing_id:
    st.info("Enter a filing ID to ask questions. Ingest a filing first on the Ingest page.")
    st.stop()

# --- Question Input ---
question = st.text_area(
    "Your Question",
    placeholder="e.g., What are the main cybersecurity risks? What legal proceedings is the company involved in?",
    max_chars=1000,
    height=100,
)

if st.button("Ask", type="primary", disabled=not question):
    with st.spinner("Searching filing and generating answer..."):
        try:
            resp = requests.post(
                f"{API_URL}/api/query",
                json={"filing_id": filing_id, "question": question},
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()

                # Display answer
                st.markdown("### Answer")
                st.markdown(data["answer"])

                # Display sources
                st.markdown("### Sources")
                for i, source in enumerate(data["sources"], 1):
                    with st.expander(f"Source {i}: {source['section']}"):
                        st.markdown(f"**Section:** {source['section']}")
                        st.markdown(f"**Excerpt:**")
                        st.code(source["excerpt"], language=None)

            elif resp.status_code == 404:
                st.error("Filing not found. Check the filing ID.")
            elif resp.status_code == 400:
                st.error(resp.json().get("detail", "Filing not ready for queries."))
            else:
                st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")

        except requests.exceptions.Timeout:
            st.error("Request timed out. Try a shorter question.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Is the API running?")

# --- Query History ---
st.markdown("---")
st.markdown("### Query History")

try:
    resp = requests.get(f"{API_URL}/api/queries/{filing_id}", timeout=10)
    if resp.status_code == 200:
        queries = resp.json()
        if queries:
            for q in queries:
                with st.expander(f"Q: {q['question'][:100]}..."):
                    st.markdown(f"**Question:** {q['question']}")
                    st.markdown(f"**Answer:** {q['answer']}")
                    st.caption(f"Asked at: {q['created_at']}")
        else:
            st.caption("No questions asked yet for this filing.")
except requests.exceptions.ConnectionError:
    st.caption("Could not load query history.")