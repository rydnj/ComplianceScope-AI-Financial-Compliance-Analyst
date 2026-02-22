"""ComplianceScope — Streamlit entry point.

Streamlit's multi-page app structure works by convention:
  - app.py is the home page
  - files in pages/ become additional pages, sorted alphabetically
  - The number prefix (1_, 2_, etc.) controls the page order in the sidebar
"""

import streamlit as st

st.set_page_config(
    page_title="ComplianceScope",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 ComplianceScope")
st.subheader("AI-Powered SEC Filing Compliance Analyzer")

st.markdown("""
---

### What is ComplianceScope?

ComplianceScope ingests SEC filings (10-K and 10-Q), automatically detects 
compliance risks, and lets you ask natural language questions about any filing — 
all powered by RAG (Retrieval-Augmented Generation) and a two-tier risk detection engine.

### How to use it

1. **Ingest Filing** — Enter a company ticker (e.g., AAPL) to fetch and process their latest filing
2. **Risk Dashboard** — View automatically detected compliance risks with severity ratings
3. **Ask Questions** — Ask natural language questions about any ingested filing
4. **Generate Report** — Create an executive risk briefing with recommended actions

### Architecture

- **Document Processing**: SEC EDGAR → HTML parsing → chunking → OpenAI embeddings → pgvector
- **Risk Detection**: Keyword matching (high-precision) + LLM classification (high-recall) across 7 categories
- **RAG Pipeline**: Vector similarity search → context retrieval → GPT-4o-mini grounded answers
""")

st.markdown("---")
st.caption("Built with FastAPI, Streamlit, LangChain, OpenAI, and PostgreSQL/pgvector")