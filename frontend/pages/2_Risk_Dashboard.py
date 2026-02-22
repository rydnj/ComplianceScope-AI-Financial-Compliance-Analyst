"""Page 2: Risk Dashboard — view and analyze compliance risks."""

import os
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="Risk Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Risk Dashboard")


def get_filings():
    """Fetch all companies and their filings for the selector."""
    try:
        resp = requests.get(f"{API_URL}/api/companies", timeout=10)
        if resp.status_code != 200:
            return []
        companies = resp.json()

        filings = []
        for company in companies:
            # Get filings for each company by listing all
            # We'll use the company info we already have
            filings.append({
                "company_name": company["name"],
                "ticker": company["ticker"],
                "company_id": company["id"],
            })
        return filings
    except Exception:
        return []


# --- Filing Selector ---
# Check if we have a filing ID from the ingest page
filing_id = st.session_state.get("current_filing_id", "")
company_name = st.session_state.get("current_company", "")

filing_id_input = st.text_input(
    "Filing ID",
    value=filing_id,
    placeholder="Paste a filing ID from the Ingest page",
)

if not filing_id_input:
    st.info("Enter a filing ID to view risks. Ingest a filing first on the Ingest page.")
    st.stop()

# --- Run Analysis Button ---
col1, col2 = st.columns([1, 3])
with col1:
    run_analysis = st.button("🔎 Run Risk Analysis", type="primary")

if run_analysis:
    with st.status("Running risk analysis...", expanded=True) as status:
        st.write("**Phase 1:** Keyword detection (scanning for known patterns)...")
        st.write("**Phase 2:** LLM classification (analyzing each chunk with GPT-4o-mini)...")
        st.write("This may take 2-3 minutes.")

        try:
            resp = requests.post(
                f"{API_URL}/api/analyze/{filing_id_input}",
                timeout=300,
            )
            if resp.status_code == 200:
                data = resp.json()
                status.update(label=data["message"], state="complete")
            else:
                status.update(label="Analysis failed", state="error")
                st.error(resp.json().get("detail", "Unknown error"))
        except requests.exceptions.Timeout:
            status.update(label="Timeout", state="error")
            st.error("Analysis timed out. The filing may have too many chunks.")

st.markdown("---")

# --- Load Risk Data ---
try:
    summary_resp = requests.get(f"{API_URL}/api/risks/{filing_id_input}/summary", timeout=10)
    risks_resp = requests.get(f"{API_URL}/api/risks/{filing_id_input}", timeout=10)
except requests.exceptions.ConnectionError:
    st.error("Could not connect to the backend.")
    st.stop()

if summary_resp.status_code != 200 or risks_resp.status_code != 200:
    st.info("No risk data found. Click 'Run Risk Analysis' above to analyze this filing.")
    st.stop()

summary = summary_resp.json()
risks = risks_resp.json()

if summary["total"] == 0:
    st.info("No risks detected. Run the analysis first.")
    st.stop()

# --- Metric Cards ---
st.markdown("### Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Risks", summary["total"])
col2.metric("🔴 High", summary["high"])
col3.metric("🟡 Medium", summary["medium"])
col4.metric("🟢 Low", summary["low"])

st.markdown("---")

# --- Charts ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Risks by Category")
    category_data = summary["by_category"]
    if category_data:
        fig = px.bar(
            x=list(category_data.keys()),
            y=list(category_data.values()),
            labels={"x": "Category", "y": "Count"},
            color=list(category_data.values()),
            color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.markdown("### Severity Distribution")
    severity_data = {
        "High": summary["high"],
        "Medium": summary["medium"],
        "Low": summary["low"],
    }
    # Filter out zero values
    severity_data = {k: v for k, v in severity_data.items() if v > 0}
    if severity_data:
        fig = px.pie(
            names=list(severity_data.keys()),
            values=list(severity_data.values()),
            color=list(severity_data.keys()),
            color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"},
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Detection Method Breakdown ---
keyword_count = sum(1 for r in risks if r["detection"] == "keyword")
llm_count = sum(1 for r in risks if r["detection"] == "llm")

st.markdown("### Detection Methods")
method_col1, method_col2 = st.columns(2)
method_col1.metric("🔑 Keyword Detection", keyword_count)
method_col2.metric("🤖 LLM Classification", llm_count)

st.markdown("---")

# --- Risk Table with Expandable Details ---
st.markdown("### Risk Flags")

# Filter controls
filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    severity_filter = st.multiselect(
        "Filter by Severity",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
with filter_col2:
    categories = list(set(r["category"] for r in risks))
    category_filter = st.multiselect(
        "Filter by Category",
        categories,
        default=categories,
    )
with filter_col3:
    detection_filter = st.multiselect(
        "Filter by Detection",
        ["keyword", "llm"],
        default=["keyword", "llm"],
    )

# Apply filters
filtered_risks = [
    r for r in risks
    if r["severity"] in severity_filter
    and r["category"] in category_filter
    and r["detection"] in detection_filter
]

st.caption(f"Showing {len(filtered_risks)} of {len(risks)} risks")

# Display each risk as an expandable card
for risk in filtered_risks:
    severity_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk["severity"], "⚪")
    detection_icon = "🔑" if risk["detection"] == "keyword" else "🤖"

    with st.expander(
        f"{severity_icon} [{risk['severity']}] {risk['title']} — {risk['category']} {detection_icon}"
    ):
        st.markdown(f"**Category:** {risk['category']}")
        st.markdown(f"**Severity:** {risk['severity']}")
        st.markdown(f"**Detection:** {risk['detection']}"
                    + (f" (confidence: {risk['confidence']:.1f})" if risk['confidence'] else ""))
        st.markdown(f"**Description:** {risk['description']}")
        st.markdown("**Source Text:**")
        st.code(risk["source_text"], language=None)