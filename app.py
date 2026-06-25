"""
AI-Powered Supermarket Sales Dashboard
app.py
Main Streamlit application.
"""

from __future__ import annotations

import streamlit as st

from utils.data_loader import (
    get_dataset,
    dataset_profile,
    missing_report,
    preview,
    validate_dataset,
)
from utils.preprocessing import preprocess
from utils.analytics import dashboard_summary
from utils.insights import generate_insights
from utils.exports import export_dashboard_package
from utils.config import STYLE_PATH

def load_css():
    if STYLE_PATH.exists():
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.set_page_config(
    page_title="AI-Powered Supermarket Sales Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛒 AI-Powered Supermarket Sales Dashboard")
st.caption("Interactive Business Intelligence Dashboard")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "xlsx", "xls"]
    )

# -------------------------
# Load & preprocess
# -------------------------
try:
    raw_df = get_dataset(uploaded)
    df = preprocess(raw_df)
except Exception as e:
    st.error(f"Unable to load dataset.\n\n{e}")
    st.stop()

st.session_state["data"] = df

# -------------------------
# Validation
# -------------------------
issues = validate_dataset(df)
if issues:
    with st.expander("⚠ Dataset Validation"):
        for issue in issues:
            st.warning(issue)

# -------------------------
# Sidebar Filters
# -------------------------
with st.sidebar:
    st.header("Filters")

    filtered = df.copy()

    if "Branch" in filtered.columns:
        branches = st.multiselect(
            "Branch",
            sorted(filtered["Branch"].dropna().unique()),
            default=sorted(filtered["Branch"].dropna().unique())
        )
        filtered = filtered[filtered["Branch"].isin(branches)]

    if "City" in filtered.columns:
        cities = st.multiselect(
            "City",
            sorted(filtered["City"].dropna().unique()),
            default=sorted(filtered["City"].dropna().unique())
        )
        filtered = filtered[filtered["City"].isin(cities)]

    if "Product Line" in filtered.columns:
        products = st.multiselect(
            "Product Line",
            sorted(filtered["Product Line"].dropna().unique()),
            default=sorted(filtered["Product Line"].dropna().unique())
        )
        filtered = filtered[filtered["Product Line"].isin(products)]

    if "Customer Type" in filtered.columns:
        ctype = st.multiselect(
            "Customer Type",
            sorted(filtered["Customer Type"].dropna().unique()),
            default=sorted(filtered["Customer Type"].dropna().unique())
        )
        filtered = filtered[filtered["Customer Type"].isin(ctype)]

st.session_state["filtered_data"] = filtered

# -------------------------
# KPI Cards
# -------------------------
summary = dashboard_summary(filtered)

st.subheader("📊 Dashboard KPIs")

c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, c8 = st.columns(4)

c1.metric("Revenue", f"₹{summary['Total Revenue']:,.2f}")
c2.metric("Orders", summary["Total Orders"])
c3.metric("Customers", summary["Total Customers"])
c4.metric("Products Sold", summary["Products Sold"])

c5.metric("Gross Income", f"₹{summary['Gross Income']:,.2f}")
c6.metric("Avg Order", f"₹{summary['Average Order Value']:,.2f}")
c7.metric("Rating", summary["Average Rating"])
c8.metric("Best Branch", summary["Best Branch"])

# -------------------------
# Dataset Info
# -------------------------
st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Dataset Preview")
    st.dataframe(preview(filtered, 15), use_container_width=True)

with right:
    st.subheader("Dataset Profile")
    st.json(dataset_profile(filtered))

# -------------------------
# Missing Report
# -------------------------
st.divider()
st.subheader("Missing Value Report")
st.dataframe(missing_report(filtered), use_container_width=True)

# -------------------------
# Insights
# -------------------------
st.divider()
st.subheader("Business Insights")

for item in generate_insights(filtered):
    st.markdown(f"- {item}")

# -------------------------
# Downloads
# -------------------------
st.divider()
st.subheader("Export")

exports = export_dashboard_package(filtered)

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        "⬇ CSV",
        exports["csv"],
        file_name="sales.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        "⬇ Excel",
        exports["excel"],
        file_name="sales.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col3:
    st.download_button(
        "⬇ PDF",
        exports["pdf"],
        file_name="dashboard_summary.pdf",
        mime="application/pdf"
    )

st.success("Application loaded successfully.")
