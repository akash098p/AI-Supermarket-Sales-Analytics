"""
AI-Powered Supermarket Sales Analytics Dashboard
pages/finance.py
Financial performance and profitability analysis page.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.analytics import dashboard_summary, gross_income, tax_summary, total_revenue
from utils.charts import bar, line, waterfall
from utils.config import APP_NAME, PRIMARY, SECONDARY, STYLE_PATH, SUCCESS, WARNING
from utils.data_loader import get_dataset, validate_dataset
from utils.exports import export_dashboard_package
from utils.preprocessing import preprocess


def load_css() -> None:
    """Inject the shared dashboard stylesheet."""
    if STYLE_PATH.exists():
        with open(STYLE_PATH, "r", encoding="utf-8") as handle:
            st.markdown(
                f"<style>{handle.read()}</style>",
                unsafe_allow_html=True,
            )


def _format_currency(value: float) -> str:
    """Format values as currency."""
    return f"₹{value:,.2f}" if pd.notna(value) else "₹0.00"


def _format_ratio(value: float) -> str:
    """Format percentage values."""
    return f"{value:.2f}%" if pd.notna(value) else "0.00%"


def _load_finance_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the active dataset while preserving session state."""
    uploaded = st.session_state.get("finance_upload")
    if uploaded is not None:
        raw_df = get_dataset(uploaded)
        prepared = preprocess(raw_df)
        st.session_state["finance_data"] = prepared
        st.session_state["finance_filtered"] = prepared.copy()
        return prepared, prepared.copy()

    if "finance_data" in st.session_state:
        data = st.session_state["finance_data"]
        filtered = st.session_state.get("finance_filtered", data.copy())
        return data, filtered

    raw_df = get_dataset()
    prepared = preprocess(raw_df)
    st.session_state["finance_data"] = prepared
    st.session_state["finance_filtered"] = prepared.copy()
    return prepared, prepared.copy()


def _prepare_finance_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize financial columns used by the page."""
    prepared = df.copy()
    if "Date" in prepared.columns:
        prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
        prepared = prepared.dropna(subset=["Date"]).reset_index(drop=True)
    if "Total" in prepared.columns:
        prepared["Total"] = pd.to_numeric(prepared["Total"], errors="coerce")
    if "Gross Income" in prepared.columns:
        prepared["Gross Income"] = pd.to_numeric(prepared["Gross Income"], errors="coerce")
    if "Tax" in prepared.columns:
        prepared["Tax"] = pd.to_numeric(prepared["Tax"], errors="coerce")
    if {"Gross Income", "Total"}.issubset(prepared.columns):
        prepared["Profit Margin %"] = (prepared["Gross Income"] / prepared["Total"] * 100).round(2)
    return prepared


def _build_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render sidebar filters and return the active selection state."""
    st.sidebar.markdown("## 🧭 Finance Filters")
    filters: Dict[str, Any] = {}

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        min_date = dates.min()
        max_date = dates.max()
        if pd.notna(min_date) and pd.notna(max_date):
            start_date, end_date = st.sidebar.date_input(
                "Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
            filters["date_start"] = pd.Timestamp(start_date)
            filters["date_end"] = pd.Timestamp(end_date)

    if "Branch" in df.columns:
        branches = sorted(df["Branch"].dropna().astype(str).unique())
        filters["branch"] = st.sidebar.multiselect("Branch", branches, default=branches)

    if "City" in df.columns:
        cities = sorted(df["City"].dropna().astype(str).unique())
        filters["city"] = st.sidebar.multiselect("City", cities, default=cities)

    if "Payment" in df.columns:
        payments = sorted(df["Payment"].dropna().astype(str).unique())
        filters["payment"] = st.sidebar.multiselect("Payment", payments, default=payments)

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"], key="finance_upload")
    if uploaded is not None:
        st.sidebar.success("Dataset updated successfully.")

    return filters


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply sidebar filters to the active dataset."""
    filtered = df.copy()

    if "date_start" in filters and "date_end" in filters and "Date" in filtered.columns:
        filtered = filtered[(filtered["Date"] >= filters["date_start"]) & (filtered["Date"] <= filters["date_end"])]

    if "branch" in filters and "Branch" in filtered.columns:
        if filters["branch"]:
            filtered = filtered[filtered["Branch"].astype(str).isin(filters["branch"])]

    if "city" in filters and "City" in filtered.columns:
        if filters["city"]:
            filtered = filtered[filtered["City"].astype(str).isin(filters["city"])]

    if "payment" in filters and "Payment" in filtered.columns:
        if filters["payment"]:
            filtered = filtered[filtered["Payment"].astype(str).isin(filters["payment"])]

    return filtered.reset_index(drop=True)


def _render_hero(df: pd.DataFrame) -> None:
    """Render the hero section for the finance page."""
    summary = dashboard_summary(df)
    tax = tax_summary(df)
    today = datetime.now().strftime("%d %b %Y")

    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    left, right = st.columns([2.2, 1.0])

    with left:
        st.markdown("<h1 style='margin-bottom:0.2rem;'>💹 Financial Performance Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.02rem; margin-top:0.25rem; opacity:0.95;'>Track revenue generation, gross income, tax burden, and profitability with executive-grade financial analysis.</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.45rem;'>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🗓️ {today}</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>📊 {len(df):,} transactions</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>💵 {tax.get('Total Tax', 0):,.0f} tax</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.metric("Revenue", _format_currency(summary["Total Revenue"]), help="Gross revenue")
        st.metric("Gross Income", _format_currency(summary["Gross Income"]), help="Gross profit")
        st.metric("Orders", f"{summary['Total Orders']:,}", help="Transaction count")
        st.metric("Profit Margin", _format_ratio(df["Profit Margin %"].mean() if "Profit Margin %" in df.columns else 0), help="Average profit margin")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_kpi_cards(df: pd.DataFrame) -> None:
    """Render executive KPI cards for finance performance."""
    summary = dashboard_summary(df)
    tax = tax_summary(df)

    metrics: List[Dict[str, Any]] = [
        {"title": "Revenue", "value": _format_currency(summary["Total Revenue"]), "icon": "💰", "accent": PRIMARY},
        {"title": "COGS", "value": _format_currency(summary["Total Revenue"] - summary["Gross Income"]), "icon": "📦", "accent": SECONDARY},
        {"title": "Tax", "value": _format_currency(tax.get("Total Tax", 0.0)), "icon": "🧾", "accent": WARNING},
        {"title": "Gross Income", "value": _format_currency(summary["Gross Income"]), "icon": "📈", "accent": SUCCESS},
        {"title": "Profit Margin", "value": _format_ratio(df["Profit Margin %"].mean() if "Profit Margin %" in df.columns else 0), "icon": "📊", "accent": PRIMARY},
        {"title": "Avg Order", "value": _format_currency(summary["Average Order Value"]), "icon": "🧮", "accent": WARNING},
    ]

    cols = st.columns(3)
    for idx, metric in enumerate(metrics):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left:6px solid {metric['accent']}; margin-bottom:1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div><div class="kpi-title">{metric['title']}</div><div class="kpi-value">{metric['value']}</div></div>
                        <div style="font-size:1.7rem;">{metric['icon']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _build_finance_sections(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare finance tables for charts and summary output."""
    frames: Dict[str, pd.DataFrame] = {}

    if "Date" in df.columns and "Total" in df.columns:
        monthly = (
            df.assign(Month=df["Date"].dt.to_period("M"))
            .groupby("Month", as_index=False)["Total"]
            .sum()
            .rename(columns={"Total": "Revenue"})
        )
        monthly["Month"] = monthly["Month"].astype(str)
        frames["monthly_revenue"] = monthly.sort_values("Month").reset_index(drop=True)
    else:
        frames["monthly_revenue"] = pd.DataFrame(columns=["Month", "Revenue"])

    if {"Date", "Gross Income"}.issubset(df.columns):
        monthly_profit = (
            df.assign(Month=df["Date"].dt.to_period("M"))
            .groupby("Month", as_index=False)["Gross Income"]
            .sum()
            .rename(columns={"Gross Income": "Gross Income"})
        )
        monthly_profit["Month"] = monthly_profit["Month"].astype(str)
        frames["monthly_profit"] = monthly_profit.sort_values("Month").reset_index(drop=True)
    else:
        frames["monthly_profit"] = pd.DataFrame(columns=["Month", "Gross Income"])

    if "Tax" in df.columns:
        frames["tax_summary"] = pd.DataFrame(
            {
                "Metric": ["Total Tax", "Average Tax", "Max Tax"],
                "Value": [df["Tax"].sum(), df["Tax"].mean(), df["Tax"].max()],
            }
        )
    else:
        frames["tax_summary"] = pd.DataFrame(columns=["Metric", "Value"])

    if {"Total", "Gross Income", "Tax"}.issubset(df.columns):
        frames["financial_snapshot"] = pd.DataFrame(
            {
                "Metric": ["Revenue", "COGS", "Tax", "Gross Income"],
                "Value": [df["Total"].sum(), max(df["Total"].sum() - df["Gross Income"].sum(), 0), df["Tax"].sum(), df["Gross Income"].sum()],
            }
        )
    else:
        frames["financial_snapshot"] = pd.DataFrame(columns=["Metric", "Value"])

    return frames


def _render_charts(df: pd.DataFrame) -> None:
    """Render financial KPIs with Plotly charts."""
    frames = _build_finance_sections(df)

    st.markdown("### 📈 Financial Analytics")
    left, right = st.columns(2)

    with left:
        if not frames["monthly_revenue"].empty:
            chart = line(frames["monthly_revenue"], "Month", "Revenue", "Monthly Revenue")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Monthly revenue chart unavailable.")

    with right:
        if not frames["monthly_profit"].empty:
            chart = line(frames["monthly_profit"], "Month", "Gross Income", "Monthly Gross Income")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Monthly profit chart unavailable.")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        if not frames["financial_snapshot"].empty:
            chart = bar(frames["financial_snapshot"], "Metric", "Value", "Financial Snapshot")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Financial snapshot unavailable.")

    with lower_right:
        if not frames["tax_summary"].empty:
            chart = bar(frames["tax_summary"], "Metric", "Value", "Tax Breakdown")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Tax breakdown unavailable.")

    st.markdown("### 🌊 Profitability Flow")
    if not frames["financial_snapshot"].empty:
        values = [float(value) for value in frames["financial_snapshot"]["Value"]]
        labels = frames["financial_snapshot"]["Metric"].tolist()
        fig = waterfall(labels, values, "Profitability Waterfall")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waterfall chart unavailable.")


def _render_finance_tables(df: pd.DataFrame) -> None:
    """Render financial summary tables."""
    frames = _build_finance_sections(df)

    st.markdown("### 🧾 Finance Tables")
    if not frames["tax_summary"].empty:
        st.markdown("#### Tax Summary")
        st.dataframe(frames["tax_summary"], use_container_width=True, hide_index=True)

    if not frames["financial_snapshot"].empty:
        st.markdown("#### Financial Snapshot")
        st.dataframe(frames["financial_snapshot"], use_container_width=True, hide_index=True)


def _render_insights(df: pd.DataFrame) -> None:
    """Render financial insights cards."""
    st.markdown("### 🧠 Financial Insights")
    summary = dashboard_summary(df)
    tax = tax_summary(df)

    insight_cards = [
        f"Revenue reached {_format_currency(summary['Total Revenue'])} with gross income of {_format_currency(summary['Gross Income'])}.",
        f"The current profitability profile includes total tax of {_format_currency(tax.get('Total Tax', 0.0))} and an average margin of {_format_ratio(df['Profit Margin %'].mean() if 'Profit Margin %' in df.columns else 0)}.",
        f"The business remains operationally healthy when gross income outpaces the combined cost and tax load.",
    ]

    for card in insight_cards:
        st.markdown(f"<div class='insight-card'>{card}</div>", unsafe_allow_html=True)


def _render_exports(df: pd.DataFrame) -> None:
    """Render export buttons for the finance view."""
    st.markdown("### ⬇️ Export Finance View")
    exports = export_dashboard_package(df)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button("Download CSV", exports["csv"], file_name="finance_view.csv", mime="text/csv")
    with c2:
        st.download_button("Download Excel", exports["excel"], file_name="finance_view.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Download PDF", exports["pdf"], file_name="finance_view_summary.pdf", mime="application/pdf")


def render_finance_page() -> None:
    """Render the complete finance analytics experience."""
    st.set_page_config(page_title=APP_NAME, page_icon="💹", layout="wide", initial_sidebar_state="expanded")
    load_css()

    data, filtered_data = _load_finance_data()
    if data.empty:
        st.error("The active dataset is empty.")
        st.stop()

    issues = validate_dataset(data)
    if issues:
        with st.expander("⚠ Dataset Validation"):
            for issue in issues:
                st.warning(issue)

    filters = _build_filters(data)
    filtered_data = _apply_filters(data, filters)
    st.session_state["finance_filtered"] = filtered_data

    if filtered_data.empty:
        st.warning("No rows match the selected filters. Please broaden the selection.")
        st.stop()

    with st.spinner("Preparing finance analytics..."):
        _render_hero(filtered_data)
        st.markdown("---")
        _render_kpi_cards(filtered_data)
        st.markdown("---")
        _render_charts(filtered_data)
        st.markdown("---")
        _render_finance_tables(filtered_data)
        st.markdown("---")
        _render_insights(filtered_data)
        st.markdown("---")
        _render_exports(filtered_data)


render_finance_page()
