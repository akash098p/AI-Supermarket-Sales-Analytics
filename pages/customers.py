

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.analytics import (
    average_order_value,
    average_rating,
    dashboard_summary,
    gross_income,
    total_customers,
    total_orders,
    total_products_sold,
    total_revenue,
)
from utils.charts import bar, box, bubble, donut, histogram, pie, scatter, violin
from utils.config import APP_NAME, PRIMARY, SECONDARY, STYLE_PATH, SUCCESS, WARNING
from utils.data_loader import get_dataset, load_page_dataset, validate_dataset
from utils.exports import export_dashboard_package
from utils.page_helpers import render_page_skeleton
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


def _format_number(value: Any) -> str:
    """Format numeric values for display."""
    if pd.isna(value):
        return "0"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    return str(value)


def _load_customer_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the active dataset while preserving session state."""
    uploaded = st.session_state.get("customers_upload")
    return load_page_dataset(
        "customers",
        lambda raw_df: preprocess(raw_df),
        uploaded_file=uploaded,
    )


def _prepare_customer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize customer-related fields used by the page."""
    prepared = df.copy()
    if "Customer ID" not in prepared.columns and "Customer Name" in prepared.columns:
        prepared["Customer ID"] = prepared["Customer Name"]
    if "Customer Type" not in prepared.columns:
        prepared["Customer Type"] = "Unknown"
    if "Gender" not in prepared.columns:
        prepared["Gender"] = "Unknown"
    if "Payment" not in prepared.columns:
        prepared["Payment"] = "Unknown"
    if "Rating" in prepared.columns:
        prepared["Rating"] = pd.to_numeric(prepared["Rating"], errors="coerce")
    if "Total" in prepared.columns:
        prepared["Total"] = pd.to_numeric(prepared["Total"], errors="coerce")
    if "Quantity" in prepared.columns:
        prepared["Quantity"] = pd.to_numeric(prepared["Quantity"], errors="coerce")
    return prepared


def _build_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render the sidebar filters for the customer page."""
    st.sidebar.markdown("## 🧭 Customer Filters")
    filters: Dict[str, Any] = {}

    if "Branch" in df.columns:
        branches = sorted(df["Branch"].dropna().astype(str).unique())
        filters["branch"] = st.sidebar.multiselect("Branch", branches, default=branches)

    if "City" in df.columns:
        cities = sorted(df["City"].dropna().astype(str).unique())
        filters["city"] = st.sidebar.multiselect("City", cities, default=cities)

    if "Customer Type" in df.columns:
        customer_types = sorted(df["Customer Type"].dropna().astype(str).unique())
        filters["customer_type"] = st.sidebar.multiselect("Customer Type", customer_types, default=customer_types)

    if "Gender" in df.columns:
        genders = sorted(df["Gender"].dropna().astype(str).unique())
        filters["gender"] = st.sidebar.multiselect("Gender", genders, default=genders)

    if "Payment" in df.columns:
        payments = sorted(df["Payment"].dropna().astype(str).unique())
        filters["payment"] = st.sidebar.multiselect("Payment", payments, default=payments)

    if "Rating" in df.columns:
        rating_min, rating_max = float(df["Rating"].min()), float(df["Rating"].max())
        filters["rating"] = st.sidebar.slider("Rating", min_value=float(rating_min), max_value=float(rating_max), value=(float(rating_min), float(rating_max)), step=0.1)

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"], key="customers_upload")
    if uploaded is not None:
        st.sidebar.success("Dataset updated successfully.")

    return filters


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply sidebar filters to the active dataset."""
    filtered = df.copy()

    if "branch" in filters and "Branch" in filtered.columns:
        if filters["branch"]:
            filtered = filtered[filtered["Branch"].astype(str).isin(filters["branch"])]

    if "city" in filters and "City" in filtered.columns:
        if filters["city"]:
            filtered = filtered[filtered["City"].astype(str).isin(filters["city"])]

    if "customer_type" in filters and "Customer Type" in filtered.columns:
        if filters["customer_type"]:
            filtered = filtered[filtered["Customer Type"].astype(str).isin(filters["customer_type"])]

    if "gender" in filters and "Gender" in filtered.columns:
        if filters["gender"]:
            filtered = filtered[filtered["Gender"].astype(str).isin(filters["gender"])]

    if "payment" in filters and "Payment" in filtered.columns:
        if filters["payment"]:
            filtered = filtered[filtered["Payment"].astype(str).isin(filters["payment"])]

    if "rating" in filters and "Rating" in filtered.columns:
        low, high = filters["rating"]
        filtered = filtered[(filtered["Rating"] >= low) & (filtered["Rating"] <= high)]

    return filtered.reset_index(drop=True)


def _render_hero(df: pd.DataFrame) -> None:
    """Render the hero section for the customer page."""
    summary = dashboard_summary(df)
    today = datetime.now().strftime("%d %b %Y")

    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    left, right = st.columns([2.2, 1.0])

    with left:
        st.markdown("<h1 style='margin-bottom:0.2rem;'>👥 Customer Intelligence Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.02rem; margin-top:0.25rem; opacity:0.95;'>Understand customer mix, channel behavior, and segmentation patterns across the retail footprint.</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.45rem;'>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🗓️ {today}</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>👤 {summary['Total Customers']:,} customers</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>⭐ {summary['Average Rating']:.1f}/10</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.metric("Revenue", _format_currency(summary["Total Revenue"]), help="Customer contribution to revenue")
        st.metric("Orders", f"{summary['Total Orders']:,}", help="Total transactions")
        st.metric("Avg Order", _format_currency(summary["Average Order Value"]), help="Average basket value")
        st.metric("Preferred Payment", summary["Preferred Payment"], help="Most used payment method")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_kpi_cards(df: pd.DataFrame) -> None:
    """Render executive KPI cards for customer analysis."""
    summary = dashboard_summary(df)
    customer_types = df.groupby("Customer Type")["Total"].sum() if "Customer Type" in df.columns else pd.Series(dtype=float)

    metrics: List[Dict[str, Any]] = [
        {"title": "Revenue", "value": _format_currency(summary["Total Revenue"]), "icon": "💰", "accent": PRIMARY},
        {"title": "Customers", "value": f"{summary['Total Customers']:,}", "icon": "👥", "accent": SECONDARY},
        {"title": "Orders", "value": f"{summary['Total Orders']:,}", "icon": "🧾", "accent": SUCCESS},
        {"title": "Avg Order", "value": _format_currency(summary["Average Order Value"]), "icon": "🧮", "accent": WARNING},
    ]

    cols = st.columns(4)
    for idx, metric in enumerate(metrics):
        with cols[idx % 4]:
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


def _build_customer_sections(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare customer segmentation and behavior aggregates."""
    frames: Dict[str, pd.DataFrame] = {}

    if "Gender" in df.columns:
        frames["gender"] = df.groupby("Gender", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
    else:
        frames["gender"] = pd.DataFrame(columns=["Gender", "Total"])

    if "Customer Type" in df.columns:
        frames["customer_type"] = df.groupby("Customer Type", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
    else:
        frames["customer_type"] = pd.DataFrame(columns=["Customer Type", "Total"])

    if "Payment" in df.columns:
        frames["payment"] = df.groupby("Payment", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
    else:
        frames["payment"] = pd.DataFrame(columns=["Payment", "Total"])

    if {"Customer ID", "Total"}.issubset(df.columns):
        customer_summary = (
            df.groupby("Customer ID", as_index=False)
            .agg(revenue=("Total", "sum"), orders=("Total", "size"), rating=("Rating", "mean"), quantity=("Quantity", "sum"))
        )
        frames["customer_ltv"] = customer_summary.sort_values("revenue", ascending=False).head(20)
    else:
        frames["customer_ltv"] = pd.DataFrame(columns=["Customer ID", "revenue", "orders", "rating", "quantity"])

    return frames


def _render_charts(df: pd.DataFrame) -> None:
    """Render customer distribution and segmentation charts."""
    frames = _build_customer_sections(df)

    st.markdown("### 📊 Customer Distribution")
    left, right = st.columns(2)

    with left:
        if not frames["gender"].empty:
            chart = pie(frames["gender"], "Gender", "Total", "Gender Distribution")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Gender chart unavailable.")

    with right:
        if not frames["customer_type"].empty:
            chart = donut(frames["customer_type"], "Customer Type", "Total", "Customer Type Distribution")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Customer type chart unavailable.")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        if not frames["payment"].empty:
            chart = bar(frames["payment"], "Payment", "Total", "Payment Preference", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Payment chart unavailable.")

    with lower_right:
        if not frames["customer_ltv"].empty:
            chart = bubble(frames["customer_ltv"], "orders", "revenue", "quantity", "rating", "Customer Revenue vs Frequency")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Customer LTV chart unavailable.")

    st.markdown("### 🧠 Segmentation Analysis")
    seg_left, seg_right = st.columns(2)
    with seg_left:
        if {"Customer ID", "Total", "Rating", "Quantity"}.issubset(df.columns):
            seg_df = df.groupby("Customer ID", as_index=False).agg(revenue=("Total", "sum"), rating=("Rating", "mean"), orders=("Total", "size"), quantity=("Quantity", "sum"))
            if len(seg_df) >= 3:
                features = seg_df[["revenue", "rating", "orders", "quantity"]].fillna(0)
                scaled = StandardScaler().fit_transform(features)
                labels = KMeans(n_clusters=min(3, len(seg_df)), random_state=42, n_init=10).fit_predict(scaled)
                seg_df["Segment"] = labels
                fig = px.scatter(seg_df, x="orders", y="revenue", size="quantity", color="Segment", hover_name="Customer ID")
                fig.update_layout(title="KMeans Customer Segments", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough customer records for segmentation.")
        else:
            st.info("Customer segmentation unavailable.")

    with seg_right:
        if "Rating" in df.columns and "Total" in df.columns:
            chart = scatter(df[["Rating", "Total"]].dropna(), "Rating", "Total", "Revenue vs Rating")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Revenue vs rating chart unavailable.")

    st.markdown("### 📈 Behavioral Insights")
    tab_left, tab_right = st.columns(2)
    with tab_left:
        if "Rating" in df.columns:
            chart = histogram(df, "Rating", "Customer Rating Distribution")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Rating histogram unavailable.")

    with tab_right:
        if "Quantity" in df.columns:
            chart = box(df, "Customer Type", "Quantity", "Quantity Spread by Customer Type") if "Customer Type" in df.columns else None
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("Box plot unavailable.")
        else:
            st.info("Box plot unavailable.")


def _render_customer_tables(df: pd.DataFrame) -> None:
    """Render customer summary tables."""
    frames = _build_customer_sections(df)

    st.markdown("### 🧾 Customer Tables")
    if not frames["customer_ltv"].empty:
        st.markdown("#### Top Customer Revenue Contributors")
        st.dataframe(frames["customer_ltv"].head(10), use_container_width=True, hide_index=True)

    if not frames["customer_type"].empty:
        st.markdown("#### Customer Type Revenue")
        st.dataframe(frames["customer_type"], use_container_width=True, hide_index=True)

    if not frames["payment"].empty:
        st.markdown("#### Payment Preferences")
        st.dataframe(frames["payment"], use_container_width=True, hide_index=True)


def _render_insights(df: pd.DataFrame) -> None:
    """Render customer insights cards."""
    st.markdown("### 🧠 Customer Insights")
    summary = dashboard_summary(df)

    insight_cards = [
        f"The customer base contributes { _format_currency(summary['Total Revenue']) } in revenue across {summary['Total Orders']:,} transactions.",
        f"Payment behavior is led by {summary['Preferred Payment']} and the average satisfaction score is {summary['Average Rating']:.1f}/10.",
        f"The current mix shows strong potential for segmentation-driven loyalty and retention strategy.",
    ]

    for card in insight_cards:
        st.markdown(f"<div class='insight-card'>{card}</div>", unsafe_allow_html=True)


def _render_exports(df: pd.DataFrame) -> None:
    """Render export buttons for the customer view."""
    st.markdown("### ⬇️ Export Customer View")
    exports = export_dashboard_package(df)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button("Download CSV", exports["csv"], file_name="customer_view.csv", mime="text/csv")
    with c2:
        st.download_button("Download Excel", exports["excel"], file_name="customer_view.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Download PDF", exports["pdf"], file_name="customer_view_summary.pdf", mime="application/pdf")


def render_customers_page() -> None:
    """Render the complete customer analytics experience."""
    st.set_page_config(page_title=APP_NAME, page_icon="👥", layout="wide", initial_sidebar_state="expanded")
    load_css()
    skeleton = st.empty()
    with skeleton.container():
        render_page_skeleton(
            "Customer Intelligence Dashboard",
            "Understand customer mix, channel behavior, and segmentation patterns across the retail footprint.",
            metric_count=4,
            section_titles=["Customer Distribution", "Segmentation Analysis"],
            chart_cards=4,
            table_cards=1,
            insight_cards=3,
        )

    data, filtered_data = _load_customer_data()
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
    st.session_state["customers_filtered"] = filtered_data

    if filtered_data.empty:
        st.warning("No rows match the selected filters. Please broaden the selection.")
        st.stop()

    with skeleton.container():
        with st.spinner("Preparing customer analytics..."):
            _render_hero(filtered_data)
            st.markdown("---")
            _render_kpi_cards(filtered_data)
            st.markdown("---")
            _render_charts(filtered_data)
            st.markdown("---")
            _render_customer_tables(filtered_data)
            st.markdown("---")
            _render_insights(filtered_data)
            st.markdown("---")
            _render_exports(filtered_data)


render_customers_page()
