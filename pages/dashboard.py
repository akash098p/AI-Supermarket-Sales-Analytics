

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.analytics import (
    average_order_value,
    average_rating,
    average_rating_by,
    correlation_matrix,
    customer_type_summary,
    dashboard_summary,
    gender_summary,
    gross_income,
    highest_sales_branch,
    hourly_sales,
    monthly_sales,
    quantity_by,
    revenue_by,
    tax_summary,
    top_products,
    total_customers,
    total_orders,
    total_products_sold,
    total_revenue,
    weekday_sales,
)
from utils.charts import (
    actual_vs_predicted,
    area,
    bar,
    box,
    bubble,
    donut,
    funnel,
    gauge,
    heatmap,
    histogram,
    kpi,
    line,
    pie,
    scatter,
    sunburst,
    treemap,
    violin,
    waterfall,
)
from utils.config import (
    APP_NAME,
    BACKGROUND,
    CHART_HEIGHT,
    DANGER,
    PRIMARY,
    SECONDARY,
    STYLE_PATH,
    SUCCESS,
    WARNING,
)
from utils.data_loader import (
    dataset_profile,
    get_dataset,
    missing_report,
    preview,
    validate_dataset,
)
from utils.exports import export_dashboard_package
from utils.insights import generate_insights
from utils.page_helpers import render_profile_summary
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
    """Format numeric values as currency."""
    return f"₹{value:,.2f}" if pd.notna(value) else "₹0.00"


def _format_number(value: Any) -> str:
    """Format numeric values for compact display."""
    if pd.isna(value):
        return "0"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    return str(value)


def _format_ratio(value: float) -> str:
    """Format percentage-like values."""
    return f"{value:.1f}%" if pd.notna(value) else "0.0%"


def _render_hero_header(df: pd.DataFrame) -> None:
    """Render the hero banner with metadata and quick overview."""
    summary = dashboard_summary(df)
    today = datetime.now().strftime("%d %b %Y")
    profile = dataset_profile(df)

    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    col_left, col_right = st.columns([2.3, 1.0])

    with col_left:
        st.markdown(
            "<h1 style='margin-bottom:0.2rem;'>📊 AI-Powered Supermarket Sales Analytics Dashboard</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:1.02rem; margin-top:0.25rem; opacity:0.95;'>"
            "Executive intelligence layer for branch performance, sales health, customer behavior, and financial outlook.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.45rem;'>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🗓️ {today}</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>📦 {profile['rows']:,} transactions</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🏬 {summary['Best Branch']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_right:
        st.metric("Revenue", _format_currency(summary["Total Revenue"]), help="Gross sales value")
        st.metric("Orders", f"{summary['Total Orders']:,}", help="Total transactions")
        st.metric("Customers", f"{summary['Total Customers']:,}", help="Unique customer count")
        st.metric("Rating", f"{summary['Average Rating']:.1f}/10", help="Average customer rating")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_kpi_cards(df: pd.DataFrame) -> None:
    """Render eight KPI cards with executive summary metrics."""
    summary = dashboard_summary(df)
    metrics: List[Dict[str, Any]] = [
        {"title": "Total Revenue", "value": _format_currency(summary["Total Revenue"]), "icon": "💰", "accent": PRIMARY},
        {"title": "Orders", "value": f"{summary['Total Orders']:,}", "icon": "🧾", "accent": SECONDARY},
        {"title": "Products Sold", "value": f"{summary['Products Sold']:,}", "icon": "📦", "accent": SUCCESS},
        {"title": "Customers", "value": f"{summary['Total Customers']:,}", "icon": "👥", "accent": WARNING},
        {"title": "Gross Income", "value": _format_currency(summary["Gross Income"]), "icon": "📈", "accent": PRIMARY},
        {"title": "Average Order", "value": _format_currency(summary["Average Order Value"]), "icon": "🧮", "accent": SECONDARY},
        {"title": "Average Rating", "value": f"{summary['Average Rating']:.1f}/10", "icon": "⭐", "accent": SUCCESS},
        {"title": "Best Branch", "value": summary["Best Branch"], "icon": "🏬", "accent": WARNING},
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


def _render_dataset_overview(df: pd.DataFrame) -> None:
    """Render preview, profile, and missing value diagnostics."""
    profile = dataset_profile(df)
    missing = missing_report(df)
    preview_df = preview(df, 12)

    st.markdown("### 🔎 Dataset Intelligence")
    left, right = st.columns([1.5, 1.0])

    with left:
        st.markdown("#### Recent Transactions")
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Dataset Profile")
        render_profile_summary(profile)
        st.caption("Quality and reliability snapshot for the active filtered view")

    st.markdown("#### Missing Values")
    st.dataframe(missing, use_container_width=True, hide_index=True)


def _render_exec_summary(df: pd.DataFrame) -> None:
    """Render an executive summary and business insights."""
    summary = dashboard_summary(df)
    insights = generate_insights(df)

    st.markdown("### 🧠 Executive Summary")

    left, right = st.columns([1.3, 1.0])

    with left:
        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Business Pulse</strong><br>
                Revenue is currently tracking at <strong>{_format_currency(summary['Total Revenue'])}</strong> across <strong>{summary['Total Orders']:,}</strong> orders.
                The strongest branch in the current view is <strong>{summary['Best Branch']}</strong>, while the average rating stands at <strong>{summary['Average Rating']:.1f}/10</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="success-card">
                <strong>Operational Highlights</strong><br>
                Customers purchased <strong>{summary['Products Sold']:,}</strong> units, with an average basket value of <strong>{_format_currency(summary['Average Order Value'])}</strong>.
                The leading payment mode is <strong>{summary['Preferred Payment']}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        for item in insights:
            st.markdown(f"<div class='insight-card'>{item}</div>", unsafe_allow_html=True)


def _render_health_snapshot(df: pd.DataFrame) -> None:
    """Render quality, segment, and coverage diagnostics."""
    st.markdown("### 🏥 Data Health & Segmentation")

    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    quality_score = max(100 - (missing_count * 0.8) - (duplicate_count * 1.2), 0)

    left, middle, right = st.columns(3)
    with left:
        st.plotly_chart(gauge(quality_score, "Data Quality Score", 0, 100), use_container_width=True)

    with middle:
        if {"Branch", "City", "Total"}.issubset(df.columns):
            segment_df = df.groupby(["Branch", "City"], as_index=False)["Total"].sum()
            chart = treemap(segment_df, path=["Branch", "City"], values="Total", title="Branch-to-City Revenue Mix")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Branch and city segmentation is not available for the current view.")

    with right:
        if {"Branch", "Product", "Total"}.issubset(df.columns):
            branch_product_df = df.groupby(["Branch", "Product"], as_index=False)["Total"].sum()
            chart = sunburst(branch_product_df, path=["Branch", "Product"], values="Total", title="Branch × Product Revenue")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Product segmentation is not available for the current view.")

    st.caption(f"Quality score is derived from missing values, duplicates, and overall dataset completeness. Missing values: {missing_count}; Duplicates: {duplicate_count}.")


def _render_charts(df: pd.DataFrame) -> None:
    """Render the main chart suite with Plotly visuals."""
    if df.empty:
        st.info("No data available for the selected filters.")
        return

    st.markdown("### 📈 Performance Analytics")

    top_left, top_right = st.columns(2)
    with top_left:
        monthly = monthly_sales(df)
        if not monthly.empty:
            chart = line(monthly, "Month Name", "Total", "Monthly Sales Trend")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Monthly sales chart unavailable for the current dataset.")

    with top_right:
        weekly = weekday_sales(df)
        if not weekly.empty:
            chart = bar(weekly, "Weekday", "Total", "Revenue by Weekday", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Weekly sales chart unavailable for the current dataset.")

    mid_left, mid_right = st.columns(2)
    with mid_left:
        branch_data = revenue_by(df, "Branch")
        if not branch_data.empty:
            chart = bar(branch_data.head(8), "Branch", "Total", "Top Branch Revenue", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Branch chart unavailable.")

    with mid_right:
        city_data = revenue_by(df, "City")
        if not city_data.empty:
            chart = bar(city_data.head(8), "City", "Total", "Top City Revenue", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("City chart unavailable.")

    lower_left, lower_middle, lower_right = st.columns(3)
    with lower_left:
        product_col = "Product" if "Product" in df.columns else "Product Line"
        product_data = quantity_by(df, product_col)
        if not product_data.empty:
            chart = bar(product_data.head(10), product_col, "Quantity", "Top Products by Quantity", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Product chart unavailable.")

    with lower_middle:
        if "Payment" in df.columns:
            payment_data = df.groupby("Payment", as_index=False)["Total"].sum()
            chart = donut(payment_data, "Payment", "Total", "Payment Distribution")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Payment distribution unavailable.")

    with lower_right:
        if "Gender" in df.columns:
            gender_data = gender_summary(df)
            chart = pie(gender_data, "Gender", "Total", "Gender Mix")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Gender distribution unavailable.")

    st.markdown("### 🌐 Channel and Customer Segmentation")
    left, right = st.columns(2)
    with left:
        if "Customer Type" in df.columns:
            customer_data = customer_type_summary(df)
            chart = bar(customer_data, "Customer Type", "Total", "Revenue by Customer Type")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Customer type analysis unavailable.")

    with right:
        if {"Rating", "Total"}.issubset(df.columns):
            rating_data = df[["Rating", "Total"]].copy()
            chart = scatter(rating_data, "Rating", "Total", "Rating vs Revenue")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Rating scatter plot unavailable.")

    st.markdown("### 🔬 Analytical Deep Dive")
    tab_one, tab_two, tab_three = st.tabs(["Correlation", "Distribution", "Financials"])

    with tab_one:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr = correlation_matrix(df)
            chart = heatmap(corr, "Correlation Heatmap")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Correlation heatmap unavailable without numeric fields.")

    with tab_two:
        if "Rating" in df.columns:
            rating_chart = histogram(df, "Rating", "Customer Rating Distribution")
            st.plotly_chart(rating_chart, use_container_width=True)

        if "Quantity" in df.columns:
            quantity_chart = histogram(df, "Quantity", "Quantity Distribution")
            st.plotly_chart(quantity_chart, use_container_width=True)

        if "Total" in df.columns and "Branch" in df.columns:
            box_chart = box(df, "Branch", "Total", "Revenue Spread by Branch")
            st.plotly_chart(box_chart, use_container_width=True)

    with tab_three:
        if {"Total", "Gross Income", "Tax"}.issubset(df.columns):
            finance_summary = tax_summary(df)
            st.metric("Total Tax", _format_currency(finance_summary.get("Total Tax", 0.0)))
            st.metric("Average Tax", _format_currency(finance_summary.get("Average Tax", 0.0)))
            st.metric("Max Tax", _format_currency(finance_summary.get("Maximum Tax", 0.0)))

            finance_frame = pd.DataFrame(
                {
                    "Metric": ["Revenue", "Gross Income", "Tax"],
                    "Value": [total_revenue(df), gross_income(df), finance_summary.get("Total Tax", 0.0)],
                }
            )
            chart = bar(finance_frame, "Metric", "Value", "Financial Snapshot")
            st.plotly_chart(chart, use_container_width=True)


def _render_recent_transactions(df: pd.DataFrame) -> None:
    """Render a recent transactions table and action cards."""
    st.markdown("### 🧾 Recent Transactions")
    recent = df.sort_values(by="Date", ascending=False).head(15).copy() if "Date" in df.columns else df.head(15).copy()

    if not recent.empty:
        display = recent[[
            c for c in ["Invoice ID", "Date", "Branch", "City", "Product", "Product Line", "Quantity", "Total", "Gross Income", "Rating", "Payment"]
            if c in recent.columns
        ]].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No transaction history is available in the current view.")

    st.markdown("#### Operational Snapshot")
    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Highest Revenue Branch", highest_sales_branch(df))
    with info_cols[1]:
        st.metric("Average Order Value", _format_currency(average_order_value(df)))
    with info_cols[2]:
        st.metric("Average Rating", f"{average_rating(df):.1f}/10")
    with info_cols[3]:
        st.metric("Total Products Sold", f"{total_products_sold(df):,}")


def _render_export_section(df: pd.DataFrame) -> None:
    """Render export actions for CSV, Excel, and PDF."""
    st.markdown("### ⬇️ Export & Share")
    exports = export_dashboard_package(df)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "Download CSV",
            exports["csv"],
            file_name="supermarket_dashboard.csv",
            mime="text/csv",
        )

    with c2:
        st.download_button(
            "Download Excel",
            exports["excel"],
            file_name="supermarket_dashboard.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with c3:
        st.download_button(
            "Download PDF",
            exports["pdf"],
            file_name="supermarket_dashboard_summary.pdf",
            mime="application/pdf",
        )


def _render_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render the sidebar filter controls and return their current values."""
    st.sidebar.markdown("## 🧭 Filters")
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
        filters["branch"] = st.sidebar.multiselect(
            "Branch",
            branches,
            default=branches,
            help="Filter by branch",
        )

    if "City" in df.columns:
        cities = sorted(df["City"].dropna().astype(str).unique())
        filters["city"] = st.sidebar.multiselect(
            "City",
            cities,
            default=cities,
            help="Filter by city",
        )

    product_col = "Product" if "Product" in df.columns else "Product Line"
    if product_col in df.columns:
        products = sorted(df[product_col].dropna().astype(str).unique())
        filters["product"] = st.sidebar.multiselect(
            product_col,
            products,
            default=products,
            help="Filter by product",
        )

    if "Customer Type" in df.columns:
        customer_types = sorted(df["Customer Type"].dropna().astype(str).unique())
        filters["customer_type"] = st.sidebar.multiselect(
            "Customer Type",
            customer_types,
            default=customer_types,
            help="Filter by customer type",
        )

    if "Payment" in df.columns:
        payments = sorted(df["Payment"].dropna().astype(str).unique())
        filters["payment"] = st.sidebar.multiselect(
            "Payment",
            payments,
            default=payments,
            help="Filter by payment method",
        )

    if "Gender" in df.columns:
        genders = sorted(df["Gender"].dropna().astype(str).unique())
        filters["gender"] = st.sidebar.multiselect(
            "Gender",
            genders,
            default=genders,
            help="Filter by customer gender",
        )

    if "Rating" in df.columns:
        rating_min, rating_max = float(df["Rating"].min()), float(df["Rating"].max())
        filters["rating"] = st.sidebar.slider(
            "Rating",
            min_value=float(rating_min),
            max_value=float(rating_max),
            value=(float(rating_min), float(rating_max)),
            step=0.1,
        )

    if "Quantity" in df.columns:
        qty_min, qty_max = int(df["Quantity"].min()), int(df["Quantity"].max())
        filters["quantity"] = st.sidebar.slider(
            "Quantity",
            min_value=qty_min,
            max_value=qty_max,
            value=(qty_min, qty_max),
            step=1,
        )

    return filters


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply sidebar filters to the active dataset."""
    filtered = df.copy()

    if "date_start" in filters and "date_end" in filters:
        filtered = filtered[
            filtered["Date"].between(filters["date_start"], filters["date_end"])
        ] if "Date" in filtered.columns else filtered

    if "branch" in filters and filtered is not None and "Branch" in filtered.columns:
        if filters["branch"]:
            filtered = filtered[filtered["Branch"].astype(str).isin(filters["branch"])]

    if "city" in filters and filtered is not None and "City" in filtered.columns:
        if filters["city"]:
            filtered = filtered[filtered["City"].astype(str).isin(filters["city"])]

    product_col = "Product" if "Product" in filtered.columns else "Product Line"
    if "product" in filters and filtered is not None and product_col in filtered.columns:
        if filters["product"]:
            filtered = filtered[filtered[product_col].astype(str).isin(filters["product"])]

    if "customer_type" in filters and filtered is not None and "Customer Type" in filtered.columns:
        if filters["customer_type"]:
            filtered = filtered[filtered["Customer Type"].astype(str).isin(filters["customer_type"])]

    if "payment" in filters and filtered is not None and "Payment" in filtered.columns:
        if filters["payment"]:
            filtered = filtered[filtered["Payment"].astype(str).isin(filters["payment"])]

    if "gender" in filters and filtered is not None and "Gender" in filtered.columns:
        if filters["gender"]:
            filtered = filtered[filtered["Gender"].astype(str).isin(filters["gender"])]

    if "rating" in filters and filtered is not None and "Rating" in filtered.columns:
        low, high = filters["rating"]
        filtered = filtered[(filtered["Rating"] >= low) & (filtered["Rating"] <= high)]

    if "quantity" in filters and filtered is not None and "Quantity" in filtered.columns:
        low, high = filters["quantity"]
        filtered = filtered[(filtered["Quantity"] >= low) & (filtered["Quantity"] <= high)]

    return filtered.reset_index(drop=True)


def _initialize_dashboard_state() -> None:
    """Load the base dataset into Streamlit session state when needed."""
    if "dashboard_data" not in st.session_state:
        raw_df = get_dataset()
        prepared_df = preprocess(raw_df)
        st.session_state["dashboard_data"] = prepared_df

    if "dashboard_filtered" not in st.session_state:
        st.session_state["dashboard_filtered"] = st.session_state["dashboard_data"].copy()


def _load_dataset_from_upload() -> pd.DataFrame:
    """Load the active dataset from either the uploader or the session state."""
    uploaded_file = st.session_state.get("dashboard_upload")
    if uploaded_file is not None:
        raw_df = get_dataset(uploaded_file)
        prepared_df = preprocess(raw_df)
        st.session_state["dashboard_data"] = prepared_df
        st.session_state["dashboard_filtered"] = prepared_df.copy()
        return prepared_df

    if "dashboard_data" in st.session_state:
        return st.session_state["dashboard_data"]

    raw_df = get_dataset()
    prepared_df = preprocess(raw_df)
    st.session_state["dashboard_data"] = prepared_df
    st.session_state["dashboard_filtered"] = prepared_df.copy()
    return prepared_df


def _render_sidebar_branding() -> None:
    """Render the left sidebar header and upload control."""
    with st.sidebar:
        st.markdown("## 🛒 Business Intelligence")
        st.markdown("### AI-Powered Supermarket Analytics")
        st.caption("Interactive performance and customer intelligence workspace")

        uploaded = st.file_uploader(
            "Upload dataset",
            type=["csv", "xlsx", "xls"],
            key="dashboard_upload",
            help="Replace the current dataset with a new CSV or Excel file.",
        )

        if uploaded is not None:
            st.success("Dataset updated successfully.")

        st.markdown("---")
        st.markdown("### Quick Notes")
        st.caption("Use the filters to focus the analysis on selected branches, products, customer segments, and time ranges.")
        st.markdown("---")


def render_dashboard() -> None:
    """Render the full dashboard experience."""
    st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    load_css()
    _initialize_dashboard_state()
    _render_sidebar_branding()

    data = _load_dataset_from_upload()
    if data.empty:
        st.error("The loaded dataset is empty.")
        st.stop()

    issues = validate_dataset(data)
    if issues:
        with st.expander("⚠ Dataset Validation"):
            for issue in issues:
                st.warning(issue)

    filters = _render_sidebar_filters(data)
    filtered_df = _apply_filters(data, filters)
    st.session_state["dashboard_filtered"] = filtered_df

    if filtered_df.empty:
        st.warning("No rows match the selected filters. Please broaden the selection.")
        st.stop()

    with st.spinner("Preparing executive dashboard..."):
        _render_hero_header(filtered_df)
        st.markdown("---")
        _render_kpi_cards(filtered_df)
        st.markdown("---")
        _render_dataset_overview(filtered_df)
        st.markdown("---")
        _render_exec_summary(filtered_df)
        st.markdown("---")
        _render_health_snapshot(filtered_df)
        st.markdown("---")
        _render_charts(filtered_df)
        st.markdown("---")
        _render_recent_transactions(filtered_df)
        st.markdown("---")
        _render_export_section(filtered_df)


render_dashboard()
