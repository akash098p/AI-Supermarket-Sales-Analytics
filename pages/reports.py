

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.analytics import (
    average_order_value,
    average_rating_by,
    correlation_matrix,
    dashboard_summary,
    quantity_by,
    revenue_by,
    total_products_sold,
    total_revenue,
)
from utils.charts import bar, heatmap, pie
from utils.config import APP_NAME, PRIMARY, SECONDARY, STYLE_PATH, SUCCESS, WARNING
from utils.data_loader import (
    dataset_profile,
    get_dataset,
    load_page_dataset,
    missing_report,
    numeric_summary,
    preview,
    validate_dataset,
)
from utils.exports import export_dashboard_package
from utils.preprocessing import preprocess


def load_css() -> None:
    """Inject the shared dashboard stylesheet."""
    if STYLE_PATH.exists():
        with open(STYLE_PATH, "r", encoding="utf-8") as handle:
            st.markdown(f"<style>{handle.read()}</style>", unsafe_allow_html=True)


def _format_currency(value: float) -> str:
    """Format numeric values as currency."""
    return f"₹{value:,.2f}" if pd.notna(value) else "₹0.00"


def _format_number(value: Any) -> str:
    """Format numbers for display."""
    if pd.isna(value):
        return "0"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    return str(value)


def _load_reports_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset with session caching for reports."""
    uploaded = st.session_state.get("reports_upload")
    return load_page_dataset(
        "reports",
        lambda raw_df: preprocess(raw_df),
        uploaded_file=uploaded,
    )


def _prepare_reports_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns used by the reports page."""
    prepared = df.copy()
    if "Date" in prepared.columns:
        prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
        prepared = prepared.dropna(subset=["Date"]).reset_index(drop=True)
    if "Total" in prepared.columns:
        prepared["Total"] = pd.to_numeric(prepared["Total"], errors="coerce")
    if "Quantity" in prepared.columns:
        prepared["Quantity"] = pd.to_numeric(prepared["Quantity"], errors="coerce")
    if "Rating" in prepared.columns:
        prepared["Rating"] = pd.to_numeric(prepared["Rating"], errors="coerce")
    return prepared


def _build_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render sidebar filters for the reporting page."""
    st.sidebar.markdown("## 🧭 Report Filters")
    filters: Dict[str, Any] = {}

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        min_date, max_date = dates.min(), dates.max()
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

    product_col = "Product" if "Product" in df.columns else "Product Line"
    if product_col in df.columns:
        products = sorted(df[product_col].dropna().astype(str).unique())
        filters["product"] = st.sidebar.multiselect(product_col, products, default=products)

    if "Payment" in df.columns:
        payments = sorted(df["Payment"].dropna().astype(str).unique())
        filters["payment"] = st.sidebar.multiselect("Payment", payments, default=payments)

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"], key="reports_upload")
    if uploaded is not None:
        st.sidebar.success("Dataset updated successfully.")

    return filters


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply sidebar filters to the active report dataset."""
    filtered = df.copy()

    if "date_start" in filters and "date_end" in filters and "Date" in filtered.columns:
        filtered = filtered[(filtered["Date"] >= filters["date_start"]) & (filtered["Date"] <= filters["date_end"])]

    if "branch" in filters and "Branch" in filtered.columns:
        if filters["branch"]:
            filtered = filtered[filtered["Branch"].astype(str).isin(filters["branch"])]

    if "city" in filters and "City" in filtered.columns:
        if filters["city"]:
            filtered = filtered[filtered["City"].astype(str).isin(filters["city"])]

    product_col = "Product" if "Product" in filtered.columns else "Product Line"
    if "product" in filters and product_col in filtered.columns:
        if filters["product"]:
            filtered = filtered[filtered[product_col].astype(str).isin(filters["product"])]

    if "payment" in filters and "Payment" in filtered.columns:
        if filters["payment"]:
            filtered = filtered[filtered["Payment"].astype(str).isin(filters["payment"])]

    return filtered.reset_index(drop=True)


def _render_hero(df: pd.DataFrame) -> None:
    """Render the reports hero section."""
    summary = dashboard_summary(df)
    profile = dataset_profile(df)
    today = datetime.now().strftime("%d %b %Y")

    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    left, right = st.columns([2.2, 1.0])

    with left:
        st.markdown("<h1 style='margin-bottom:0.2rem;'>📑 Performance Reports & Diagnostics</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.02rem; margin-top:0.25rem; opacity:0.95;'>Build operational intelligence reports with performance benchmarks, data quality diagnostics, and export-ready output.</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.45rem;'>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🗓️ {today}</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>📦 {profile['rows']:,} records</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🧠 {profile['numeric_columns'].__len__()} numeric fields</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.metric("Total Revenue", _format_currency(summary["Total Revenue"]), help="Revenue included in the report")
        st.metric("Orders", f"{summary['Total Orders']:,}", help="Transactions in the report")
        st.metric("Products Sold", f"{summary['Products Sold']:,}", help="Units sold")
        st.metric("Average Rating", f"{summary['Average Rating']:.2f}/10", help="Customer satisfaction")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_kpi_cards(df: pd.DataFrame) -> None:
    """Render report KPIs for revenue, quality, and coverage."""
    summary = dashboard_summary(df)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    metrics: List[Dict[str, Any]] = [
        {"title": "Revenue", "value": _format_currency(summary["Total Revenue"]), "icon": "💰", "accent": PRIMARY},
        {"title": "Coverage", "value": f"{summary['Total Orders']:,}", "icon": "🧾", "accent": SECONDARY},
        {"title": "Data Gaps", "value": f"{missing:,}", "icon": "⚠️", "accent": WARNING},
        {"title": "Duplicates", "value": f"{duplicates:,}", "icon": "♻️", "accent": SUCCESS},
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


def _build_report_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare report frames for charts and tabular output."""
    frames: Dict[str, pd.DataFrame] = {}

    if "Branch" in df.columns and "Total" in df.columns:
        frames["revenue_by_branch"] = revenue_by(df, "Branch")
    else:
        frames["revenue_by_branch"] = pd.DataFrame(columns=["Branch", "Total"])

    if "City" in df.columns and "Total" in df.columns:
        frames["revenue_by_city"] = revenue_by(df, "City")
    else:
        frames["revenue_by_city"] = pd.DataFrame(columns=["City", "Total"])

    product_col = "Product" if "Product" in df.columns else "Product Line"
    if product_col in df.columns and "Quantity" in df.columns:
        frames["product_volume"] = quantity_by(df, product_col).head(10)
    else:
        frames["product_volume"] = pd.DataFrame(columns=[product_col, "Quantity"])

    if "Rating" in df.columns:
        frames["rating_by_branch"] = average_rating_by(df, "Branch") if "Branch" in df.columns else pd.DataFrame(columns=["Branch", "Rating"])
        frames["rating_by_city"] = average_rating_by(df, "City") if "City" in df.columns else pd.DataFrame(columns=["City", "Rating"])
    else:
        frames["rating_by_branch"] = pd.DataFrame(columns=["Branch", "Rating"])
        frames["rating_by_city"] = pd.DataFrame(columns=["City", "Rating"])

    if {"Total", "Quantity"}.issubset(df.columns):
        frames["financial_summary"] = pd.DataFrame(
            [
                {"Metric": "Revenue", "Value": _format_currency(total_revenue(df))},
                {"Metric": "Products Sold", "Value": f"{total_products_sold(df):,}"},
                {"Metric": "Average Order", "Value": _format_currency(average_order_value(df))},
                {"Metric": "Revenue per Unit", "Value": _format_currency(total_revenue(df) / total_products_sold(df) if total_products_sold(df) else 0)},
            ]
        )
    else:
        frames["financial_summary"] = pd.DataFrame(columns=["Metric", "Value"])

    frames["dataset_profile"] = pd.DataFrame.from_dict(dataset_profile(df), orient="index", columns=["Value"]).reset_index().rename(columns={"index": "Metric"})
    frames["missing_report"] = missing_report(df)
    frames["numeric_summary"] = numeric_summary(df)
    frames["top_products"] = revenue_by(df, product_col).head(10) if product_col in df.columns and "Total" in df.columns else pd.DataFrame(columns=[product_col, "Total"])

    if {"Total", "Rating"}.issubset(df.columns):
        frames["rating_revenue"] = df.groupby("Rating", as_index=False)["Total"].sum().sort_values("Rating")
    else:
        frames["rating_revenue"] = pd.DataFrame(columns=["Rating", "Total"])

    return frames


def _render_charts(df: pd.DataFrame) -> None:
    """Render reporting visualizations."""
    frames = _build_report_frames(df)

    st.markdown("### 📈 Performance Charts")
    col1, col2 = st.columns(2)

    with col1:
        if not frames["revenue_by_branch"].empty:
            chart = bar(frames["revenue_by_branch"].head(10), "Branch", "Total", "Top Branch Revenue", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Branch revenue chart unavailable.")

    with col2:
        if not frames["revenue_by_city"].empty:
            chart = bar(frames["revenue_by_city"].head(10), "City", "Total", "Top City Revenue", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("City revenue chart unavailable.")

    st.markdown("### 🧾 Report Diagnostics")
    diag_left, diag_right = st.columns(2)

    with diag_left:
        if not frames["product_volume"].empty:
            chart = pie(frames["product_volume"], frames["product_volume"].columns[0], "Quantity", "Top Product Volume")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Top product volume chart unavailable.")

    with diag_right:
        if not frames["rating_revenue"].empty:
            chart = bar(frames["rating_revenue"], "Rating", "Total", "Revenue by Rating")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Revenue by rating chart unavailable.")

    side_left, side_right = st.columns(2)

    with side_left:
        if not frames["rating_by_branch"].empty:
            chart = bar(frames["rating_by_branch"].head(10), "Branch", "Rating", "Branch Rating Benchmark", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)

    with side_right:
        if not frames["rating_by_city"].empty:
            chart = bar(frames["rating_by_city"].head(10), "City", "Rating", "City Rating Benchmark", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)


def _render_tables(df: pd.DataFrame) -> None:
    """Render tables used for report details."""
    frames = _build_report_frames(df)

    st.markdown("### 📋 Report Tables")
    left, right = st.columns(2)

    with left:
        st.markdown("#### Dataset Quality")
        st.dataframe(frames["dataset_profile"], use_container_width=True, hide_index=True)
        st.markdown("#### Missing Values")
        st.dataframe(frames["missing_report"], use_container_width=True, hide_index=True)

    with right:
        if not frames["financial_summary"].empty:
            st.markdown("#### Financial Summary")
            st.dataframe(frames["financial_summary"], use_container_width=True, hide_index=True)

        if not frames["top_products"].empty:
            st.markdown("#### Top Revenue Products")
            st.dataframe(frames["top_products"].head(10), use_container_width=True, hide_index=True)

    if not frames["numeric_summary"].empty:
        st.markdown("#### Numeric Summary")
        st.dataframe(frames["numeric_summary"], use_container_width=True, hide_index=True)

    st.markdown("#### Sample Records")
    st.dataframe(preview(df, rows=10), use_container_width=True)


def _render_insights(df: pd.DataFrame) -> None:
    """Render concise reporting insights."""
    st.markdown("### 🧠 Report Insights")
    summary = dashboard_summary(df)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    top_branch = revenue_by(df, "Branch").iloc[0]["Branch"] if "Branch" in df.columns and not revenue_by(df, "Branch").empty else "N/A"

    insights = [
        f"The filtered report covers {summary['Total Orders']:,} transactions and {summary['Products Sold']:,} sold units.",
        f"Dataset quality check identified {missing:,} missing values and {duplicates:,} duplicate rows.",
        f"Highest revenue contribution comes from {top_branch} when branch data is available.",
    ]

    for text in insights:
        st.markdown(f"<div class='insight-card'>{text}</div>", unsafe_allow_html=True)


def _render_exports(df: pd.DataFrame) -> None:
    """Render export buttons for report output."""
    st.markdown("### ⬇️ Export Report")
    exports = export_dashboard_package(df)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button("Download CSV", exports["csv"], file_name="reports_view.csv", mime="text/csv")
    with c2:
        st.download_button("Download Excel", exports["excel"], file_name="reports_view.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Download PDF", exports["pdf"], file_name="reports_view_summary.pdf", mime="application/pdf")


def render_reports_page() -> None:
    """Render the full operational reports experience."""
    st.set_page_config(page_title=APP_NAME, page_icon="📑", layout="wide", initial_sidebar_state="expanded")
    load_css()

    data, filtered_data = _load_reports_data()
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
    st.session_state["reports_filtered"] = filtered_data

    if filtered_data.empty:
        st.warning("No rows match the selected filters. Please broaden the selection.")
        st.stop()

    prepared = _prepare_reports_frame(filtered_data)

    with st.spinner("Preparing report assets..."):
        _render_hero(prepared)
        st.markdown("---")
        _render_kpi_cards(prepared)
        st.markdown("---")
        _render_charts(prepared)
        st.markdown("---")
        _render_tables(prepared)
        st.markdown("---")
        _render_insights(prepared)
        st.markdown("---")
        _render_exports(prepared)


render_reports_page()
