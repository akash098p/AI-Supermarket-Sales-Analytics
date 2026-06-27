"""
AI-Powered Supermarket Sales Analytics Dashboard
pages/ratings.py
Customer rating and satisfaction analytics page.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.analytics import (
    average_order_value,
    average_rating,
    average_rating_by,
    dashboard_summary,
    gross_income,
    quantity_by,
    revenue_by,
    total_customers,
    total_orders,
    total_products_sold,
    total_revenue,
)
from utils.charts import bar, donut, heatmap, histogram, pie, treemap
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


def _format_number(value: Any) -> str:
    """Format numeric values for display."""
    if pd.isna(value):
        return "0"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    return str(value)


def _load_ratings_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and cache the dataset for the ratings view."""
    uploaded = st.session_state.get("ratings_upload")
    if uploaded is not None:
        raw_df = get_dataset(uploaded)
        prepared = preprocess(raw_df)
        st.session_state["ratings_data"] = prepared
        st.session_state["ratings_filtered"] = prepared.copy()
        return prepared, prepared.copy()

    if "ratings_data" in st.session_state:
        data = st.session_state["ratings_data"]
        filtered = st.session_state.get("ratings_filtered", data.copy())
        return data, filtered

    raw_df = get_dataset()
    prepared = preprocess(raw_df)
    st.session_state["ratings_data"] = prepared
    st.session_state["ratings_filtered"] = prepared.copy()
    return prepared, prepared.copy()


def _prepare_ratings_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize rating-specific fields used by the page."""
    prepared = df.copy()
    if "Date" in prepared.columns:
        prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
        prepared = prepared.dropna(subset=["Date"]).reset_index(drop=True)
    if "Rating" in prepared.columns:
        prepared["Rating"] = pd.to_numeric(prepared["Rating"], errors="coerce")
    if "Total" in prepared.columns:
        prepared["Total"] = pd.to_numeric(prepared["Total"], errors="coerce")
    if "Quantity" in prepared.columns:
        prepared["Quantity"] = pd.to_numeric(prepared["Quantity"], errors="coerce")
    return prepared


def _build_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render the rating analytics filters in the sidebar."""
    st.sidebar.markdown("## 🧭 Ratings Filters")
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

    if "Customer Type" in df.columns:
        customer_types = sorted(df["Customer Type"].dropna().astype(str).unique())
        filters["customer_type"] = st.sidebar.multiselect("Customer Type", customer_types, default=customer_types)

    if "Gender" in df.columns:
        genders = sorted(df["Gender"].dropna().astype(str).unique())
        filters["gender"] = st.sidebar.multiselect("Gender", genders, default=genders)

    if "Rating" in df.columns:
        min_rating = float(df["Rating"].min())
        max_rating = float(df["Rating"].max())
        filters["rating"] = st.sidebar.slider(
            "Rating Range",
            min_value=min_rating,
            max_value=max_rating,
            value=(min_rating, max_rating),
            step=0.1,
        )

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"], key="ratings_upload")
    if uploaded is not None:
        st.sidebar.success("Dataset updated successfully.")

    return filters


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply sidebar filters to the active ratings view."""
    filtered = df.copy()

    if "date_start" in filters and "date_end" in filters and "Date" in filtered.columns:
        filtered = filtered[(filtered["Date"] >= filters["date_start"]) & (filtered["Date"] <= filters["date_end"]) ]

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

    if "customer_type" in filters and "Customer Type" in filtered.columns:
        if filters["customer_type"]:
            filtered = filtered[filtered["Customer Type"].astype(str).isin(filters["customer_type"])]

    if "gender" in filters and "Gender" in filtered.columns:
        if filters["gender"]:
            filtered = filtered[filtered["Gender"].astype(str).isin(filters["gender"])]

    if "rating" in filters and "Rating" in filtered.columns:
        low, high = filters["rating"]
        filtered = filtered[(filtered["Rating"] >= low) & (filtered["Rating"] <= high)]

    return filtered.reset_index(drop=True)


def _render_hero(df: pd.DataFrame) -> None:
    """Render page hero with recommendation and metrics."""
    summary = dashboard_summary(df)
    rating_avg = average_rating(df)
    today = datetime.now().strftime("%d %b %Y")

    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    left, right = st.columns([2.2, 1.0])

    with left:
        st.markdown("<h1 style='margin-bottom:0.2rem;'>⭐ Customer Ratings & Reputation</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.02rem; margin-top:0.25rem; opacity:0.95;'>Track satisfaction, product sentiment, branch reputation, and service quality across the supermarket dataset.</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.45rem;'>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🗓️ {today}</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>📊 {len(df):,} reviews</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>⭐ {rating_avg:.2f} average</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.metric("Average Rating", f"{rating_avg:.2f}/10", help="Mean customer satisfaction")
        st.metric("Total Reviews", f"{len(df):,}", help="Total rating records")
        st.metric("Revenue", _format_currency(summary["Total Revenue"]), help="Revenue covered by rating data")
        st.metric("Avg Order", _format_currency(summary["Average Order Value"]), help="Average basket value")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_kpi_cards(df: pd.DataFrame) -> None:
    """Render KPI cards for rating performance."""
    summary = dashboard_summary(df)
    metrics: List[Dict[str, Any]] = [
        {"title": "Average Rating", "value": f"{average_rating(df):.2f}/10", "icon": "⭐", "accent": PRIMARY},
        {"title": "High Ratings", "value": f"{len(df[df['Rating'] >= 8]) if 'Rating' in df.columns else 0:,}", "icon": "💎", "accent": SUCCESS},
        {"title": "Low Ratings", "value": f"{len(df[df['Rating'] <= 4]) if 'Rating' in df.columns else 0:,}", "icon": "⚠️", "accent": WARNING},
        {"title": "Revenue", "value": _format_currency(summary["Total Revenue"]), "icon": "💰", "accent": SECONDARY},
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


def _build_rating_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare rating analytics frames for charts and tables."""
    frames: Dict[str, pd.DataFrame] = {}

    if "Rating" in df.columns:
        distribution = df["Rating"].value_counts().reset_index()
        distribution.columns = ["Rating", "Count"]
        frames["distribution"] = distribution.sort_values("Rating")
        frames["rating_bands"] = (
            df.assign(RatingBand=pd.cut(df["Rating"], bins=[0,4,7,8.5,10], labels=["Poor", "Fair", "Good", "Excellent"], include_lowest=True))
              .groupby("RatingBand", as_index=False)["Rating"].count()
              .rename(columns={"Rating": "Count"})
        )
    else:
        frames["distribution"] = pd.DataFrame(columns=["Rating", "Count"])
        frames["rating_bands"] = pd.DataFrame(columns=["RatingBand", "Count"])

    if "Branch" in df.columns:
        frames["branch_ratings"] = average_rating_by(df, "Branch")
    else:
        frames["branch_ratings"] = pd.DataFrame(columns=["Branch", "Rating"])

    if "City" in df.columns:
        frames["city_ratings"] = average_rating_by(df, "City")
    else:
        frames["city_ratings"] = pd.DataFrame(columns=["City", "Rating"])

    product_col = "Product" if "Product" in df.columns else "Product Line"
    if product_col in df.columns:
        frames["product_ratings"] = average_rating_by(df, product_col).head(10)
    else:
        frames["product_ratings"] = pd.DataFrame(columns=[product_col, "Rating"])

    if "Rating" in df.columns and "Total" in df.columns:
        frames["rating_revenue"] = (
            df.groupby("Rating", as_index=False)["Total"].sum().sort_values("Rating")
        )
    else:
        frames["rating_revenue"] = pd.DataFrame(columns=["Rating", "Total"])

    if {"Rating", "Total", "Quantity"}.issubset(df.columns):
        numeric = df[["Rating", "Total", "Quantity"]].select_dtypes(include=[np.number]).corr()
        frames["rating_corr"] = numeric
    else:
        frames["rating_corr"] = pd.DataFrame()

    return frames


def _render_charts(df: pd.DataFrame) -> None:
    """Render rating analytics visualizations."""
    frames = _build_rating_frames(df)

    st.markdown("### 📈 Rating Analytics")
    top, bottom = st.columns(2)

    with top:
        if not frames["distribution"].empty:
            chart = bar(frames["distribution"], "Rating", "Count", "Rating Distribution")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Rating distribution chart unavailable.")

    with bottom:
        if not frames["rating_bands"].empty:
            chart = pie(frames["rating_bands"], "RatingBand", "Count", "Rating Band Share")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Rating band chart unavailable.")

    left, right = st.columns(2)
    with left:
        if not frames["branch_ratings"].empty:
            chart = bar(frames["branch_ratings"].head(10), "Branch", "Rating", "Average Rating by Branch", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Branch ratings chart unavailable.")

    with right:
        if not frames["city_ratings"].empty:
            chart = bar(frames["city_ratings"].head(10), "City", "Rating", "Average Rating by City", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("City ratings chart unavailable.")

    st.markdown("### 🧾 Rating-Linked Revenue")
    dual_left, dual_right = st.columns(2)

    with dual_left:
        if not frames["rating_revenue"].empty:
            chart = bar(frames["rating_revenue"], "Rating", "Total", "Revenue by Rating")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Revenue-by-rating chart unavailable.")

    with dual_right:
        if not frames["rating_corr"].empty:
            chart = heatmap(frames["rating_corr"], title="Rating Correlation Matrix")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Rating correlation matrix unavailable.")

    if not frames["product_ratings"].empty:
        st.markdown("### 🏆 Product Sentiment")
        chart = treemap(frames["product_ratings"], path=[frames['product_ratings'].columns[0]], values="Rating", title="Top Product Ratings")
        st.plotly_chart(chart, use_container_width=True)


def _render_tables(df: pd.DataFrame) -> None:
    """Render rating summary tables for action review."""
    frames = _build_rating_frames(df)

    st.markdown("### 🧾 Rating Insights")
    if not frames["branch_ratings"].empty:
        st.markdown("#### Branch Rating Ranking")
        st.dataframe(frames["branch_ratings"].sort_values("Rating", ascending=False), use_container_width=True, hide_index=True)

    if not frames["product_ratings"].empty:
        st.markdown("#### Product Rating Ranking")
        st.dataframe(frames["product_ratings"].sort_values("Rating", ascending=False), use_container_width=True, hide_index=True)

    if not frames["rating_revenue"].empty:
        st.markdown("#### Revenue by Rating")
        st.dataframe(frames["rating_revenue"].sort_values("Rating"), use_container_width=True, hide_index=True)


def _render_insights(df: pd.DataFrame) -> None:
    """Render actionable rating insights."""
    st.markdown("### 🧠 Sentiment Insights")
    score = average_rating(df)
    ratio_high = float(len(df[df["Rating"] >= 8]) / len(df)) if "Rating" in df.columns and len(df) else 0.0
    ratio_low = float(len(df[df["Rating"] <= 4]) / len(df)) if "Rating" in df.columns and len(df) else 0.0

    insights = [
        f"Average customer satisfaction is {score:.2f}/10; high-rating reviews account for {ratio_high:.0%} of the dataset.",
        f"Low-score feedback represents {ratio_low:.0%} of responses and should be a focus for service improvement.",
        "Top-rated branches and products can be used as case studies for operational excellence and merchandising best practices.",
    ]

    for item in insights:
        st.markdown(f"<div class='insight-card'>{item}</div>", unsafe_allow_html=True)


def _render_exports(df: pd.DataFrame) -> None:
    """Render export actions for the ratings page."""
    st.markdown("### ⬇️ Export Ratings View")
    exports = export_dashboard_package(df)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button("Download CSV", exports["csv"], file_name="ratings_view.csv", mime="text/csv")
    with c2:
        st.download_button("Download Excel", exports["excel"], file_name="ratings_view.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Download PDF", exports["pdf"], file_name="ratings_view_summary.pdf", mime="application/pdf")


def render_ratings_page() -> None:
    """Render the complete ratings analytics experience."""
    st.set_page_config(page_title=APP_NAME, page_icon="⭐", layout="wide", initial_sidebar_state="expanded")
    load_css()

    data, filtered_data = _load_ratings_data()
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
    st.session_state["ratings_filtered"] = filtered_data

    if filtered_data.empty:
        st.warning("No rows match the selected filters. Please broaden the selection.")
        st.stop()

    prepared = _prepare_ratings_frame(filtered_data)

    with st.spinner("Preparing rating intelligence..."):
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


render_ratings_page()
