

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.analytics import (
    average_order_value,
    average_rating,
    dashboard_summary,
    gross_income,
    quantity_by,
    revenue_by,
    total_customers,
    total_orders,
    total_products_sold,
    total_revenue,
)
from utils.charts import bar, donut, heatmap, pie, treemap
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


def _load_branch_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the active dataset while preserving session state."""
    uploaded = st.session_state.get("branches_upload")
    return load_page_dataset(
        "branches",
        lambda raw_df: preprocess(raw_df),
        uploaded_file=uploaded,
    )


def _prepare_branch_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize branch-related columns for the page."""
    prepared = df.copy()
    if "Branch" not in prepared.columns:
        prepared["Branch"] = "Unknown"
    if "City" not in prepared.columns:
        prepared["City"] = "Unknown"
    if "Date" in prepared.columns:
        prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
        prepared = prepared.dropna(subset=["Date"]).reset_index(drop=True)
    return prepared


def _build_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render sidebar filters and return the active selection state."""
    st.sidebar.markdown("## 🧭 Branch Filters")
    filters: Dict[str, Any] = {}

    if "Branch" in df.columns:
        branches = sorted(df["Branch"].dropna().astype(str).unique())
        filters["branch"] = st.sidebar.multiselect("Branch", branches, default=branches)

    if "City" in df.columns:
        cities = sorted(df["City"].dropna().astype(str).unique())
        filters["city"] = st.sidebar.multiselect("City", cities, default=cities)

    if "Product" in df.columns:
        products = sorted(df["Product"].dropna().astype(str).unique())
        filters["product"] = st.sidebar.multiselect("Product", products, default=products)
    elif "Product Line" in df.columns:
        products = sorted(df["Product Line"].dropna().astype(str).unique())
        filters["product"] = st.sidebar.multiselect("Product Line", products, default=products)

    if "Customer Type" in df.columns:
        customer_types = sorted(df["Customer Type"].dropna().astype(str).unique())
        filters["customer_type"] = st.sidebar.multiselect("Customer Type", customer_types, default=customer_types)

    if "Payment" in df.columns:
        payments = sorted(df["Payment"].dropna().astype(str).unique())
        filters["payment"] = st.sidebar.multiselect("Payment", payments, default=payments)

    if "Rating" in df.columns:
        rating_min, rating_max = float(df["Rating"].min()), float(df["Rating"].max())
        filters["rating"] = st.sidebar.slider(
            "Rating",
            min_value=float(rating_min),
            max_value=float(rating_max),
            value=(float(rating_min), float(rating_max)),
            step=0.1,
        )

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"], key="branches_upload")
    if uploaded is not None:
        st.sidebar.success("Dataset updated successfully.")

    return filters


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply the sidebar filters to the active dataset."""
    filtered = df.copy()

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

    if "customer_type" in filters and "Customer Type" in filtered.columns:
        if filters["customer_type"]:
            filtered = filtered[filtered["Customer Type"].astype(str).isin(filters["customer_type"])]

    if "payment" in filters and "Payment" in filtered.columns:
        if filters["payment"]:
            filtered = filtered[filtered["Payment"].astype(str).isin(filters["payment"])]

    if "rating" in filters and "Rating" in filtered.columns:
        low, high = filters["rating"]
        filtered = filtered[(filtered["Rating"] >= low) & (filtered["Rating"] <= high)]

    return filtered.reset_index(drop=True)


def _render_hero(df: pd.DataFrame) -> None:
    """Render the header hero section for the branch page."""
    summary = dashboard_summary(df)
    today = datetime.now().strftime("%d %b %Y")

    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    left, right = st.columns([2.2, 1.0])

    with left:
        st.markdown("<h1 style='margin-bottom:0.2rem;'>🏬 Branch Performance Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.02rem; margin-top:0.25rem; opacity:0.95;'>Evaluate branch profitability, customer demand, and outlet excellence with a focused operational view.</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.45rem;'>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🗓️ {today}</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🏪 {len(df):,} transactions</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>⭐ {summary['Average Rating']:.1f}/10</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.metric("Revenue", _format_currency(summary["Total Revenue"]), help="Revenue generated by the filter")
        st.metric("Orders", f"{summary['Total Orders']:,}", help="Transaction count")
        st.metric("Customers", f"{summary['Total Customers']:,}", help="Unique customers")
        st.metric("Best Branch", summary["Best Branch"], help="Top revenue branch")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_kpi_cards(df: pd.DataFrame) -> None:
    """Render executive KPI cards for branch performance."""
    summary = dashboard_summary(df)
    branch_summary = revenue_by(df, "Branch")
    best_branch = branch_summary.iloc[0]["Branch"] if not branch_summary.empty else "-"
    worst_branch = branch_summary.iloc[-1]["Branch"] if not branch_summary.empty else "-"

    metrics: List[Dict[str, Any]] = [
        {"title": "Revenue", "value": _format_currency(summary["Total Revenue"]), "icon": "💰", "accent": PRIMARY},
        {"title": "Orders", "value": f"{summary['Total Orders']:,}", "icon": "🧾", "accent": SECONDARY},
        {"title": "Gross Income", "value": _format_currency(summary["Gross Income"]), "icon": "📈", "accent": SUCCESS},
        {"title": "Avg Rating", "value": f"{summary['Average Rating']:.1f}/10", "icon": "⭐", "accent": WARNING},
        {"title": "Best Branch", "value": best_branch, "icon": "🏆", "accent": PRIMARY},
        {"title": "Worst Branch", "value": worst_branch, "icon": "⚠️", "accent": WARNING},
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


def _build_branch_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare summary tables for branch-level analysis."""
    frames: Dict[str, pd.DataFrame] = {}

    empty_revenue = pd.DataFrame(columns=["Branch", "Total"])
    empty_orders = pd.DataFrame(columns=["Branch", "Orders"])
    empty_profit = pd.DataFrame(columns=["Branch", "Gross Income"])
    empty_rating = pd.DataFrame(columns=["Branch", "Avg Rating"])
    empty_customers = pd.DataFrame(columns=["Branch", "Customers"])
    empty_quantity = pd.DataFrame(columns=["Branch", "Units Sold"])

    if "Branch" not in df.columns:
        frames["branch_revenue"] = empty_revenue
        frames["branch_orders"] = empty_orders
        frames["branch_profit"] = empty_profit
        frames["branch_rating"] = empty_rating
        frames["branch_customers"] = empty_customers
        frames["branch_quantity"] = empty_quantity
        return frames

    frames["branch_revenue"] = revenue_by(df, "Branch") if "Total" in df.columns else empty_revenue
    frames["branch_orders"] = df.groupby("Branch").size().reset_index(name="Orders")
    frames["branch_profit"] = (
        df.groupby("Branch", as_index=False)["Gross Income"].sum().rename(columns={"Gross Income": "Gross Income"})
        if "Gross Income" in df.columns
        else empty_profit
    )
    frames["branch_rating"] = (
        df.groupby("Branch", as_index=False)["Rating"].mean().rename(columns={"Rating": "Avg Rating"})
        if "Rating" in df.columns
        else empty_rating
    )
    frames["branch_customers"] = (
        df.groupby("Branch", as_index=False)["Customer ID"].nunique().rename(columns={"Customer ID": "Customers"})
        if "Customer ID" in df.columns
        else empty_customers
    )
    frames["branch_quantity"] = (
        df.groupby("Branch", as_index=False)["Quantity"].sum().rename(columns={"Quantity": "Units Sold"})
        if "Quantity" in df.columns
        else empty_quantity
    )

    return frames


def _render_charts(df: pd.DataFrame) -> None:
    """Render the branch analytics charts."""
    frames = _build_branch_metrics(df)

    st.markdown("### 📈 Branch Performance")

    left, right = st.columns(2)
    with left:
        if not frames["branch_revenue"].empty:
            chart = bar(frames["branch_revenue"], "Branch", "Total", "Revenue by Branch", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Revenue by branch chart unavailable.")

    with right:
        if not frames["branch_orders"].empty:
            chart = bar(frames["branch_orders"], "Branch", "Orders", "Orders by Branch", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Orders by branch chart unavailable.")

    lower_left, lower_middle, lower_right = st.columns(3)
    with lower_left:
        if not frames["branch_profit"].empty:
            chart = bar(frames["branch_profit"], "Branch", "Gross Income", "Gross Income by Branch", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Gross income by branch chart unavailable.")

    with lower_middle:
        if not frames["branch_rating"].empty:
            chart = bar(frames["branch_rating"], "Branch", "Avg Rating", "Average Rating by Branch", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Average rating by branch chart unavailable.")

    with lower_right:
        if not frames["branch_customers"].empty:
            chart = bar(frames["branch_customers"], "Branch", "Customers", "Customer Count by Branch", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Customer count by branch chart unavailable.")

    st.markdown("### 🌐 Branch Geography & Ranking")
    geo_left, geo_right = st.columns(2)
    with geo_left:
        if not frames["branch_revenue"].empty:
            chart = treemap(frames["branch_revenue"], path=["Branch"], values="Total", title="Branch Revenue Treemap")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Treemap unavailable.")

    with geo_right:
        if not frames["branch_revenue"].empty:
            chart = pie(frames["branch_revenue"], "Branch", "Total", "Branch Revenue Share")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Branch share chart unavailable.")


def _render_branch_tables(df: pd.DataFrame) -> None:
    """Render concise branch ranking tables."""
    frames = _build_branch_metrics(df)

    st.markdown("### 🧾 Branch Rankings")
    left, right = st.columns(2)

    with left:
        if not frames["branch_revenue"].empty:
            st.markdown("#### Revenue Ranking")
            st.dataframe(frames["branch_revenue"].sort_values("Total", ascending=False), use_container_width=True, hide_index=True)

    with right:
        if not frames["branch_rating"].empty:
            st.markdown("#### Rating Ranking")
            st.dataframe(frames["branch_rating"].sort_values("Avg Rating", ascending=False), use_container_width=True, hide_index=True)

    if not frames["branch_quantity"].empty:
        st.markdown("#### Sales Volume")
        st.dataframe(frames["branch_quantity"].sort_values("Units Sold", ascending=False), use_container_width=True, hide_index=True)


def _render_insights(df: pd.DataFrame) -> None:
    """Render operational insights and branch recommendations."""
    st.markdown("### 🧠 Branch Insights")
    summary = dashboard_summary(df)
    frames = _build_branch_metrics(df)

    insight_cards = [
        f"The strongest outlet is {summary['Best Branch']} with the highest revenue contribution in the current selection.",
        f"The current branch mix records {summary['Total Orders']:,} orders and {summary['Products Sold']:,} units sold.",
        f"Customer satisfaction averages {summary['Average Rating']:.1f}/10, providing a strong baseline for comparison across branches.",
    ]

    for card in insight_cards:
        st.markdown(f"<div class='insight-card'>{card}</div>", unsafe_allow_html=True)

    if not frames["branch_revenue"].empty:
        st.markdown("#### Top & Bottom Branches")
        st.dataframe(
            pd.concat([
                frames["branch_revenue"].head(3).assign(Performance="Top"),
                frames["branch_revenue"].tail(3).assign(Performance="Bottom"),
            ]).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


def _render_exports(df: pd.DataFrame) -> None:
    """Render export buttons for the branch view."""
    st.markdown("### ⬇️ Export Branch View")
    exports = export_dashboard_package(df)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button("Download CSV", exports["csv"], file_name="branch_view.csv", mime="text/csv")
    with c2:
        st.download_button("Download Excel", exports["excel"], file_name="branch_view.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Download PDF", exports["pdf"], file_name="branch_view_summary.pdf", mime="application/pdf")


def render_branches_page() -> None:
    """Render the complete branch analytics experience."""
    st.set_page_config(page_title=APP_NAME, page_icon="🏬", layout="wide", initial_sidebar_state="expanded")
    load_css()
    skeleton = st.empty()
    with skeleton.container():
        render_page_skeleton(
            "Branch Performance Intelligence",
            "Evaluate branch profitability, customer demand, and outlet excellence with a focused operational view.",
            metric_count=4,
            section_titles=["Branch Performance", "Branch Geography & Ranking"],
            chart_cards=4,
            table_cards=1,
            insight_cards=3,
        )

    data, filtered_data = _load_branch_data()
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
    st.session_state["branches_filtered"] = filtered_data

    if filtered_data.empty:
        st.warning("No rows match the selected filters. Please broaden the selection.")
        st.stop()

    with skeleton.container():
        with st.spinner("Preparing branch analytics..."):
            _render_hero(filtered_data)
            st.markdown("---")
            _render_kpi_cards(filtered_data)
            st.markdown("---")
            _render_charts(filtered_data)
            st.markdown("---")
            _render_branch_tables(filtered_data)
            st.markdown("---")
            _render_insights(filtered_data)
            st.markdown("---")
            _render_exports(filtered_data)


render_branches_page()
