

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.analytics import (
    average_rating,
    dashboard_summary,
    gross_income,
    quantity_by,
    revenue_by,
    top_products,
    total_products_sold,
    total_revenue,
)
from utils.charts import bar, donut, histogram, pie, sunburst, treemap
from utils.config import APP_NAME, PRIMARY, SECONDARY, STYLE_PATH, SUCCESS, WARNING
from utils.data_loader import get_dataset, load_page_dataset, validate_dataset
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


def _load_products_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the active dataset while keeping session state in sync."""
    uploaded = st.session_state.get("products_upload")
    return load_page_dataset(
        "products",
        lambda raw_df: _prepare_products_frame(preprocess(raw_df)),
        uploaded_file=uploaded,
    )


def _prepare_products_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns used by the product page."""
    prepared = df.copy()
    product_col = "Product" if "Product" in prepared.columns else "Product Line"
    if product_col not in prepared.columns:
        return prepared

    if "Date" in prepared.columns:
        prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
        prepared = prepared.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if "Brand" not in prepared.columns:
        prepared["Brand"] = "Unknown"

    if "Category" not in prepared.columns and "Product Line" in prepared.columns:
        prepared["Category"] = prepared["Product Line"]

    return prepared


def _build_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render sidebar filters and return the active selection state."""
    st.sidebar.markdown("## 🧭 Product Filters")
    filters: Dict[str, Any] = {}

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

    if "Brand" in df.columns:
        brands = sorted(df["Brand"].dropna().astype(str).unique())
        filters["brand"] = st.sidebar.multiselect("Brand", brands, default=brands)

    if "Category" in df.columns:
        categories = sorted(df["Category"].dropna().astype(str).unique())
        filters["category"] = st.sidebar.multiselect("Category", categories, default=categories)

    if "Customer Type" in df.columns:
        customer_types = sorted(df["Customer Type"].dropna().astype(str).unique())
        filters["customer_type"] = st.sidebar.multiselect("Customer Type", customer_types, default=customer_types)

    if "Payment" in df.columns:
        payments = sorted(df["Payment"].dropna().astype(str).unique())
        filters["payment"] = st.sidebar.multiselect("Payment", payments, default=payments)

    if "Rating" in df.columns:
        rating_min, rating_max = float(df["Rating"].min()), float(df["Rating"].max())
        filters["rating"] = st.sidebar.slider("Rating", min_value=float(rating_min), max_value=float(rating_max), value=(float(rating_min), float(rating_max)), step=0.1)

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"], key="products_upload")
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

    if "brand" in filters and "Brand" in filtered.columns:
        if filters["brand"]:
            filtered = filtered[filtered["Brand"].astype(str).isin(filters["brand"])]

    if "category" in filters and "Category" in filtered.columns:
        if filters["category"]:
            filtered = filtered[filtered["Category"].astype(str).isin(filters["category"])]

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
    """Render the hero banner and overview metrics."""
    summary = dashboard_summary(df)
    today = datetime.now().strftime("%d %b %Y")

    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    left, right = st.columns([2.2, 1.0])

    with left:
        st.markdown("<h1 style='margin-bottom:0.2rem;'>📦 Product Performance Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.02rem; margin-top:0.25rem; opacity:0.95;'>Analyze product demand, revenue contribution, assortment mix, and category health.</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.45rem;'>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🗓️ {today}</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>📊 {len(df):,} transactions</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>⭐ {summary['Average Rating']:.1f}/10</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.metric("Revenue", _format_currency(summary["Total Revenue"]), help="Revenue generated by the filtered assortment")
        st.metric("Units Sold", f"{summary['Products Sold']:,}", help="Total quantity sold")
        st.metric("Orders", f"{summary['Total Orders']:,}", help="Number of transactions")
        st.metric("Best Product", summary["Best Product"], help="Highest-selling product")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_kpi_cards(df: pd.DataFrame) -> None:
    """Render KPI cards for product performance."""
    summary = dashboard_summary(df)
    product_col = "Product" if "Product" in df.columns else "Product Line"
    top_product = top_products(df, 1).iloc[0][product_col] if not top_products(df, 1).empty else "-"

    metrics: List[Dict[str, Any]] = [
        {"title": "Revenue", "value": _format_currency(summary["Total Revenue"]), "icon": "💰", "accent": PRIMARY},
        {"title": "Units Sold", "value": f"{summary['Products Sold']:,}", "icon": "📦", "accent": SECONDARY},
        {"title": "Top Product", "value": top_product, "icon": "🏆", "accent": SUCCESS},
        {"title": "Gross Income", "value": _format_currency(summary["Gross Income"]), "icon": "📈", "accent": WARNING},
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


def _build_product_sections(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare summary tables for top products, bottom products, and category revenue."""
    product_col = "Product" if "Product" in df.columns else "Product Line"
    frames: Dict[str, pd.DataFrame] = {}

    if product_col in df.columns and "Quantity" in df.columns:
        frames["top_products"] = (
            df.groupby(product_col, as_index=False)["Quantity"]
            .sum()
            .sort_values("Quantity", ascending=False)
            .head(10)
        )
        frames["bottom_products"] = (
            df.groupby(product_col, as_index=False)["Quantity"]
            .sum()
            .sort_values("Quantity", ascending=True)
            .head(10)
        )
    else:
        frames["top_products"] = pd.DataFrame(columns=[product_col, "Quantity"])
        frames["bottom_products"] = pd.DataFrame(columns=[product_col, "Quantity"])

    if product_col in df.columns and "Total" in df.columns:
        frames["revenue_by_product"] = (
            df.groupby(product_col, as_index=False)["Total"]
            .sum()
            .sort_values("Total", ascending=False)
            .head(12)
        )
    else:
        frames["revenue_by_product"] = pd.DataFrame(columns=[product_col, "Total"])

    if "Category" in df.columns and "Total" in df.columns:
        frames["category_revenue"] = df.groupby("Category", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
    elif product_col in df.columns and "Product Line" in df.columns and "Total" in df.columns:
        frames["category_revenue"] = (
            df.groupby("Product Line", as_index=False)["Total"]
            .sum()
            .sort_values("Total", ascending=False)
            .rename(columns={"Product Line": "Category"})
        )
    elif product_col in df.columns and "Total" in df.columns:
        frames["category_revenue"] = (
            df.groupby(product_col, as_index=False)["Total"]
            .sum()
            .sort_values("Total", ascending=False)
            .rename(columns={product_col: "Category"})
        )
    else:
        frames["category_revenue"] = pd.DataFrame(columns=["Category", "Total"])

    if "Brand" in df.columns and "Total" in df.columns:
        frames["brand_revenue"] = df.groupby("Brand", as_index=False)["Total"].sum().sort_values("Total", ascending=False).head(12)
    else:
        frames["brand_revenue"] = pd.DataFrame(columns=["Brand", "Total"])

    return frames


def _render_charts(df: pd.DataFrame) -> None:
    """Render the main product analytics charts."""
    frames = _build_product_sections(df)
    product_col = "Product" if "Product" in df.columns else "Product Line"

    st.markdown("### 📈 Product Analytics")

    top_left, top_right = st.columns(2)
    with top_left:
        if not frames["top_products"].empty:
            chart = bar(frames["top_products"].head(10), product_col, "Quantity", "Top Products by Quantity", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Top products chart unavailable.")

    with top_right:
        if not frames["bottom_products"].empty:
            chart = bar(frames["bottom_products"].head(10), product_col, "Quantity", "Bottom Products by Quantity", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Bottom products chart unavailable.")

    mid_left, mid_right = st.columns(2)
    with mid_left:
        if not frames["revenue_by_product"].empty:
            chart = bar(frames["revenue_by_product"], product_col, "Total", "Revenue by Product", horizontal=True)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Revenue by product chart unavailable.")

    with mid_right:
        if not frames["category_revenue"].empty:
            chart = pie(frames["category_revenue"], "Category", "Total", "Category Revenue Distribution")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Category distribution chart unavailable.")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        if not frames["brand_revenue"].empty:
            chart = donut(frames["brand_revenue"], "Brand", "Total", "Brand Revenue Distribution")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Brand analysis unavailable.")

    with lower_right:
        if not frames["top_products"].empty and not frames["revenue_by_product"].empty:
            combined = pd.merge(frames["top_products"], frames["revenue_by_product"], on=product_col, how="outer")
            combined = combined.fillna(0)
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=combined[product_col], y=combined["Quantity"], mode="markers", name="Units Sold", marker=dict(size=combined["Quantity"].astype(float) / 2, color=PRIMARY)))
            chart.add_trace(go.Scatter(x=combined[product_col], y=combined["Total"], mode="lines+markers", name="Revenue"))
            chart.update_layout(title="Quantity vs Revenue", template="plotly_white", hovermode="x unified")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Quantity vs revenue analysis unavailable.")

    st.markdown("### 🌐 Assortment Intelligence")
    left, right = st.columns(2)
    with left:
        if not frames["revenue_by_product"].empty:
            chart = treemap(frames["revenue_by_product"], path=[product_col], values="Total", title="Product Revenue Treemap")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Treemap unavailable.")

    with right:
        if not frames["category_revenue"].empty:
            chart = sunburst(frames["category_revenue"], path=["Category"], values="Total", title="Category Sunburst")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Sunburst unavailable.")


def _render_product_tables(df: pd.DataFrame) -> None:
    """Render tables for top and bottom products plus category context."""
    frames = _build_product_sections(df)
    product_col = "Product" if "Product" in df.columns else "Product Line"

    st.markdown("### 🧾 Assortment Tables")
    left, right = st.columns(2)

    with left:
        if not frames["top_products"].empty:
            st.markdown("#### Top Products")
            st.dataframe(frames["top_products"].head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No top product data available.")

    with right:
        if not frames["bottom_products"].empty:
            st.markdown("#### Bottom Products")
            st.dataframe(frames["bottom_products"].head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No bottom product data available.")

    if not frames["category_revenue"].empty:
        st.markdown("#### Category Revenue")
        st.dataframe(frames["category_revenue"], use_container_width=True, hide_index=True)

    if not frames["brand_revenue"].empty:
        st.markdown("#### Brand Revenue")
        st.dataframe(frames["brand_revenue"], use_container_width=True, hide_index=True)


def _render_insights(df: pd.DataFrame) -> None:
    """Render product insights."""
    st.markdown("### 🧠 Product Insights")
    summary = dashboard_summary(df)
    product_col = "Product" if "Product" in df.columns else "Product Line"
    top_product = top_products(df, 1).iloc[0][product_col] if not top_products(df, 1).empty else "-"

    insight_cards = [
        f"The highest-selling item in the current view is {top_product}.",
        f"The filtered assortment generated { _format_currency(summary['Total Revenue']) } in revenue and {summary['Products Sold']:,} units sold.",
        f"Average customer satisfaction for this selection is {summary['Average Rating']:.1f}/10.",
    ]

    for card in insight_cards:
        st.markdown(f"<div class='insight-card'>{card}</div>", unsafe_allow_html=True)


def _render_exports(df: pd.DataFrame) -> None:
    """Render export buttons for the product view."""
    st.markdown("### ⬇️ Export Product View")
    exports = export_dashboard_package(df)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button("Download CSV", exports["csv"], file_name="product_view.csv", mime="text/csv")
    with c2:
        st.download_button("Download Excel", exports["excel"], file_name="product_view.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Download PDF", exports["pdf"], file_name="product_view_summary.pdf", mime="application/pdf")


def render_products_page() -> None:
    """Render the complete product analytics dashboard."""
    st.set_page_config(page_title=APP_NAME, page_icon="📦", layout="wide", initial_sidebar_state="expanded")
    load_css()

    data, filtered_data = _load_products_data()
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
    st.session_state["products_filtered"] = filtered_data

    if filtered_data.empty:
        st.warning("No rows match the selected filters. Please broaden the selection.")
        st.stop()

    with st.spinner("Preparing product analytics..."):
        _render_hero(filtered_data)
        st.markdown("---")
        _render_kpi_cards(filtered_data)
        st.markdown("---")
        _render_charts(filtered_data)
        st.markdown("---")
        _render_product_tables(filtered_data)
        st.markdown("---")
        _render_insights(filtered_data)
        st.markdown("---")
        _render_exports(filtered_data)


render_products_page()
