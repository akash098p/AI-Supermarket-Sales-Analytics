"""
AI-Powered Supermarket Sales Analytics Dashboard
pages/sales.py
Sales performance analytics page.
"""

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
    highest_sales_branch,
    monthly_sales,
    total_orders,
    total_products_sold,
    total_revenue,
    weekday_sales,
)
from utils.charts import area, bar, line
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


def _load_sales_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the active dataset while keeping session state in sync."""
    uploaded = st.session_state.get("sales_upload")
    if uploaded is not None:
        raw_df = get_dataset(uploaded)
        prepared = preprocess(raw_df)
        st.session_state["sales_data"] = prepared
        st.session_state["sales_filtered"] = prepared.copy()
        return prepared, prepared.copy()

    if "sales_data" in st.session_state:
        data = st.session_state["sales_data"]
        filtered = st.session_state.get("sales_filtered", data.copy())
        return data, filtered

    raw_df = get_dataset()
    prepared = preprocess(raw_df)
    st.session_state["sales_data"] = prepared
    st.session_state["sales_filtered"] = prepared.copy()
    return prepared, prepared.copy()


def _prepare_sales_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the sales frame for time-based analysis."""
    prepared = df.copy()

    if "Date" in prepared.columns:
        prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
        prepared = prepared.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if "Date" in prepared.columns and "Month Name" not in prepared.columns:
        prepared["Month Name"] = prepared["Date"].dt.month_name()

    if "Date" in prepared.columns and "Weekday" not in prepared.columns:
        prepared["Weekday"] = prepared["Date"].dt.day_name()

    if "Time" in prepared.columns and "Hour" not in prepared.columns:
        prepared["Hour"] = pd.to_datetime(prepared["Time"], errors="coerce").dt.hour

    if "Date" in prepared.columns and "Day" not in prepared.columns:
        prepared["Day"] = prepared["Date"].dt.day

    return prepared


def _build_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render sidebar filters and return the active selection state."""
    st.sidebar.markdown("## 🧭 Sales Filters")
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
        quantity_min, quantity_max = int(df["Quantity"].min()), int(df["Quantity"].max())
        filters["quantity"] = st.sidebar.slider(
            "Quantity",
            min_value=quantity_min,
            max_value=quantity_max,
            value=(quantity_min, quantity_max),
            step=1,
        )

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"],
        key="sales_upload",
        help="Load a new CSV or Excel file for analysis.",
    )
    if uploaded is not None:
        st.sidebar.success("Dataset updated successfully.")

    return filters


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply the sidebar filters to the active dataset."""
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

    if "customer_type" in filters and "Customer Type" in filtered.columns:
        if filters["customer_type"]:
            filtered = filtered[filtered["Customer Type"].astype(str).isin(filters["customer_type"])]

    if "payment" in filters and "Payment" in filtered.columns:
        if filters["payment"]:
            filtered = filtered[filtered["Payment"].astype(str).isin(filters["payment"])]

    if "gender" in filters and "Gender" in filtered.columns:
        if filters["gender"]:
            filtered = filtered[filtered["Gender"].astype(str).isin(filters["gender"])]

    if "rating" in filters and "Rating" in filtered.columns:
        low, high = filters["rating"]
        filtered = filtered[(filtered["Rating"] >= low) & (filtered["Rating"] <= high)]

    if "quantity" in filters and "Quantity" in filtered.columns:
        low, high = filters["quantity"]
        filtered = filtered[(filtered["Quantity"] >= low) & (filtered["Quantity"] <= high)]

    return filtered.reset_index(drop=True)


def _render_hero(df: pd.DataFrame) -> None:
    """Render the sales hero banner."""
    summary = dashboard_summary(df)
    today = datetime.now().strftime("%d %b %Y")

    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    left, right = st.columns([2.2, 1.0])

    with left:
        st.markdown(
            "<h1 style='margin-bottom:0.2rem;'>📈 Sales Performance Intelligence</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:1.02rem; margin-top:0.25rem; opacity:0.95;'>"
            "A detailed view of sales cadence, revenue momentum, seasonality, and forecast direction.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.45rem;'>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🗓️ {today}</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>📦 {len(df):,} transactions</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>🏬 {summary['Best Branch']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.metric("Revenue", _format_currency(summary["Total Revenue"]), help="Revenue in the filtered period")
        st.metric("Orders", f"{summary['Total Orders']:,}", help="Number of transactions")
        st.metric("Avg Order", _format_currency(summary["Average Order Value"]), help="Average basket value")
        st.metric("Rating", f"{summary['Average Rating']:.1f}/10", help="Average customer rating")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_kpi_cards(df: pd.DataFrame) -> None:
    """Render the executive KPI cards for sales performance."""
    summary = dashboard_summary(df)
    metrics: List[Dict[str, Any]] = [
        {"title": "Revenue", "value": _format_currency(summary["Total Revenue"]), "icon": "💰", "accent": PRIMARY},
        {"title": "Orders", "value": f"{summary['Total Orders']:,}", "icon": "🧾", "accent": SECONDARY},
        {"title": "Products Sold", "value": f"{summary['Products Sold']:,}", "icon": "📦", "accent": SUCCESS},
        {"title": "Gross Income", "value": _format_currency(summary["Gross Income"]), "icon": "📈", "accent": WARNING},
        {"title": "Average Order", "value": _format_currency(summary["Average Order Value"]), "icon": "🧮", "accent": PRIMARY},
        {"title": "Average Rating", "value": f"{summary['Average Rating']:.1f}/10", "icon": "⭐", "accent": SECONDARY},
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


def _build_timeseries_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare the core time-series objects used by the sales page."""
    frames: Dict[str, pd.DataFrame] = {}

    if "Date" in df.columns:
        monthly = (
            df.assign(Month_Key=df["Date"].dt.to_period("M"))
            .groupby("Month_Key", as_index=False)["Total"]
            .sum()
        )
        monthly["Month_Key"] = monthly["Month_Key"].astype(str)
        monthly = monthly.sort_values("Month_Key").reset_index(drop=True)
        frames["monthly"] = monthly

        daily = (
            df.groupby("Date", as_index=False)["Total"]
            .sum()
            .sort_values("Date")
            .reset_index(drop=True)
        )
        frames["daily"] = daily

        weekly = (
            df.assign(Weekday=df["Date"].dt.day_name())
            .groupby("Weekday", as_index=False)["Total"]
            .sum()
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly = weekly.set_index("Weekday").reindex(weekday_order).reset_index().rename(columns={"index": "Weekday"})
        weekly = weekly.dropna(subset=["Total"]).reset_index(drop=True)
        frames["weekly"] = weekly

        if "Hour" in df.columns:
            hourly = df.groupby("Hour", as_index=False)["Total"].sum().sort_values("Hour")
            frames["hourly"] = hourly
        else:
            frames["hourly"] = pd.DataFrame(columns=["Hour", "Total"])
    else:
        frames["monthly"] = pd.DataFrame(columns=["Month_Key", "Total"])
        frames["daily"] = pd.DataFrame(columns=["Date", "Total"])
        frames["weekly"] = pd.DataFrame(columns=["Weekday", "Total"])
        frames["hourly"] = pd.DataFrame(columns=["Hour", "Total"])

    return frames


def _render_trend_overview(df: pd.DataFrame) -> None:
    """Render the overview charts for monthly, weekly, and daily trend analysis."""
    frames = _build_timeseries_frames(df)

    st.markdown("### 📊 Sales Trend Overview")
    top_left, top_right = st.columns(2)

    with top_left:
        if not frames["monthly"].empty:
            chart = line(frames["monthly"], "Month_Key", "Total", "Monthly Sales")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Monthly sales trend is unavailable for the current dataset.")

    with top_right:
        if not frames["weekly"].empty:
            chart = bar(frames["weekly"], "Weekday", "Total", "Revenue by Weekday")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Weekly revenue trend is unavailable.")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        if not frames["daily"].empty:
            chart = line(frames["daily"], "Date", "Total", "Daily Sales")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Daily sales trend is unavailable.")

    with lower_right:
        if not frames["hourly"].empty:
            chart = bar(frames["hourly"], "Hour", "Total", "Hourly Sales Volume")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Hourly sales analysis is unavailable.")


def _render_moving_analytics(df: pd.DataFrame) -> None:
    """Render rolling average, moving average, growth, and forecast charts."""
    frames = _build_timeseries_frames(df)
    st.markdown("### 🔄 Momentum & Forecast")

    left, right = st.columns(2)

    with left:
        if not frames["daily"].empty:
            daily = frames["daily"].copy()
            daily = daily.set_index("Date")
            daily["Rolling Average"] = daily["Total"].rolling(window=7, min_periods=1).mean()
            daily = daily.reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Total"], mode="lines", name="Revenue"))
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Rolling Average"], mode="lines", name="7-Day Rolling Avg"))
            fig.update_layout(title="Rolling Average", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rolling average chart is unavailable.")

    with right:
        if not frames["daily"].empty:
            daily = frames["daily"].copy()
            daily = daily.set_index("Date")
            daily["Moving Average"] = daily["Total"].rolling(window=30, min_periods=1).mean()
            daily = daily.reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Total"], mode="lines", name="Revenue"))
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Moving Average"], mode="lines", name="30-Day Moving Avg"))
            fig.update_layout(title="Moving Average", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Moving average chart is unavailable.")

    lower_left, lower_middle, lower_right = st.columns(3)

    with lower_left:
        if not frames["monthly"].empty:
            growth = frames["monthly"].copy()
            growth["Growth %"] = growth["Total"].pct_change() * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=growth["Month_Key"], y=growth["Growth %"], mode="lines+markers", name="Growth %"))
            fig.update_layout(title="Revenue Growth", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue growth is unavailable.")

    with lower_middle:
        if "Date" in df.columns and {"Month Name", "Weekday"}.issubset(df.columns):
            seasonality = (
                df.assign(Month=df["Date"].dt.month_name(), Weekday=df["Date"].dt.day_name())
                .pivot_table(index="Weekday", columns="Month", values="Total", aggfunc="sum", fill_value=0)
            )
            ordered_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            ordered_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            seasonality = seasonality.reindex(index=ordered_weekdays, columns=ordered_months, fill_value=0)
            fig = go.Figure(data=go.Heatmap(z=seasonality.values, x=seasonality.columns, y=seasonality.index, colorscale="Viridis"))
            fig.update_layout(title="Seasonality Heatmap", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Seasonality analysis is unavailable.")

    with lower_right:
        if not frames["monthly"].empty:
            monthly = frames["monthly"].copy()
            x = np.arange(len(monthly))
            y = monthly["Total"].astype(float).values
            slope, intercept = np.polyfit(x, y, 1)
            forecast_x = np.arange(len(monthly) + 3)
            forecast_y = slope * forecast_x + intercept
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["Month_Key"], y=monthly["Total"], mode="lines+markers", name="Observed"))
            fig.add_trace(go.Scatter(x=[*monthly["Month_Key"], *[f"Forecast {i}" for i in range(1, 4)]], y=[*monthly["Total"], *forecast_y[len(monthly):]], mode="lines+markers", name="Forecast"))
            fig.update_layout(title="Forecast Trend", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Forecast trend is unavailable.")


def _render_insights(df: pd.DataFrame) -> None:
    """Render sales insights and a compact summary table."""
    st.markdown("### 🧠 Sales Insights")
    summary = dashboard_summary(df)
    frames = _build_timeseries_frames(df)

    insight_cards = [
        f"Revenue is concentrated around {summary['Best Branch']} with an average basket value of { _format_currency(summary['Average Order Value']) }.",
        f"The current filter returns {summary['Total Orders']:,} transactions and {summary['Products Sold']:,} units sold.",
        f"Customer sentiment remains healthy, with an average rating of {summary['Average Rating']:.1f}/10.",
    ]

    for card in insight_cards:
        st.markdown(f"<div class='insight-card'>{card}</div>", unsafe_allow_html=True)

    if not frames["monthly"].empty:
        st.markdown("#### Monthly Summary")
        st.dataframe(frames["monthly"].tail(12), use_container_width=True, hide_index=True)


def _render_exports(df: pd.DataFrame) -> None:
    """Render export buttons for CSV, Excel, and PDF."""
    st.markdown("### ⬇️ Export Sales View")
    exports = export_dashboard_package(df)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button("Download CSV", exports["csv"], file_name="sales_view.csv", mime="text/csv")
    with c2:
        st.download_button("Download Excel", exports["excel"], file_name="sales_view.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Download PDF", exports["pdf"], file_name="sales_view_summary.pdf", mime="application/pdf")


def render_sales_page() -> None:
    """Render the complete sales analytics experience."""
    st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    load_css()

    data, filtered_data = _load_sales_data()
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
    st.session_state["sales_filtered"] = filtered_data

    if filtered_data.empty:
        st.warning("No rows match the selected filters. Please broaden the selection.")
        st.stop()

    with st.spinner("Preparing sales analytics..."):
        _render_hero(filtered_data)
        st.markdown("---")
        _render_kpi_cards(filtered_data)
        st.markdown("---")
        _render_trend_overview(filtered_data)
        st.markdown("---")
        _render_moving_analytics(filtered_data)
        st.markdown("---")
        _render_insights(filtered_data)
        st.markdown("---")
        _render_exports(filtered_data)


render_sales_page()
