

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from utils.analytics import dashboard_summary
from utils.charts import actual_vs_predicted, bar
from utils.config import PRIMARY, SECONDARY, SUCCESS, WARNING
from utils.data_loader import get_dataset, validate_dataset
from utils.ml_models import evaluate, prepare_features
from utils.page_helpers import (
    configure_page,
    load_shared_css,
    render_active_scope,
    render_export_buttons,
    render_sidebar_uploader,
    render_validation_issues,
    stop_for_empty_data,
    stop_for_empty_filters,
)
from utils.preprocessing import preprocess


def _format_currency(value: float) -> str:
    """Format monetary values."""
    return f"Rs.{value:,.2f}" if pd.notna(value) else "Rs.0.00"


def _load_prediction_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset with session-state caching."""
    uploaded = st.session_state.get("prediction_upload")
    if uploaded is not None:
        raw_df = get_dataset(uploaded)
        prepared = preprocess(raw_df)
        st.session_state["prediction_data"] = prepared
        st.session_state["prediction_filtered"] = prepared.copy()
        return prepared, prepared.copy()

    if "prediction_data" in st.session_state:
        data = st.session_state["prediction_data"]
        filtered = st.session_state.get("prediction_filtered", data.copy())
        return data, filtered

    raw_df = get_dataset()
    prepared = preprocess(raw_df)
    st.session_state["prediction_data"] = prepared
    st.session_state["prediction_filtered"] = prepared.copy()
    return prepared, prepared.copy()


def _prepare_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the data includes the columns needed by the prediction workflow."""
    prepared = df.copy()
    for numeric_column in ["Total", "Quantity", "Rating"]:
        if numeric_column in prepared.columns:
            prepared[numeric_column] = pd.to_numeric(prepared[numeric_column], errors="coerce")

    fallback_columns = {
        "Branch": "Unknown",
        "City": "Unknown",
        "Payment": "Unknown",
        "Customer Type": "Unknown",
        "Gender": "Unknown",
        "Product Line": "Unknown",
    }
    for column, fallback in fallback_columns.items():
        if column not in prepared.columns:
            prepared[column] = fallback

    return prepared


def _build_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render sidebar filters for the prediction page."""
    st.sidebar.markdown("## Prediction Filters")
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

    for column in ["Branch", "City", "Payment", "Customer Type", "Gender", "Product Line"]:
        if column in df.columns:
            values = sorted(df[column].dropna().astype(str).unique())
            filters[column.lower().replace(" ", "_")] = st.sidebar.multiselect(column, values, default=values)

    render_sidebar_uploader(
        "prediction_upload",
        help_text="Load a new CSV or Excel file for the forecasting page.",
    )

    return filters


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply sidebar filters to the active dataset."""
    filtered = df.copy()

    if "date_start" in filters and "date_end" in filters and "Date" in filtered.columns:
        filtered = filtered[(filtered["Date"] >= filters["date_start"]) & (filtered["Date"] <= filters["date_end"])]

    for column in ["Branch", "City", "Payment", "Customer Type", "Gender", "Product Line"]:
        filter_key = column.lower().replace(" ", "_")
        if filters.get(filter_key) and column in filtered.columns:
            filtered = filtered[filtered[column].astype(str).isin(filters[filter_key])]

    return filtered.reset_index(drop=True)


def _get_default_features(df: pd.DataFrame) -> List[str]:
    """Select a sensible subset of features for prediction modeling."""
    candidates = [
        "Quantity",
        "Rating",
        "Branch",
        "City",
        "Payment",
        "Customer Type",
        "Gender",
        "Product Line",
        "Month",
        "Year",
        "Hour",
        "Weekday",
    ]
    return [column for column in candidates if column in df.columns]


def _get_feature_defaults(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """Construct default values for a single prediction row."""
    defaults: Dict[str, Any] = {}
    for feature in features:
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            defaults[feature] = float(series.median()) if not series.dropna().empty else 0.0
        elif pd.api.types.is_datetime64_any_dtype(series):
            defaults[feature] = series.dropna().max() if not series.dropna().empty else pd.Timestamp.today()
        else:
            defaults[feature] = str(series.mode().iloc[0]) if not series.mode().empty else "Unknown"
    return defaults


def _build_prediction_frame(features: List[str], defaults: Dict[str, Any]) -> pd.DataFrame:
    """Create a single-row feature frame for prediction."""
    row = {feature: defaults.get(feature) for feature in features}
    frame = pd.DataFrame([row])
    frame["Total"] = 0.0
    return frame


def _train_model(df: pd.DataFrame, features: List[str], model_name: str) -> Dict[str, Any]:
    """Train a regression model on the filtered dataset."""
    model_df = df[features + ["Total"]].copy().dropna(subset=["Total"]).reset_index(drop=True)
    if model_df.empty or len(model_df) < 10:
        raise ValueError("Not enough rows to train a reliable model.")

    X, y = prepare_features(model_df, target="Total")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": evaluate(y_test, predictions),
        "feature_columns": list(X.columns),
    }


def _render_hero(df: pd.DataFrame, summary: Dict[str, Any]) -> None:
    """Render the hero banner for the prediction page."""
    today = datetime.now().strftime("%d %b %Y")
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    left, right = st.columns([2.2, 1.0])

    with left:
        st.markdown("<h1 style='margin-bottom:0.2rem;'>Demand Forecasting Studio</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:1.02rem; margin-top:0.25rem; opacity:0.95;'>"
            "Train regression models, compare forecast quality, and run what-if transaction scenarios from the active filtered dataset.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='display:flex; gap:0.7rem; flex-wrap:wrap; margin-top:0.45rem;'>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>{today}</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>{len(df):,} rows</span>"
            f"<span style='background:rgba(255,255,255,0.18); padding:0.35rem 0.7rem; border-radius:999px;'>{_format_currency(summary['Total Revenue'])} revenue</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.metric("Revenue", _format_currency(summary["Total Revenue"]), help="Revenue available for the current selection")
        st.metric("Avg Order", _format_currency(summary["Average Order Value"]), help="Typical basket value")
        st.metric("Orders", f"{summary['Total Orders']:,}", help="Transactions available for modeling")
        st.metric("Customers", f"{summary['Total Customers']:,}", help="Distinct customers in the filtered view")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_kpi_cards(df: pd.DataFrame) -> None:
    """Render prediction KPI cards."""
    summary = dashboard_summary(df)
    metrics: List[Dict[str, Any]] = [
        {"title": "Revenue", "value": _format_currency(summary["Total Revenue"]), "accent": PRIMARY},
        {"title": "Orders", "value": f"{summary['Total Orders']:,}", "accent": SECONDARY},
        {"title": "Avg Order", "value": _format_currency(summary["Average Order Value"]), "accent": SUCCESS},
        {"title": "Customers", "value": f"{summary['Total Customers']:,}", "accent": WARNING},
    ]

    cols = st.columns(4)
    for idx, metric in enumerate(metrics):
        with cols[idx % 4]:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left:6px solid {metric['accent']}; margin-bottom:1rem;">
                    <div class="kpi-title">{metric['title']}</div>
                    <div class="kpi-value">{metric['value']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_model_panel(df: pd.DataFrame) -> None:
    """Train and evaluate a forecasting model."""
    st.markdown("### Model Training")
    selected_features = st.session_state.get("prediction_features") or _get_default_features(df)
    if not selected_features:
        st.info("No compatible features are available for modeling in the current dataset.")
        return

    model_name = st.selectbox("Model", ["Linear Regression", "Random Forest"], key="prediction_model")
    if st.button("Train Model"):
        with st.spinner("Training the forecasting model..."):
            try:
                result = _train_model(df, selected_features, model_name)
                st.session_state["prediction_result"] = result
                st.session_state["prediction_features"] = selected_features
                st.success("Model trained successfully.")
            except Exception as exc:  # pragma: no cover - UI fallback
                st.error(f"Model training failed: {exc}")

    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]
        metrics = result["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R2", metrics["R2"])
        c2.metric("MAE", _format_currency(metrics["MAE"]))
        c3.metric("RMSE", _format_currency(metrics["RMSE"]))
        c4.metric("Features", len(result["feature_columns"]))

        actual = result["y_test"].to_numpy()
        predicted = result["predictions"]
        st.plotly_chart(actual_vs_predicted(actual, predicted, "Actual vs Predicted"), use_container_width=True)

        if model_name == "Random Forest":
            feature_importance = (
                pd.DataFrame({"Feature": result["feature_columns"], "Importance": result["model"].feature_importances_})
                .sort_values("Importance", ascending=False)
                .head(10)
            )
            st.markdown("#### Feature Importance")
            st.dataframe(feature_importance, use_container_width=True, hide_index=True)
            st.plotly_chart(bar(feature_importance, "Feature", "Importance", "Top Feature Importance"), use_container_width=True)


def _render_forecast_form(df: pd.DataFrame) -> None:
    """Capture a what-if scenario and generate a prediction."""
    st.markdown("### What-if Scenario")
    if "prediction_result" not in st.session_state:
        st.info("Train a model first to enable scenario forecasting.")
        return

    features = st.session_state.get("prediction_features") or _get_default_features(df)
    defaults = _get_feature_defaults(df, features)
    form = st.form("prediction_form")
    inputs: Dict[str, Any] = {}

    for feature in features:
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            inputs[feature] = form.number_input(feature, value=float(defaults[feature]), step=1.0)
        elif pd.api.types.is_datetime64_any_dtype(series):
            inputs[feature] = form.date_input(feature, value=pd.Timestamp(defaults[feature]).date())
        else:
            values = sorted(df[feature].dropna().astype(str).unique())
            inputs[feature] = form.selectbox(feature, values, index=0 if values else None)

    if form.form_submit_button("Predict"):
        prediction_frame = _build_prediction_frame(features, inputs)
        model = st.session_state["prediction_result"]["model"]
        X_row, _ = prepare_features(prediction_frame, target="Total")
        prediction = float(model.predict(X_row)[0])
        st.success(f"Estimated transaction total: {_format_currency(prediction)}")
        st.caption("This forecast uses the trained model and the values from the scenario form.")


def _render_insights(df: pd.DataFrame) -> None:
    """Render insight cards for actionability."""
    st.markdown("### Forecast Insights")
    summary = dashboard_summary(df)
    cards = [
        f"The current selection contains {summary['Total Orders']:,} transactions with an average order value of {_format_currency(summary['Average Order Value'])}.",
        "Forecast quality improves when you include stronger contextual features such as branch, payment method, and product line.",
        "Use the what-if form to estimate how a change in quantity, region, or payment behavior might affect revenue.",
    ]
    for card in cards:
        st.markdown(f"<div class='insight-card'>{card}</div>", unsafe_allow_html=True)


def _render_exports(df: pd.DataFrame) -> None:
    """Render export actions for the prediction view."""
    render_export_buttons(df, "prediction_view", "Export Forecast View")


def render_prediction_page() -> None:
    """Render the complete prediction analytics experience."""
    configure_page("🔮")
    load_shared_css()

    data, _ = _load_prediction_data()
    if data.empty:
        stop_for_empty_data("The active dataset is empty.")

    render_validation_issues(validate_dataset(data))

    filters = _build_filters(data)
    filtered_data = _apply_filters(data, filters)
    st.session_state["prediction_filtered"] = filtered_data

    if filtered_data.empty:
        stop_for_empty_filters()

    prepared_df = _prepare_prediction_frame(filtered_data)
    summary = dashboard_summary(prepared_df)

    with st.spinner("Preparing forecast workspace..."):
        render_active_scope(prepared_df)
        _render_hero(prepared_df, summary)
        st.markdown("---")
        _render_kpi_cards(prepared_df)
        st.markdown("---")
        _render_model_panel(prepared_df)
        st.markdown("---")
        _render_forecast_form(prepared_df)
        st.markdown("---")
        _render_insights(prepared_df)
        st.markdown("---")
        _render_exports(prepared_df)


render_prediction_page()
