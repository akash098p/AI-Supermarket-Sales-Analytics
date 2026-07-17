"""
Shared Streamlit page helpers for consistent UI, UX, and reliability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

from utils.config import APP_NAME, STYLE_PATH
from utils.exports import export_dashboard_package


def configure_page(page_icon: str) -> None:
    """Apply a consistent Streamlit page configuration."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def load_shared_css(style_path: Path = STYLE_PATH) -> None:
    """Inject the shared stylesheet when available."""
    if style_path.exists():
        st.markdown(
            f"<style>{style_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def render_validation_issues(issues: Iterable[str], title: str = "Dataset Validation") -> None:
    """Display dataset validation messages in a single expandable section."""
    issue_list = [issue for issue in issues if issue]
    if not issue_list:
        return

    with st.expander(f"Warning: {title}", expanded=False):
        st.markdown(
            "These checks are advisory. The app will still try to render with the data that is available."
        )
        for issue in issue_list:
            st.warning(issue)


def stop_for_empty_data(message: str) -> None:
    """Render a consistent hard-stop state for empty datasets."""
    st.error(message)
    st.stop()


def stop_for_empty_filters(message: str = "No rows match the selected filters.") -> None:
    """Render a more helpful empty-state when filters remove all rows."""
    st.markdown(
        f"""
        <div class="warning-card">
            <strong>No matching data</strong><br>
            {message} Try widening the date range or re-enabling one of the sidebar filters.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


def render_active_scope(df: pd.DataFrame, label: str = "transactions") -> None:
    """Show a compact status strip describing the current filtered scope."""
    total_rows = len(df)
    branch_count = df["Branch"].nunique() if "Branch" in df.columns else 0
    city_count = df["City"].nunique() if "City" in df.columns else 0

    date_text = "Date range unavailable"
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if not dates.empty:
            date_text = f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}"

    st.caption(
        f"Active scope: {total_rows:,} {label} | {branch_count} branches | {city_count} cities | {date_text}"
    )


def render_profile_summary(profile: dict, *, expanded: bool = False) -> None:
    """Render a compact dataset profile without dumping long JSON arrays."""
    summary_rows = [
        {"Metric": "Rows", "Value": f"{profile.get('rows', 0):,}"},
        {"Metric": "Columns", "Value": f"{profile.get('columns', 0):,}"},
        {"Metric": "Memory (MB)", "Value": f"{profile.get('memory_mb', 0):.2f}"},
        {"Metric": "Duplicates", "Value": f"{profile.get('duplicates', 0):,}"},
        {"Metric": "Missing Values", "Value": f"{profile.get('missing', 0):,}"},
        {"Metric": "Numeric Columns", "Value": f"{len(profile.get('numeric_columns', [])):,}"},
        {"Metric": "Categorical Columns", "Value": f"{len(profile.get('categorical_columns', [])):,}"},
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    with st.expander("Column Details", expanded=expanded):
        st.markdown("**Numeric Columns**")
        numeric_columns = profile.get("numeric_columns", [])
        st.write(", ".join(map(str, numeric_columns)) if numeric_columns else "None")

        st.markdown("**Categorical Columns**")
        categorical_columns = profile.get("categorical_columns", [])
        st.write(", ".join(map(str, categorical_columns)) if categorical_columns else "None")


def render_sidebar_uploader(
    upload_key: str,
    label: str = "Upload Dataset",
    help_text: str = "Load a CSV or Excel file for this page.",
) -> Optional[object]:
    """Render a consistent sidebar upload control."""
    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader(
        label,
        type=["csv", "xlsx", "xls"],
        key=upload_key,
        help=help_text,
    )
    st.sidebar.caption("Supported formats: CSV, XLSX, XLS")
    if uploaded is not None:
        st.sidebar.success("Dataset updated for this page.")
    return uploaded


def render_export_buttons(df: pd.DataFrame, file_stem: str, heading: str) -> None:
    """Render consistent export actions for the active filtered view."""
    st.markdown(f"### {heading}")
    exports = export_dashboard_package(df)
    col_csv, col_excel, col_pdf = st.columns(3)

    with col_csv:
        st.download_button(
            "Download CSV",
            exports["csv"],
            file_name=f"{file_stem}.csv",
            mime="text/csv",
        )

    with col_excel:
        st.download_button(
            "Download Excel",
            exports["excel"],
            file_name=f"{file_stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col_pdf:
        st.download_button(
            "Download PDF",
            exports["pdf"],
            file_name=f"{file_stem}_summary.pdf",
            mime="application/pdf",
        )
