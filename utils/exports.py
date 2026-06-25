"""
AI-Powered Supermarket Sales Dashboard
utils/exports.py
Export utilities for CSV, Excel and PDF reports.
"""

from __future__ import annotations

from io import BytesIO
from datetime import datetime
from typing import Dict

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Return dataframe as CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel(df: pd.DataFrame, sheet_name: str = "Sales") -> bytes:
    """Return dataframe as Excel bytes."""
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    # Header
    for c, col in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=c, value=col)
        cell.font = Font(bold=True)

    # Data
    for row in df.itertuples(index=False):
        ws.append(list(row))

    stream = BytesIO()
    wb.save(stream)
    stream.seek(0)
    return stream.getvalue()


def summary_dict(df: pd.DataFrame) -> Dict[str, str]:
    """Generate basic dashboard summary."""
    revenue = df["Total"].sum() if "Total" in df.columns else 0
    orders = len(df)
    qty = df["Quantity"].sum() if "Quantity" in df.columns else 0
    rating = round(df["Rating"].mean(), 2) if "Rating" in df.columns else 0

    return {
        "Generated": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "Total Revenue": f"₹{revenue:,.2f}",
        "Total Orders": f"{orders:,}",
        "Products Sold": f"{qty:,}",
        "Average Rating": f"{rating}/10",
    }


def summary_to_pdf(summary: Dict[str, str], title="Dashboard Summary") -> bytes:
    """Generate PDF report from summary dictionary."""
    stream = BytesIO()

    doc = SimpleDocTemplate(stream)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    table_data = [["Metric", "Value"]]
    for k, v in summary.items():
        table_data.append([k, str(v)])

    table = Table(table_data, colWidths=[180, 250])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2F5597")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,1), (-1,-1), colors.beige),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
    ]))

    story.append(table)
    doc.build(story)

    stream.seek(0)
    return stream.getvalue()


def export_dashboard_package(df: pd.DataFrame):
    """
    Returns a dictionary containing:
        csv
        excel
        pdf
    """
    summary = summary_dict(df)

    return {
        "csv": dataframe_to_csv(df),
        "excel": dataframe_to_excel(df),
        "pdf": summary_to_pdf(summary),
    }
