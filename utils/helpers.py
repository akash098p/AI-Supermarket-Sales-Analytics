"""
===========================================================
AI-Powered Supermarket Sales Dashboard
Module: Helper Functions
Author: Akash Pramanik
===========================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime


# =========================================================
# Currency Formatting
# =========================================================

def format_currency(value):

    if pd.isna(value):
        return "₹0"

    return f"₹{value:,.2f}"


# =========================================================
# Integer Formatting
# =========================================================

def format_number(value):

    if pd.isna(value):
        return "0"

    return f"{int(value):,}"


# =========================================================
# Percentage Formatting
# =========================================================

def format_percentage(value):

    if pd.isna(value):
        return "0%"

    return f"{value:.2f}%"


# =========================================================
# Large Number Formatting
# =========================================================

def compact_number(number):

    if pd.isna(number):
        return "0"

    if number >= 10000000:
        return f"{number/10000000:.2f} Cr"

    if number >= 100000:
        return f"{number/100000:.2f} L"

    if number >= 1000:
        return f"{number/1000:.2f} K"

    return str(round(number,2))


# =========================================================
# Growth Percentage
# =========================================================

def growth_percentage(current, previous):

    if previous == 0:
        return 0

    return round(
        ((current - previous) / previous) * 100,
        2
    )


# =========================================================
# KPI Card Color
# =========================================================

def growth_color(value):

    if value > 0:
        return "green"

    elif value < 0:
        return "red"

    return "gray"


# =========================================================
# Rating Stars
# =========================================================

def rating_stars(rating):

    rating = round(rating)

    return "⭐" * rating


# =========================================================
# Greeting
# =========================================================

def greeting():

    hour = datetime.now().hour

    if hour < 12:
        return "Good Morning ☀️"

    elif hour < 17:
        return "Good Afternoon 🌤️"

    return "Good Evening 🌙"


# =========================================================
# Data Quality Score
# =========================================================

def calculate_quality_score(df):

    score = 100

    missing = df.isnull().sum().sum()

    duplicate = df.duplicated().sum()

    score -= missing * 0.02

    score -= duplicate * 0.05

    score = max(score, 0)

    return round(score, 2)


# =========================================================
# Top N
# =========================================================

def top_records(df, column, n=10):

    return (
        df[column]
        .value_counts()
        .head(n)
    )


# =========================================================
# Bottom N
# =========================================================

def bottom_records(df, column, n=10):

    return (
        df[column]
        .value_counts()
        .tail(n)
    )


# =========================================================
# Revenue by Group
# =========================================================

def revenue_by(df, group_column):

    if (
        group_column not in df.columns
        or
        "Total" not in df.columns
    ):
        return pd.DataFrame()

    return (
        df.groupby(group_column)["Total"]
        .sum()
        .reset_index()
        .sort_values(
            "Total",
            ascending=False
        )
    )


# =========================================================
# Quantity by Group
# =========================================================

def quantity_by(df, group_column):

    if (
        group_column not in df.columns
        or
        "Quantity" not in df.columns
    ):
        return pd.DataFrame()

    return (
        df.groupby(group_column)["Quantity"]
        .sum()
        .reset_index()
        .sort_values(
            "Quantity",
            ascending=False
        )
    )


# =========================================================
# Average Rating
# =========================================================

def average_rating(df):

    if "Rating" not in df.columns:
        return 0

    return round(
        df["Rating"].mean(),
        2
    )


# =========================================================
# Revenue
# =========================================================

def total_revenue(df):

    if "Total" not in df.columns:
        return 0

    return df["Total"].sum()


# =========================================================
# Gross Income
# =========================================================

def gross_income(df):

    if "Gross Income" not in df.columns:
        return 0

    return df["Gross Income"].sum()


# =========================================================
# Total Orders
# =========================================================

def total_orders(df):

    return len(df)


# =========================================================
# Total Products Sold
# =========================================================

def products_sold(df):

    if "Quantity" not in df.columns:
        return 0

    return df["Quantity"].sum()


# =========================================================
# Average Order Value
# =========================================================

def average_order(df):

    if "Total" not in df.columns:
        return 0

    return round(
        df["Total"].mean(),
        2
    )


# =========================================================
# Highest Sales Branch
# =========================================================

def highest_branch(df):

    if (
        "Branch" not in df.columns
        or
        "Total" not in df.columns
    ):
        return "-"

    return (
        df.groupby("Branch")["Total"]
        .sum()
        .idxmax()
    )


# =========================================================
# Best Product
# =========================================================

def best_product(df):

    if (
        "Product" not in df.columns
        or
        "Quantity" not in df.columns
    ):
        return "-"

    return (
        df.groupby("Product")["Quantity"]
        .sum()
        .idxmax()
    )


# =========================================================
# Random Color Generator
# =========================================================

def random_color():

    colors = [
        "#3498db",
        "#2ecc71",
        "#9b59b6",
        "#e74c3c",
        "#f39c12",
        "#1abc9c",
        "#34495e",
        "#e67e22",
        "#16a085",
        "#8e44ad"
    ]

    return np.random.choice(colors)