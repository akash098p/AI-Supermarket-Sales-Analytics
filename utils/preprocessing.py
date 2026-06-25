"""
AI-Powered Supermarket Sales Dashboard
utils/preprocessing.py
Production-ready preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Basic Cleaning
# ---------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all")

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes(include="number").columns
    cat = df.select_dtypes(exclude="number").columns

    for c in num:
        df[c] = df[c].fillna(df[c].median())

    for c in cat:
        mode = df[c].mode()
        value = mode.iloc[0] if not mode.empty else "Unknown"
        df[c] = df[c].fillna(value)

    return df

# ---------------------------
# Data Types
# ---------------------------

def convert_date(df: pd.DataFrame, column="Date") -> pd.DataFrame:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df

def convert_time(df: pd.DataFrame, column="Time") -> pd.DataFrame:
    if column in df.columns:
        try:
            df[column] = pd.to_datetime(df[column], format="%H:%M")
        except Exception:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df

# ---------------------------
# Feature Engineering
# ---------------------------

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        return df

    d = df["Date"]

    df["Year"] = d.dt.year
    df["Quarter"] = d.dt.quarter
    df["Month"] = d.dt.month
    df["Month Name"] = d.dt.month_name()
    df["Week"] = d.dt.isocalendar().week.astype("Int64")
    df["Day"] = d.dt.day
    df["Weekday"] = d.dt.day_name()
    df["Weekend"] = d.dt.dayofweek >= 5

    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Time" not in df.columns:
        return df

    h = df["Time"].dt.hour
    df["Hour"] = h

    bins = [0,6,12,17,21,24]
    labels = ["Night","Morning","Afternoon","Evening","Late Evening"]

    df["Time Slot"] = pd.cut(
        h.fillna(0),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    return df

def calculate_profit_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if {"Gross Income","Total"}.issubset(df.columns):
        df["Profit Margin %"] = (
            df["Gross Income"] / df["Total"] * 100
        ).round(2)

    if {"Quantity","Total"}.issubset(df.columns):
        df["Average Basket Value"] = (
            df["Total"] / df["Quantity"].replace(0,np.nan)
        ).round(2)

    return df

def revenue_category(df: pd.DataFrame) -> pd.DataFrame:
    if "Total" not in df.columns:
        return df
    try:
        df["Revenue Category"] = pd.qcut(
            df["Total"],
            q=3,
            labels=["Low","Medium","High"]
        )
    except Exception:
        pass
    return df

# ---------------------------
# Outliers
# ---------------------------

def detect_outliers_iqr(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return pd.DataFrame()

    q1 = df[column].quantile(.25)
    q3 = df[column].quantile(.75)
    iqr = q3 - q1

    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr

    return df[(df[column] < lower) | (df[column] > upper)]

def remove_outliers(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return df

    q1 = df[column].quantile(.25)
    q3 = df[column].quantile(.75)
    iqr = q3 - q1

    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr

    return df[(df[column] >= lower) & (df[column] <= upper)]

# ---------------------------
# Encoding
# ---------------------------

def encode_categorical(df: pd.DataFrame):
    df = df.copy()
    encoders = {}

    for c in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    return df, encoders

# ---------------------------
# Quality
# ---------------------------

def quality_score(df: pd.DataFrame) -> float:
    score = 100
    score -= df.duplicated().sum() * 0.05
    score -= df.isna().sum().sum() * 0.02
    return round(max(score,0),2)

# ---------------------------
# Dataset Summary
# ---------------------------

def dataset_summary(df: pd.DataFrame):
    return {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing": int(df.isna().sum().sum()),
        "Duplicates": int(df.duplicated().sum()),
        "Quality Score": quality_score(df),
    }

# ---------------------------
# Full Pipeline
# ---------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_empty_rows(df)
    df = remove_duplicates(df)
    df = fill_missing(df)
    df = convert_date(df)
    df = convert_time(df)
    df = create_date_features(df)
    df = create_time_features(df)
    df = calculate_profit_metrics(df)
    df = revenue_category(df)

    if "Total" in df.columns:
        df = remove_outliers(df, "Total")

    return df.reset_index(drop=True)
