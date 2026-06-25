"""
AI-Powered Supermarket Sales Dashboard
utils/analytics.py
Business analytics and KPI calculations.
"""

from __future__ import annotations
import pandas as pd


# ---------- Generic KPI ----------

def total_revenue(df: pd.DataFrame) -> float:
    return float(df["Total"].sum()) if "Total" in df.columns else 0.0

def total_orders(df: pd.DataFrame) -> int:
    return len(df)

def average_order_value(df: pd.DataFrame) -> float:
    return float(df["Total"].mean()) if "Total" in df.columns else 0.0

def total_products_sold(df: pd.DataFrame) -> int:
    return int(df["Quantity"].sum()) if "Quantity" in df.columns else 0

def gross_income(df: pd.DataFrame) -> float:
    return float(df["Gross Income"].sum()) if "Gross Income" in df.columns else 0.0

def average_rating(df: pd.DataFrame) -> float:
    return round(float(df["Rating"].mean()),2) if "Rating" in df.columns else 0.0

def total_customers(df: pd.DataFrame) -> int:
    if "Customer ID" in df.columns:
        return df["Customer ID"].nunique()
    if "Customer Name" in df.columns:
        return df["Customer Name"].nunique()
    return len(df)

# ---------- Group Analytics ----------

def revenue_by(df: pd.DataFrame, column: str):
    if column not in df.columns or "Total" not in df.columns:
        return pd.DataFrame()
    return (df.groupby(column,as_index=False)["Total"]
              .sum()
              .sort_values("Total",ascending=False))

def quantity_by(df: pd.DataFrame, column: str):
    if column not in df.columns or "Quantity" not in df.columns:
        return pd.DataFrame()
    return (df.groupby(column,as_index=False)["Quantity"]
              .sum()
              .sort_values("Quantity",ascending=False))

def average_rating_by(df: pd.DataFrame, column: str):
    if column not in df.columns or "Rating" not in df.columns:
        return pd.DataFrame()
    return (df.groupby(column,as_index=False)["Rating"]
              .mean()
              .sort_values("Rating",ascending=False))

# ---------- Dashboard Cards ----------

def highest_sales_branch(df):
    if "Branch" not in df.columns:
        return "-"
    return revenue_by(df,"Branch").iloc[0]["Branch"]

def highest_sales_city(df):
    if "City" not in df.columns:
        return "-"
    return revenue_by(df,"City").iloc[0]["City"]

def best_product(df):
    col = "Product" if "Product" in df.columns else "Product Line"
    if col not in df.columns:
        return "-"
    return quantity_by(df,col).iloc[0][col]

def preferred_payment(df):
    if "Payment" not in df.columns:
        return "-"
    return df["Payment"].mode().iat[0]

# ---------- Time Analytics ----------

def monthly_sales(df):
    if "Month Name" not in df.columns:
        return pd.DataFrame()
    return revenue_by(df,"Month Name")

def weekday_sales(df):
    if "Weekday" not in df.columns:
        return pd.DataFrame()
    return revenue_by(df,"Weekday")

def hourly_sales(df):
    if "Hour" not in df.columns:
        return pd.DataFrame()
    return revenue_by(df,"Hour")

# ---------- Product ----------

def top_products(df,n=10):
    col="Product" if "Product" in df.columns else "Product Line"
    return quantity_by(df,col).head(n)

def bottom_products(df,n=10):
    col="Product" if "Product" in df.columns else "Product Line"
    return quantity_by(df,col).tail(n)

# ---------- Customer ----------

def customer_type_summary(df):
    if "Customer Type" not in df.columns:
        return pd.DataFrame()
    return revenue_by(df,"Customer Type")

def gender_summary(df):
    if "Gender" not in df.columns:
        return pd.DataFrame()
    return revenue_by(df,"Gender")

# ---------- Finance ----------

def tax_summary(df):
    if "Tax" not in df.columns:
        return {}
    return {
        "Total Tax": float(df["Tax"].sum()),
        "Average Tax": float(df["Tax"].mean()),
        "Maximum Tax": float(df["Tax"].max()),
    }

# ---------- Correlation ----------

def correlation_matrix(df):
    return df.select_dtypes("number").corr()

# ---------- Summary ----------

def dashboard_summary(df):
    return {
        "Total Revenue": total_revenue(df),
        "Total Orders": total_orders(df),
        "Average Order Value": average_order_value(df),
        "Total Customers": total_customers(df),
        "Products Sold": total_products_sold(df),
        "Gross Income": gross_income(df),
        "Average Rating": average_rating(df),
        "Best Branch": highest_sales_branch(df),
        "Best City": highest_sales_city(df),
        "Best Product": best_product(df),
        "Preferred Payment": preferred_payment(df),
    }
