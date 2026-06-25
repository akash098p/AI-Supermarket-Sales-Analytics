"""
AI-Powered Supermarket Sales Dashboard
utils/insights.py
Automatic business insight generation.
"""

from __future__ import annotations
import pandas as pd

def _safe_mode(series):
    m = series.mode()
    return m.iat[0] if not m.empty else "-"

def highest_revenue_branch(df):
    if {"Branch","Total"}.issubset(df.columns):
        s=df.groupby("Branch")["Total"].sum()
        return s.idxmax(), float(s.max())
    return "-",0

def lowest_revenue_branch(df):
    if {"Branch","Total"}.issubset(df.columns):
        s=df.groupby("Branch")["Total"].sum()
        return s.idxmin(), float(s.min())
    return "-",0

def best_product(df):
    col="Product" if "Product" in df.columns else "Product Line"
    if {col,"Quantity"}.issubset(df.columns):
        s=df.groupby(col)["Quantity"].sum()
        return s.idxmax(), int(s.max())
    return "-",0

def worst_product(df):
    col="Product" if "Product" in df.columns else "Product Line"
    if {col,"Quantity"}.issubset(df.columns):
        s=df.groupby(col)["Quantity"].sum()
        return s.idxmin(), int(s.min())
    return "-",0

def peak_hour(df):
    if {"Hour","Total"}.issubset(df.columns):
        s=df.groupby("Hour")["Total"].sum()
        return int(s.idxmax()), float(s.max())
    return None,0

def best_payment(df):
    if {"Payment","Total"}.issubset(df.columns):
        s=df.groupby("Payment")["Total"].sum()
        return s.idxmax(), float(s.max())
    return "-",0

def highest_rated_product(df):
    col="Product" if "Product" in df.columns else "Product Line"
    if {col,"Rating"}.issubset(df.columns):
        s=df.groupby(col)["Rating"].mean()
        return s.idxmax(), round(float(s.max()),2)
    return "-",0

def average_basket(df):
    if {"Total","Quantity"}.issubset(df.columns):
        return round((df["Total"]/df["Quantity"].replace(0,pd.NA)).mean(),2)
    return 0

def revenue_growth(df):
    if {"Date","Total"}.issubset(df.columns):
        d=df.copy()
        d["Month"]=pd.to_datetime(d["Date"]).dt.to_period("M")
        m=d.groupby("Month")["Total"].sum()
        if len(m)>=2:
            prev=m.iloc[-2]
            curr=m.iloc[-1]
            if prev!=0:
                return round((curr-prev)/prev*100,2)
    return 0

def generate_insights(df):
    insights=[]

    b,v=highest_revenue_branch(df)
    insights.append(f"🏆 Highest revenue branch: {b} (₹{v:,.2f})")

    b,v=lowest_revenue_branch(df)
    insights.append(f"📉 Lowest revenue branch: {b} (₹{v:,.2f})")

    p,q=best_product(df)
    insights.append(f"📦 Best-selling product: {p} ({q:,} units)")

    p,q=worst_product(df)
    insights.append(f"⚠️ Lowest-selling product: {p} ({q:,} units)")

    h,_=peak_hour(df)
    if h is not None:
        insights.append(f"🕒 Peak shopping hour: {h}:00")

    pay,val=best_payment(df)
    insights.append(f"💳 Most profitable payment: {pay} (₹{val:,.2f})")

    prod,r=highest_rated_product(df)
    insights.append(f"⭐ Highest rated product: {prod} ({r}/10)")

    insights.append(f"🛒 Average basket value: ₹{average_basket(df):,.2f}")

    insights.append(f"📈 Latest monthly growth: {revenue_growth(df)}%")

    if "Customer Type" in df.columns:
        insights.append(f"👥 Most loyal customer type: {_safe_mode(df['Customer Type'])}")

    insights.extend([
        "💡 Increase stock for top-selling products.",
        "💡 Run promotions during off-peak hours.",
        "💡 Reward loyal customers with exclusive offers.",
        "💡 Bundle slow-moving items with popular products.",
        "💡 Monitor branch performance monthly."
    ])

    return insights

def executive_summary(df):
    rev=df["Total"].sum() if "Total" in df.columns else 0
    orders=len(df)
    rating=round(df["Rating"].mean(),2) if "Rating" in df.columns else 0
    branch,_=highest_revenue_branch(df)
    return (
        f"The supermarket processed {orders:,} orders with total revenue of "
        f"₹{rev:,.2f}. The highest-performing branch is {branch}. "
        f"Average customer rating is {rating}/10. "
        "Focus on inventory optimization, customer retention, and targeted promotions "
        "to improve profitability."
    )
