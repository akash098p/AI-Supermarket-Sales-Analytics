"""
AI-Powered Supermarket Sales Dashboard
utils/charts.py
Reusable Plotly chart library.
"""

from __future__ import annotations
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

TEMPLATE = "plotly_white"

def _layout(fig,title):
    fig.update_layout(
        title=title,
        template=TEMPLATE,
        hovermode="x unified",
        margin=dict(l=20,r=20,t=50,b=20),
        legend_title=None
    )
    return fig

def bar(df,x,y,title,color=None,horizontal=False):
    fig = px.bar(df,x=y if horizontal else x,
                 y=x if horizontal else y,
                 color=color,
                 orientation="h" if horizontal else "v",
                 text_auto=True)
    return _layout(fig,title)

def line(df,x,y,title,color=None):
    fig=px.line(df,x=x,y=y,color=color,markers=True)
    return _layout(fig,title)

def area(df,x,y,title,color=None):
    fig=px.area(df,x=x,y=y,color=color)
    return _layout(fig,title)

def pie(df,names,values,title,hole=0):
    fig=px.pie(df,names=names,values=values,hole=hole)
    return _layout(fig,title)

def donut(df,names,values,title):
    return pie(df,names,values,title,hole=.5)

def scatter(df,x,y,title,color=None,size=None):
    fig=px.scatter(df,x=x,y=y,color=color,size=size)
    return _layout(fig,title)

def bubble(df,x,y,size,color,title):
    fig=px.scatter(df,x=x,y=y,size=size,color=color)
    return _layout(fig,title)

def histogram(df,column,title):
    fig=px.histogram(df,x=column)
    return _layout(fig,title)

def box(df,x,y,title,color=None):
    fig=px.box(df,x=x,y=y,color=color)
    return _layout(fig,title)

def violin(df,x,y,title,color=None):
    fig=px.violin(df,x=x,y=y,color=color,box=True)
    return _layout(fig,title)

def heatmap(corr,title="Correlation Heatmap"):
    fig=px.imshow(corr,text_auto=".2f",aspect="auto")
    return _layout(fig,title)

def treemap(df,path,values,title):
    fig=px.treemap(df,path=path,values=values)
    return _layout(fig,title)

def sunburst(df,path,values,title):
    fig=px.sunburst(df,path=path,values=values)
    return _layout(fig,title)

def funnel(df,x,y,title):
    fig=px.funnel(df,x=x,y=y)
    return _layout(fig,title)

def waterfall(labels,values,title):
    fig=go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=["relative"]*len(values)
    ))
    return _layout(fig,title)

def gauge(value,title,min_value=0,max_value=100):
    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text":title},
        gauge={"axis":{"range":[min_value,max_value]}}
    ))
    return _layout(fig,title)

def kpi(value,title,prefix="",suffix=""):
    fig=go.Figure(go.Indicator(
        mode="number",
        value=value,
        number={"prefix":prefix,"suffix":suffix},
        title={"text":title}
    ))
    return _layout(fig,title)

def actual_vs_predicted(actual,predicted,title="Actual vs Predicted"):
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=actual,name="Actual",mode="lines"))
    fig.add_trace(go.Scatter(y=predicted,name="Predicted",mode="lines"))
    return _layout(fig,title)

def sales_trend(df,date_col="Date",value_col="Total"):
    data=df.groupby(date_col,as_index=False)[value_col].sum()
    return line(data,date_col,value_col,"Sales Trend")

def branch_revenue(df):
    data=df.groupby("Branch",as_index=False)["Total"].sum()
    return bar(data,"Branch","Total","Revenue by Branch")

def city_revenue(df):
    data=df.groupby("City",as_index=False)["Total"].sum()
    return bar(data,"City","Total","Revenue by City")

def product_revenue(df):
    col="Product" if "Product" in df.columns else "Product Line"
    data=df.groupby(col,as_index=False)["Total"].sum()
    return bar(data,col,"Total","Revenue by Product")

def payment_distribution(df):
    data=df.groupby("Payment",as_index=False)["Total"].sum()
    return donut(data,"Payment","Total","Payment Distribution")

def rating_distribution(df):
    return histogram(df,"Rating","Rating Distribution")
