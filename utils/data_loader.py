"""
AI-Powered Supermarket Sales Dashboard
utils/data_loader.py
Production-ready data loading utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import csv
import logging

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data") / "supermarket_sales.csv"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

COLUMN_ALIASES = {
    "date": ["date","bill date","invoice date"],
    "sales": ["total","sales","revenue","amount","net sales"],
    "quantity": ["quantity","qty","units"],
    "product": ["product","item","product name"],
    "branch": ["branch","store","outlet"],
    "city": ["city"],
    "customer": ["customer","customer name","customer id"],
    "payment": ["payment","payment method"],
    "rating": ["rating","review"],
}

def _normalize(name: str) -> str:
    return name.strip().lower().replace("_"," ").replace("-"," ")

def detect_encoding(path: Path) -> str:
    for enc in ("utf-8","utf-8-sig","latin1"):
        try:
            with open(path,"r",encoding=enc) as f:
                f.readline()
            return enc
        except Exception:
            pass
    return "latin1"

def detect_delimiter(path: Path, encoding: str) -> str:
    with open(path,"r",encoding=encoding,newline="") as f:
        sample = f.read(2048)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return ","

@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    if not DEFAULT_DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DEFAULT_DATASET}")
    return load_file(DEFAULT_DATASET)

@st.cache_data(show_spinner=False)
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx",".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type.")

def load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        enc = detect_encoding(path)
        delim = detect_delimiter(path, enc)
        return pd.read_csv(path, encoding=enc, sep=delim)
    if suffix in {".xlsx",".xls"}:
        return pd.read_excel(path)
    raise ValueError("Unsupported file type.")

def get_dataset(uploaded_file=None) -> pd.DataFrame:
    return load_uploaded_file(uploaded_file) if uploaded_file else load_default_dataset()

def smart_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping={}
    normalized={c:_normalize(c) for c in df.columns}
    for logical, aliases in COLUMN_ALIASES.items():
        mapping[logical]=None
        for col,n in normalized.items():
            if n in aliases:
                mapping[logical]=col
                break
    return mapping

def dataset_profile(df: pd.DataFrame) -> Dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum()/1024/1024,2),
        "duplicates": int(df.duplicated().sum()),
        "missing": int(df.isna().sum().sum()),
        "numeric_columns": list(df.select_dtypes("number").columns),
        "categorical_columns": list(df.select_dtypes(include="object").columns),
    }

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rep=pd.DataFrame({
        "Column":df.columns,
        "Missing":df.isna().sum().values,
        "Missing %":(df.isna().mean()*100).round(2).values
    })
    return rep.sort_values("Missing",ascending=False)

def validate_dataset(df: pd.DataFrame)->List[str]:
    issues=[]
    if df.empty:
        issues.append("Dataset is empty.")
    if len(df.columns)<5:
        issues.append("Very few columns detected.")
    m=smart_column_mapping(df)
    for key in ("sales","date","quantity"):
        if m.get(key) is None:
            issues.append(f"Missing expected column: {key}")
    return issues

def quality_score(df: pd.DataFrame)->float:
    score=100.0
    score-=df.duplicated().sum()*0.05
    score-=df.isna().sum().sum()*0.02
    return round(max(score,0),2)

def preview(df: pd.DataFrame, rows:int=10)->pd.DataFrame:
    return df.head(rows)

def numeric_summary(df: pd.DataFrame)->pd.DataFrame:
    return df.describe(include="number").T

def categorical_summary(df: pd.DataFrame)->pd.DataFrame:
    obj=df.select_dtypes(include="object")
    return obj.describe().T if not obj.empty else pd.DataFrame()
