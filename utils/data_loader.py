"""
AI-Powered Supermarket Sales Dashboard
utils/data_loader.py
Production-ready data loading utilities.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import csv
import hashlib
import logging

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data") / "supermarket_sales.csv"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
ACTIVE_UPLOAD_BYTES_KEY = "active_upload_bytes"
ACTIVE_UPLOAD_NAME_KEY = "active_upload_name"
ACTIVE_DATASET_TOKEN_KEY = "active_dataset_token"

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


class _SessionUpload(BytesIO):
    """In-memory uploaded file compatible with pandas readers."""

    def __init__(self, data: bytes, name: str) -> None:
        super().__init__(data)
        self.name = name

@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    if not DEFAULT_DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DEFAULT_DATASET}")
    return load_file(DEFAULT_DATASET)

def load_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix or 'unknown'}. Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    uploaded_file.seek(0)
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type.")


def set_active_uploaded_file(uploaded_file: Any) -> None:
    """Persist the active uploaded dataset across page navigations."""
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    st.session_state[ACTIVE_UPLOAD_BYTES_KEY] = file_bytes
    st.session_state[ACTIVE_UPLOAD_NAME_KEY] = uploaded_file.name
    digest = hashlib.md5(file_bytes).hexdigest()[:12]
    st.session_state[ACTIVE_DATASET_TOKEN_KEY] = f"upload:{uploaded_file.name}:{len(file_bytes)}:{digest}"


def get_active_uploaded_file() -> Optional[_SessionUpload]:
    """Return the currently active uploaded file from session state."""
    file_bytes = st.session_state.get(ACTIVE_UPLOAD_BYTES_KEY)
    file_name = st.session_state.get(ACTIVE_UPLOAD_NAME_KEY)
    if not file_bytes or not file_name:
        return None
    return _SessionUpload(file_bytes, file_name)


def get_active_dataset_token() -> str:
    """Return a stable token representing the current app-wide dataset source."""
    return st.session_state.get(
        ACTIVE_DATASET_TOKEN_KEY,
        f"default:{DEFAULT_DATASET.resolve()}",
    )

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
    if uploaded_file is not None:
        set_active_uploaded_file(uploaded_file)
        return load_uploaded_file(uploaded_file)

    active_upload = get_active_uploaded_file()
    if active_upload is not None:
        return load_uploaded_file(active_upload)

    return load_default_dataset()


def load_page_dataset(
    page_key: str,
    loader: Callable[[pd.DataFrame], pd.DataFrame],
    uploaded_file: Any = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and cache the active dataset for a page while tracking global upload changes."""
    data_key = f"{page_key}_data"
    filtered_key = f"{page_key}_filtered"
    token_key = f"{page_key}_token"

    if uploaded_file is not None:
        raw_df = get_dataset(uploaded_file)
        prepared = loader(raw_df)
        token = get_active_dataset_token()
        st.session_state[data_key] = prepared
        st.session_state[filtered_key] = prepared.copy()
        st.session_state[token_key] = token
        return prepared, prepared.copy()

    token = get_active_dataset_token()
    if data_key in st.session_state and st.session_state.get(token_key) == token:
        data = st.session_state[data_key]
        filtered = st.session_state.get(filtered_key, data.copy())
        return data, filtered

    raw_df = get_dataset()
    prepared = loader(raw_df)
    st.session_state[data_key] = prepared
    st.session_state[filtered_key] = prepared.copy()
    st.session_state[token_key] = token
    return prepared, prepared.copy()

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
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        issues.append(f"Duplicate column names detected: {', '.join(map(str, duplicate_columns[:5]))}")

    m=smart_column_mapping(df)
    for key in ("sales","date","quantity"):
        if m.get(key) is None:
            aliases = ", ".join(COLUMN_ALIASES[key][:3])
            issues.append(f"Missing expected {key} column. Example accepted names: {aliases}.")
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
