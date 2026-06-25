"""
AI-Powered Supermarket Sales Dashboard
utils/ml_models.py
Machine learning utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def prepare_features(df: pd.DataFrame,
                     target: str = "Total") -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare numeric feature matrix and target."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    data = df.copy()

    # Encode categoricals
    for c in data.select_dtypes(include="object").columns:
        data[c] = data[c].astype("category").cat.codes

    # Convert datetime columns
    for c in data.select_dtypes(include=["datetime64[ns]"]).columns:
        data[c] = data[c].view("int64") // 10**9

    data = data.fillna(0)

    X = data.drop(columns=[target])
    X = X.select_dtypes(include=np.number)

    y = data[target]

    return X, y


def evaluate(y_true, y_pred) -> Dict[str, float]:
    """Return evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "R2": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(rmse, 4),
    }


def train_linear_regression(df: pd.DataFrame,
                            target: str = "Total") -> Dict:
    X, y = prepare_features(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    joblib.dump(model, MODEL_DIR / "linear_regression.pkl")

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": pred,
        "metrics": evaluate(y_test, pred),
    }


def train_random_forest(df: pd.DataFrame,
                        target: str = "Total") -> Dict:
    X, y = prepare_features(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    importance = (
        pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    joblib.dump(model, MODEL_DIR / "random_forest.pkl")

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": pred,
        "metrics": evaluate(y_test, pred),
        "feature_importance": importance,
    }


def customer_segmentation(df: pd.DataFrame,
                          n_clusters: int = 3) -> pd.DataFrame:
    """KMeans clustering using Total, Quantity and Rating."""
    required = ["Total", "Quantity"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    features = ["Total", "Quantity"]

    if "Rating" in df.columns:
        features.append("Rating")

    X = df[features].fillna(0)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X)

    result = df.copy()
    result["Cluster"] = clusters

    labels = {
        0: "Low Value",
        1: "Medium Value",
        2: "High Value"
    }

    if n_clusters == 3:
        means = result.groupby("Cluster")["Total"].mean().sort_values()
        ordered = list(means.index)
        mapping = {
            ordered[0]: "Low Value",
            ordered[1]: "Medium Value",
            ordered[2]: "High Value",
        }
        result["Segment"] = result["Cluster"].map(mapping)
    else:
        result["Segment"] = result["Cluster"].map(labels).fillna("Cluster")

    joblib.dump(model, MODEL_DIR / "kmeans.pkl")

    return result


def load_model(model_name: str):
    """Load saved model."""
    path = MODEL_DIR / model_name
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def predict(model, feature_df: pd.DataFrame):
    """Predict using a trained model."""
    data = feature_df.copy()

    for c in data.select_dtypes(include="object").columns:
        data[c] = data[c].astype("category").cat.codes

    for c in data.select_dtypes(include=["datetime64[ns]"]).columns:
        data[c] = data[c].view("int64") // 10**9

    data = data.select_dtypes(include=np.number).fillna(0)

    return model.predict(data)
