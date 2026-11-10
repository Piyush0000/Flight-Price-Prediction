"""
Feature engineering and transformation pipeline for flight price prediction.
Includes leak-free target encoding, domain feature creation, and preprocessors for sklearn and CatBoost.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import ProjectConfig
from src.utils import setup_logger

logger = setup_logger("feature_engineering")


class RouteTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Leak-free route target encoder. Computes the route mean target strictly
    on training data during fit(), with empirical Bayes smoothing towards the global mean.
    """

    def __init__(self, route_col: str = "route", smoothing_weight: float = 10.0):
        self.route_col = route_col
        self.smoothing_weight = smoothing_weight
        self.global_mean_: float = 0.0
        self.route_means_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "RouteTargetEncoder":
        y_series = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y
        self.global_mean_ = float(y_series.mean())

        # Group stats
        grouped = y_series.groupby(X[self.route_col])
        counts = grouped.count()
        means = grouped.mean()

        # Empirical Bayes smoothing: (count * mean + weight * global_mean) / (count + weight)
        smoothed = (counts * means + self.smoothing_weight * self.global_mean_) / (
            counts + self.smoothing_weight
        )
        self.route_means_ = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        encoded = (
            X[self.route_col]
            .map(self.route_means_)
            .fillna(self.global_mean_)
            .astype(float)
            .values.reshape(-1, 1)
        )
        return encoded


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct all domain-specific features required by project specification:
    - route: string of source_destination
    - is_direct: 1 if stops is zero, else 0
    - duration: numeric duration in hours
    - days_left_bucket: categorical binned days_left (0-3, 4-7, 8-14, 15-30, 31+)
    - is_last_minute: 1 if days_left <= 3, else 0
    - is_red_eye: 1 if departure is Late_Night/Night and arrival is Early_Morning/Morning, else 0
    - airline_route: interaction string between airline and route
    """
    df_feat = df.copy()

    # Route feature
    df_feat["route"] = (
        df_feat["source_city"].astype(str).str.strip()
        + "_"
        + df_feat["destination_city"].astype(str).str.strip()
    )

    # Direct flight indicator
    stops_clean = df_feat["stops"].astype(str).str.strip().str.lower()
    df_feat["is_direct"] = stops_clean.isin(["zero", "0", "non-stop", "direct"]).astype(int)

    # Ensure duration is numeric
    df_feat["duration"] = pd.to_numeric(df_feat["duration"], errors="coerce").fillna(0.0)

    # Days left bucket
    bins = [-1, 3, 7, 14, 30, 999]
    labels = ["0-3", "4-7", "8-14", "15-30", "31+"]
    df_feat["days_left_bucket"] = pd.cut(
        df_feat["days_left"], bins=bins, labels=labels
    ).astype(str)

    # Last minute indicator
    df_feat["is_last_minute"] = (df_feat["days_left"] <= 3).astype(int)

    # Red eye indicator
    dep_str = df_feat["departure_time"].astype(str).str.strip().str.lower()
    arr_str = df_feat["arrival_time"].astype(str).str.strip().str.lower()
    night_dep = dep_str.isin(["late_night", "night"])
    morning_arr = arr_str.isin(["early_morning", "morning"])
    df_feat["is_red_eye"] = (night_dep & morning_arr).astype(int)

    # Airline x Route interaction
    df_feat["airline_route"] = (
        df_feat["airline"].astype(str).str.strip() + "_" + df_feat["route"]
    )

    # Ensure string categoricals
    cat_cols = [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "arrival_time",
        "stops",
        "class",
        "route",
        "airline_route",
        "days_left_bucket",
    ]
    for col in cat_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype(str).str.strip()

    logger.info("Engineered features: %d columns generated.", len(df_feat.columns))
    return df_feat


def get_feature_column_names(config: ProjectConfig) -> Tuple[List[str], List[str], List[str]]:
    """Return categorized column lists: categorical, numerical, binary."""
    cat_cols = [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "arrival_time",
        "stops",
        "class",
        "route",
        "airline_route",
        "days_left_bucket",
    ]
    num_cols = ["duration", "days_left"]
    bin_cols = ["is_direct", "is_last_minute", "is_red_eye"]
    return cat_cols, num_cols, bin_cols


def build_sklearn_preprocessor(
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    use_one_hot: bool = True,
) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for linear and tree models.
    Scales numerical features and encodes categorical features without leakage.
    """
    if use_one_hot:
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

    num_transformer = StandardScaler()

    transformers = [
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
        ("bin", "passthrough", bin_cols),
    ]

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor
