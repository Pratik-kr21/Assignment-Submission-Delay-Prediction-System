"""Feature engineering and preprocessing pipeline."""
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


NUMERIC_FEATURES: List[str] = [
    "hist_avg_delay_days",
    "hist_late_rate",
    "hist_max_delay_days",
    "clicks_last_7_days",
    "clicks_last_14_days",
    "activity_decay_ratio",
    "days_to_first_vle_access",
    "concurrent_deadlines_window",
    "studied_credits",
    "assessment_weight",
]

CATEGORICAL_FEATURES: List[str] = [
    "assessment_type",
]


def create_preprocessor() -> ColumnTransformer:
    """Creates a scikit-learn ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def prepare_features(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into features and target, and creates train/test sets."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_late"]
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
