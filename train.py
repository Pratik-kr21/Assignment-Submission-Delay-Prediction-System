import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "assessment_type",
]

TARGET_COLUMN = "is_late"


def generate_synthetic_data(num_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    student_ids = [f"STU_{1000 + i}" for i in range(num_samples)]

    hist_avg_delay_days = np.random.normal(loc=0.5, scale=2.0, size=num_samples)
    hist_late_rate = np.clip(np.random.beta(a=2, b=5, size=num_samples), 0.0, 1.0)
    hist_max_delay_days = np.maximum(
        hist_avg_delay_days,
        hist_avg_delay_days + np.random.exponential(scale=1.5, size=num_samples),
    )

    clicks_last_7_days = np.random.negative_binomial(n=5, p=0.08, size=num_samples)
    clicks_last_14_days = clicks_last_7_days + np.random.negative_binomial(n=8, p=0.07, size=num_samples)
    activity_decay_ratio = np.clip(
        (clicks_last_7_days + 1) / (clicks_last_14_days - clicks_last_7_days + 1), 0.0, 5.0
    )
    days_to_first_vle_access = np.clip(np.random.gamma(shape=2.5, scale=2.0, size=num_samples), 0.0, 20.0)

    concurrent_deadlines_window = np.random.poisson(lam=1.2, size=num_samples)
    studied_credits = np.random.choice([30, 60, 90, 120], p=[0.2, 0.5, 0.2, 0.1], size=num_samples)
    assessment_weight = np.random.choice([10.0, 20.0, 30.0, 50.0], p=[0.3, 0.4, 0.2, 0.1], size=num_samples)
    assessment_type = np.random.choice(["TMA", "CMA", "Exam"], p=[0.7, 0.2, 0.1], size=num_samples)

    logit = (
        0.8 * hist_avg_delay_days
        + 2.2 * hist_late_rate
        - 0.04 * clicks_last_7_days
        - 0.6 * activity_decay_ratio
        + 0.15 * days_to_first_vle_access
        + 0.45 * concurrent_deadlines_window
        - 0.02 * assessment_weight
        - 1.5
    )
    delay_prob = 1 / (1 + np.exp(-logit))
    is_late = (np.random.rand(num_samples) < delay_prob).astype(int)

    return pd.DataFrame({
        "student_id": student_ids,
        "hist_avg_delay_days": np.round(hist_avg_delay_days, 2),
        "hist_late_rate": np.round(hist_late_rate, 3),
        "hist_max_delay_days": np.round(hist_max_delay_days, 2),
        "clicks_last_7_days": clicks_last_7_days,
        "clicks_last_14_days": clicks_last_14_days,
        "activity_decay_ratio": np.round(activity_decay_ratio, 3),
        "days_to_first_vle_access": np.round(days_to_first_vle_access, 1),
        "concurrent_deadlines_window": concurrent_deadlines_window,
        "studied_credits": studied_credits,
        "assessment_weight": assessment_weight,
        "assessment_type": assessment_type,
        "is_late": is_late,
    })


def load_data(data_path: str = "data.csv") -> pd.DataFrame:
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print(f"'{data_path}' not found. Generating synthetic dataset...")
        df = generate_synthetic_data()
        df.to_csv(data_path, index=False)
        print(f"Saved dataset with {len(df)} records to '{data_path}'.")
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
        ]
    )

    classifier = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss",
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train(data_path: str = "data.csv", model_output_path: str = "model.joblib") -> Tuple[Pipeline, Dict[str, float]]:
    print("Loading dataset...")
    df = load_data(data_path)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    print("\n--- Model Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k.upper():<12}: {v:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(pipeline, model_output_path)
    print(f"Successfully saved trained model to '{model_output_path}'.")

    return pipeline, metrics


if __name__ == "__main__":
    train()
