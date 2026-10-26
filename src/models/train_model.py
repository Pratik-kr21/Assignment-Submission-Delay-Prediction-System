"""Model training and MLflow experiment tracking script."""
import os
import joblib
import yaml
import numpy as np
import pandas as pd
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

from src.data.make_dataset import load_or_create_dataset
from src.features.build_features import create_preprocessor, prepare_features


def load_config(config_path: str = "config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def train(model_type: str = "xgboost"):
    config = load_config()
    mlflow_config = config.get("mlflow", {})
    exp_name = mlflow_config.get("experiment_name", "Assignment_Submission_Delay_Prediction")
    
    mlflow.set_experiment(exp_name)
    
    df = load_or_create_dataset()
    X_train, X_test, y_train, y_test = prepare_features(df)
    
    preprocessor = create_preprocessor()
    
    if model_type == "xgboost":
        clf = XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric="logloss"
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)
    
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    with mlflow.start_run(run_name=f"{model_type}_run"):
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(clf.get_params())
        
        # Train model
        full_pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = full_pipeline.predict(X_test)
        y_proba = full_pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        print(f"[{model_type.upper()}] Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        
        # Save local model artifact
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "model.joblib")
        joblib.dump(full_pipeline, model_path)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(full_pipeline, "model")
        print(f"Model successfully saved to {model_path} and logged to MLflow.")
        
    return full_pipeline


if __name__ == "__main__":
    train()
