"""Unit tests for data generation and feature preprocessing."""
import pytest
import pandas as pd
from src.data.make_dataset import generate_synthetic_data
from src.features.build_features import prepare_features, create_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def test_synthetic_data_generation():
    df = generate_synthetic_data(num_samples=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "is_late" in df.columns
    assert set(df["is_late"].unique()).issubset({0, 1})


def test_prepare_features_split():
    df = generate_synthetic_data(num_samples=200)
    X_train, X_test, y_train, y_test = prepare_features(df, test_size=0.25)
    
    assert len(X_train) == 150
    assert len(X_test) == 50
    assert len(y_train) == 150
    assert len(y_test) == 50
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(X_train.columns)


def test_preprocessor_transformation():
    df = generate_synthetic_data(num_samples=50)
    preprocessor = create_preprocessor()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_transformed = preprocessor.fit_transform(X)
    
    assert X_transformed.shape[0] == 50
    assert X_transformed.shape[1] >= len(NUMERIC_FEATURES)
