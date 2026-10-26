"""Monitoring and Data Drift detection module using statistical checks and Evidently AI."""
import pandas as pd
import numpy as np
from typing import Dict, Any


def compute_ks_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, feature_col: str) -> Dict[str, Any]:
    """
    Computes Kolmogorov-Smirnov test for continuous numeric feature drift detection.
    """
    from scipy import stats
    
    ref_vals = reference_data[feature_col].dropna()
    curr_vals = current_data[feature_col].dropna()
    
    statistic, p_value = stats.ks_2samp(ref_vals, curr_vals)
    drift_detected = bool(p_value < 0.05)
    
    return {
        "feature": feature_col,
        "ks_statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 5),
        "drift_detected": drift_detected
    }


def generate_drift_summary(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
    """Generates drift metrics across key behavioral features."""
    features = [
        "hist_avg_delay_days",
        "clicks_last_7_days",
        "activity_decay_ratio",
        "concurrent_deadlines_window"
    ]
    
    results = {}
    drift_count = 0
    for f in features:
        if f in reference_data.columns and f in current_data.columns:
            res = compute_ks_drift(reference_data, current_data, f)
            results[f] = res
            if res["drift_detected"]:
                drift_count += 1
                
    return {
        "features_analyzed": len(results),
        "drifted_features_count": drift_count,
        "dataset_drift_detected": bool(drift_count > 0),
        "details": results
    }
