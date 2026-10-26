"""Data loading and preprocessing module."""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict


def generate_synthetic_data(num_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic student submission and LMS engagement data mirroring the OULAD schema.
    Used for local pipeline development, testing, and continuous integration.
    """
    np.random.seed(random_seed)
    
    student_ids = [f"STU_{1000 + i}" for i in range(num_samples)]
    
    # Historical habits
    hist_avg_delay_days = np.random.normal(loc=0.5, scale=2.0, size=num_samples)
    hist_late_rate = np.clip(np.random.beta(a=2, b=5, size=num_samples), 0.0, 1.0)
    hist_max_delay_days = np.maximum(hist_avg_delay_days, hist_avg_delay_days + np.random.exponential(scale=1.5, size=num_samples))
    
    # Engagement in VLE (clicks)
    clicks_last_7_days = np.random.negative_binomial(n=5, p=0.08, size=num_samples)
    clicks_last_14_days = clicks_last_7_days + np.random.negative_binomial(n=8, p=0.07, size=num_samples)
    activity_decay_ratio = np.clip((clicks_last_7_days + 1) / (clicks_last_14_days - clicks_last_7_days + 1), 0.0, 5.0)
    days_to_first_vle_access = np.clip(np.random.gamma(shape=2.5, scale=2.0, size=num_samples), 0.0, 20.0)
    
    # Academic & Workload context
    concurrent_deadlines_window = np.random.poisson(lam=1.2, size=num_samples)
    studied_credits = np.random.choice([30, 60, 90, 120], p=[0.2, 0.5, 0.2, 0.1], size=num_samples)
    assessment_weight = np.random.choice([10.0, 20.0, 30.0, 50.0], p=[0.3, 0.4, 0.2, 0.1], size=num_samples)
    assessment_type = np.random.choice(["TMA", "CMA", "Exam"], p=[0.7, 0.2, 0.1], size=num_samples)
    
    # Generate realistic latent delay probability
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
    
    df = pd.DataFrame({
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
        "is_late": is_late
    })
    
    return df


def load_or_create_dataset(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Loads existing dataset or creates benchmark data if raw folder is empty."""
    os.makedirs(raw_dir, exist_ok=True)
    csv_path = os.path.join(raw_dir, "student_submission_data.csv")
    
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    
    df = generate_synthetic_data()
    df.to_csv(csv_path, index=False)
    print(f"Created initial dataset with {len(df)} records at {csv_path}")
    return df


if __name__ == "__main__":
    load_or_create_dataset()
