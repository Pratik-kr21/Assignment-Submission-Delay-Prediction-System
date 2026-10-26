"""Inference module for assignment submission delay prediction."""
import os
import joblib
import pandas as pd
from typing import Dict, Any, List


class DelayPredictor:
    def __init__(self, model_path: str = "models/model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_model.py first.")
        self.pipeline = joblib.load(model_path)

    def predict_single(self, student_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts delay probability for a single student record."""
        df = pd.DataFrame([student_features])
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        prediction = int(prob >= 0.5)
        
        # Risk tier assignment
        if prob >= 0.75:
            risk_tier = "High"
            action = "Immediate instructor intervention & automated email nudge recommended."
        elif prob >= 0.40:
            risk_tier = "Moderate"
            action = "Friendly study reminder & office hours alert suggested."
        else:
            risk_tier = "Low"
            action = "Student on track. No action needed."
            
        return {
            "is_late_prediction": prediction,
            "delay_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "recommended_action": action
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs batch predictions on a DataFrame."""
        result_df = df.copy()
        probs = self.pipeline.predict_proba(df)[:, 1]
        result_df["delay_probability"] = probs.round(4)
        result_df["is_late_prediction"] = (probs >= 0.5).astype(int)
        result_df["risk_tier"] = pd.cut(
            probs,
            bins=[-0.1, 0.4, 0.75, 1.0],
            labels=["Low", "Moderate", "High"]
        )
        return result_df
