import os
import joblib
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Assignment Submission Delay Prediction API",
    description="REST API service to predict student assignment submission delays and evaluate risk tiers.",
    version="1.0.0",
)

MODEL_PATH = "model.joblib"
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            from train import train
            _model, _ = train(model_output_path=MODEL_PATH)
        else:
            _model = joblib.load(MODEL_PATH)
    return _model


class StudentFeaturesInput(BaseModel):
    student_id: Optional[str] = Field("STU_UNKNOWN", description="Unique student identifier")
    hist_avg_delay_days: float = Field(..., description="Historical average submission delay in days")
    hist_late_rate: float = Field(..., ge=0.0, le=1.0, description="Historical proportion of late submissions (0.0 - 1.0)")
    hist_max_delay_days: float = Field(..., description="Historical maximum delay in days")
    clicks_last_7_days: int = Field(..., ge=0, description="Total VLE / LMS click activity in the last 7 days")
    clicks_last_14_days: int = Field(..., ge=0, description="Total VLE / LMS click activity in the last 14 days")
    activity_decay_ratio: float = Field(..., ge=0.0, description="Ratio of recent to past LMS engagement")
    days_to_first_vle_access: float = Field(..., ge=0.0, description="Days between course opening and first student access")
    concurrent_deadlines_window: int = Field(..., ge=0, description="Other course assessments due within ±3 days")
    studied_credits: int = Field(..., description="Total module credits enrolled (e.g. 30, 60, 90, 120)")
    assessment_weight: float = Field(..., ge=0.0, le=100.0, description="Assessment contribution to final grade percentage")
    assessment_type: str = Field(..., description="Assessment category (e.g., TMA, CMA, Exam)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "student_id": "STU_1001",
                "hist_avg_delay_days": 1.5,
                "hist_late_rate": 0.45,
                "hist_max_delay_days": 3.0,
                "clicks_last_7_days": 12,
                "clicks_last_14_days": 55,
                "activity_decay_ratio": 0.28,
                "days_to_first_vle_access": 5.0,
                "concurrent_deadlines_window": 2,
                "studied_credits": 60,
                "assessment_weight": 25.0,
                "assessment_type": "TMA",
            }
        }
    }


class PredictionResponse(BaseModel):
    student_id: str
    is_late_prediction: int
    delay_probability: float
    risk_tier: str
    recommended_action: str


class BatchPredictionRequest(BaseModel):
    students: List[StudentFeaturesInput]


class BatchPredictionResponse(BaseModel):
    total_evaluated: int
    high_risk_count: int
    predictions: List[PredictionResponse]


def evaluate_risk(prob: float) -> tuple[str, str]:
    if prob >= 0.75:
        return "High", "Immediate instructor intervention & automated email nudge recommended."
    elif prob >= 0.40:
        return "Moderate", "Friendly study reminder & office hours alert suggested."
    else:
        return "Low", "Student on track. No action needed."


@app.get("/")
def root():
    return {
        "service": "Assignment Submission Delay Prediction API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    model = get_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentFeaturesInput):
    model = get_model()
    data_dict = data.model_dump()
    student_id = data_dict.pop("student_id", "STU_UNKNOWN")

    df = pd.DataFrame([data_dict])
    prob = float(model.predict_proba(df)[0, 1])
    is_late = int(prob >= 0.5)
    risk_tier, action = evaluate_risk(prob)

    return PredictionResponse(
        student_id=student_id,
        is_late_prediction=is_late,
        delay_probability=round(prob, 4),
        risk_tier=risk_tier,
        recommended_action=action,
    )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(batch: BatchPredictionRequest):
    if not batch.students:
        raise HTTPException(status_code=400, detail="Student list cannot be empty.")

    model = get_model()
    records = [s.model_dump() for s in batch.students]
    df = pd.DataFrame(records)
    student_ids = df.pop("student_id")

    probs = model.predict_proba(df)[:, 1]

    predictions = []
    high_risk_count = 0
    for sid, prob in zip(student_ids, probs):
        p_val = float(prob)
        is_late = int(p_val >= 0.5)
        risk_tier, action = evaluate_risk(p_val)
        if risk_tier == "High":
            high_risk_count += 1

        predictions.append(
            PredictionResponse(
                student_id=str(sid),
                is_late_prediction=is_late,
                delay_probability=round(p_val, 4),
                risk_tier=risk_tier,
                recommended_action=action,
            )
        )

    return BatchPredictionResponse(
        total_evaluated=len(predictions),
        high_risk_count=high_risk_count,
        predictions=predictions,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
