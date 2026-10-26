"""FastAPI microservice for Assignment Submission Delay Prediction."""
from fastapi import FastAPI, HTTPException
import os
import pandas as pd
from api.schemas import (
    StudentFeaturesInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from src.models.predict_model import DelayPredictor

from contextlib import asynccontextmanager

# Global predictor instance
predictor = None


def get_predictor() -> DelayPredictor:
    global predictor
    if predictor is None:
        model_path = os.path.join("models", "model.joblib")
        if not os.path.exists(model_path):
            from src.models.train_model import train
            train(model_type="xgboost")
        predictor = DelayPredictor(model_path=model_path)
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()
    yield


app = FastAPI(
    title="Assignment Submission Delay Prediction API",
    description="MLOps microservice forecasting student assignment submission lateness.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "service": "Assignment Submission Delay Prediction API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
def healthcheck():
    pred = get_predictor()
    return {
        "status": "healthy",
        "model_loaded": pred is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(data: StudentFeaturesInput):
    pred = get_predictor()
    input_dict = data.model_dump()
    student_id = input_dict.pop("student_id", "STU_UNKNOWN")
    
    res = pred.predict_single(input_dict)
    res["student_id"] = student_id
    return res


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    pred = get_predictor()
        
    records = [s.model_dump() for s in batch.students]
    df = pd.DataFrame(records)
    student_ids = df.pop("student_id")
    
    batch_results = predictor.predict_batch(df)
    batch_results["student_id"] = student_ids
    
    predictions = []
    high_risk_count = 0
    for _, row in batch_results.iterrows():
        tier = str(row["risk_tier"])
        if tier == "High":
            high_risk_count += 1
            action = "Immediate instructor intervention & automated email nudge recommended."
        elif tier == "Moderate":
            action = "Friendly study reminder & office hours alert suggested."
        else:
            action = "Student on track. No action needed."
            
        predictions.append(
            PredictionResponse(
                student_id=str(row["student_id"]),
                is_late_prediction=int(row["is_late_prediction"]),
                delay_probability=float(row["delay_probability"]),
                risk_tier=tier,
                recommended_action=action
            )
        )
        
    return BatchPredictionResponse(
        total_evaluated=len(predictions),
        high_risk_count=high_risk_count,
        predictions=predictions
    )
