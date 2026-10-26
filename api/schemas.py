"""Pydantic schemas for API request and response validation."""
from pydantic import BaseModel, Field
from typing import List, Optional


class StudentFeaturesInput(BaseModel):
    student_id: Optional[str] = Field(default="STU_9999", description="Unique student identifier")
    hist_avg_delay_days: float = Field(..., ge=-10.0, le=30.0, description="Average delay on prior assignments (days)")
    hist_late_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of past submissions that were late (0.0 to 1.0)")
    hist_max_delay_days: float = Field(..., ge=-10.0, le=60.0, description="Maximum past delay in days")
    clicks_last_7_days: int = Field(..., ge=0, description="LMS portal clicks in the last 7 days")
    clicks_last_14_days: int = Field(..., ge=0, description="LMS portal clicks in the last 14 days")
    activity_decay_ratio: float = Field(..., ge=0.0, description="Ratio of clicks last 7d vs previous 7d")
    days_to_first_vle_access: float = Field(..., ge=0.0, description="Days after assignment release before first access")
    concurrent_deadlines_window: int = Field(..., ge=0, le=15, description="Other deadlines due within +/- 3 days")
    studied_credits: int = Field(default=60, description="Total credits currently enrolled")
    assessment_weight: float = Field(default=20.0, ge=0.0, le=100.0, description="Weight of this assignment (% of final grade)")
    assessment_type: str = Field(default="TMA", description="Assessment type: TMA (Tutor-Marked), CMA, or Exam")

    model_config = {
        "json_schema_extra": {
            "example": {
                "student_id": "STU_1042",
                "hist_avg_delay_days": 2.5,
                "hist_late_rate": 0.65,
                "hist_max_delay_days": 5.0,
                "clicks_last_7_days": 12,
                "clicks_last_14_days": 85,
                "activity_decay_ratio": 0.16,
                "days_to_first_vle_access": 8.0,
                "concurrent_deadlines_window": 3,
                "studied_credits": 60,
                "assessment_weight": 25.0,
                "assessment_type": "TMA"
            }
        }
    }


class PredictionResponse(BaseModel):
    student_id: Optional[str]
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
