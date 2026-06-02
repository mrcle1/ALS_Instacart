"""Pydantic models for the FastAPI surface."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    raw_path: Optional[str] = Field(
        default=None,
        description="Override path to the raw Instacart file. Defaults to config.ini.",
    )
    config_path: Optional[str] = Field(
        default=None,
        description="Override path to config.ini.",
    )


class TrainResponse(BaseModel):
    encoders: str
    als_model: str
    fpgrowth_model: str
    train_split: str
    test_split: str
    eval_summary: str
    training_dashboard: Optional[str]
    n_evaluated: int
    elapsed_s: float


class RecommendRequest(BaseModel):
    algorithm: Literal["als", "fpgrowth"] = "als"
    user_ids: Optional[List[int]] = Field(
        default=None,
        description="Cohort of user ids. If omitted, uses the held-out test users.",
    )
    N: int = Field(default=10, ge=1, le=500)


class RecommendationItem(BaseModel):
    rank: int
    product_id: int
    score: Optional[float]


class UserRecommendation(BaseModel):
    user_id: int
    items: List[RecommendationItem]


class RecommendResponse(BaseModel):
    predictions_csv: str
    inference_dashboard: str
    n_users: int
    n_evaluated: Optional[int]
    elapsed_s: float
    recommendations: List[UserRecommendation]
    latencies_s: List[float]
    n_values: List[int]


class HealthResponse(BaseModel):
    status: str
    artifacts_present: bool
    n_users: int
    n_products: int


class MetricsResponse(BaseModel):
    algorithm: str
    rows: List[dict]

