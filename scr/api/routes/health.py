"""``GET /health`` — lightweight liveness probe."""
import os
from pathlib import Path

from fastapi import APIRouter

from ..schemas import HealthResponse
from ...config import get_config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg = get_config()
    als = Path(cfg.paths.models_dir) / "als_model.pkl"
    fpg = Path(cfg.paths.models_dir) / "fpgrowth_model.pkl"
    enc = Path(cfg.paths.models_dir) / "encoders.pkl"
    present = als.exists() and fpg.exists() and enc.exists()

    n_users, n_products = 0, 0
    if present:
        try:
            from ...data.encoders import UserItemEncoder
            from ...models.als_model import ALSRecommender

            encoder = UserItemEncoder.load(str(enc))
            n_users, n_products = encoder.n_users, encoder.n_products
        except Exception:  # noqa: BLE001
            pass
    return HealthResponse(
        status="ok" if present else "models-not-trained",
        artifacts_present=present,
        n_users=n_users, n_products=n_products,
    )