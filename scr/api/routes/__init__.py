"""Route registration for the FastAPI app."""
from .health import router as health_router
from .train import router as train_router
from .recommend import router as recommend_router
from .metrics import router as metrics_router

__all__ = ["health_router", "train_router", "recommend_router", "metrics_router"]