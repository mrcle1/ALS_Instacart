"""FastAPI application entry point.

Run locally with::

    uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import health_router, metrics_router, recommend_router, train_router
from ..config import get_config
from ..logger import get_logger, setup_logging

log = get_logger(__name__) 


def _build_app() -> FastAPI:
    cfg = get_config()
    setup_logging(cfg.paths.logs_dir)

    app = FastAPI(
        title=cfg.api.title,
        version=cfg.api.version,
        description=(
            "Instacart Recommender API — wraps the ALS and FPGrowth "
            "pipelines, exposes training and inference endpoints."
        ),
    )
    # Permissive CORS — the API is intended to be called from a
    # dashboard; tighten this in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(train_router)
    app.include_router(recommend_router)
    app.include_router(metrics_router)

    log.info("FastAPI app created | %s v%s | %d routes",
             cfg.api.title, cfg.api.version, len(app.routes))
    return app


app = _build_app()