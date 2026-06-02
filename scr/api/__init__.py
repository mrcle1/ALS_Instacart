"""FastAPI surface that wraps the recommendation pipelines.

Exposes three routes:

* ``GET  /health``         — liveness probe
* ``POST /train``          — run the training pipeline
* ``POST /recommend``      — run the inference pipeline for a user cohort
* ``GET  /metrics``        — last evaluation summary (if any)
"""
from .main import app

__all__ = ["app"]