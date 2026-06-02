"""``POST /recommend`` — run inference for a cohort and return top-N items."""
import os
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..schemas import (
    RecommendRequest, RecommendResponse, RecommendationItem, UserRecommendation,
)
from ...config import get_config
from ...logger import get_logger
from ...pipeline.infer_pipeline import run_inference_pipeline

router = APIRouter(prefix="/recommend", tags=["recommend"])
log = get_logger(__name__)


@router.post("", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        cfg = get_config()
        artefacts = run_inference_pipeline(
            algorithm=req.algorithm, user_ids=req.user_ids, N=req.N, cfg=cfg,
        )
    except FileNotFoundError as exc:
        log.error("Inference failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}")

    # Hydrate the per-user response from the predictions CSV.
    pred_df = pd.read_csv(artefacts["predictions"])
    grouped: List[UserRecommendation] = []
    for uid, grp in pred_df.groupby("user_id"):
        items = [
            RecommendationItem(
                rank=int(row["rank"]),
                product_id=int(row["product_id"]),
                score=None if pd.isna(row.get("score")) else float(row["score"]),
            )
            for _, row in grp.sort_values("rank").iterrows()
        ]
        grouped.append(UserRecommendation(user_id=int(uid), items=items))

    return RecommendResponse(
        predictions_csv=artefacts["predictions"],
        inference_dashboard=artefacts["inference_dashboard"],
        n_users=artefacts["n_users"],
        n_evaluated=artefacts["n_evaluated"],
        elapsed_s=artefacts["elapsed_s"],
        recommendations=grouped,
        latencies_s=artefacts["latencies_s"],
        n_values=artefacts["n_values"],
    )

