"""``GET /metrics`` — return the most recent evaluation summary."""
import os
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..schemas import MetricsResponse
from ...config import get_config
from ...logger import get_logger

router = APIRouter(prefix="/metrics", tags=["metrics"])
log = get_logger(__name__)


@router.get("", response_model=List[MetricsResponse])
def metrics() -> List[MetricsResponse]:
    cfg = get_config()
    preds_dir = cfg.paths.predictions_dir
    candidates = [
        ("als", os.path.join(preds_dir, "als_eval_summary.csv")),
        ("als", os.path.join(preds_dir, "inference_eval_als.csv")),
        ("fpgrowth", os.path.join(preds_dir, "inference_eval_fpgrowth.csv")),
    ]
    out: List[MetricsResponse] = []
    for algo, path in candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to read %s: %s", path, exc)
            continue
        out.append(MetricsResponse(algorithm=algo, rows=df.to_dict(orient="records")))
    if not out:
        raise HTTPException(status_code=404, detail="no evaluation summaries found")
    return out