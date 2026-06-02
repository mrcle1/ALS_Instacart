"""``POST /train`` — kick off the training pipeline."""

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..schemas import TrainRequest, TrainResponse
from ...config import get_config
from ...logger import get_logger
from ...pipeline.train_pipeline import run_training_pipeline

router = APIRouter(prefix="/train", tags=["train"])
log = get_logger(__name__)


@router.post("", response_model=TrainResponse)
def train(req: TrainRequest, background: BackgroundTasks) -> TrainResponse:
    """Run the training pipeline synchronously and return the artefact paths.

    The synchronous variant keeps the call simple for notebook /
    curl users. A real production deployment would dispatch this to a
    background worker queue (Celery, RQ, etc.) — wire it through
    :class:`fastapi.BackgroundTasks` if you want fire-and-forget.
    """
    try:
        cfg = get_config(path=req.config_path, reload=req.config_path is not None)
        artefacts = run_training_pipeline(raw_path=req.raw_path, cfg=cfg)
    except FileNotFoundError as exc:
        log.error("Training failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        log.error("Training failed (bad input): %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Training failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"training failed: {exc}")
    return TrainResponse(**artefacts)