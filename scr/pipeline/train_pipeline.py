"""Training pipeline.

This is the single command an operator runs to (re)build every
artefact the API will later serve:

1. Load raw Instacart data.
2. Build a 80/20 per-user product split (parallel via joblib).
3. Encode labels and build the user×item confidence matrix.
4. Build the test-set ground truth.
5. Train the ALS model.
6. Train the FPGrowth model.
7. Evaluate ALS on the ground truth.
8. Render the training dashboard and write the eval summary CSV.
9. Persist encoders, ALS model, FPGrowth model, and the split to disk.

Every step uses the project logger, the project tqdm bar, and the
config-ini driven hyperparameters.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import AppConfig, get_config
from ..data.encoders import (
    build_ground_truth, build_user_item_matrix, encode_train_split,
)
from ..data.loader import load_instacart_raw
from ..data.splitter import build_train_test_split_parallel
from ..evaluation.evaluator import evaluate_recommender
from ..logger import get_logger, setup_logging
from ..models.als_model import ALSRecommender
from ..models.fpgrowth_model import FPGrowthRecommender
from ..progress import instacart_tqdm
from ..visualization.training_viz import render_training_dashboard

log = get_logger(__name__)


def _ensure_dirs(cfg: AppConfig) -> None:
    for d in (
        cfg.paths.artifacts_dir,
        cfg.paths.models_dir,
        cfg.paths.plots_dir,
        cfg.paths.logs_dir,
        cfg.paths.predictions_dir,
    ):
        Path(d).mkdir(parents=True, exist_ok=True)


def run_training_pipeline(
    raw_path: Optional[str] = None,
    cfg: Optional[AppConfig] = None,
) -> dict:
    """Run the entire training pipeline. Returns a dict of artefact paths."""
    cfg = cfg or get_config()
    setup_logging(cfg.paths.logs_dir)
    _ensure_dirs(cfg)

    raw_path = raw_path or cfg.paths.raw_data
    if not raw_path:
        log.error("No raw_data path configured (set config.ini [paths] raw_data).")
        raise ValueError("raw_data path missing — set [paths] raw_data in config.ini")

    log.info("=== Training pipeline start ===")
    t_total = time.time()

    # ---- 1. Load ----
    df = load_instacart_raw(raw_path)

    # ---- 2. Split ----
    split = build_train_test_split_parallel(
        df,
        test_ratio=cfg.data.test_ratio,
        random_state=cfg.data.random_state,
        min_user_items=cfg.data.min_user_items,
        n_jobs=cfg.eval.n_jobs,
    )
    train_df, test_df = split.train, split.test

    # Persist the split for reproducible re-runs / evaluation audits.
    train_csv = os.path.join(cfg.paths.predictions_dir, "train_split.csv")
    test_csv = os.path.join(cfg.paths.predictions_dir, "test_split.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    log.info("Split persisted: %s & %s", train_csv, test_csv)

    # ---- 3. Encode ----
    encoder, encoded = encode_train_split(train_df)
    encoder_path = os.path.join(cfg.paths.models_dir, "encoders.pkl")
    encoder.save(encoder_path)

    # ---- 4. Sparse matrix ----
    user_item, item_user = build_user_item_matrix(encoded, alpha=cfg.als.alpha)
    ground_truth = build_ground_truth(test_df, encoder)

    # ---- 5. ALS ----
    als = ALSRecommender(params=cfg.als).fit(user_item, encoder)
    als_path = os.path.join(cfg.paths.models_dir, "als_model.pkl")
    als.save(als_path)

    # ---- 6. FPGrowth ----
    fpg = FPGrowthRecommender(params=cfg.fpgrowth).fit(train_df)
    fpg_path = os.path.join(cfg.paths.models_dir, "fpgrowth_model.pkl")
    fpg.save(fpg_path)

    # ---- 7. Evaluate ALS ----
    def recommend_fn(user_idx: int, N: int, _filter: bool):
        # The encoded user index may be larger than the model's encoder
        # coverage (some test users are filtered out by the encoder fit).
        # The recommender returns (ids, scores) safely.
        return als.model.recommend(
            user_idx, user_item[user_idx], N=N,
            filter_already_liked_items=_filter,
        )

    eval_res = evaluate_recommender(
        recommend_fn=recommend_fn,
        ground_truth=ground_truth,
        k_values=cfg.eval.k_values,
        sample_size=cfg.eval.sample_size,
        n_jobs=cfg.eval.n_jobs,
    )
    eval_csv = os.path.join(cfg.paths.predictions_dir, "als_eval_summary.csv")
    eval_res.summary.to_csv(eval_csv, index=False)
    log.info("Eval summary saved → %s", eval_csv)

    # ---- 8. Dashboard ----
    plot_path = render_training_dashboard(
        eval_df=eval_res.summary,
        per_user_ndcg=eval_res.per_user_ndcg,
        cfg=cfg.viz,
        save_path=os.path.join(cfg.paths.plots_dir, "training_dashboard.png"),
    )

    log.info("=== Training pipeline done in %.1fs ===", time.time() - t_total)
    return {
        "encoders": encoder_path,
        "als_model": als_path,
        "fpgrowth_model": fpg_path,
        "train_split": train_csv,
        "test_split": test_csv,
        "eval_summary": eval_csv,
        "training_dashboard": plot_path,
        "n_evaluated": eval_res.n_evaluated,
        "elapsed_s": time.time() - t_total,
    }