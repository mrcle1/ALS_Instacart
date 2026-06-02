"""Inference pipeline.

Loads artefacts produced by :mod:`train_pipeline`, runs batch
recommendation for a cohort (default = the test split), evaluates if a
ground truth is available, persists predictions, and renders the
inference dashboard.
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import AppConfig, get_config
from ..data.encoders import UserItemEncoder, build_ground_truth
from ..data.loader import load_instacart_raw
from ..data.splitter import build_train_test_split_parallel
from ..evaluation.evaluator import evaluate_recommender
from ..logger import get_logger, setup_logging
from ..models.als_model import ALSRecommender
from ..models.fpgrowth_model import FPGrowthRecommender
from ..progress import instacart_tqdm
from ..visualization.inference_viz import render_inference_dashboard

log = get_logger(__name__)


def _load_artefacts(cfg: AppConfig) -> Tuple[ALSRecommender,
                                              FPGrowthRecommender,
                                              UserItemEncoder,
                                              Optional[pd.DataFrame],
                                              Optional[pd.DataFrame]]:
    als_path = os.path.join(cfg.paths.models_dir, "als_model.pkl")
    fpg_path = os.path.join(cfg.paths.models_dir, "fpgrowth_model.pkl")
    enc_path = os.path.join(cfg.paths.models_dir, "encoders.pkl")

    als = ALSRecommender.load(als_path)
    fpg = FPGrowthRecommender.load(fpg_path)
    encoder = UserItemEncoder.load(enc_path)

    train_csv = os.path.join(cfg.paths.predictions_dir, "train_split.csv")
    test_csv = os.path.join(cfg.paths.predictions_dir, "test_split.csv")
    train_df = pd.read_csv(train_csv) if os.path.exists(train_csv) else None
    test_df = pd.read_csv(test_csv) if os.path.exists(test_csv) else None
    return als, fpg, encoder, train_df, test_df


def _user_histories(train_df: pd.DataFrame) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    if train_df is None:
        return out
    grouped = train_df.groupby("user_id")["product_id"]
    for uid, grp in instacart_tqdm(grouped, desc="hist", total=len(grouped)):
        out[int(uid)] = grp.astype(int).tolist()
    return out


def run_inference_pipeline(
    *,
    algorithm: str = "als",
    user_ids: Optional[Sequence[int]] = None,
    N: int = 10,
    cfg: Optional[AppConfig] = None,
) -> dict:
    """Run the inference pipeline.

    Parameters
    ----------
    algorithm : {"als", "fpgrowth"}
        Which model to use.
    user_ids : optional sequence
        Cohort to recommend for. If omitted, the held-out test users are
        used.
    N : int, default 10
        Number of recommendations per user.
    cfg : AppConfig, optional
        Configuration object; defaults to the cached singleton.
    """
    cfg = cfg or get_config()
    setup_logging(cfg.paths.logs_dir)

    log.info("=== Inference pipeline start (algorithm=%s, N=%d) ===", algorithm, N)
    t0 = time.time()

    als, fpg, encoder, train_df, test_df = _load_artefacts(cfg)
    user_item = als.user_item
    user_histories = _user_histories(train_df) if train_df is not None else {}

    if user_ids is None:
        if test_df is None:
            log.error("No cohort supplied and no test split on disk.")
            raise RuntimeError("Provide user_ids or run training first.")
        user_ids = (
            test_df[test_df["user_id"].isin(encoder.user.classes_)]
            ["user_id"].unique().tolist()
        )
    log.info("Cohort size: %d users", len(user_ids))

    recommendations: Dict[int, List[int]] = {}
    scores_map: Dict[int, List[float]] = {}

    if algorithm == "als":
        log.info("Running ALS batch recommend (n_jobs=%d)...", cfg.eval.n_jobs)
        results = als.recommend_batch(
            list(user_ids), N=N, filter_already_liked=False, n_jobs=cfg.eval.n_jobs,
        )
        for uid, (ids, sc) in results.items():
            prods = encoder.inverse_transform_products(ids)
            recommendations[int(uid)] = [int(p) for p in prods]
            scores_map[int(uid)] = [float(s) for s in sc]
    elif algorithm == "fpgrowth":
        log.info("Running FPGrowth batch recommend (n_jobs=%d)...", cfg.eval.n_jobs)
        with_hist = {int(u): user_histories.get(int(u), []) for u in user_ids}
        results = fpg.recommend_batch(with_hist, N=N, n_jobs=cfg.eval.n_jobs)
        for uid, (ids, sc) in results.items():
            recommendations[int(uid)] = [int(p) for p in ids]
            scores_map[int(uid)] = [float(s) for s in sc]
    else:
        log.error("Unknown algorithm: %s", algorithm)
        raise ValueError(f"Unknown algorithm '{algorithm}' — use 'als' or 'fpgrowth'")

    # ---- Latency vs N sweep ----
    n_values = [5, 10, 20, 50]
    latencies: List[float] = []
    log.info("Measuring latency at N=%s", n_values)
    for n in n_values:
        t_n = time.time()
        if algorithm == "als":
            _ = als.recommend_batch(
                list(user_ids)[:min(1000, len(user_ids))],
                N=n, filter_already_liked=False, n_jobs=cfg.eval.n_jobs,
            )
        else:
            with_hist = {int(u): user_histories.get(int(u), []) for u in user_ids[:1000]}
            _ = fpg.recommend_batch(with_hist, N=n, n_jobs=cfg.eval.n_jobs)
        latencies.append(time.time() - t_n)

    # ---- Persist predictions ----
    rows = []
    for uid, recs in recommendations.items():
        for rank, pid in enumerate(recs, start=1):
            score = scores_map.get(uid, [None] * len(recs))[rank - 1]
            rows.append({"user_id": uid, "rank": rank, "product_id": pid, "score": score})
    pred_df = pd.DataFrame(rows)
    pred_csv = os.path.join(
        cfg.paths.predictions_dir, f"predictions_{algorithm}.csv",
    )
    pred_df.to_csv(pred_csv, index=False)
    log.info("Predictions saved → %s (%d rows)", pred_csv, len(pred_df))

    # ---- Evaluate (if we have ground truth) ----
    gt_map = None
    if test_df is not None and encoder is not None:
        gt_map = build_ground_truth(test_df, encoder)
    eval_summary_path = None
    if gt_map:
        log.info("Evaluating inference output against ground truth...")

        def _eval_rec(user_idx: int, n: int, _filt: bool):
            # Map encoded idx back to raw user id then to recommendation list.
            try:
                raw = int(encoder.user.inverse_transform([user_idx])[0])
            except Exception:  # noqa: BLE001
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            recs = recommendations.get(raw, [])
            return np.asarray(recs, dtype=np.int64), np.asarray(
                scores_map.get(raw, [0.0] * len(recs)), dtype=np.float32,
            )

        # Restrict the eval cohort to users that appear in both recs and GT.
        raw_to_idx = dict(zip(encoder.user.classes_.astype(int),
                              range(len(encoder.user.classes_))))
        cohort = {
            raw_to_idx[u]: gt_map[raw_to_idx[u]]
            for u in user_ids
            if u in raw_to_idx and raw_to_idx[u] in gt_map
        }
        eval_res = evaluate_recommender(
            _eval_rec, cohort,
            k_values=cfg.eval.k_values,
            sample_size=cfg.eval.sample_size,
            n_jobs=cfg.eval.n_jobs,
        )
        eval_summary_path = os.path.join(
            cfg.paths.predictions_dir, f"inference_eval_{algorithm}.csv",
        )
        eval_res.summary.to_csv(eval_summary_path, index=False)
        log.info("Inference eval summary → %s", eval_summary_path)

    # ---- Inference dashboard ----
    plot_path = render_inference_dashboard(
        recommendations=recommendations,
        ground_truth=gt_map,
        scores=scores_map,
        n_values=n_values,
        latencies_s=latencies,
        cfg=cfg.viz,
        save_path=os.path.join(
            cfg.paths.plots_dir, f"inference_dashboard_{algorithm}.png",
        ),
    )

    log.info("=== Inference pipeline done in %.1fs ===", time.time() - t0)
    return {
        "predictions": pred_csv,
        "eval_summary": eval_summary_path,
        "inference_dashboard": plot_path,
        "n_users": len(recommendations),
        "n_evaluated": gt_map is not None and len(gt_map),
        "elapsed_s": time.time() - t0,
        "latencies_s": latencies,
        "n_values": n_values,
    }

