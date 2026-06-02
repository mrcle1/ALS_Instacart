"""Parallel evaluation orchestrator.

This module is the *only* place that knows how to call a recommender
inside an evaluation loop. The implementation is parallel by default
(``joblib`` with ``n_jobs=-1``) because per-user ``recommend`` is
independent and the per-user metric computation is non-trivial.

Two public entry points:

* :func:`evaluate_recommender` — drives the full eval, returns a
  summary DataFrame plus the per-user NDCG@10 vector (for the
  histograms in the inference dashboard).
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..logger import get_logger
from ..progress import instacart_tqdm
from .metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    """All artefacts produced by an evaluation run."""
    summary: pd.DataFrame
    per_user_ndcg: List[float] = field(default_factory=list)
    per_user_recall: Dict[int, List[float]] = field(default_factory=dict)
    n_evaluated: int = 0
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Per-user worker
# ---------------------------------------------------------------------------
def _score_one_user(
    user_idx: int,
    rel: Set[int],
    rec_fn: Callable[[int, int, bool], Tuple[np.ndarray, np.ndarray]],
    max_k: int,
    k_values: List[int],
) -> Dict[str, List[float]]:
    """Compute every K-metric for a single user. Pure function → safe in
    ``joblib`` workers.
    """
    try:
        ids, _ = rec_fn(user_idx, max_k, False)
    except Exception as exc:  # noqa: BLE001
        log.warning("recommend() failed for user_idx=%s: %s", user_idx, exc)
        return {f"{m}@{k}": [0.0] for k in k_values for m in ("ndcg", "mrr", "recall", "precision")}
    rec = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    out: Dict[str, List[float]] = {}
    for k in k_values:
        out[f"ndcg@{k}"].append(ndcg_at_k(rec, rel, k))
        out[f"mrr@{k}"].append(mrr_at_k(rec, rel, k))
        out[f"recall@{k}"].append(recall_at_k(rec, rel, k))
        out[f"precision@{k}"].append(precision_at_k(rec, rel, k))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def evaluate_recommender(
    recommend_fn: Callable[[int, int, bool], Tuple[np.ndarray, np.ndarray]],
    ground_truth: Dict[int, Set[int]],
    *,
    k_values: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> EvalResult:
    """Evaluate a recommender against a ground-truth map.

    Parameters
    ----------
    recommend_fn : callable
        ``recommend_fn(user_idx, N, filter_already_liked) -> (ids, scores)``.
        The caller wires this up to ALS or FPGrowth.
    ground_truth : dict
        ``{user_idx: {product_idx, ...}}``.
    k_values : list of int, optional
        Defaults to ``[5, 10, 20]``.
    sample_size : int or None
        If set, subsample users to this count.
    n_jobs : int, default -1
        joblib parallelism level.
    """
    k_values = sorted(set(k_values or [5, 10, 20]))
    max_k = max(k_values)
    log.info("Evaluation | k=%s | n_jobs=%d", k_values, n_jobs)

    users = list(ground_truth.keys())
    if sample_size is not None and sample_size < len(users):
        rng = np.random.default_rng(random_state)
        users = rng.choice(users, size=sample_size, replace=False).tolist()
        log.info("Subsampled to %d users", len(users))

    t0 = time.time()

    # ----- parallel map -----
    user_gt_pairs = [(u, ground_truth[u]) for u in users if ground_truth[u]]
    log.info("Dispatching per-user scoring for %d users...", len(user_gt_pairs))

    parallel = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)
    results = parallel(
        delayed(_score_one_user)(u, gt, recommend_fn, max_k, k_values)
        for u, gt in instacart_tqdm(
            user_gt_pairs, desc="eval-users", total=len(user_gt_pairs),
        )
    )

    # ----- reduce -----
    accum: Dict[str, List[float]] = {
        f"{m}@{k}": [] for k in k_values for m in ("ndcg", "mrr", "recall", "precision")
    }
    for r in results:
        for key, vals in r.items():
            accum[key].extend(vals)

    rows = []
    for k in k_values:
        rows.append({
            "K": k,
            "NDCG@K": float(np.mean(accum[f"ndcg@{k}"])),
            "MRR@K": float(np.mean(accum[f"mrr@{k}"])),
            "Recall@K": float(np.mean(accum[f"recall@{k}"])),
            "Precision@K": float(np.mean(accum[f"precision@{k}"])),
            "n_users": len(accum[f"ndcg@{k}"]),
        })
    summary = pd.DataFrame(rows)

    elapsed = time.time() - t0
    log.info("Evaluation done in %.1fs | n_evaluated=%d", elapsed, len(accum[f"ndcg@{max_k}"]))

    return EvalResult(
        summary=summary,
        per_user_ndcg=list(accum[f"ndcg@{10 if 10 in k_values else max_k}"]),
        per_user_recall={k: list(accum[f"recall@{k}"]) for k in k_values},
        n_evaluated=len(accum[f"ndcg@{max_k}"]),
        elapsed_s=elapsed,
    )

