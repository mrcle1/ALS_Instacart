"""Ranking metrics at K.

All four metrics follow the same calling convention::

    metric(recommended_ids, relevant_set, k) -> float

* :func:`ndcg_at_k` — Normalised Discounted Cumulative Gain.
* :func:`mrr_at_k` — Mean Reciprocal Rank (returns the rank of the first
  hit, 0 if no hit).
* :func:`recall_at_k` — fraction of the relevant set that appears in the
  top-K recommendations.
* :func:`precision_at_k` — fraction of the top-K that are relevant.

These are *per-user* metrics — the caller is expected to average them
across the evaluation cohort.
"""
import math
from typing import Iterable, Set

import numpy as np


def ndcg_at_k(rec: Iterable[int], rel: Set[int], k: int) -> float:
    r = list(rec)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(r) if x in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(rec: Iterable[int], rel: Set[int], k: int) -> float:
    for i, x in enumerate(list(rec)[:k]):
        if x in rel:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(rec: Iterable[int], rel: Set[int], k: int) -> float:
    r = set(list(rec)[:k])
    if not rel:
        return 0.0
    return len(r & rel) / len(rel)


def precision_at_k(rec: Iterable[int], rel: Set[int], k: int) -> float:
    r = set(list(rec)[:k])
    if k <= 0:
        return 0.0
    return len(r & rel) / k