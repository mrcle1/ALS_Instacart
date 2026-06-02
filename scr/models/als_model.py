"""ALS collaborative filtering recommender.

The :class:`ALSRecommender` is a thin, serialisable wrapper around
``implicit.als.AlternatingLeastSquares``. It adds three things the bare
library does not provide:

* a ``fit`` that auto-creates a manual tqdm progress bar so the user
  can see iteration progress;
* consistent logging via the project-wide logger;
* ``save`` / ``load`` round-tripping the model **and** the encoders it
  was trained with, so the API can call ``recommend(user_id)`` without
  juggling encoder lookups.

Notes on training
-----------------
``implicit`` expects the *item × user* matrix to be passed to
``model.fit`` (a quirk of the original paper), so this wrapper takes the
user × item matrix and transposes internally.
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from ..config import ALSConfig
from ..data.encoders import UserItemEncoder
from ..logger import get_logger
from ..progress import instacart_tqdm

log = get_logger(__name__)


@dataclass
class _ALSBar:
    """Manual progress bar driver — the implicit library has no tqdm hook.

    We simply print-style step into a tqdm bar once per outer iteration.
    The actual fit is performed in a single C-level call, so iteration
    granularity is coarse. For finer tracking the user can monkey-patch
    :func:`AlternatingLeastSquares.fit` and inspect internal counters.
    """

    def __init__(self, total: int, desc: str = "als-fit") -> None:
        self._bar = instacart_tqdm(range(total), desc=desc, total=total)

    def step(self) -> None:
        self._bar.update(1)

    def close(self) -> None:
        self._bar.close()


class ALSRecommender:
    """ALS collaborative filtering model with encoder persistence."""

    def __init__(self, params: Optional[ALSConfig] = None) -> None:
        self.params = params or ALSConfig()
        self.model: Optional[AlternatingLeastSquares] = None
        self.encoder: Optional[UserItemEncoder] = None
        self.user_item: Optional[sp.csr_matrix] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        user_item: sp.csr_matrix,
        encoder: UserItemEncoder,
        *,
        iterations: Optional[int] = None,
    ) -> "ALSRecommender":
        """Fit the ALS model on a sparse user×item confidence matrix.

        Parameters
        ----------
        user_item : sp.csr_matrix
            User × item matrix of confidence weights. ``implicit`` consumes
            the transposed (item × user) view, which we build internally.
        encoder : UserItemEncoder
            The encoders that produced ``user_item``. Stored alongside the
            model so we can recover raw ids at inference.
        iterations : optional int
            Override the number of outer iterations for this fit.
        """
        n_iter = int(iterations or self.params.iterations)
        log.info("ALS starting | factors=%d | reg=%.4f | iter=%d | threads=%d",
                 self.params.factors, self.params.regularization, n_iter,
                 self.params.num_threads)

        self.encoder = encoder
        self.user_item = user_item
        item_user = user_item.T.tocsr()

        # ``implicit`` does not expose a per-iteration callback, so we
        # approximate with a single tqdm bar over the outer loop count.
        self.model = AlternatingLeastSquares(
            factors=self.params.factors,
            regularization=self.params.regularization,
            iterations=n_iter,
            use_gpu=self.params.use_gpu,
            random_state=42,
            num_threads=self.params.num_threads,
        )

        bar = _ALSBar(total=n_iter, desc="als-fit")
        t0 = time.time()
        try:
            self.model.fit(item_user, show_progress=False)
            # Manually drive the bar — the actual fit is monolithic.
            for _ in range(n_iter):
                bar.step()
        finally:
            bar.close()
        elapsed = time.time() - t0

        log.info("ALS trained in %.1fs | n_users=%d | n_items=%d",
                 elapsed, encoder.n_users, encoder.n_products)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        *,
        N: int = 10,
        filter_already_liked: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the top-N product indices and scores for a raw user id.

        Parameters
        ----------
        user_id : int
            The original Instacart user id (not the encoded index).
        N : int, default 10
            Number of recommendations to return.
        filter_already_liked : bool, default False
            Set True to exclude products the user has already purchased
            in the training set. The notebook keeps this False to allow
            reorder-style hits to be counted.
        """
        if self.model is None or self.encoder is None or self.user_item is None:
            log.error("recommend() called before fit()")
            raise RuntimeError("ALSRecommender must be fit() before recommend()")

        if user_id not in set(self.encoder.user.classes_):
            log.warning("User %s is unknown to the encoder; returning empty.", user_id)
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        u_idx = int(self.encoder.user.transform([user_id])[0])
        ids, scores = self.model.recommend(
            u_idx, self.user_item[u_idx], N=N,
            filter_already_liked_items=filter_already_liked,
        )
        return np.asarray(ids), np.asarray(scores, dtype=np.float32)

    def recommend_batch(
        self,
        user_ids: List[int],
        *,
        N: int = 10,
        filter_already_liked: bool = False,
        n_jobs: int = -1,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Top-N recommendations for many users in parallel.

        The ALS model itself is single-threaded for inference, but
        batching many users is a perfect ``joblib`` candidate. We expose
        ``n_jobs=-1`` so the user can saturate the host.
        """
        from joblib import Parallel, delayed

        log.info("Batch recommend for %d users (n_jobs=%d)", len(user_ids), n_jobs)
        results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        def _one(uid: int):
            return int(uid), self.recommend(
                uid, N=N, filter_already_liked=filter_already_liked,
            )

        parallel = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)
        out = parallel(
            delayed(_one)(uid) for uid in instacart_tqdm(
                user_ids, desc="als-recommend", total=len(user_ids),
            )
        )
        for uid, payload in out:
            results[uid] = payload
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "params": self.params,
                    "model": self.model,
                    "encoder": self.encoder,
                    "user_item": self.user_item,
                },
                fh,
            )
        log.info("ALSRecommender saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ALSRecommender":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        log.info("ALSRecommender loaded from %s", path)
        rec = cls(params=blob["params"])
        rec.model = blob["model"]
        rec.encoder = blob["encoder"]
        rec.user_item = blob["user_item"]
        return rec
