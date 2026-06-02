"""FP-Growth association-rule recommender.

This module wraps `mlxtend.fpm.fpgrowth` to mine association rules over
user baskets and uses those rules to score candidates for any given
user. The output format is intentionally compatible with the ALS
recommender (``recommend()`` returns ``(ids, scores)``) so the API and
the evaluation code can stay algorithm-agnostic.

Why FPGrowth?
-------------
ALS generalises from sparse implicit feedback but ignores sequential /
co-occurrence signals. FPGrowth captures "people who bought X also
bought Y" patterns that are complementary and useful for cold baskets
or for explaining *why* a product was suggested.

Implementation notes
--------------------
* Mining happens once during :meth:`fit`; recommendation uses a cached
  in-memory rule table for O(1) look-ups.
* The ``use_colnames`` flag keeps the rule antecedents / consequents as
  the *original* product ids (i.e. the integer labels from the
  ``LabelEncoder``), which we then map back to Instacart product ids at
  inference.
* :func:`joblib.Parallel` parallelises the per-user scoring so the
  inference path scales with the same ``n_jobs=-1`` ergonomics as ALS.
"""

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlxtend.fpm import fpgrowth, association_rules

from ..config import FPGrowthConfig
from ..logger import get_logger
from ..progress import instacart_tqdm

log = get_logger(__name__)


@dataclass
class _RuleIndex:
    """Pre-computed look-up tables for fast recommendation.

    ``antecedent_to_rules`` maps each antecedent product to the list of
    rule indices whose LHS contains it. ``antecedent_sets`` is the
    parallel list of frozensets (so we can do O(len(LHS)) intersections
    with a user's basket).
    """
    antecedent_to_rules: Dict[int, List[int]] = field(default_factory=dict)
    antecedent_sets: List[frozenset] = field(default_factory=list)
    rule_consequents: List[frozenset] = field(default_factory=list)
    rule_confidences: List[float] = field(default_factory=list)
    rule_supports: List[float] = field(default_factory=list)


class FPGrowthRecommender:
    """FP-Growth association-rule recommender."""

    def __init__(self, params: Optional[FPGrowthConfig] = None) -> None:
        self.params = params or FPGrowthConfig()
        self.frequent_itemsets: Optional[pd.DataFrame] = None
        self.rules: Optional[pd.DataFrame] = None
        self._index: Optional[_RuleIndex] = None
        self._product_to_idx: Optional[Dict[int, int]] = None
        self._idx_to_product: Optional[Dict[int, int]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_basket_dataframe(train_df: pd.DataFrame) -> pd.DataFrame:
        """Group ``train_df`` (user_id, product_id) into a one-hot basket
        matrix compatible with mlxtend's FP-Growth.
        """
        baskets: Dict[int, List[int]] = {}
        for uid, grp in train_df.groupby("user_id"):
            baskets[int(uid)] = grp["product_id"].astype(int).tolist()
        log.info("Built %d user baskets for FPGrowth", len(baskets))
        return pd.DataFrame(
            [{"Transaction": k, "Items": v} for k, v in baskets.items()],
        )

    @staticmethod
    def _one_hot_encode(basket_df: pd.DataFrame) -> pd.DataFrame:
        """Convert the per-basket ``Items`` lists into a one-hot frame.

        The columns are the unique product ids present in the input —
        mlxtend requires the column index to mirror the item label.
        """
        from mlxtend.preprocessing import TransactionEncoder

        te = TransactionEncoder()
        # Filter out empty baskets defensively
        transactions = [t for t in basket_df["Items"].tolist() if t]
        te_ary = te.fit_transform(transactions)
        unique_products = sorted({p for tx in transactions for p in tx})
        return pd.DataFrame(te_ary, columns=unique_products)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "FPGrowthRecommender":
        """Mine frequent itemsets & rules from the train split.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must contain ``user_id`` and ``product_id`` columns.
        """
        if not {"user_id", "product_id"}.issubset(train_df.columns):
            log.error("train_df missing user_id/product_id")
            raise ValueError("train_df must contain user_id and product_id columns")

        log.info(
            "FPGrowth starting | min_support=%.4f | min_confidence=%.4f | max_len=%d",
            self.params.min_support, self.params.min_confidence, self.params.max_len,
        )

        t0 = time.time()
        basket_df = self._build_basket_dataframe(train_df)
        one_hot = self._one_hot_encode(basket_df)
        # Cap column count — mlxtend + sklearn cannot handle 30k+ items in one frame.
        if one_hot.shape[1] > 8000:
            log.warning(
                "Basket one-hot has %d columns; trimming to top 8000 most frequent.",
                one_hot.shape[1],
            )
            col_sums = one_hot.sum(axis=0).sort_values(ascending=False)
            keep = col_sums.head(8000).index
            one_hot = one_hot[keep]

        log.info("Mining frequent itemsets (shape=%s)...", one_hot.shape)
        self.frequent_itemsets = fpgrowth(
            one_hot,
            min_support=self.params.min_support,
            use_colnames=True,
            max_len=self.params.max_len,
            verbose=0,
        )
        log.info("Frequent itemsets: %d", len(self.frequent_itemsets))

        if self.frequent_itemsets.empty:
            log.warning("No frequent itemsets found at min_support=%.4f. "
                        "Lower it and re-fit.", self.params.min_support)
            self.rules = pd.DataFrame()
            self._index = _RuleIndex()
            return self

        log.info("Generating association rules...")
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.params.min_confidence,
            num_itemsets=len(self.frequent_itemsets),
        )
        if not self.rules.empty:
            self.rules = self.rules[self.rules["lift"] >= self.params.min_lift]

        log.info("Rules kept: %d (lift>=%.2f)", len(self.rules), self.params.min_lift)
        self._build_index()
        log.info("FPGrowth fit in %.1fs", time.time() - t0)
        return self

    def _build_index(self) -> None:
        """Pre-compute per-antecedent lookups for O(rules-touched) scoring."""
        idx = _RuleIndex()
        if self.rules is None or self.rules.empty:
            self._index = idx
            return

        for rid, row in self.rules.reset_index(drop=True).iterrows():
            ant = frozenset(int(x) for x in row["antecedents"])
            con = frozenset(int(x) for x in row["consequents"])
            idx.antecedent_sets.append(ant)
            idx.rule_consequents.append(con)
            idx.rule_confidences.append(float(row["confidence"]))
            idx.rule_supports.append(float(row["support"]))
            for a in ant:
                idx.antecedent_to_rules.setdefault(a, []).append(rid)
        self._index = idx
        log.info("Rule index built: %d antecedents covered",
                 len(idx.antecedent_to_rules))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        *,
        N: int = 10,
        user_history: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Score candidates for a single user.

        Parameters
        ----------
        user_id : int
            The Instacart user id. Currently unused for FPGrowth (rules
            are global) but accepted for symmetry with the ALS
            recommender.
        N : int, default 10
            Top-N to return.
        user_history : list of int, optional
            The user's training-set products. Required for FPGrowth since
            rule antecedents are product ids. If omitted, the method
            returns an empty array (with a warning).
        """
        if self._index is None:
            log.error("recommend() called before fit()")
            raise RuntimeError("FPGrowthRecommender must be fit() before recommend()")

        if not user_history:
            log.warning("No user_history supplied for FPGrowth recommend; returning [].")
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Aggregate scores per consequent product
        score_map: Dict[int, float] = {}
        basket = set(int(p) for p in user_history)

        for ant_product in basket:
            rule_ids = self._index.antecedent_to_rules.get(ant_product, [])
            for rid in rule_ids:
                # Only fire if the user already owns the entire antecedent
                if not self._index.antecedent_sets[rid].issubset(basket):
                    continue
                conf = self._index.rule_confidences[rid]
                for con in self._index.rule_consequents[rid]:
                    if con in basket:  # don't recommend things they already bought
                        continue
                    score_map[con] = score_map.get(con, 0.0) + conf

        if not score_map:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Sort and return top-N
        sorted_items = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:N]
        ids = np.array([k for k, _ in sorted_items], dtype=np.int64)
        scores = np.array([v for _, v in sorted_items], dtype=np.float32)
        return ids, scores

    def recommend_batch(
        self,
        users_with_history: Dict[int, List[int]],
        *,
        N: int = 10,
        n_jobs: int = -1,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Parallel top-N for many users.

        ``users_with_history`` maps Instacart ``user_id`` → list of
        product ids in the user's training basket. The function is
        embarrassingly parallel: each user is scored independently.
        """
        log.info("FPGrowth batch recommend: %d users (n_jobs=%d)",
                 len(users_with_history), n_jobs)

        def _one(uid: int, history: List[int]):
            return int(uid), self.recommend(uid, N=N, user_history=history)

        parallel = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)
        out = parallel(
            delayed(_one)(uid, hist) for uid, hist in instacart_tqdm(
                users_with_history.items(),
                desc="fpg-recommend", total=len(users_with_history),
            )
        )
        return {uid: payload for uid, payload in out}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "params": self.params,
                    "frequent_itemsets": self.frequent_itemsets,
                    "rules": self.rules,
                    "index": self._index,
                },
                fh,
            )
        log.info("FPGrowthRecommender saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "FPGrowthRecommender":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        log.info("FPGrowthRecommender loaded from %s", path)
        rec = cls(params=blob["params"])
        rec.frequent_itemsets = blob["frequent_itemsets"]
        rec.rules = blob["rules"]
        rec._index = blob["index"]
        return rec

