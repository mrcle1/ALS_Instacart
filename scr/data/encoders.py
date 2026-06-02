import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from ..logger import get_logger
from ..progress import instacart_tqdm

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
@dataclass
class UserItemEncoder:
    """Persists the user and product label encoders for reuse at inference."""
    user: LabelEncoder = field(default_factory=LabelEncoder)
    product: LabelEncoder = field(default_factory=LabelEncoder)

    # ---- save / load ----
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"user": self.user, "product": self.product}, fh)
        log.info("Encoders saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "UserItemEncoder":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        log.info("Encoders loaded from %s", path)
        return cls(user=blob["user"], product=blob["product"])

    # ---- introspection ----
    @property
    def n_users(self) -> int:
        return int(self.user.classes_.shape[0])

    @property
    def n_products(self) -> int:
        return int(self.product.classes_.shape[0])

    def transform_user(self, user_ids) -> np.ndarray:
        """Transform raw user ids; unknown ids map to -1."""
        known = set(self.user.classes_.tolist())
        out = np.full(len(user_ids), -1, dtype=np.int64)
        for i, u in enumerate(user_ids):
            if u in known:
                out[i] = int(self.user.transform([u])[0])
        return out

    def transform_product(self, product_ids) -> np.ndarray:
        known = set(self.product.classes_.tolist())
        out = np.full(len(product_ids), -1, dtype=np.int64)
        for i, p in enumerate(product_ids):
            if p in known:
                out[i] = int(self.product.transform([p])[0])
        return out

    def inverse_transform_products(self, indices) -> np.ndarray:
        return self.product.inverse_transform(indices)


def encode_train_split(train_df: pd.DataFrame) -> Tuple[UserItemEncoder, pd.DataFrame]:
    """Fit a fresh encoder pair on the train split and return them with the
    integer-indexed frame the rest of the pipeline consumes.
    """
    encoder = UserItemEncoder()
    encoder.user.fit(train_df["user_id"].unique())
    encoder.product.fit(train_df["product_id"].unique())
    log.info(
        "Fit encoders: users=%d | products=%d",
        encoder.n_users, encoder.n_products,
    )

    out = train_df.copy()
    out["user_idx"] = encoder.user.transform(out["user_id"])
    out["product_idx"] = encoder.product.transform(out["product_id"])
    return encoder, out


# ---------------------------------------------------------------------------
# Sparse matrices
# ---------------------------------------------------------------------------
def build_user_item_matrix(
    freq_df: pd.DataFrame,
    *,
    alpha: float = 40.0,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build (user×item, item×user) sparse matrices with confidence weights.

    The ``confidence`` weight follows the standard ``implicit`` library
    recipe: ``1 + alpha * purchase_count``. Higher ``alpha`` sharpens the
    difference between casual and repeat buyers.
    """
    if not {"user_idx", "product_idx", "purchase_count"}.issubset(freq_df.columns):
        log.error("freq_df missing required columns: %s",
                  {"user_idx", "product_idx", "purchase_count"} - set(freq_df.columns))
        raise ValueError("freq_df must contain user_idx, product_idx, purchase_count")

    n_users = int(freq_df["user_idx"].max()) + 1
    n_items = int(freq_df["product_idx"].max()) + 1
    confidence = 1.0 + alpha * freq_df["purchase_count"].astype(float).values

    user_item = sp.csr_matrix(
        (confidence, (freq_df["user_idx"].values, freq_df["product_idx"].values)),
        shape=(n_users, n_items),
    )
    item_user = user_item.T.tocsr()
    sparsity = 1.0 - user_item.nnz / (n_users * n_items)

    log.info(
        "Sparse matrix built | shape=%s | nnz=%s | sparsity=%.4f%% | alpha=%.1f",
        user_item.shape, f"{user_item.nnz:,}", sparsity * 100, alpha,
    )
    return user_item, item_user


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def build_ground_truth(
    test_df: pd.DataFrame,
    encoder: UserItemEncoder,
) -> Dict[int, Set[int]]:
    """Map each test user to the set of held-out product indices."""
    valid = test_df[
        test_df["user_id"].isin(encoder.user.classes_) &
        test_df["product_id"].isin(encoder.product.classes_)
    ].copy()

    valid["user_idx"] = encoder.user.transform(valid["user_id"])
    valid["product_idx"] = encoder.product.transform(valid["product_id"])

    grouped: Dict[int, Set[int]] = {}
    for uid, grp in instacart_tqdm(
        valid.groupby("user_idx"), desc="build-gt", total=valid["user_idx"].nunique(),
    ):
        grouped[int(uid)] = set(int(p) for p in grp["product_idx"].unique())

    sizes = [len(v) for v in grouped.values()]
    if sizes:
        log.info(
            "Ground truth built | users=%d | empty=%d | mean=%.2f | min=%d | max=%d",
            len(grouped), sum(1 for s in sizes if s == 0),
            float(np.mean(sizes)), int(np.min(sizes)), int(np.max(sizes)),
        )
    return grouped
