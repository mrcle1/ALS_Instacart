"""Train / test split strategies.

The Instacart file we receive already contains a single order per user
(one order_id per user_id), so a *time-aware* split is not possible —
every record is contemporaneous. The notebook therefore resorts to a
**random per-user product split**: for each user we hold out 20% of the
distinct products they purchased and put them in the test fold, leaving
the other 80% in train. The two folds are guaranteed to be disjoint for
any (user, product) pair, which is the property the ALS recommender
relies on.

Two flavours are exposed:

* :func:`build_train_test_split` — pure pandas, single thread. Useful for
  small slices & unit tests.
* :func:`build_train_test_split_parallel` — uses :mod:`joblib` to shuffle
  per-user product arrays in parallel. This is the production path for
  the full 1.3M-row dataset.
"""


from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..logger import get_logger
from ..progress import instacart_tqdm

log = get_logger(__name__)


@dataclass
class SplitResult:
    """Container bundling the three artefacts the downstream layers need."""
    train: pd.DataFrame
    test: pd.DataFrame
    pair_df: pd.DataFrame
    freq_df: pd.DataFrame


def _filter_active_users(pair_df: pd.DataFrame, min_products: int) -> pd.DataFrame:
    counts = pair_df.groupby("user_id").size()
    keep = counts[counts >= min_products].index
    before = pair_df["user_id"].nunique()
    out = pair_df[pair_df["user_id"].isin(keep)].copy()
    log.info(
        "Active-user filter: %s -> %s users (min_products=%d)",
        f"{before:,}", f"{out['user_id'].nunique():,}", min_products,
    )
    return out


def _shuffle_split_one_user(uid: int, prods: np.ndarray,
                            test_ratio: float, rng_seed: int) -> Tuple[list, list]:
    """Per-user deterministic shuffle + 80/20 split. Pure function."""
    rng = np.random.default_rng(rng_seed + int(uid))
    prods = prods.copy()
    rng.shuffle(prods)
    n_test = max(1, int(len(prods) * test_ratio))
    train = [(uid, int(p)) for p in prods[:-n_test]]
    test = [(uid, int(p)) for p in prods[-n_test:]]
    return train, test


def build_train_test_split(
    df: pd.DataFrame,
    *,
    test_ratio: float = 0.2,
    random_state: int = 42,
    min_user_items: int = 10,
) -> SplitResult:
    """Sequential 80/20 per-user product split."""
    log.info("Building sequential train/test split (ratio=%.2f)", test_ratio)
    pair_df = df[["user_id", "product_id"]].drop_duplicates().copy()
    freq_df = (
        df.groupby(["user_id", "product_id"])
        .size().reset_index(name="purchase_count")
    )
    pair_df = _filter_active_users(pair_df, min_user_items)

    train_rows: list = []
    test_rows: list = []

    grouped = pair_df.groupby("user_id")
    for uid, grp in instacart_tqdm(grouped, desc="split-per-user", total=len(grouped)):
        train, test = _shuffle_split_one_user(
            int(uid), grp["product_id"].values, test_ratio, random_state,
        )
        train_rows.extend(train)
        test_rows.extend(test)

    train_df = pd.DataFrame(train_rows, columns=["user_id", "product_id"])
    test_df = pd.DataFrame(test_rows, columns=["user_id", "product_id"])

    train_df = train_df.merge(freq_df, on=["user_id", "product_id"], how="left")
    train_df["purchase_count"] = train_df["purchase_count"].fillna(1).astype(int)

    overlap = train_df[["user_id", "product_id"]].merge(
        test_df[["user_id", "product_id"]], on=["user_id", "product_id"],
    )
    if len(overlap):
        log.error("Train/test overlap detected: %d pairs", len(overlap))
        raise AssertionError(f"Found {len(overlap)} overlapping pairs in split")
    log.info("No overlap between train (%s) and test (%s).",
             f"{len(train_df):,}", f"{len(test_df):,}")

    return SplitResult(train=train_df, test=test_df, pair_df=pair_df, freq_df=freq_df)


def build_train_test_split_parallel(
    df: pd.DataFrame,
    *,
    test_ratio: float = 0.2,
    random_state: int = 42,
    min_user_items: int = 10,
    n_jobs: int = -1,
) -> SplitResult:
    """Parallel 80/20 per-user product split using joblib.

    This is materially faster than the sequential variant on the full
    Instacart file because per-user shuffles are independent.
    """
    log.info("Building parallel train/test split (n_jobs=%d)", n_jobs)
    pair_df = df[["user_id", "product_id"]].drop_duplicates().copy()
    freq_df = (
        df.groupby(["user_id", "product_id"])
        .size().reset_index(name="purchase_count")
    )
    pair_df = _filter_active_users(pair_df, min_user_items)

    # groupby → dict of arrays for embarrassingly parallel mapping
    by_user = {uid: grp["product_id"].values for uid, grp in pair_df.groupby("user_id")}

    log.info("Dispatching %d per-user shuffles across %d worker(s)...",
             len(by_user), n_jobs)

    parallel_output = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_shuffle_split_one_user)(int(uid), prods, test_ratio, random_state)
        for uid, prods in instacart_tqdm(
            by_user.items(), desc="split-parallel", total=len(by_user),
        )
    )

    train_rows: list = []
    test_rows: list = []
    for train, test in parallel_output:
        train_rows.extend(train)
        test_rows.extend(test)

    train_df = pd.DataFrame(train_rows, columns=["user_id", "product_id"])
    test_df = pd.DataFrame(test_rows, columns=["user_id", "product_id"])

    train_df = train_df.merge(freq_df, on=["user_id", "product_id"], how="left")
    train_df["purchase_count"] = train_df["purchase_count"].fillna(1).astype(int)

    overlap = train_df[["user_id", "product_id"]].merge(
        test_df[["user_id", "product_id"]], on=["user_id", "product_id"],
    )
    if len(overlap):
        log.error("Train/test overlap detected: %d pairs", len(overlap))
        raise AssertionError(f"Found {len(overlap)} overlapping pairs in split")
    log.info(
        "Parallel split OK | train=%s | test=%s | users=%s",
        f"{len(train_df):,}", f"{len(test_df):,}",
        f"{train_df['user_id'].nunique():,}",
    )

    return SplitResult(train=train_df, test=test_df, pair_df=pair_df, freq_df=freq_df)