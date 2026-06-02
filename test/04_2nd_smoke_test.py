"""Re-run the smoke test with the fixes."""
import sys, os, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from src.logger import setup_logging, get_logger

log = get_logger("smoke2")
setup_logging("artifacts/logs")

n_users, n_products, n_rows = 200, 50, 5000
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "order_id": rng.integers(0, n_rows, size=n_rows),
    "user_id": rng.integers(1, n_users + 1, size=n_rows),
    "product_id": rng.integers(1, n_products + 1, size=n_rows),
    "aisle_id": rng.integers(1, 20, size=n_rows),
    "department_id": rng.integers(1, 5, size=n_rows),
    "order_number": rng.integers(1, 10, size=n_rows),
    "order_dow": rng.integers(0, 7, size=n_rows),
    "order_hour_of_day": rng.integers(0, 24, size=n_rows),
    "days_since_prior_order": rng.uniform(0, 30, size=n_rows),
    "add_to_cart_order": rng.integers(1, 20, size=n_rows),
    "reordered": rng.integers(0, 2, size=n_rows),
})
df = df.drop_duplicates(subset=["user_id", "product_id"])
counts = df.groupby("user_id").size()
good_users = counts[counts >= 10].index
df = df[df["user_id"].isin(good_users)].copy()

with tempfile.TemporaryDirectory() as td:
    pq_path = os.path.join(td, "synthetic.parquet")
    df.to_parquet(pq_path, index=False)

    from src.data.loader import load_instacart_raw
    from src.data.splitter import build_train_test_split_parallel
    from src.data.encoders import (
        encode_train_split, build_user_item_matrix, build_ground_truth,
    )
    from src.config import ALSConfig, FPGrowthConfig
    from src.models.als_model import ALSRecommender
    from src.models.fpgrowth_model import FPGrowthRecommender
    from src.evaluation.evaluator import evaluate_recommender

    df_loaded = load_instacart_raw(pq_path)
    split = build_train_test_split_parallel(df_loaded, n_jobs=2)
    enc, encoded = encode_train_split(split.train)
    uim, _ = build_user_item_matrix(encoded, alpha=40.0)
    gt = build_ground_truth(split.test, enc)

    als = ALSRecommender(params=ALSConfig(factors=8, iterations=2, num_threads=2))
    als.fit(uim, enc)

    sample_user = int(enc.user.classes_[0])
    ids, scores = als.recommend(sample_user, N=5)
    log.info("ALS recommend OK | user=%s, ids=%s", sample_user, ids.tolist())

    def rec_fn(user_idx, N, filt):
        return als.model.recommend(user_idx, uim[user_idx], N=N, filter_already_liked_items=filt)
    eval_res = evaluate_recommender(rec_fn, gt, k_values=[5, 10], sample_size=50, n_jobs=2)
    log.info("Eval summary:\n%s", eval_res.summary.to_string(index=False))

    fpg = FPGrowthRecommender(params=FPGrowthConfig(min_support=0.05, min_confidence=0.3))
    fpg.fit(split.train)
    log.info("FPGrowth fit OK | rules=%d", len(fpg.rules) if fpg.rules is not None else 0)

    # FPGrowth recommend needs user history
    user_hist = split.train.groupby("user_id")["product_id"].apply(list).to_dict()
    sample_uid = list(user_hist.keys())[0]
    ids2, scores2 = fpg.recommend(sample_uid, N=5, user_history=user_hist[sample_uid])
    log.info("FPGrowth recommend OK | user=%s, ids=%s", sample_uid, ids2.tolist())

log.info("==== SMOKE TEST 2 PASSED ====")