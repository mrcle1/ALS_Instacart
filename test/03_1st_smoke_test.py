"""End-to-end smoke test with a tiny synthetic Instacart-shaped dataset."""
import sys, os, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from src.config import get_config
from src.logger import setup_logging, get_logger
from src.progress import instacart_tqdm, INSTACART_GREEN, BAR_FORMAT

log = get_logger("smoke")
setup_logging("artifacts/logs")

cfg = get_config()
log.info("Config OK | sample_size=%s | k_values=%s", cfg.data.sample_size, cfg.eval.k_values)

# Build a tiny synthetic Instacart frame
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
log.info("Synthetic dataset: rows=%d, users=%d, products=%d",
         len(df), df.user_id.nunique(), df.product_id.nunique())

with tempfile.TemporaryDirectory() as td:
    pq_path = os.path.join(td, "synthetic.parquet")
    df.to_parquet(pq_path, index=False)

    from src.data.loader import load_instacart_raw
    df_loaded = load_instacart_raw(pq_path)
    log.info("Loader round-trip OK | rows=%d", len(df_loaded))

    from src.data.splitter import build_train_test_split_parallel
    from src.data.encoders import (
        encode_train_split, build_user_item_matrix, build_ground_truth,
    )
    from src.config import DataConfig, ALSConfig
    split = build_train_test_split_parallel(df_loaded, n_jobs=2)
    log.info("Split OK | train=%d test=%d", len(split.train), len(split.test))
    enc, encoded = encode_train_split(split.train)
    uim, ium = build_user_item_matrix(encoded, alpha=40.0)
    log.info("Matrix OK | shape=%s nnz=%d", uim.shape, uim.nnz)
    gt = build_ground_truth(split.test, enc)
    log.info("GT OK | users=%d", len(gt))

    from src.models.als_model import ALSRecommender
    als = ALSRecommender(params=ALSConfig(factors=8, iterations=2, num_threads=2))
    als.fit(uim, enc)
    log.info("ALS fit OK")

    sample_user = int(enc.user.classes_[0])
    ids, scores = als.recommend(sample_user, N=5)
    log.info("Recommend OK | user=%s, ids=%s", sample_user, ids.tolist())

    from src.config import FPGrowthConfig
    from src.models.fpgrowth_model import FPGrowthRecommender
    fpg = FPGrowthRecommender(params=FPGrowthConfig(min_support=0.05, min_confidence=0.3))
    fpg.fit(split.train)
    log.info("FPGrowth fit OK | rules=%d", len(fpg.rules) if fpg.rules is not None else 0)

    from src.evaluation.evaluator import evaluate_recommender
    def rec_fn(user_idx, N, filt):
        return als.model.recommend(user_idx, uim[user_idx], N=N,
                                   filter_already_liked_items=filt)
    eval_res = evaluate_recommender(
        rec_fn, gt, k_values=[5, 10], sample_size=50, n_jobs=2,
    )
    log.info("Eval summary:\n%s", eval_res.summary.to_string(index=False))

    from src.visualization.training_viz import render_training_dashboard
    plot_path = render_training_dashboard(eval_res.summary, eval_res.per_user_ndcg)
    log.info("Training dashboard saved → %s", plot_path)

    from src.visualization.inference_viz import render_inference_dashboard
    batch = als.recommend_batch(
        enc.user.classes_[:30].astype(int).tolist(),
        N=10, filter_already_liked=False, n_jobs=2,
    )
    recs = {u: enc.inverse_transform_products(v[0]).tolist() for u, v in batch.items()}
    scmap = {u: v[1].tolist() for u, v in batch.items()}
    inf_plot = render_inference_dashboard(
        recs, ground_truth={}, scores=scmap,
        n_values=[5, 10, 20], latencies_s=[0.01, 0.02, 0.03],
    )
    log.info("Inference dashboard saved → %s", inf_plot)

    als.save(os.path.join(td, "als.pkl"))
    als2 = ALSRecommender.load(os.path.join(td, "als.pkl"))
    log.info("Save/load round-trip OK | loaded n_users=%d", als2.encoder.n_users)

log.info("==== SMOKE TEST PASSED ====")