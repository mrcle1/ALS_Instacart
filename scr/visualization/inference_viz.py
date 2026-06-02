"""Inference-time dashboard.

Generated after a batch ``recommend`` call. Six panels summarise what
came out:

1. **Top-N popularity** — which products were recommended most often
2. **Score distribution** — KDE of recommendation scores
3. **Hits @ N** — per-user hit count histogram
4. **Catalog coverage** — fraction of products ever recommended
5. **User activity** — histogram of per-user basket sizes
6. **Latency** — recommendation time per N (if provided)
"""

import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ..config import VizConfig
from ..logger import get_logger
from ..progress import instacart_tqdm
from ._style import DASH_FIG_BG, PANEL_BG, apply_global_style

log = get_logger(__name__)


def _hits_per_user(recommendations: Mapping[int, Sequence[int]],
                   ground_truth: Mapping[int, set]) -> np.ndarray:
    hits = []
    for uid, recs in recommendations.items():
        gt = ground_truth.get(uid, set())
        if not gt:
            continue
        hits.append(len(set(recs) & gt))
    return np.asarray(hits, dtype=int)


def _popularity(recommendations: Mapping[int, Sequence[int]]) -> pd.Series:
    counter: dict = {}
    for recs in recommendations.values():
        for p in recs:
            counter[int(p)] = counter.get(int(p), 0) + 1
    return pd.Series(counter).sort_values(ascending=False)


def render_inference_dashboard(
    recommendations: Mapping[int, Sequence[int]],
    ground_truth: Optional[Mapping[int, set]] = None,
    *,
    scores: Optional[Mapping[int, Sequence[float]]] = None,
    n_values: Optional[Sequence[int]] = None,
    latencies_s: Optional[Sequence[float]] = None,
    cfg: Optional[VizConfig] = None,
    save_path: Optional[str] = None,
) -> Optional[str]:
    """Render the inference dashboard.

    Parameters
    ----------
    recommendations : mapping
        ``{user_id: [product_id, ...]}``.
    ground_truth : mapping, optional
        ``{user_id: {product_id, ...}}`` — used for the hit-rate panel.
    scores : mapping, optional
        ``{user_id: [score, ...]}`` — used for the score distribution
        panel.
    n_values : sequence, optional
        N values used to compute ``latencies_s`` — used to label the
        x-axis on the latency panel.
    latencies_s : sequence, optional
        One latency per N (seconds).
    cfg : VizConfig, optional
        Visual configuration.
    save_path : str, optional
        Defaults to ``artifacts/plots/inference_dashboard.png``.
    """
    cfg = cfg or VizConfig()
    apply_global_style()
    save_path = save_path or "artifacts/plots/inference_dashboard.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("Rendering inference dashboard → %s | users=%d",
             save_path, len(recommendations))

    fig = plt.figure(figsize=tuple(cfg.figsize_inference))
    fig.patch.set_facecolor(DASH_FIG_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle(
        "Recommendation Inference — Dashboard",
        fontsize=18, fontweight="bold", color="#1A1A2E", y=0.98,
    )

    # ---- Panel 1: Top-20 product popularity ----
    ax1 = fig.add_subplot(gs[0, 0])
    pop = _popularity(recommendations)
    top = pop.head(20)
    ax1.barh(range(len(top))[::-1], top.values, color="#05ad46", edgecolor="white")
    ax1.set_yticks(range(len(top))[::-1])
    ax1.set_yticklabels([str(p) for p in top.index], fontsize=8)
    ax1.set_title("Top-20 Produk Paling Direkomendasikan", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Count", fontsize=9)
    ax1.set_facecolor(PANEL_BG)

    # ---- Panel 2: Score distribution ----
    ax2 = fig.add_subplot(gs[0, 1])
    if scores:
        flat = np.concatenate([
            np.asarray(s, dtype=float) for s in instacart_tqdm(
                scores.values(), desc="viz-scores", total=len(scores),
            ) if len(s)
        ]) if scores else np.array([])
    else:
        flat = np.array([])
    if flat.size:
        ax2.hist(flat, bins=40, color="#3A86FF", edgecolor="white", alpha=0.78, density=True)
        try:
            kde = gaussian_kde(flat, bw_method=0.3)
            xl = np.linspace(flat.min(), flat.max(), 300)
            ax2.plot(xl, kde(xl), color="#F77F00", linewidth=2.5, label="KDE")
        except Exception as exc:  # noqa: BLE001
            log.warning("Score KDE failed: %s", exc)
        ax2.set_title("Distribusi Skor Rekomendasi", fontsize=11, fontweight="bold", pad=8)
        ax2.set_xlabel("Score", fontsize=9)
        ax2.set_ylabel("Density", fontsize=9)
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No scores available", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=10)
        ax2.set_title("Distribusi Skor Rekomendasi", fontsize=11, fontweight="bold", pad=8)
    ax2.set_facecolor(PANEL_BG)

    # ---- Panel 3: Hits @ N ----
    ax3 = fig.add_subplot(gs[0, 2])
    if ground_truth:
        hits = _hits_per_user(recommendations, ground_truth)
        if hits.size:
            ax3.hist(hits, bins=np.arange(hits.max() + 2) - 0.5,
                     color="#06D6A0", edgecolor="white", alpha=0.85)
            ax3.axvline(hits.mean(), color="#E63946", linestyle="--",
                        linewidth=1.6, label=f"Mean={hits.mean():.2f}")
            ax3.set_title("Hits @ N per User", fontsize=11, fontweight="bold", pad=8)
            ax3.set_xlabel("Hits", fontsize=9)
            ax3.set_ylabel("Users", fontsize=9)
            ax3.legend(fontsize=8)
        else:
            ax3.text(0.5, 0.5, "No ground truth", ha="center", va="center",
                     transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, "No ground truth", ha="center", va="center",
                 transform=ax3.transAxes)
    ax3.set_facecolor(PANEL_BG)

    # ---- Panel 4: Catalog coverage ----
    ax4 = fig.add_subplot(gs[1, 0])
    n_unique = len({int(p) for recs in recommendations.values() for p in recs})
    n_total = len(set().union(*[set() for _ in range(1)]) or {0})  # placeholder 0
    # We can't know the global catalog size here; plot the absolute number
    # as a bar chart with a "n_recommended" annotation.
    sizes = [n_unique]
    ax4.bar(["Unique recommended"], sizes, color="#4361EE", edgecolor="white")
    ax4.text(0, sizes[0] + max(sizes) * 0.01, f"{sizes[0]:,}",
             ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax4.set_title("Catalog Coverage", fontsize=11, fontweight="bold", pad=8)
    ax4.set_ylabel("Unique product ids", fontsize=9)
    ax4.set_facecolor(PANEL_BG)

    # ---- Panel 5: User activity ----
    ax5 = fig.add_subplot(gs[1, 1])
    sizes = np.array([len(r) for r in recommendations.values()])
    if sizes.size:
        ax5.hist(sizes, bins=30, color="#F77F00", edgecolor="white", alpha=0.85)
        ax5.axvline(sizes.mean(), color="#1A237E", linestyle="--",
                    linewidth=1.6, label=f"Mean={sizes.mean():.1f}")
        ax5.set_title("Per-User Recommendation Count", fontsize=11, fontweight="bold", pad=8)
        ax5.set_xlabel("Recommendations per user", fontsize=9)
        ax5.set_ylabel("Users", fontsize=9)
        ax5.legend(fontsize=8)
    ax5.set_facecolor(PANEL_BG)

    # ---- Panel 6: Latency vs N ----
    ax6 = fig.add_subplot(gs[1, 2])
    if latencies_s is not None and n_values is not None and len(latencies_s) == len(n_values):
        ax6.plot(n_values, latencies_s, marker="o", markersize=8, color="#05ad46",
                 linewidth=2.2, label="Latency")
        for n, lat in zip(n_values, latencies_s):
            ax6.annotate(f"{lat:.2f}s", (n, lat), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color="#05ad46",
                         fontweight="bold")
        ax6.set_title("Latency vs N", fontsize=11, fontweight="bold", pad=8)
        ax6.set_xlabel("N", fontsize=9)
        ax6.set_ylabel("Seconds", fontsize=9)
        ax6.set_xticks(list(n_values))
        ax6.legend(fontsize=8)
    else:
        ax6.text(0.5, 0.5, "No latency data", ha="center", va="center",
                 transform=ax6.transAxes, fontsize=10)
        ax6.set_title("Latency vs N", fontsize=11, fontweight="bold", pad=8)
    ax6.set_facecolor(PANEL_BG)

    plt.savefig(save_path, dpi=cfg.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Inference dashboard saved → %s", os.path.abspath(save_path))
    return os.path.abspath(save_path)