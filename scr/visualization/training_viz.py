"""Training-time dashboard.

A faithful 6-panel recreation of the notebook visualisation:

1. Grouped bar — every metric at every K
2. Line — metric trend vs K
3. Radar — snapshot at K=10
4. Heatmap — K × metric, annotated
5. Histogram + KDE — per-user NDCG@10 distribution
6. Summary table — readable on its own
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from ..config import VizConfig
from ..logger import get_logger
from ._style import (
    CMAP, DASH_FIG_BG, METRICS, PALETTE, PANEL_BG, RADAR_BG, apply_global_style,
)

log = get_logger(__name__)


def render_training_dashboard(
    eval_df: pd.DataFrame,
    per_user_ndcg,
    *,
    cfg: Optional[VizConfig] = None,
    save_path: Optional[str] = None,
) -> Optional[str]:
    """Render the training dashboard to disk and return the saved path.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Output of :func:`src.evaluation.evaluator.evaluate_recommender` —
        must contain ``K``, ``NDCG@K``, ``MRR@K``, ``Recall@K``,
        ``Precision@K`` columns.
    per_user_ndcg : sequence of float
        Per-user NDCG@K (typically NDCG@10) used by the histogram panel.
    cfg : VizConfig, optional
        Visual configuration.
    save_path : str, optional
        If omitted, the figure is written to
        ``artifacts/plots/training_dashboard.png``.
    """
    cfg = cfg or VizConfig()
    apply_global_style()
    save_path = save_path or "artifacts/plots/training_dashboard.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    K_VALS = eval_df["K"].tolist()
    log.info("Rendering training dashboard → %s", save_path)

    fig = plt.figure(figsize=tuple(cfg.figsize_dashboard))
    fig.patch.set_facecolor(DASH_FIG_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle(
        "ALS Collaborative Filtering — Evaluation Dashboard",
        fontsize=18, fontweight="bold", color="#1A1A2E", y=0.98,
    )

    # ---- Panel 1: Grouped Bar ----
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(K_VALS))
    w = 0.18
    offs = np.linspace(-(len(METRICS) - 1) / 2 * w, (len(METRICS) - 1) / 2 * w, len(METRICS))
    for off, m in zip(offs, METRICS):
        vals = eval_df[m].values
        bars = ax1.bar(
            x + off, vals, width=w, label=m,
            color=PALETTE[m], edgecolor="white", linewidth=0.6, alpha=0.92,
        )
        for b in bars:
            h = b.get_height()
            ax1.text(
                b.get_x() + b.get_width() / 2, h + 0.001, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#333",
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"K={k}" for k in K_VALS], fontsize=10)
    ax1.set_title("Semua Metrik per K", fontsize=12, fontweight="bold", pad=8)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_ylim(0, eval_df[METRICS].max().max() * 1.3)
    ax1.set_facecolor(PANEL_BG)

    # ---- Panel 2: Line ----
    ax2 = fig.add_subplot(gs[0, 1])
    for m in METRICS:
        vals = eval_df[m].values
        ax2.plot(K_VALS, vals, marker="o", markersize=8, linewidth=2.2,
                 label=m, color=PALETTE[m])
        for k, v in zip(K_VALS, vals):
            ax2.annotate(
                f"{v:.3f}", (k, v), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color=PALETTE[m], fontweight="bold",
            )
    ax2.set_xticks(K_VALS)
    ax2.set_xticklabels([f"K={k}" for k in K_VALS], fontsize=10)
    ax2.set_title("Tren Metrik vs K", fontsize=12, fontweight="bold", pad=8)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_xlabel("K", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_facecolor(PANEL_BG)

    # ---- Panel 3: Radar ----
    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    if 10 in K_VALS:
        row10 = eval_df[eval_df["K"] == 10].iloc[0]
    else:
        row10 = eval_df.iloc[len(eval_df) // 2]
    r_vals = [row10[m] for m in METRICS] + [row10[METRICS[0]]]
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist() + [0]
    ax3.plot(angles, r_vals, "o-", linewidth=2, color="#4361EE")
    ax3.fill(angles, r_vals, alpha=0.25, color="#4361EE")
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(METRICS, fontsize=9, fontweight="bold")
    ax3.set_title("Radar Chart @ K=10", fontsize=12, fontweight="bold", pad=20)
    ax3.set_facecolor(RADAR_BG)

    # ---- Panel 4: Heatmap ----
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        eval_df.set_index("K")[METRICS],
        annot=True, fmt=".4f", cmap=CMAP,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 10, "weight": "bold"},
        ax=ax4, cbar_kws={"shrink": 0.8},
    )
    ax4.set_title("Heatmap (K × Metrik)", fontsize=12, fontweight="bold", pad=8)
    ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, fontsize=10)

    # ---- Panel 5: NDCG distribution ----
    ax5 = fig.add_subplot(gs[1, 1])
    arr = np.asarray(per_user_ndcg, dtype=float)
    ax5.hist(arr, bins=40, color="#4361EE", edgecolor="white", alpha=0.75, density=True)
    if len(arr) > 10:
        try:
            kde = gaussian_kde(arr, bw_method=0.3)
            xl = np.linspace(0, 1, 300)
            ax5.plot(xl, kde(xl), color="#F77F00", linewidth=2.5, label="KDE")
        except Exception as exc:  # noqa: BLE001
            log.warning("KDE failed: %s", exc)
    ax5.axvline(arr.mean(), color="#E63946", linestyle="--", linewidth=1.8,
                label=f"Mean={arr.mean():.3f}")
    ax5.axvline(np.median(arr), color="#06D6A0", linestyle=":", linewidth=1.8,
                label=f"Median={np.median(arr):.3f}")
    ax5.set_title("Distribusi NDCG@10 per User", fontsize=12, fontweight="bold", pad=8)
    ax5.set_xlabel("NDCG@10", fontsize=10)
    ax5.set_ylabel("Density", fontsize=10)
    ax5.legend(fontsize=8)
    ax5.set_facecolor(PANEL_BG)

    # ---- Panel 6: Summary table ----
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    alt = [["#EEF2FF"] * 5, ["#FFFFFF"] * 5]
    rows_data = [
        [
            f"{int(r['K'])}", f"{r['NDCG@K']:.4f}", f"{r['MRR@K']:.4f}",
            f"{r['Recall@K']:.4f}", f"{r['Precision@K']:.4f}",
        ]
        for _, r in eval_df.iterrows()
    ]
    cell_colors = [alt[i % 2] for i in range(len(rows_data))]
    tbl = ax6.table(
        cellText=rows_data,
        colLabels=["K", "NDCG@K", "MRR@K", "Recall@K", "Prec@K"],
        cellLoc="center", loc="center", cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 1.8)
    for j in range(5):
        c = tbl[0, j]
        c.set_facecolor("#1A237E")
        c.set_text_props(color="white", fontweight="bold")
    ax6.set_title("Tabel Ringkasan", fontsize=12, fontweight="bold", pad=8)

    plt.savefig(save_path, dpi=cfg.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Training dashboard saved → %s", os.path.abspath(save_path))
    return os.path.abspath(save_path)
