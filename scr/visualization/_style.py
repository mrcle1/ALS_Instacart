import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# The palette mirrors the notebook: muted indigo / blue / green / orange.
PALETTE = {
    "NDCG@K": "#4361EE",
    "MRR@K": "#3A86FF",
    "Recall@K": "#06D6A0",
    "Precision@K": "#F77F00",
}
METRICS = ["NDCG@K", "MRR@K", "Recall@K", "Precision@K"]

# Diverging light→dark blue colormap used by the heatmap panel.
CMAP = LinearSegmentedColormap.from_list("instacart_blu", ["#EEF2FF", "#1A237E"])

DASH_FIG_BG = "#F8F9FA"
PANEL_BG = "#FFFFFF"
RADAR_BG = "#FAFAFA"


def apply_global_style() -> None:
    """Set sensible defaults once per process."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 130,
    })