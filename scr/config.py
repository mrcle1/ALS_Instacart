import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Resolve the project root regardless of where Python is launched from.
# config.ini lives at the project root, two levels above this file.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.ini"


# ---------------------------------------------------------------------------
# Typed sub-configs
# ---------------------------------------------------------------------------
@dataclass
class PathsConfig:
    raw_data: str = ""
    artifacts_dir: str = "artifacts"
    models_dir: str = "artifacts/models"
    plots_dir: str = "artifacts/plots"
    logs_dir: str = "artifacts/logs"
    predictions_dir: str = "artifacts/predictions"


@dataclass
class DataConfig:
    min_user_items: int = 10
    test_ratio: float = 0.2
    random_state: int = 42
    sample_size: Optional[int] = 10_000
    active_user_min_products: int = 10


@dataclass
class ALSConfig:
    factors: int = 128
    regularization: float = 0.01
    iterations: int = 30
    alpha: float = 40.0
    use_gpu: bool = False
    num_threads: int = 4


@dataclass
class FPGrowthConfig:
    min_support: float = 0.01
    min_confidence: float = 0.2
    min_lift: float = 1.0
    max_len: int = 3
    top_n: int = 10
    use_colnames: bool = True


@dataclass
class EvalConfig:
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    n_jobs: int = -1
    sample_size: Optional[int] = 10_000


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "Instacart Recommender API"
    version: str = "0.1.0"


@dataclass
class VizConfig:
    palette_ndcg: str = "#4361EE"
    palette_mrr: str = "#3A86FF"
    palette_recall: str = "#06D6A0"
    palette_precision: str = "#F77F00"
    dpi: int = 150
    figsize_dashboard: List[int] = field(default_factory=lambda: [20, 14])
    figsize_inference: List[int] = field(default_factory=lambda: [18, 12])
    brand_color: str = "#05ad46"  # Instacart green — used for tqdm bar


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    als: ALSConfig = field(default_factory=ALSConfig)
    fpgrowth: FPGrowthConfig = field(default_factory=FPGrowthConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    api: APIConfig = field(default_factory=APIConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _get_list(parser: configparser.ConfigParser, section: str, key: str,
              fallback: List[str]) -> List[str]:
    raw = parser.get(section, key, fallback=",".join(fallback))
    return [item.strip() for item in raw.split(",") if item.strip()]


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config(path: Optional[os.PathLike] = None) -> AppConfig:
    """Load and parse the INI file into a typed ``AppConfig`` object.

    Parameters
    ----------
    path : optional path-like
        Path to ``config.ini``. Defaults to the project-root file.
    """
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config.ini not found at {cfg_path}. "
            "Create it at the project root or pass an explicit path."
        )

    parser = configparser.ConfigParser()
    parser.read(cfg_path, encoding="utf-8")

    cfg = AppConfig()

    # -- paths --
    if parser.has_section("paths"):
        cfg.paths.raw_data = parser.get("paths", "raw_data",
                                        fallback=cfg.paths.raw_data)
        cfg.paths.artifacts_dir = parser.get("paths", "artifacts_dir",
                                             fallback=cfg.paths.artifacts_dir)
        cfg.paths.models_dir = parser.get("paths", "models_dir",
                                          fallback=cfg.paths.models_dir)
        cfg.paths.plots_dir = parser.get("paths", "plots_dir",
                                         fallback=cfg.paths.plots_dir)
        cfg.paths.logs_dir = parser.get("paths", "logs_dir",
                                        fallback=cfg.paths.logs_dir)
        cfg.paths.predictions_dir = parser.get("paths", "predictions_dir",
                                               fallback=cfg.paths.predictions_dir)

    # -- data --
    if parser.has_section("data"):
        cfg.data.min_user_items = parser.getint("data", "min_user_items",
                                                fallback=cfg.data.min_user_items)
        cfg.data.test_ratio = parser.getfloat("data", "test_ratio",
                                              fallback=cfg.data.test_ratio)
        cfg.data.random_state = parser.getint("data", "random_state",
                                              fallback=cfg.data.random_state)
        cfg.data.active_user_min_products = parser.getint(
            "data", "active_user_min_products",
            fallback=cfg.data.active_user_min_products,
        )
        sample_raw = parser.get("data", "sample_size", fallback="10000")
        cfg.data.sample_size = None if sample_raw.lower() in {"none", "null", ""} \
            else int(sample_raw)

    # -- als --
    if parser.has_section("als"):
        cfg.als.factors = parser.getint("als", "factors", fallback=cfg.als.factors)
        cfg.als.regularization = parser.getfloat("als", "regularization",
                                                 fallback=cfg.als.regularization)
        cfg.als.iterations = parser.getint("als", "iterations",
                                           fallback=cfg.als.iterations)
        cfg.als.alpha = parser.getfloat("als", "alpha", fallback=cfg.als.alpha)
        cfg.als.use_gpu = _as_bool(parser.get("als", "use_gpu",
                                              fallback=str(cfg.als.use_gpu)))
        cfg.als.num_threads = parser.getint("als", "num_threads",
                                            fallback=cfg.als.num_threads)

    # -- fpgrowth --
    if parser.has_section("fpgrowth"):
        cfg.fpgrowth.min_support = parser.getfloat("fpgrowth", "min_support",
                                                   fallback=cfg.fpgrowth.min_support)
        cfg.fpgrowth.min_confidence = parser.getfloat("fpgrowth", "min_confidence",
                                                     fallback=cfg.fpgrowth.min_confidence)
        cfg.fpgrowth.min_lift = parser.getfloat("fpgrowth", "min_lift",
                                                fallback=cfg.fpgrowth.min_lift)
        cfg.fpgrowth.max_len = parser.getint("fpgrowth", "max_len",
                                             fallback=cfg.fpgrowth.max_len)
        cfg.fpgrowth.top_n = parser.getint("fpgrowth", "top_n",
                                           fallback=cfg.fpgrowth.top_n)
        cfg.fpgrowth.use_colnames = _as_bool(parser.get(
            "fpgrowth", "use_colnames", fallback=str(cfg.fpgrowth.use_colnames),
        ))

    # -- eval --
    if parser.has_section("eval"):
        cfg.eval.k_values = [
            int(x) for x in _get_list(parser, "eval", "k_values",
                                      [str(k) for k in cfg.eval.k_values])
        ]
        cfg.eval.n_jobs = parser.getint("eval", "n_jobs", fallback=cfg.eval.n_jobs)
        sample_raw = parser.get("eval", "sample_size", fallback="10000")
        cfg.eval.sample_size = None if sample_raw.lower() in {"none", "null", ""} \
            else int(sample_raw)

    # -- api --
    if parser.has_section("api"):
        cfg.api.host = parser.get("api", "host", fallback=cfg.api.host)
        cfg.api.port = parser.getint("api", "port", fallback=cfg.api.port)
        cfg.api.title = parser.get("api", "title", fallback=cfg.api.title)
        cfg.api.version = parser.get("api", "version", fallback=cfg.api.version)

    # -- viz --
    if parser.has_section("viz"):
        cfg.viz.dpi = parser.getint("viz", "dpi", fallback=cfg.viz.dpi)
        cfg.viz.palette_ndcg = parser.get("viz", "palette_ndcg",
                                          fallback=cfg.viz.palette_ndcg)
        cfg.viz.palette_mrr = parser.get("viz", "palette_mrr",
                                         fallback=cfg.viz.palette_mrr)
        cfg.viz.palette_recall = parser.get("viz", "palette_recall",
                                            fallback=cfg.viz.palette_recall)
        cfg.viz.palette_precision = parser.get("viz", "palette_precision",
                                               fallback=cfg.viz.palette_precision)
        cfg.viz.brand_color = parser.get("viz", "brand_color",
                                         fallback=cfg.viz.brand_color)
        cfg.viz.figsize_dashboard = [
            int(x) for x in _get_list(
                parser, "viz", "figsize_dashboard",
                [str(v) for v in cfg.viz.figsize_dashboard],
            )
        ]
        cfg.viz.figsize_inference = [
            int(x) for x in _get_list(
                parser, "viz", "figsize_inference",
                [str(v) for v in cfg.viz.figsize_inference],
            )
        ]

    # Make sure artifact directories actually exist.
    for d in [cfg.paths.artifacts_dir, cfg.paths.models_dir, cfg.paths.plots_dir,
              cfg.paths.logs_dir, cfg.paths.predictions_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    return cfg


# Module-level singleton accessor — keeps call sites short.
_cached: Optional[AppConfig] = None


def get_config(path: Optional[os.PathLike] = None,
               reload: bool = False) -> AppConfig:
    """Return a memoized :class:`AppConfig`. Pass ``reload=True`` to re-read."""
    global _cached
    if _cached is None or reload:
        _cached = load_config(path)
    return _cached