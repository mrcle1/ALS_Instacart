from pathlib import Path
from typing import Union

import pandas as pd

from ..logger import get_logger
from ..progress import instacart_tqdm

log = get_logger(__name__)

PathLike = Union[str, Path]

_REQUIRED_COLUMNS = (
    "order_id",
    "user_id",
    "product_id",
    "aisle_id",
    "department_id",
    "order_number",
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order",
    "add_to_cart_order",
    "reordered",
)


def _read_any(path: Path) -> pd.DataFrame:
    """Dispatch on file suffix to the appropriate pandas reader."""
    suffix = path.suffix.lower()
    log.info("Reading %s (suffix=%s)", path, suffix or "<none>")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format for {path!r}")


def _coerce_schema(df: pd.DataFrame, source: PathLike) -> pd.DataFrame:
    """Cast columns to the canonical dtypes and warn on missing ones."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        log.warning("Source %s is missing expected columns: %s", source, missing)
        return df  # caller may still want to use the partial frame

    df = df.copy()
    for col in (
        "order_id", "user_id", "product_id", "aisle_id", "department_id",
        "order_number", "order_dow", "order_hour_of_day",
        "add_to_cart_order", "reordered",
    ):
        df[col] = df[col].astype("int64")
    return df


def load_instacart_raw(path: PathLike,
                       *, validate: bool = True) -> pd.DataFrame:
    """Load an Instacart source file into a normalised DataFrame.

    Parameters
    ----------
    path : str or Path
        Local path to the parquet/csv/etc. file.
    validate : bool, default True
        If True, raises :class:`ValueError` when required columns are
        missing. If False, the frame is returned as-is and a warning is
        logged.
    """
    p = Path(path)
    if not p.exists():
        log.error("Source path does not exist: %s", p)
        raise FileNotFoundError(p)

    log.info("Ingesting Instacart source: %s", p)
    df = _read_any(p)

    if validate:
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            log.error("Missing required columns: %s", missing)
            raise ValueError(
                f"File {p} is missing required Instacart columns: {missing}"
            )
        df = _coerce_schema(df, p)
    else:
        log.warning("Skipping schema validation for %s", p)

    # Wrap the trivial in-memory step in a tqdm bar so the user sees the
    # same green bar style used everywhere else in the project.
    rows = 0
    for _ in instacart_tqdm(
        range(1), desc="load-instacart", total=1,
    ):
        rows = len(df)
    log.info(
        "Loaded %s rows | users=%s | products=%s | columns=%d",
        f"{rows:,}",
        f"{df['user_id'].nunique():,}" if "user_id" in df.columns else "?",
        f"{df['product_id'].nunique():,}" if "product_id" in df.columns else "?",
        df.shape[1],
    )
    return df

