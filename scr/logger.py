import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ROOT_NAME = "instacart_recommender"


def _resolve_log_file(logs_dir: str) -> Path:
    p = Path(logs_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / "instacart_recommender.log"


def setup_logging(logs_dir: str = "artifacts/logs",
                  level: int = logging.INFO,
                  log_to_file: bool = True) -> None:
    """Configure the root project logger in-place.

    Safe to call multiple times — handlers are wiped first to avoid
    double-emission when modules re-import.
    """
    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Wipe any previously attached handlers (covers re-imports and tests).
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:  # noqa: BLE001 — best effort cleanup
            pass

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ---- console handler (stderr) ----
    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setLevel(level)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    # ---- rotating file handler ----
    if log_to_file:
        log_path = _resolve_log_file(logs_dir)
        rotating = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
        )
        rotating.setLevel(level)
        rotating.setFormatter(formatter)
        logger.addHandler(rotating)

    logger.debug("Logger initialised at level %s", logging.getLevelName(level))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the project root.

    Usage::

        from src.logger import get_logger
        log = get_logger(__name__)
        log.info("ready")
    """
    if name is None or name == _ROOT_NAME or not name.startswith(_ROOT_NAME):
        # Always namespace under the project root so configuration sticks.
        name = f"{_ROOT_NAME}.{name}" if name and name != _ROOT_NAME else _ROOT_NAME
    return logging.getLogger(name)

