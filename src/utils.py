import logging
from pathlib import Path

from src.config import LOGS_DIR


def ensure_directories() -> None:
    """Create runtime folders if they do not already exist."""
    for folder in [LOGS_DIR, LOGS_DIR.parent / "data", LOGS_DIR.parent / "models"]:
        folder.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    """Configure console and file logging for the pipeline."""
    ensure_directories()

    logger = logging.getLogger("stock_etl_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOGS_DIR / "pipeline.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def project_path(path: Path) -> str:
    """Return a friendly path string for logs and print messages."""
    return str(path.resolve())
