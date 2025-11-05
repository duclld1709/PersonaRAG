"""Logging helpers for the backend application."""

import logging
from pathlib import Path

from .config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure application-level logging."""

    log_dir: Path = settings.logs_path
    log_dir.mkdir(parents=True, exist_ok=True)

    logfile = log_dir / "backend.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
