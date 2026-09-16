"""Logging utilities with clean format for academic and production runs."""

import logging
import sys


def setup_logger(name: str = "smart_waste", level: int = logging.INFO) -> logging.Logger:
    """Create and configure a standard console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on re-initialization
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
