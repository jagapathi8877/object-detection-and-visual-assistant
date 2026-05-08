"""
Centralised Logging Configuration for the AI Vision Assistant.

Provides a configured logger with both console and rotating file handlers.
All modules should import get_logger() rather than using print() or
creating their own logging setup.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Camera started successfully")
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


# Directory where log files are written (project root / logs)
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create and return a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level as string (e.g. 'DEBUG', 'INFO'). If None,
               defaults to INFO.
        log_file: Optional log filename. If None, defaults to
                  'vision_assistant.log' in the logs/ directory.

    Returns:
        A logging.Logger configured with console + file handlers.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)

    # ── Formatter ──────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler ────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File Handler (rotating, max 5 MB × 3 backups) ─────────
    os.makedirs(_LOG_DIR, exist_ok=True)
    file_path = os.path.join(_LOG_DIR, log_file or "vision_assistant.log")
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent log propagation to the root logger (avoids duplicate output)
    logger.propagate = False

    return logger
