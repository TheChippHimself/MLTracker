"""Structured logging setup for PoseSense.

Configures a console handler (concise format) and an optional rotating
file handler under ``data/logs/``.  All loggers in the ``posesense``
namespace inherit from the root ``posesense`` logger.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONSOLE_FORMAT = "%(asctime)s [%(levelname)-5.5s] %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_FILE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB per log file
_BACKUP_COUNT = 3


def setup_logging(
    *,
    level: int = logging.INFO,
    log_dir: Path | None = None,
    log_to_file: bool = True,
) -> logging.Logger:
    """Configure the ``posesense`` logger hierarchy.

    Args:
        level: Logging level for the root ``posesense`` logger.
        log_dir: Directory for log files.  Defaults to ``data/logs/``.
        log_to_file: Whether to write logs to a rotating file.

    Returns:
        The configured root ``posesense`` :class:`logging.Logger`.
    """
    logger = logging.getLogger("posesense")
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    # ── Console handler ──────────────────────────────────────────────
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console)

    # ── File handler (optional) ──────────────────────────────────────
    if log_to_file:
        if log_dir is None:
            log_dir = Path("data") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "posesense.log",
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FORMAT)
        )
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid double output.
    logger.propagate = False

    return logger
