"""Tests for the logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from posesense.logging_setup import setup_logging


class TestLoggingSetup:
    """Test logger configuration."""

    def test_returns_logger(self) -> None:
        logger = setup_logging(level=logging.DEBUG, log_to_file=False)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "posesense"
        # Cleanup
        logger.handlers.clear()

    def test_console_handler_added(self) -> None:
        logger = setup_logging(level=logging.INFO, log_to_file=False)
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1
        # Cleanup
        logger.handlers.clear()

    def test_file_handler_created(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        logger = setup_logging(level=logging.DEBUG, log_dir=log_dir, log_to_file=True)
        assert log_dir.exists()
        assert (log_dir / "posesense.log").exists()
        # Cleanup
        logger.handlers.clear()

    def test_no_duplicate_handlers(self) -> None:
        """Calling setup_logging twice should not add duplicate handlers."""
        logger = setup_logging(level=logging.INFO, log_to_file=False)
        n = len(logger.handlers)
        setup_logging(level=logging.INFO, log_to_file=False)
        assert len(logger.handlers) == n
        # Cleanup
        logger.handlers.clear()

    def test_child_logger_inherits(self) -> None:
        setup_logging(level=logging.DEBUG, log_to_file=False)
        child = logging.getLogger("posesense.camera")
        assert child.getEffectiveLevel() == logging.DEBUG
        # Cleanup
        logging.getLogger("posesense").handlers.clear()
