"""Tests for the logging module."""

from __future__ import annotations

from misalign_fv.utils.logging import configure_logging, logger


class TestLogging:
    def test_logger_importable(self) -> None:
        assert logger is not None

    def test_configure_idempotent(self) -> None:
        configure_logging(level="DEBUG")
        configure_logging(level="WARNING")
        # Should not raise; second call is a no-op.
