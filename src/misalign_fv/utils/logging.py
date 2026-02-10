"""Thread-safe structured logging for misalign-fv.

Usage::

    from misalign_fv.utils.logging import logger

    logger.info("Starting training", step=10, reward=0.5)
"""

from __future__ import annotations

import sys
import threading
from typing import Any

from loguru import logger as _loguru_logger

_setup_lock = threading.Lock()
_is_configured = False


def configure_logging(
    *,
    level: str = "INFO",
    json: bool = False,
    sink: Any = None,
) -> None:
    """Configure the global logger.

    Safe to call multiple times — only the first call takes effect.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json: If True, emit JSON-formatted log lines.
        sink: Output sink. Defaults to ``sys.stderr``.
    """
    global _is_configured
    with _setup_lock:
        if _is_configured:
            return
        _loguru_logger.remove()
        target = sink if sink is not None else sys.stderr
        _loguru_logger.add(
            target,
            level=level.upper(),
            serialize=json,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            enqueue=True,  # thread-safe via internal queue
        )
        _is_configured = True


# Auto-configure with defaults on import so callers can just use `logger`
# directly without explicit setup.
configure_logging()

logger = _loguru_logger

__all__ = ["configure_logging", "logger"]
