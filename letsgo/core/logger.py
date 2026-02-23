"""
WasteBot Logger
================
Centralised logging configuration for the entire robot.

Log Levels:
    DEBUG   → per-frame data, servo angles, motor PWM values
    INFO    → state transitions, decisions, periodic summaries
    WARNING → lost targets, retries, fallbacks
    ERROR   → hardware failures, exceptions

Usage:
    from core.logger import get_logger
    log = get_logger("module_name")
    log.info("Something happened")
"""

import logging
import sys


def setup_logging(level: str = "DEBUG") -> None:
    """
    Configure the root WasteBot logger.
    Call this once at startup before any other imports.

    Args:
        level: "DEBUG", "INFO", "WARNING", "ERROR"
    """
    log_level = getattr(logging, level.upper(), logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)-5s] %(name)-12s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger("wastebot")
    root.setLevel(log_level)

    # Avoid duplicate handlers on re-import
    if not root.handlers:
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the 'wastebot' namespace.

    Args:
        name: Module name (e.g. "motor", "scanner", "approach").

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(f"wastebot.{name}")
