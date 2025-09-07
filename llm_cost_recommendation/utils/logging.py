"""
Enhanced logging configuration for better readability and debugging.
"""

import sys
import structlog
from typing import Dict, Any
import os
from datetime import datetime


def should_use_human_readable() -> bool:
    """Determine if we should use human-readable output"""
    # Use human readable if:
    # 1. Running in a terminal (TTY)
    # 2. Not explicitly disabled via environment variable
    # 3. Not running in JSON mode

    if os.getenv("LOG_FORMAT") == "json":
        return False

    if os.getenv("LOG_FORMAT") == "human":
        return True

    # Auto-detect: use human readable if stdout is a TTY
    return sys.stdout.isatty()


def configure_logging(level: str = "INFO", format_type: str = "auto") -> None:
    """
    Configure structured logging with enhanced readability

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Output format ("auto", "json", "human")
    """
    # Set format type from environment or parameter
    if format_type == "auto":
        use_human = should_use_human_readable()
    else:
        use_human = format_type == "human"

    # Configure processors based on output format
    if use_human:
        # Human-readable format for interactive use
        processors = [
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON format for logging systems
        processors = [
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ]

    # Convert level string to number
    level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    log_level = level_map.get(level.upper(), 20)

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None):
    """Get a configured logger instance"""
    return structlog.get_logger(name)


# Convenience function for quick setup
def setup_logging(verbose: bool = False, json_format: bool = False) -> None:
    """
    Quick setup function for common use cases

    Args:
        verbose: Enable DEBUG level logging
        json_format: Force JSON output format
    """
    level = "DEBUG" if verbose else "INFO"
    format_type = "json" if json_format else "auto"
    configure_logging(level=level, format_type=format_type)
