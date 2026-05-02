"""Tiny helper utilities used across the project."""
import logging

_LOGGER_INITIALIZED = False


def get_logger(name: str = "ned") -> logging.Logger:
    global _LOGGER_INITIALIZED
    if not _LOGGER_INITIALIZED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        _LOGGER_INITIALIZED = True
    return logging.getLogger(name)
