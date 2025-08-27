# logger_module.py

import logging
from contextlib import contextmanager
from typing import Generator

chipiron_logger: logging.Logger = logging.getLogger("chipiron_app")
chipiron_logger.setLevel(logging.DEBUG)

# Avoid duplicate handlers if this module is imported multiple times
if not chipiron_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)

    chipiron_logger.addHandler(console_handler)
    chipiron_logger.propagate = False


@contextmanager
def suppress_logging(
    logger: logging.Logger, level: int = logging.WARNING
) -> Generator[None, None, None]:
    """
    Context manager to temporarily suppress logging for a specific logger to a given level.

    Sets the logger's level to the specified value for the duration of the context, then restores
    its original level afterwards. Useful for silencing output from a particular logger during
    benchmarking or other operations.

    Args:
        logger (logging.Logger): The logger to suppress.
        level (int): The logging level to set (e.g., logging.ERROR, logging.WARNING).

    Yields:
        None
    """
    previous_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


# Suppress all logging from all loggers (global)
@contextmanager
def suppress_all_logging(level: int = logging.ERROR) -> Generator[None, None, None]:
    """
    Context manager to temporarily suppress logging from all loggers to a specified level.

    This sets the level of all loggers (including the root logger) to the given level for the duration
    of the context, then restores their original levels afterwards. Useful for benchmarking or
    situations where you want to silence all logging output temporarily.

    Args:
        level (int): The logging level to set (e.g., logging.ERROR, logging.WARNING).

    Yields:
        None
    """
    logger_dict = logging.getLogger().manager.loggerDict
    original_levels = {}

    for name in logger_dict:
        logger = logging.getLogger(name)
        original_levels[name] = logger.level
        logger.setLevel(level)

    root_logger = logging.getLogger()
    original_root_level = root_logger.level
    root_logger.setLevel(level)

    try:
        yield
    finally:
        for name, original_level in original_levels.items():
            logging.getLogger(name).setLevel(original_level)
        root_logger.setLevel(original_root_level)
