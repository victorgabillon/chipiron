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
    previous_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous_level)
