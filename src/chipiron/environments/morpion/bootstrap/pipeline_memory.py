"""Cheap always-on RSS checkpoints for Morpion artifact-pipeline stages."""

from __future__ import annotations

import logging
import os
import resource
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def current_rss_mb() -> float | None:
    """Return current process RSS in MiB when available."""
    if sys.platform.startswith("linux"):
        try:
            statm = Path("/proc/self/statm").read_text(encoding="utf-8").split()
            resident_pages = int(statm[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return resident_pages * page_size / (1024 * 1024)
        except (OSError, ValueError, IndexError):
            pass

    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None
    if rss <= 0:
        return None
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def format_metric(value: object | None) -> object | None:
    """Render one optional metric compactly for structured logs."""
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 3)
    return value


def log_pipeline_memory(
    *,
    stage: str,
    event: str,
    generation: int | None = None,
    **metadata: object,
) -> None:
    """Log one lightweight pipeline memory marker."""
    metadata_text = " ".join(
        f"{key}={format_metric(value)}" for key, value in metadata.items()
    )
    if metadata_text:
        LOGGER.info(
            "[pipeline-memory] stage=%s generation=%s event=%s rss_mb=%s %s",
            stage,
            format_metric(generation),
            event,
            format_metric(current_rss_mb()),
            metadata_text,
        )
        return
    LOGGER.info(
        "[pipeline-memory] stage=%s generation=%s event=%s rss_mb=%s",
        stage,
        format_metric(generation),
        event,
        format_metric(current_rss_mb()),
    )


__all__ = ["current_rss_mb", "format_metric", "log_pipeline_memory"]
