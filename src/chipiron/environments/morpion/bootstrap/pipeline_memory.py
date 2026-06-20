"""Cheap always-on RSS checkpoints for Morpion artifact-pipeline stages."""

from __future__ import annotations

import logging
import os
import resource
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _available_ram_mb_from_meminfo_text(text: str) -> float | None:
    """Parse Linux /proc/meminfo text and return MemAvailable in MiB."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] != "MemAvailable:":
            continue
        try:
            available_kib = float(parts[1])
        except ValueError:
            return None
        return available_kib / 1024
    return None


def available_ram_mb() -> float | None:
    """Return system-wide available RAM in MiB when cheaply discoverable."""
    if sys.platform.startswith("linux"):
        try:
            available = _available_ram_mb_from_meminfo_text(
                Path("/proc/meminfo").read_text(encoding="utf-8")
            )
        except OSError:
            available = None
        if available is not None:
            return available

    try:
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if available_pages <= 0 or page_size <= 0:
        return None
    return available_pages * page_size / (1024 * 1024)


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


def _ram_guard_enabled(required_mb: int | None) -> bool:
    """Return whether a minimum-available-RAM guard is configured."""
    return required_mb is not None and required_mb > 0


def has_min_available_ram(required_mb: int | None) -> bool:
    """Return whether the host has enough available RAM for a guarded action."""
    if not _ram_guard_enabled(required_mb):
        return True
    available = available_ram_mb()
    return available is None or available >= required_mb


def log_available_ram_guard(
    *,
    stage: str,
    generation: int | None,
    action: str,
    required_mb: int | None,
) -> bool:
    """Log and return the available-RAM guard decision for one heavy action."""
    if not _ram_guard_enabled(required_mb):
        return True
    available = available_ram_mb()
    should_run = available is None or available >= required_mb
    LOGGER.info(
        "[ram-guard] stage=%s generation=%s action=%s available_mb=%s required_mb=%s decision=%s",
        stage,
        format_metric(generation),
        action,
        format_metric(available),
        required_mb,
        "run" if should_run else "skip",
    )
    return should_run


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


__all__ = [
    "available_ram_mb",
    "current_rss_mb",
    "format_metric",
    "has_min_available_ram",
    "log_available_ram_guard",
    "log_pipeline_memory",
]
