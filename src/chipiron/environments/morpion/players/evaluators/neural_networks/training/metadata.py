"""Metadata and timing helpers for Morpion neural-network training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    MORPION_MANIFEST_FILE_NAME,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping

    from chipiron.learning.timing import PhaseDurations


def add_phase_durations(
    target: PhaseDurations,
    phase_durations: Mapping[str, float],
) -> None:
    """Accumulate one phase-duration mapping into another timer."""
    for phase, seconds in phase_durations.items():
        target.add_duration(phase, seconds)


def add_timing_metrics(
    metrics: dict[str, float | str | None],
    *,
    prefix: str,
    phase_durations: Mapping[str, float],
) -> None:
    """Add flat timing fields to a Morpion training metrics mapping."""
    for phase, seconds in phase_durations.items():
        key = f"timing_{prefix}_{phase}_s" if prefix else f"timing_{phase}_s"
        metrics[key] = float(seconds)


def rows_per_second(row_count: int, elapsed_seconds: float) -> float:
    """Return one safe row-throughput value for logs."""
    if elapsed_seconds <= 0.0:
        return 0.0
    return row_count / elapsed_seconds


def update_saved_manifest_metadata(
    output_dir: str | os.PathLike[str],
    metadata: dict[str, object],
) -> None:
    """Update a saved Morpion manifest with final post-save metadata."""
    manifest_path = Path(output_dir) / MORPION_MANIFEST_FILE_NAME
    with open(manifest_path, encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    if isinstance(manifest_payload, dict):
        manifest_payload["metadata"] = metadata
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest_payload, handle, indent=2, sort_keys=True)
