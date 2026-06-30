"""Candidate checkpoint loading helpers for Morpion artifact-pipeline growth."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    CandidateCheckpointLoadDeferredError,
    CandidateCheckpointLoadProfile,
)
from chipiron.environments.morpion.bootstrap.pipeline_memory import (
    current_rss_mb,
    format_metric,
    log_available_ram_guard,
    log_candidate_checkpoint_load_memory_forecast,
)
from chipiron.environments.morpion.bootstrap.runtime.checkpoint_codec import (
    load_morpion_search_checkpoint_payload,
)
from chipiron.environments.morpion.bootstrap.runtime.restore_memory_logging import (
    restore_memory_logger_for_checkpoint_path,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from anemone.checkpoints import SearchRuntimeCheckpointPayload

    from chipiron.environments.morpion.bootstrap.bootstrap_args import (
        MorpionBootstrapArgs,
    )

LOGGER = logging.getLogger(__name__)


def runtime_checkpoint_artifact_bytes(path: Path) -> int | None:
    """Return best-effort byte size for file or sharded directory checkpoint."""
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(
                item.stat().st_size for item in path.rglob("*") if item.is_file()
            )
    except OSError:
        return None
    return None


def log_before_candidate_checkpoint_load(
    *,
    args: MorpionBootstrapArgs,
    generation: int,
    candidate_path: Path,
) -> bool:
    """Log an opt-in profile marker before candidate checkpoint validation load."""
    if args.growth_memory_profile:
        try:
            checkpoint_bytes = candidate_path.stat().st_size
        except OSError:
            checkpoint_bytes = None
        LOGGER.info(
            "[growth-profile] event=before_candidate_checkpoint_load "
            "generation=%s rss_mb=%s checkpoint_bytes=%s path=%s",
            generation,
            format_metric(current_rss_mb()),
            checkpoint_bytes,
            str(candidate_path),
        )
    return True


def should_load_candidate_checkpoint(
    *,
    args: MorpionBootstrapArgs,
    generation: int,
    source: str,
    candidate_path: Path,
) -> bool:
    """Return whether candidate checkpoint validation may load the payload."""
    log_before_candidate_checkpoint_load(
        args=args,
        generation=generation,
        candidate_path=candidate_path,
    )
    forecast = log_candidate_checkpoint_load_memory_forecast(
        stage="growth",
        generation=generation,
        action="candidate_checkpoint_load",
        checkpoint_path=candidate_path,
        min_available_ram_mb=args.min_available_ram_mb,
        headroom_factor=args.candidate_checkpoint_load_headroom_factor,
        min_headroom_mb=args.candidate_checkpoint_load_min_headroom_mb,
    )
    if forecast.decision == "skip":
        raise CandidateCheckpointLoadDeferredError(
            source=source,
            artifact_path=candidate_path,
            action="candidate_checkpoint_load_forecast",
        )
    return log_available_ram_guard(
        stage="growth",
        generation=generation,
        action="candidate_checkpoint_load",
        required_mb=args.min_available_ram_mb,
    )


def candidate_checkpoint_payload_loader(
    args: MorpionBootstrapArgs,
) -> Callable[[Path], SearchRuntimeCheckpointPayload]:
    """Build the optional instrumented candidate-checkpoint loader."""

    def load(path: Path) -> SearchRuntimeCheckpointPayload:
        restore_memory_logger = restore_memory_logger_for_checkpoint_path(
            path,
            enabled=True,
            recursive_enabled=args.growth_memory_profile_recursive,
            recursive_max_objects=args.growth_memory_profile_recursive_max_objects,
            recursive_max_depth=args.growth_memory_profile_recursive_max_depth,
        )
        if restore_memory_logger is not None:
            restore_memory_logger.log(
                "before_checkpoint_file_load",
                raw_checkpoint_referenced=False,
                typed_checkpoint_referenced=False,
                cache="candidate_validation",
            )
        return load_morpion_search_checkpoint_payload(
            path,
            restore_memory_logger=restore_memory_logger,
        )

    return load


def log_candidate_checkpoint_load_profile(
    profile: CandidateCheckpointLoadProfile,
) -> None:
    """Log the RSS delta observed while validating one candidate checkpoint."""
    rss_delta_mb = None
    if profile.rss_before_mb is not None and profile.rss_after_mb is not None:
        rss_delta_mb = profile.rss_after_mb - profile.rss_before_mb
    checkpoint_bytes_per_node = None
    if (
        profile.checkpoint_bytes is not None
        and profile.node_count is not None
        and profile.node_count > 0
    ):
        checkpoint_bytes_per_node = profile.checkpoint_bytes / profile.node_count
    LOGGER.info(
        "[growth-profile] event=candidate_checkpoint_load_done generation=%s "
        "source=%s rss_before_mb=%s rss_after_mb=%s rss_delta_mb=%s "
        "checkpoint_bytes=%s nodes=%s checkpoint_bytes_per_node=%s "
        "load_elapsed=%.3fs path=%s",
        profile.generation,
        profile.source,
        format_metric(profile.rss_before_mb),
        format_metric(profile.rss_after_mb),
        format_metric(rss_delta_mb),
        profile.checkpoint_bytes,
        profile.node_count,
        format_metric(checkpoint_bytes_per_node),
        profile.elapsed_s,
        str(profile.path),
    )
