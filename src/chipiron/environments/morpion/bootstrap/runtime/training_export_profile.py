"""Training-export profiling helpers for Morpion runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anemone.checkpoints.state_handles import CheckpointBackedStateHandle

from .checkpoint_io import _metric_value

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.sharded_training_export import (
        MorpionShardedTrainingExportStats,
    )

LOGGER = logging.getLogger("chipiron.environments.morpion.bootstrap.runtime.runner")

__all__ = [
    "MorpionTrainingExportProfile",
    "format_optional_seconds",
    "format_optional_seconds_with_unit",
    "log_sharded_training_export_stats",
    "log_training_export_profile",
    "sharded_training_export_stats_to_dict",
    "training_export_profile_to_dict",
    "value_to_scalar",
]


@dataclass(slots=True)
class MorpionTrainingExportProfile:
    """Aggregate profiling for one Morpion training/tree export build."""

    node_count: int = 0
    state_ref_count: int = 0
    payload_build_s: float = 0.0
    node_traversal_s: float = 0.0
    state_ref_serialization_s: float = 0.0
    node_payload_total_s: float = 0.0
    node_metadata_total_s: float = 0.0
    node_value_total_s: float = 0.0
    node_children_total_s: float = 0.0
    node_state_access_total_s: float = 0.0
    state_ref_conversion_total_s: float = 0.0
    checkpoint_backed_state_handles: int = 0
    reusable_checkpoint_payloads: int = 0
    plain_or_materialized_states: int = 0
    state_access_calls: int = 0

    def observe_state_handle(self, node: object) -> None:
        """Classify one raw state handle without forcing state resolution."""
        raw_handle: object = getattr(node, "state_handle", None)
        if isinstance(raw_handle, CheckpointBackedStateHandle):
            self.checkpoint_backed_state_handles += 1
        reusable_payload = (
            raw_handle.checkpoint_payload_for_reuse_or_none()
            if isinstance(raw_handle, CheckpointBackedStateHandle)
            else None
        )
        if reusable_payload is not None:
            self.reusable_checkpoint_payloads += 1
            return
        self.plain_or_materialized_states += 1

    def record_state_access(self, elapsed_s: float, *, state_present: bool) -> None:
        """Record one ``node.state`` access wall time."""
        del state_present
        self.state_access_calls += 1
        self.node_state_access_total_s += elapsed_s
        self.state_ref_serialization_s += elapsed_s

    def record_state_ref_conversion(self, elapsed_s: float) -> None:
        """Record one state-ref conversion wall time."""
        self.state_ref_count += 1
        self.state_ref_conversion_total_s += elapsed_s
        self.state_ref_serialization_s += elapsed_s

    def record_node_children(self, elapsed_s: float) -> None:
        """Record linkage extraction wall time."""
        self.node_children_total_s += elapsed_s

    def record_node_value(self, elapsed_s: float) -> None:
        """Record value payload extraction wall time."""
        self.node_value_total_s += elapsed_s

    def record_node_metadata(self, elapsed_s: float) -> None:
        """Record metadata payload extraction wall time."""
        self.node_metadata_total_s += elapsed_s

    def record_node_payload(self, elapsed_s: float) -> None:
        """Record one full node payload build wall time."""
        self.node_count += 1
        self.node_payload_total_s += elapsed_s

    def record_node_traversal(self, elapsed_s: float) -> None:
        """Record the total traversal wall time across all nodes."""
        self.node_traversal_s += elapsed_s


def training_export_profile_to_dict(
    profile: MorpionTrainingExportProfile,
) -> dict[str, object]:
    """Return JSON-friendly training-export profile metrics."""
    return {
        "node_count": profile.node_count,
        "state_ref_count": profile.state_ref_count,
        "payload_build_s": profile.payload_build_s,
        "node_traversal_s": profile.node_traversal_s,
        "state_ref_serialization_s": profile.state_ref_serialization_s,
        "node_payload_total_s": profile.node_payload_total_s,
        "node_metadata_total_s": profile.node_metadata_total_s,
        "node_value_total_s": profile.node_value_total_s,
        "node_children_total_s": profile.node_children_total_s,
        "node_state_access_total_s": profile.node_state_access_total_s,
        "state_ref_conversion_total_s": profile.state_ref_conversion_total_s,
        "checkpoint_backed_state_handles": profile.checkpoint_backed_state_handles,
        "reusable_checkpoint_payloads": profile.reusable_checkpoint_payloads,
        "plain_or_materialized_states": profile.plain_or_materialized_states,
        "state_access_calls": profile.state_access_calls,
        "state_ref_avg_ms": _average_ms(
            profile.state_ref_serialization_s,
            profile.state_ref_count,
        ),
        "state_access_avg_ms": _average_ms(
            profile.node_state_access_total_s,
            profile.state_access_calls,
        ),
        "state_ref_conversion_avg_ms": _average_ms(
            profile.state_ref_conversion_total_s,
            profile.state_ref_count,
        ),
    }


def sharded_training_export_stats_to_dict(
    stats: MorpionShardedTrainingExportStats,
) -> dict[str, object]:
    """Return JSON-friendly sharded training-export write metrics."""
    rss_delta_mb = (
        None
        if stats.rss_before_mb is None or stats.rss_after_mb is None
        else stats.rss_after_mb - stats.rss_before_mb
    )
    return {
        "export_mode": "sharded",
        "generation": stats.generation,
        "node_count": stats.node_count,
        "new_node_count": stats.new_node_count,
        "reused_node_count": stats.reused_node_count,
        "rows_written": stats.rows_written,
        "shards_written": stats.shards_written,
        "bytes_written": stats.bytes_written,
        "row_build_s": stats.row_build_s,
        "json_encode_s": stats.json_encode_s,
        "write_s": stats.write_s,
        "total_s": stats.total_s,
        "rss_before_mb": stats.rss_before_mb,
        "rss_after_mb": stats.rss_after_mb,
        "rss_delta_mb": rss_delta_mb,
    }


def format_optional_seconds(value: object) -> str:
    """Format one optional duration for stable timing logs."""
    return f"{float(value):.6f}" if isinstance(value, int | float) else "unknown"


def format_optional_seconds_with_unit(value: object) -> str:
    """Format one optional duration for concise human-facing logs."""
    return f"{float(value):.3f}s" if isinstance(value, int | float) else "unknown"


def value_to_scalar(value: object) -> float | None:
    """Extract a raw numeric score from one Anemone value-like object."""
    if value is None:
        return None
    score = getattr(value, "score", None)
    return float(score) if isinstance(score, int | float) else None


def _average_ms(total_s: float, count: int) -> float:
    """Return a stable milliseconds average for non-empty sample counts."""
    if count <= 0:
        return 0.0
    return total_s * 1000.0 / count


def _format_training_export_profile(profile: MorpionTrainingExportProfile) -> str:
    """Format one stable aggregate training-export profile log line."""
    return " ".join(
        (
            f"node_count={profile.node_count}",
            f"state_ref_count={profile.state_ref_count}",
            f"payload_build_s={_metric_value(profile.payload_build_s)}",
            f"node_traversal_s={_metric_value(profile.node_traversal_s)}",
            (
                "state_ref_serialization_s="
                f"{_metric_value(profile.state_ref_serialization_s)}"
            ),
            f"node_payload_total_s={_metric_value(profile.node_payload_total_s)}",
            f"node_metadata_total_s={_metric_value(profile.node_metadata_total_s)}",
            f"node_value_total_s={_metric_value(profile.node_value_total_s)}",
            f"node_children_total_s={_metric_value(profile.node_children_total_s)}",
            (
                "node_state_access_total_s="
                f"{_metric_value(profile.node_state_access_total_s)}"
            ),
            (
                "state_ref_conversion_total_s="
                f"{_metric_value(profile.state_ref_conversion_total_s)}"
            ),
            (
                "checkpoint_backed_state_handles="
                f"{profile.checkpoint_backed_state_handles}"
            ),
            f"reusable_checkpoint_payloads={profile.reusable_checkpoint_payloads}",
            f"plain_or_materialized_states={profile.plain_or_materialized_states}",
            f"state_access_calls={profile.state_access_calls}",
        )
    )


def _format_training_export_profile_rates(profile: MorpionTrainingExportProfile) -> str:
    """Format one stable aggregate training-export profile rates log line."""
    state_ref_avg_ms = _average_ms(
        profile.state_ref_serialization_s,
        profile.state_ref_count,
    )
    node_state_access_avg_ms = _average_ms(
        profile.node_state_access_total_s,
        profile.state_access_calls,
    )
    state_ref_conversion_avg_ms = _average_ms(
        profile.state_ref_conversion_total_s,
        profile.state_ref_count,
    )
    return " ".join(
        (
            f"state_ref_avg_ms={state_ref_avg_ms:.6f}",
            f"state_access_avg_ms={node_state_access_avg_ms:.6f}",
            f"node_state_access_avg_ms={node_state_access_avg_ms:.6f}",
            f"conversion_avg_ms={state_ref_conversion_avg_ms:.6f}",
            f"state_ref_conversion_avg_ms={state_ref_conversion_avg_ms:.6f}",
        )
    )


def log_training_export_profile(profile: MorpionTrainingExportProfile) -> None:
    """Emit stable aggregate profile logs for one training export build."""
    LOGGER.info(
        "[training-export-profile] %s", _format_training_export_profile(profile)
    )
    LOGGER.info(
        "[training-export-profile-rates] %s",
        _format_training_export_profile_rates(profile),
    )


def log_sharded_training_export_stats(
    stats: MorpionShardedTrainingExportStats,
) -> None:
    """Emit one stable summary line for sharded training-export writes."""
    LOGGER.info(
        "[sharded-training-export] generation=%s nodes=%s new_nodes=%s reused_nodes=%s rows=%s bytes=%s row_build_s=%.6f json_encode_s=%.6f write_s=%.6f total_s=%.6f",
        stats.generation,
        stats.node_count,
        stats.new_node_count,
        stats.reused_node_count,
        stats.rows_written,
        stats.bytes_written,
        stats.row_build_s,
        stats.json_encode_s,
        stats.write_s,
        stats.total_s,
    )
