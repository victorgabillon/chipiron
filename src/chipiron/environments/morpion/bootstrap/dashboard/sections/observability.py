"""Dashboard memory/export observability section rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_seconds as _format_seconds,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    format_value as _format_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    mapping_value as _mapping_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    numeric_value as _numeric_value,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    percentage as _percentage,
)
from chipiron.environments.morpion.bootstrap.dashboard.formatting import (
    ratio as _ratio,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["render_observability_section"]


def _export_fast_path_health(
    profile: Mapping[str, object],
) -> str:
    """Return a compact health label for checkpoint-backed export state access."""
    state_access_calls = _numeric_value(profile.get("state_access_calls"))
    if state_access_calls is None:
        return "unknown"
    plain_states = _numeric_value(profile.get("plain_or_materialized_states"))
    node_count = _numeric_value(profile.get("node_count"))
    tolerance = 1.0 if node_count is None else max(1.0, node_count * 0.005)
    if plain_states is not None and state_access_calls <= plain_states + tolerance:
        return "good"
    if (
        node_count is not None
        and node_count > 0
        and state_access_calls >= node_count * 0.5
    ):
        return "warning"
    return "review"


def _observability_summary_from_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Build dashboard-ready observability fields from status metadata."""
    tree = _mapping_value(metadata, "tree")
    memory = _mapping_value(metadata, "memory")
    state_eviction = _mapping_value(metadata, "state_eviction")
    checkpoint = _mapping_value(metadata, "checkpoint")
    training_export = _mapping_value(metadata, "training_export")
    training_export_profile = _mapping_value(metadata, "training_export_profile")
    node_count = tree.get("node_count") or training_export_profile.get("node_count")
    compact_payload_count = state_eviction.get("compact_payload_count")
    delta_payload_count = state_eviction.get("delta_payload_count")
    checkpoint_backed_handles = training_export_profile.get(
        "checkpoint_backed_state_handles"
    )
    plain_states = training_export_profile.get("plain_or_materialized_states")
    return {
        "tree": dict(tree),
        "memory": dict(memory),
        "state_eviction": dict(state_eviction),
        "checkpoint": dict(checkpoint),
        "training_export": dict(training_export),
        "training_export_profile": dict(training_export_profile),
        "delta_payload_ratio": _ratio(delta_payload_count, compact_payload_count),
        "checkpoint_backed_ratio": _ratio(checkpoint_backed_handles, node_count),
        "materialized_ratio": _ratio(plain_states, node_count),
        "export_fast_path_health": _export_fast_path_health(training_export_profile),
    }


def render_observability_section(
    *,
    st: Any,
    summary: Mapping[str, object],
) -> None:
    """Render memory/checkpoint/export observability from latest status metadata."""
    tree = cast("Mapping[str, object]", summary.get("tree", {}))
    memory = cast("Mapping[str, object]", summary.get("memory", {}))
    state_eviction = cast("Mapping[str, object]", summary.get("state_eviction", {}))
    checkpoint = cast("Mapping[str, object]", summary.get("checkpoint", {}))
    training_export = cast("Mapping[str, object]", summary.get("training_export", {}))
    profile = cast("Mapping[str, object]", summary.get("training_export_profile", {}))

    metric_columns = st.columns(7)
    metric_columns[0].metric("Nodes", _format_value(tree.get("node_count")))
    metric_columns[1].metric("Branches", _format_value(tree.get("branch_count")))
    metric_columns[2].metric("RSS MB", _format_value(memory.get("rss_mb")))
    metric_columns[3].metric(
        "Checkpoint-backed",
        _percentage(summary.get("checkpoint_backed_ratio")),
    )
    metric_columns[4].metric(
        "Delta payloads",
        _percentage(summary.get("delta_payload_ratio")),
    )
    metric_columns[5].metric(
        "Checkpoint save",
        _format_seconds(checkpoint.get("total_s")),
    )
    metric_columns[6].metric(
        "Training export",
        "skipped"
        if training_export.get("status") == "skipped"
        else _format_seconds(training_export.get("total_s")),
    )

    st.caption(
        "Checkpoint-backed nodes store compact payloads instead of full states. "
        "Delta payloads store parent plus move/delta. Export health warns if "
        "checkpoint-backed nodes start resolving full state during export."
    )
    st.dataframe(
        [
            {
                "metric": "Materialized/plain %",
                "value": _percentage(summary.get("materialized_ratio")),
            },
            {
                "metric": "State access calls",
                "value": _format_value(profile.get("state_access_calls")),
            },
            {
                "metric": "Plain/materialized states",
                "value": _format_value(profile.get("plain_or_materialized_states")),
            },
            {
                "metric": "Reusable checkpoint payloads",
                "value": _format_value(profile.get("reusable_checkpoint_payloads")),
            },
            {
                "metric": "Export fast-path health",
                "value": _format_value(summary.get("export_fast_path_health")),
            },
            {
                "metric": "Eviction successes",
                "value": _format_value(state_eviction.get("eviction_success_count")),
            },
            {
                "metric": "Eviction skips",
                "value": _format_value(state_eviction.get("eviction_skipped_count")),
            },
            {
                "metric": "Delta fallbacks",
                "value": _format_value(
                    state_eviction.get("delta_payload_fallback_count")
                ),
            },
            {
                "metric": "Rematerializations",
                "value": _format_value(state_eviction.get("rematerialization_count")),
            },
            {
                "metric": "Checkpoint bytes",
                "value": _format_value(checkpoint.get("bytes")),
            },
            {
                "metric": "Rows written",
                "value": _format_value(training_export.get("rows_written")),
            },
            {
                "metric": "Export bytes",
                "value": _format_value(training_export.get("bytes_written")),
            },
        ],
        width="stretch",
        hide_index=True,
    )
