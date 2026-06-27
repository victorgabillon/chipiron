"""Real Anemone-backed Morpion search runner for bootstrap cycles."""

from __future__ import annotations

import gc
import logging
import time
from collections import OrderedDict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    CheckpointNodeStatePayload,
    DeltaCheckpointStatePayload,
    LinooSelectorCheckpointPayload,
    RestoreMemoryPhaseLogger,
    SearchRuntimeCheckpointPayload,
    build_search_checkpoint_payload,
    load_checkpoint_json_payload,
    load_search_from_checkpoint_payload,
    load_search_from_sharded_checkpoint,
    read_sharded_checkpoint_manifest,
    write_checkpoint_json_payload,
    write_sharded_search_checkpoint,
)
from anemone.checkpoints.state_handles import (
    CheckpointBackedStateHandle,
)
from anemone.factory import (
    SearchArgs,
    create_tree_and_value_exploration_with_tree_eval_factory,
)
from anemone.node_evaluation.tree.single_agent.factory import (
    NodeMaxEvaluationFactory,
)
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.linoo import LinooArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.nodes.state_handles import MaterializedStateHandle
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimit,
    TreeBranchLimitArgs,
)
from anemone.recommender_rule.recommender_rule import AlmostEqualLogistic
from anemone.training_export import (
    TrainingTreeSnapshot,
    build_training_tree_snapshot,
    save_training_tree_snapshot,
)
from anemone.tree_manager import (
    OpeningExpansionConfig,
    OpeningExpansionKind,
    RolloutActionSelectorKind,
    RolloutExpansionConfig,
)
from anemone.value_updates import NodeValueUpdate, NodeValueUpdateResult
from atomheart.games.morpion import MorpionStateCheckpointCodec, initial_state
from dacite import Config, from_dict
from valanga.evaluations import Certainty, Value

from chipiron.environments.morpion.players.evaluators.morpion_state_evaluator import (
    MorpionMasterEvaluator,
    MorpionOverEventDetector,
    MorpionStateEvaluator,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_GRAPH_MODEL_KIND,
    MorpionGraphTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState

if TYPE_CHECKING:
    from torch import Tensor

from .config import DEFAULT_MORPION_TREE_BRANCH_LIMIT, MorpionBootstrapRolloutConfig
from .control import MorpionBootstrapEffectiveRuntimeConfig
from .cycle_timing import timestamp_utc_from_unix_s as _timestamp_utc_from_unix_s
from .history import MorpionBootstrapTreeStatus
from .linoo_selection_table import (
    linoo_selection_table_from_report,
    save_linoo_selection_table,
)
from .pipeline_memory import current_rss_mb, log_pipeline_memory
from .search_runner_protocol import MorpionSearchRunner
from .sharded_training_export import (
    MorpionShardedTrainingExportStats,
    save_morpion_sharded_training_tree_from_live_nodes,
)

if TYPE_CHECKING:
    from .pipeline_artifacts import (
        MorpionReevaluationPatch,
        MorpionReevaluationPatchRow,
    )

LOGGER = logging.getLogger(__name__)


def _live_compact_state_payload_cycle_error(node_id: int) -> RuntimeError:
    """Return the stable live compact payload-cycle error."""
    return RuntimeError(f"Cycle in live compact state payload chain at {node_id}.")


def _invalid_checkpoint_payload_mapping_error() -> TypeError:
    """Return the stable invalid checkpoint-payload mapping error."""
    return TypeError("checkpoint payload must be a string-keyed mapping")


_TREE_BRANCH_LIMIT_ARGS_REQUIRED_MESSAGE = (
    "Morpion bootstrap runtime reconfiguration currently supports only "
    "TreeBranchLimitArgs stopping criteria."
)
_LIVE_TREE_BRANCH_LIMIT_REQUIRED_MESSAGE = (
    "Morpion bootstrap runtime reconfiguration currently supports only "
    "tree-branch-limit stopping criteria on the live runtime."
)
_LIVE_TREE_REQUIRED_MESSAGE = "Anemone runtime must expose a live tree."


@dataclass(frozen=True, slots=True)
class CheckpointIoMetrics:
    """Compact checkpoint I/O metrics for stable structured logging."""

    path: str
    bytes: int | None = None
    file_format: str | None = None
    encoder: str | None = None
    payload_build_s: float | None = None
    jsonable_s: float | None = None
    json_encode_s: float | None = None
    compress_s: float | None = None
    write_s: float | None = None
    json_load_s: float | None = None
    payload_decode_s: float | None = None
    runtime_rebuild_s: float | None = None
    total_s: float | None = None
    uncompressed_bytes: int | None = None
    compression_ratio: float | None = None
    rss_before_mb: float | None = None
    rss_after_mb: float | None = None
    node_count: int | None = None
    anchor_count: int | None = None
    delta_count: int | None = None
    cache: str | None = None
    runtime_checkpoint_format: str | None = None


@dataclass(frozen=True, slots=True)
class _ValidatedCheckpointPayloadCacheEntry:
    """Decoded checkpoint payload retained briefly between validation and restore."""

    path: Path
    bytes: int
    mtime_ns: int
    payload: SearchRuntimeCheckpointPayload


_validated_checkpoint_payload_cache: _ValidatedCheckpointPayloadCacheEntry | None = None


@dataclass(slots=True)
class _ReevaluationBlendMetrics:
    """Aggregate diagnostics for smoothed reevaluation patch updates."""

    count: int = 0
    old_sum: float = 0.0
    new_sum: float = 0.0
    blended_sum: float = 0.0

    def record(
        self,
        *,
        old_value: float,
        new_value: float,
        blended_value: float,
    ) -> None:
        """Record one actually blended node update."""
        self.count += 1
        self.old_sum += old_value
        self.new_sum += new_value
        self.blended_sum += blended_value


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


def checkpoint_io_metrics_to_dict(metrics: CheckpointIoMetrics) -> dict[str, object]:
    """Return JSON-friendly checkpoint I/O metrics."""
    return {
        "path": metrics.path,
        "bytes": metrics.bytes,
        "format": metrics.file_format,
        "encoder": metrics.encoder,
        "payload_build_s": metrics.payload_build_s,
        "jsonable_s": metrics.jsonable_s,
        "json_encode_s": metrics.json_encode_s,
        "compress_s": metrics.compress_s,
        "write_s": metrics.write_s,
        "json_load_s": metrics.json_load_s,
        "payload_decode_s": metrics.payload_decode_s,
        "runtime_rebuild_s": metrics.runtime_rebuild_s,
        "total_s": metrics.total_s,
        "uncompressed_bytes": metrics.uncompressed_bytes,
        "compression_ratio": metrics.compression_ratio,
        "rss_before_mb": metrics.rss_before_mb,
        "rss_after_mb": metrics.rss_after_mb,
        "node_count": metrics.node_count,
        "anchor_count": metrics.anchor_count,
        "delta_count": metrics.delta_count,
        "cache": metrics.cache,
        "runtime_checkpoint_format": metrics.runtime_checkpoint_format,
    }


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


def _format_optional_seconds(value: object) -> str:
    """Format one optional duration for stable timing logs."""
    return f"{float(value):.6f}" if isinstance(value, int | float) else "unknown"


def _value_to_scalar(value: object) -> float | None:
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


def _log_training_export_profile(profile: MorpionTrainingExportProfile) -> None:
    """Emit stable aggregate profile logs for one training export build."""
    LOGGER.info(
        "[training-export-profile] %s", _format_training_export_profile(profile)
    )
    LOGGER.info(
        "[training-export-profile-rates] %s",
        _format_training_export_profile_rates(profile),
    )


def _log_sharded_training_export_stats(
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


def _format_optional_int_log(value: object) -> str:
    """Format one optional integer for stable structured logs."""
    return str(value) if isinstance(value, int) else "unknown"


def _selector_report_row_count(selector_report: object | None) -> int | None:
    """Return the selector report row count when exposed by the report."""
    if selector_report is None:
        return None
    depth_row_count = getattr(selector_report, "depth_row_count", None)
    if isinstance(depth_row_count, int):
        return depth_row_count
    depth_rows = getattr(selector_report, "depth_rows", None)
    if depth_rows is None:
        return None
    try:
        row_count = len(depth_rows)
    except TypeError:
        return None
    return row_count


def _selector_growth_diagnostic_fields(
    selector_report: object | None,
) -> dict[str, object]:
    """Return stable optional selector diagnostics for growth-step logs."""
    return {
        "selector_state_rebuilt": getattr(selector_report, "state_rebuilt", None),
        "selector_nodes_incrementally_updated": getattr(
            selector_report,
            "nodes_incrementally_updated",
            None,
        ),
        "selector_total_nodes_scanned": getattr(
            selector_report,
            "total_nodes_scanned",
            None,
        ),
        "selector_frontier_nodes_scanned": getattr(
            selector_report,
            "frontier_nodes_scanned",
            None,
        ),
    }


def _selector_heap_detail_fields(selector_report: object | None) -> dict[str, object]:
    """Return optional Linoo heap detail counters for growth-step logs."""
    return {
        "candidate_count": getattr(
            selector_report, "heap_update_candidate_count", None
        ),
        "push_count": getattr(selector_report, "heap_update_push_count", None),
        "pop_count": getattr(selector_report, "heap_update_pop_count", None),
        "stale_skip_count": getattr(
            selector_report, "heap_update_stale_skip_count", None
        ),
        "signature_check_count": getattr(
            selector_report, "heap_update_signature_check_count", None
        ),
        "signature_recompute_count": getattr(
            selector_report, "heap_update_signature_recompute_count", None
        ),
        "version_mismatch_count": getattr(
            selector_report, "heap_update_version_mismatch_count", None
        ),
        "total_heap_entries": getattr(
            selector_report, "heap_update_total_heap_entries", None
        ),
        "max_heap_size": getattr(selector_report, "heap_update_max_heap_size", None),
        "depth_count": getattr(selector_report, "heap_update_depth_count", None),
        "frontier_node_count_seen": getattr(
            selector_report, "heap_update_frontier_node_count_seen", None
        ),
    }


def _phase_delta(
    before: Mapping[str, int] | Mapping[str, float],
    after: Mapping[str, int] | Mapping[str, float],
    *,
    prefix: str,
) -> dict[str, int] | dict[str, float]:
    """Return changed phase counters for phases matching ``prefix``."""
    delta: dict[str, int] | dict[str, float] = {}
    phase_names = {
        phase
        for phase in set(before) | set(after)
        if phase == prefix or phase.startswith(f"{prefix}.")
    }
    for phase in sorted(phase_names):
        phase_delta = after.get(phase, 0) - before.get(phase, 0)
        if phase_delta:
            delta[phase] = phase_delta
    return delta


def _checkpoint_selector_state_fields(
    payload: object,
    *,
    prefix: str,
) -> dict[str, object]:
    """Return stable selector-state presence fields for checkpoint logs."""
    selector_state = getattr(payload, "selector_state", None)
    selector_state_type = getattr(selector_state, "type", None)
    selector_state_version = getattr(selector_state, "version", None)
    return {
        f"{prefix}_selector_state_present": selector_state is not None,
        f"{prefix}_selector_state_type": selector_state_type,
        f"{prefix}_selector_state_version": selector_state_version,
    }


def _invalidate_selector_cache_if_supported(runtime: object) -> bool:
    """Invalidate selector caches when the runtime or selector exposes a hook."""
    invalidate_runtime = getattr(
        runtime, "_invalidate_selector_cache_if_supported", None
    )
    if callable(invalidate_runtime):
        return bool(invalidate_runtime())

    selector = getattr(runtime, "node_selector", None)
    invalidate_selector = getattr(selector, "invalidate", None)
    if callable(invalidate_selector):
        invalidate_selector()
        return True

    return False


def _opening_expansion_config_from_rollout(
    rollout: MorpionBootstrapRolloutConfig | None,
) -> OpeningExpansionConfig:
    """Build Anemone opening-expansion config from persisted Morpion rollout."""
    if rollout is None or not rollout.enabled:
        return OpeningExpansionConfig()

    return OpeningExpansionConfig(
        kind=OpeningExpansionKind.ROLLOUT,
        rollout=RolloutExpansionConfig(
            max_extra_steps=rollout.max_extra_steps,
            action_selector_kind=RolloutActionSelectorKind(
                rollout.action_selector_kind
            ),
            random_seed=rollout.random_seed,
            stop_on_existing_node=rollout.stop_on_existing_node,
        ),
    )


def _default_search_args(
    *,
    rollout: MorpionBootstrapRolloutConfig | None = None,
) -> SearchArgs:
    """Build the default Morpion tree-search args used by the bootstrap runner."""
    return SearchArgs(
        node_selector=ComposedNodeSelectorArgs(
            type=NodeSelectorType.COMPOSED,
            priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
            base=LinooArgs(type=NodeSelectorType.LINOO),
        ),
        opening_type=OpeningType.ALL_CHILDREN,
        recommender_rule=AlmostEqualLogistic(
            type="almost_equal_logistic",
            temperature=1.0,
        ),
        stopping_criterion=TreeBranchLimitArgs(
            type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
            tree_branch_limit=DEFAULT_MORPION_TREE_BRANCH_LIMIT,
        ),
        opening_expansion=_opening_expansion_config_from_rollout(rollout),
    )


def _opening_type_name(search_args: SearchArgs) -> str:
    """Return a stable operator-facing opening-type label for logs."""
    opening_type = search_args.opening_type
    value = getattr(opening_type, "value", None)
    return value if isinstance(value, str) else str(opening_type)


def _opening_expansion_kind_name(search_args: SearchArgs) -> str:
    """Return a stable operator-facing opening-expansion kind label."""
    opening_expansion = search_args.opening_expansion
    kind = getattr(opening_expansion, "kind", None)
    value = getattr(kind, "value", None)
    return value if isinstance(value, str) else str(kind)


def _log_search_rollout_config(search_args: SearchArgs) -> None:
    """Emit the effective rollout expansion config for operator logs."""
    opening_expansion = search_args.opening_expansion
    rollout = getattr(opening_expansion, "rollout", None)
    enabled = getattr(opening_expansion, "kind", None) == OpeningExpansionKind.ROLLOUT
    LOGGER.info(
        "[search] rollout enabled=%s max_extra_steps=%s action_selector=%s random_seed=%s stop_on_existing_node=%s",
        enabled,
        _metric_value(getattr(rollout, "max_extra_steps", None)),
        _metric_value(getattr(rollout, "action_selector_kind", None)),
        _metric_value(getattr(rollout, "random_seed", None)),
        _metric_value(getattr(rollout, "stop_on_existing_node", None)),
    )


def _log_latest_rollout_report(runtime: object) -> None:
    """Emit the latest Anemone rollout report when the runtime exposes one."""
    tree_manager = getattr(runtime, "tree_manager", None)
    report = getattr(tree_manager, "latest_rollout_report", None)
    if report is None:
        return
    LOGGER.info(
        "[rollout] total_edges=%s initial_edges=%s extra_edges=%s traversals=%s stops=%s",
        _metric_value(getattr(report, "total_edge_count", None)),
        _metric_value(getattr(report, "initial_edge_count", None)),
        _metric_value(getattr(report, "extra_edge_count", None)),
        _metric_value(getattr(report, "traversal_count", None)),
        _metric_value(getattr(report, "stop_reason_counts", None)),
    )
    path_reports = _rollout_path_reports(report)
    if not path_reports:
        return
    LOGGER.info(
        "[rollout-lengths] count=%s total_lengths=%s extra_lengths=%s stops=%s",
        len(path_reports),
        [
            getattr(path_report, "total_edge_count", None)
            for path_report in path_reports
        ],
        [
            getattr(path_report, "extra_edge_count", None)
            for path_report in path_reports
        ],
        _rollout_path_stop_counts(path_reports),
    )
    for rollout_index, path_report in enumerate(path_reports):
        LOGGER.info(
            "[rollout-detail] rollout_index=%s start_node_id=%s start_depth=%s "
            "end_node_id=%s end_depth=%s total_edges=%s initial_edges=%s "
            "extra_edges=%s traversals=%s stop_reason=%s end_terminal=%s "
            "end_exact=%s end_created_node=%s end_existing_node=%s "
            "end_legal_actions=%s end_openable_actions=%s end_opened_actions=%s "
            "end_non_opened_branches=%s no_legal_but_not_terminal=%s",
            rollout_index,
            _metric_value(getattr(path_report, "start_node_id", None)),
            _metric_value(getattr(path_report, "start_depth", None)),
            _metric_value(getattr(path_report, "end_node_id", None)),
            _metric_value(getattr(path_report, "end_depth", None)),
            _metric_value(getattr(path_report, "total_edge_count", None)),
            _metric_value(getattr(path_report, "initial_edge_count", None)),
            _metric_value(getattr(path_report, "extra_edge_count", None)),
            _metric_value(getattr(path_report, "traversal_count", None)),
            _metric_value(_rollout_path_stop_reason(path_report)),
            _metric_value(getattr(path_report, "end_is_terminal", None)),
            _metric_value(getattr(path_report, "end_is_exact", None)),
            _metric_value(getattr(path_report, "end_was_created_node", None)),
            _metric_value(getattr(path_report, "end_was_existing_node", None)),
            _metric_value(getattr(path_report, "end_legal_action_count", None)),
            _metric_value(getattr(path_report, "end_openable_action_count", None)),
            _metric_value(getattr(path_report, "end_opened_action_count", None)),
            _metric_value(getattr(path_report, "end_non_opened_branch_count", None)),
            _metric_value(_rollout_no_legal_but_not_terminal(path_report)),
        )
        if _rollout_no_legal_but_not_terminal(path_report):
            LOGGER.warning(
                "[rollout-warning] no_legal_actions_but_not_terminal "
                "rollout_index=%s end_node_id=%s end_depth=%s "
                "end_legal_actions=%s end_non_opened_branches=%s",
                rollout_index,
                _metric_value(getattr(path_report, "end_node_id", None)),
                _metric_value(getattr(path_report, "end_depth", None)),
                _metric_value(getattr(path_report, "end_legal_action_count", None)),
                _metric_value(
                    getattr(path_report, "end_non_opened_branch_count", None)
                ),
            )


def _rollout_path_reports(report: object) -> tuple[object, ...]:
    """Return path reports from an Anemone rollout report when available."""
    path_reports = getattr(report, "path_reports", ())
    if path_reports is None:
        return ()
    try:
        return tuple(path_reports)
    except TypeError:
        return ()


def _rollout_path_stop_counts(path_reports: tuple[object, ...]) -> dict[str, int]:
    """Aggregate stop reasons from rollout path reports."""
    stop_counts: dict[str, int] = {}
    for path_report in path_reports:
        stop_reason = _rollout_path_stop_reason(path_report)
        stop_counts[stop_reason] = stop_counts.get(stop_reason, 0) + 1
    return stop_counts


def _rollout_path_stop_reason(path_report: object) -> str:
    """Return one stable rollout path stop-reason token."""
    stop_reason = getattr(path_report, "stop_reason", None)
    value = getattr(stop_reason, "value", None)
    if isinstance(value, str):
        return value
    if stop_reason is None:
        return "none"
    return str(stop_reason)


def _rollout_no_legal_but_not_terminal(path_report: object) -> bool:
    """Return whether a path stopped with no legal actions but is not terminal."""
    reported_flag = getattr(path_report, "no_legal_actions_but_not_terminal", None)
    if isinstance(reported_flag, bool):
        return reported_flag
    return (
        _rollout_path_stop_reason(path_report) == "no_legal_actions"
        and getattr(path_report, "end_is_terminal", None) is False
    )


def _current_rss_mb() -> float | None:
    """Return current process RSS in MB when available."""
    return current_rss_mb()


def log_morpion_checkpoint_memory_phase(
    phase: str,
    *,
    path: str | Path | None = None,
    nodes: int | None = None,
    generation: int | None = None,
) -> None:
    """Log one lightweight current-RSS checkpoint memory marker."""
    parts = [
        f"phase={phase}",
        f"rss_mb={_metric_value(_current_rss_mb())}",
    ]
    if nodes is not None:
        parts.append(f"nodes={nodes}")
    if generation is not None:
        parts.append(f"generation={generation}")
    if path is not None:
        parts.append(f"path={path}")
    LOGGER.info("[memory] %s", " ".join(parts))


@dataclass(slots=True)
class _RestoreMemoryLogger:
    """Opt-in checkpoint restore RSS and object-size phase logger."""

    checkpoint_path: Path
    compressed_checkpoint_bytes: int | None
    recursive_enabled: bool = False
    recursive_max_objects: int | None = None
    recursive_max_depth: int | None = None
    started_at: float = field(default_factory=time.perf_counter)

    def callback(self, phase: str, metadata: Mapping[str, object]) -> None:
        """Receive one Anemone restore phase callback."""
        self.log(phase, **metadata)

    def log(
        self,
        phase: str,
        *,
        raw_payload: object | None = None,
        typed_payload: SearchRuntimeCheckpointPayload | None = None,
        raw_checkpoint_referenced: bool | None = None,
        typed_checkpoint_referenced: bool | None = None,
        **metadata: object,
    ) -> None:
        """Emit one structured restore-memory phase line."""
        node_count = _restore_metadata_value(metadata, "node_count", "nodes")
        if node_count is None:
            node_count = _raw_checkpoint_node_count(raw_payload)
        if node_count is None and typed_payload is not None:
            node_count = len(typed_payload.tree.nodes)

        branch_count = _restore_metadata_value(metadata, "branch_count", "branches")
        if branch_count is None and typed_payload is not None:
            branch_count = getattr(typed_payload.tree, "branch_count", None)

        parts = [
            f"phase={phase}",
            f"rss_mb={_metric_value(_current_rss_mb())}",
            f"elapsed_s={_metric_value(time.perf_counter() - self.started_at)}",
            f"path={self.checkpoint_path}",
            f"compressed_checkpoint_bytes={_metric_value(self.compressed_checkpoint_bytes)}",
            f"node_count={_metric_value(node_count)}",
            f"branch_count={_metric_value(branch_count)}",
            f"raw_checkpoint_referenced={_metric_value(raw_checkpoint_referenced)}",
            f"typed_checkpoint_referenced={_metric_value(typed_checkpoint_referenced)}",
        ]
        gc_counts = gc.get_count()
        parts.extend(
            [
                f"gc_count0={gc_counts[0]}",
                f"gc_count1={gc_counts[1]}",
                f"gc_count2={gc_counts[2]}",
            ]
        )
        raw_recursive_mb = self._recursive_size_mb(raw_payload)
        if raw_recursive_mb is not None:
            parts.append(f"raw_decoded_recursive_mb={_metric_value(raw_recursive_mb)}")
        typed_recursive_mb = self._recursive_size_mb(typed_payload)
        if typed_recursive_mb is not None:
            parts.append(
                f"typed_checkpoint_payload_recursive_mb={_metric_value(typed_recursive_mb)}"
            )
        for key, value in metadata.items():
            if key in {
                "node_count",
                "nodes",
                "branch_count",
                "branches",
            }:
                continue
            parts.append(f"{key}={_metric_value(value)}")
        LOGGER.info("[restore-memory] %s", " ".join(parts))

    def _recursive_size_mb(self, value: object | None) -> float | None:
        if not self.recursive_enabled or value is None:
            return None
        from .recursive_memory_profile import DeepSizeStats, deep_size

        stats = DeepSizeStats(max_objects=self.recursive_max_objects)
        byte_count = deep_size(
            value,
            seen=set(),
            max_depth=self.recursive_max_depth,
            stats=stats,
        )
        return byte_count / (1024 * 1024)


def _restore_metadata_value(
    metadata: Mapping[str, object],
    *keys: str,
) -> object | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return None


def _raw_checkpoint_node_count(raw_payload: object | None) -> int | None:
    if not isinstance(raw_payload, Mapping):
        return None
    raw_tree = raw_payload.get("tree")
    if not isinstance(raw_tree, Mapping):
        return None
    raw_nodes = raw_tree.get("nodes")
    if not isinstance(raw_nodes, list):
        return None
    return len(raw_nodes)


def _restore_memory_logger_for_path(
    args: AnemoneMorpionSearchRunnerArgs,
    checkpoint_path: Path,
) -> _RestoreMemoryLogger | None:
    return restore_memory_logger_for_checkpoint_path(
        checkpoint_path,
        enabled=args.restore_memory_profile,
        recursive_enabled=args.restore_memory_profile_recursive,
        recursive_max_objects=args.restore_memory_profile_recursive_max_objects,
        recursive_max_depth=args.restore_memory_profile_recursive_max_depth,
    )


def restore_memory_logger_for_checkpoint_path(
    checkpoint_path: Path,
    *,
    enabled: bool,
    recursive_enabled: bool = False,
    recursive_max_objects: int | None = None,
    recursive_max_depth: int | None = None,
) -> _RestoreMemoryLogger | None:
    """Build the opt-in checkpoint restore-memory logger for a path."""
    if not enabled:
        return None
    checkpoint_bytes = _checkpoint_artifact_bytes(checkpoint_path)
    return _RestoreMemoryLogger(
        checkpoint_path=checkpoint_path,
        compressed_checkpoint_bytes=checkpoint_bytes,
        recursive_enabled=recursive_enabled,
        recursive_max_objects=recursive_max_objects,
        recursive_max_depth=recursive_max_depth,
    )


def _is_sharded_runtime_checkpoint_path(path: str | Path) -> bool:
    """Return whether a path points to a sharded runtime checkpoint directory."""
    resolved_path = Path(path)
    return resolved_path.is_dir() and (resolved_path / "manifest.json").is_file()


def _checkpoint_artifact_bytes(path: str | Path) -> int | None:
    """Return compressed bytes for a file or best-effort shard bytes for a directory."""
    resolved_path = Path(path)
    try:
        if resolved_path.is_file():
            return resolved_path.stat().st_size
        if _is_sharded_runtime_checkpoint_path(resolved_path):
            manifest = read_sharded_checkpoint_manifest(resolved_path / "manifest.json")
            shard_bytes = sum(
                shard.compressed_bytes or 0
                for shard in manifest.shards
                if shard.compressed_bytes is not None
            )
            return (resolved_path / "manifest.json").stat().st_size + shard_bytes
    except OSError:
        return None
    return None


def _checkpoint_node_counts(
    payload: SearchRuntimeCheckpointPayload,
) -> tuple[int, int, int]:
    """Return total, anchor, and delta node counts for one checkpoint payload."""
    nodes = payload.tree.nodes
    anchor_count = sum(
        1
        for node_payload in nodes
        if isinstance(node_payload.state_payload, AnchorCheckpointStatePayload)
    )
    delta_count = sum(
        1
        for node_payload in nodes
        if isinstance(node_payload.state_payload, DeltaCheckpointStatePayload)
    )
    return len(nodes), anchor_count, delta_count


def _metric_value(value: object) -> str:
    """Render one metric field as a stable log token."""
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _log_checkpoint_metrics(operation: str, metrics: CheckpointIoMetrics) -> None:
    """Emit one stable checkpoint metrics log line."""
    parts = [
        f"operation={operation}",
        f"path={metrics.path}",
        f"bytes={_metric_value(metrics.bytes)}",
        f"nodes={_metric_value(metrics.node_count)}",
        f"anchors={_metric_value(metrics.anchor_count)}",
        f"deltas={_metric_value(metrics.delta_count)}",
    ]
    if metrics.file_format is not None:
        parts.append(f"format={metrics.file_format}")
    if metrics.encoder is not None:
        parts.append(f"encoder={metrics.encoder}")
    if metrics.cache is not None:
        parts.append(f"cache={metrics.cache}")
    if metrics.runtime_checkpoint_format is not None:
        parts.append(f"runtime_checkpoint_format={metrics.runtime_checkpoint_format}")
    parts.extend(
        [
            f"payload_build_s={_metric_value(metrics.payload_build_s)}",
            f"jsonable_s={_metric_value(metrics.jsonable_s)}",
            f"json_encode_s={_metric_value(metrics.json_encode_s)}",
            f"compress_s={_metric_value(metrics.compress_s)}",
            f"write_s={_metric_value(metrics.write_s)}",
            f"json_load_s={_metric_value(metrics.json_load_s)}",
            f"payload_decode_s={_metric_value(metrics.payload_decode_s)}",
            f"runtime_rebuild_s={_metric_value(metrics.runtime_rebuild_s)}",
            f"total_s={_metric_value(metrics.total_s)}",
            f"uncompressed_bytes={_metric_value(metrics.uncompressed_bytes)}",
            f"compression_ratio={_metric_value(metrics.compression_ratio)}",
            f"rss_before_mb={_metric_value(metrics.rss_before_mb)}",
            f"rss_after_mb={_metric_value(metrics.rss_after_mb)}",
        ]
    )
    LOGGER.info("[checkpoint-metrics] %s", " ".join(parts))


def _checkpoint_payload_cache_identity(path: str | Path) -> tuple[Path, int, int]:
    """Return the identity fields that make a cached payload safe to reuse."""
    resolved_path = Path(path).resolve()
    path_stat = resolved_path.stat()
    return resolved_path, path_stat.st_size, path_stat.st_mtime_ns


def cache_morpion_search_checkpoint_payload_for_restore(
    path: str | Path,
    payload: SearchRuntimeCheckpointPayload,
) -> None:
    """Retain one validated payload for an immediately following restore."""
    global _validated_checkpoint_payload_cache
    try:
        resolved_path, bytes_loaded, mtime_ns = _checkpoint_payload_cache_identity(path)
    except FileNotFoundError:
        _validated_checkpoint_payload_cache = None
        return
    _validated_checkpoint_payload_cache = _ValidatedCheckpointPayloadCacheEntry(
        path=resolved_path,
        bytes=bytes_loaded,
        mtime_ns=mtime_ns,
        payload=payload,
    )


def _pop_cached_morpion_search_checkpoint_payload_for_restore(
    path: str | Path,
) -> tuple[SearchRuntimeCheckpointPayload, int] | None:
    """Return and clear the matching validated payload cache entry, if any."""
    global _validated_checkpoint_payload_cache
    entry = _validated_checkpoint_payload_cache
    if entry is None:
        return None
    try:
        resolved_path, bytes_loaded, mtime_ns = _checkpoint_payload_cache_identity(path)
    except FileNotFoundError:
        _validated_checkpoint_payload_cache = None
        return None
    if (
        entry.path != resolved_path
        or entry.bytes != bytes_loaded
        or entry.mtime_ns != mtime_ns
    ):
        _validated_checkpoint_payload_cache = None
        return None
    _validated_checkpoint_payload_cache = None
    return entry.payload, entry.bytes


class UninitializedMorpionSearchRunnerError(RuntimeError):
    """Raised when a runner method requires a live runtime that does not exist."""

    def __init__(self) -> None:
        """Initialize the missing-runtime error."""
        super().__init__(
            "AnemoneMorpionSearchRunner has no live runtime. Call load_or_create() first."
        )


def _uninitialized_reevaluation_patch_runtime_error() -> RuntimeError:
    """Build the stable missing-runtime error for live patch application."""
    return RuntimeError(
        "Cannot apply Morpion reevaluation patch before the Anemone search runtime is initialized."
    )


class InvalidMorpionSearchCheckpointError(ValueError):
    """Raised when a persisted Anemone checkpoint payload is invalid."""

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize the checkpoint validation error."""
        super().__init__(f"Invalid Morpion search checkpoint at {path!s}: {reason}")


@dataclass(slots=True)
class _CheckpointCodecProfile:
    """Aggregate fallback profiling for Morpion checkpoint codec calls."""

    anchor_calls: int = 0
    anchor_total_s: float = 0.0
    delta_calls: int = 0
    delta_total_s: float = 0.0
    summary_calls: int = 0
    summary_total_s: float = 0.0


@dataclass(slots=True)
class _ChipironMorpionStateCheckpointCodec:
    """Thin adapter from Atomheart checkpoint codecs to Chipiron Morpion states."""

    inner: MorpionStateCheckpointCodec
    dynamics: MorpionDynamics
    profile_checkpoint: bool = False
    _profile: _CheckpointCodecProfile = field(
        default_factory=_CheckpointCodecProfile,
        init=False,
    )

    def dump_state_ref(self, state: MorpionState) -> object:
        """Serialize a state-ref payload for training-export compatibility only."""
        return self.inner.dump_state_ref(state.to_atomheart_state())

    def load_state_ref(self, payload: object) -> MorpionState:
        """Restore one legacy state-ref payload into Chipiron state form."""
        return self.dynamics.wrap_atomheart_state(self.inner.load_state_ref(payload))

    def dump_anchor_ref(self, state: MorpionState) -> object:
        """Serialize one full anchor snapshot for the incremental checkpoint path."""
        started_at = perf_counter()
        result = self.inner.dump_anchor_ref(state.to_atomheart_state())
        self._record_profile(
            call_count_attr="anchor_calls",
            total_s_attr="anchor_total_s",
            started_at=started_at,
        )
        return result

    def dump_delta_from_parent(
        self,
        *,
        parent_state: MorpionState,
        child_state: MorpionState,
        branch_from_parent: object | None = None,
    ) -> object:
        """Serialize one child state as a parent-relative delta."""
        started_at = perf_counter()
        result = self.inner.dump_delta_from_parent(
            parent_state=parent_state.to_atomheart_state(),
            child_state=child_state.to_atomheart_state(),
            # Chipiron branch keys may differ in orientation from Atomheart move
            # encoding; delta payload carries the canonical move already.
            branch_from_parent=None,
        )
        self._record_profile(
            call_count_attr="delta_calls",
            total_s_attr="delta_total_s",
            started_at=started_at,
        )
        return result

    def load_anchor_ref(self, anchor_ref: object) -> MorpionState:
        """Restore one anchor snapshot through Atomheart, then wrap it for Chipiron."""
        return self.dynamics.wrap_atomheart_state(
            self.inner.load_anchor_ref(anchor_ref)
        )

    def load_child_from_delta(
        self,
        *,
        parent_state: MorpionState,
        delta_ref: object,
        branch_from_parent: object | None = None,
    ) -> MorpionState:
        """Restore one child state from its parent's concrete Chipiron state."""
        return self.dynamics.wrap_atomheart_state(
            self.inner.load_child_from_delta(
                parent_state=parent_state.to_atomheart_state(),
                delta_ref=delta_ref,
                # Chipiron branch keys may differ in orientation from Atomheart move
                # encoding; delta payload carries the canonical move already.
                branch_from_parent=None,
            )
        )

    def dump_state_summary(self, state: MorpionState) -> object:
        """Serialize optional lightweight checkpoint summary metadata."""
        started_at = perf_counter()
        result = self.inner.dump_state_summary(state.to_atomheart_state())
        self._record_profile(
            call_count_attr="summary_calls",
            total_s_attr="summary_total_s",
            started_at=started_at,
        )
        return result

    def dump_state_parent_branch_for_checkpoint(
        self,
        branch_from_parent: object | None,
    ) -> object | None:
        """Bridge optional compact state-parent branch payload serialization.

        This wrapper must preserve the hook contract and forward the actual
        branch argument. Morpion's Atomheart codec may still choose to ignore it
        and return ``None`` because its compact delta payload already stores the
        canonical move needed for reconstruction.
        """
        inner_hook = getattr(
            self.inner, "dump_state_parent_branch_for_checkpoint", None
        )
        if callable(inner_hook):
            return inner_hook(branch_from_parent)
        return None

    def checkpoint_profile_snapshot(self) -> dict[str, object]:
        """Return aggregate checkpoint profiling from the inner codec or fallback."""
        inner_snapshot = getattr(self.inner, "checkpoint_profile_snapshot", None)
        if callable(inner_snapshot):
            snapshot = inner_snapshot()
            snapshot_mapping = _string_key_mapping_or_none(snapshot)
            if snapshot_mapping is not None:
                return dict(snapshot_mapping)
        return {
            "chipiron_morpion_anchor_avg_ms": _checkpoint_profile_average_ms(
                self._profile.anchor_total_s,
                self._profile.anchor_calls,
            ),
            "chipiron_morpion_anchor_calls": self._profile.anchor_calls,
            "chipiron_morpion_anchor_total_s": self._profile.anchor_total_s,
            "chipiron_morpion_delta_avg_ms": _checkpoint_profile_average_ms(
                self._profile.delta_total_s,
                self._profile.delta_calls,
            ),
            "chipiron_morpion_delta_calls": self._profile.delta_calls,
            "chipiron_morpion_delta_total_s": self._profile.delta_total_s,
            "chipiron_morpion_summary_avg_ms": _checkpoint_profile_average_ms(
                self._profile.summary_total_s,
                self._profile.summary_calls,
            ),
            "chipiron_morpion_summary_calls": self._profile.summary_calls,
            "chipiron_morpion_summary_total_s": self._profile.summary_total_s,
        }

    def reset_checkpoint_profile(self) -> None:
        """Clear aggregate checkpoint profiling counters between builds."""
        inner_reset = getattr(self.inner, "reset_checkpoint_profile", None)
        if callable(inner_reset):
            inner_reset()
            return
        self._profile = _CheckpointCodecProfile()

    def _record_profile(
        self,
        *,
        call_count_attr: str,
        total_s_attr: str,
        started_at: float,
    ) -> None:
        """Record one fallback codec timing when Atomheart profiling is absent."""
        if not self.profile_checkpoint:
            return
        if callable(getattr(self.inner, "checkpoint_profile_snapshot", None)):
            return
        elapsed_s = perf_counter() - started_at
        setattr(
            self._profile,
            call_count_attr,
            getattr(self._profile, call_count_attr) + 1,
        )
        setattr(
            self._profile,
            total_s_attr,
            getattr(self._profile, total_s_attr) + elapsed_s,
        )


def _checkpoint_profile_average_ms(total_s: float, count: int) -> float:
    """Return a stable milliseconds average for checkpoint profile logs."""
    if count <= 0:
        return 0.0
    return 1000.0 * total_s / count


def _new_morpion_state_checkpoint_codec(
    *, profile_checkpoint: bool
) -> MorpionStateCheckpointCodec:
    """Create the Morpion checkpoint codec with optional profiling when supported."""
    try:
        return MorpionStateCheckpointCodec(profile_checkpoint=profile_checkpoint)
    except TypeError:
        return MorpionStateCheckpointCodec()


class MorpionStateToTensorConverter(Protocol):
    """Minimal interface shared by Morpion neural input converters."""

    def state_to_tensor(self, state: MorpionState) -> Tensor:
        """Convert one Morpion state to the model input tensor."""
        ...


@dataclass(frozen=True, slots=True)
class AnemoneMorpionSearchRunnerArgs:
    """Configuration for the real Anemone-backed Morpion runner."""

    search_args: SearchArgs = field(default_factory=_default_search_args)
    random_seed: int = 0
    reevaluation_scope: str = "leaves"
    restore_memory_profile: bool = False
    restore_memory_profile_recursive: bool = False
    restore_memory_profile_recursive_max_objects: int | None = None
    restore_memory_profile_recursive_max_depth: int | None = None
    runtime_checkpoint_format: str = "json-zst"
    growth_state_eviction_policy: str = "none"
    growth_state_eviction_recent_window: int = 1000
    growth_state_rematerialization_cache_size: int = 10000
    growth_state_eviction_scan_interval_steps: int = 100
    growth_state_eviction_scan_node_limit: int = 5000
    growth_state_eviction_payload_mode: str = "anchor"
    growth_state_eviction_delta_chain_max_depth: int = 32


@dataclass(frozen=True, slots=True)
class _ParentDeltaContext:
    """Single unambiguous parent edge eligible for live delta payloads."""

    parent_node: object
    parent_node_id: int
    branch_from_parent: object


@dataclass(frozen=True, slots=True)
class _LiveEvictionPayload:
    """Built payload plus live-chain metadata for one evicted node."""

    payload: CheckpointNodeStatePayload
    kind: str
    chain_depth: int


@dataclass(slots=True)
class MorpionGrowthStateEvictionMetrics:
    """Counters for experimental growth-time state eviction."""

    state_eviction_policy: str = "none"
    state_eviction_payload_mode: str = "anchor"
    state_eviction_delta_chain_max_depth: int = 32
    eviction_attempt_count: int = 0
    eviction_success_count: int = 0
    evicted_materialized_state_count: int = 0
    compact_payload_count: int = 0
    anchor_payload_count: int = 0
    delta_payload_count: int = 0
    rematerialization_count: int = 0
    rematerialization_cache_hit: int = 0
    rematerialization_cache_miss: int = 0
    rematerialization_count_by_phase: dict[str, int] = field(default_factory=dict)
    rematerialization_cache_hit_by_phase: dict[str, int] = field(default_factory=dict)
    rematerialization_cache_miss_by_phase: dict[str, int] = field(default_factory=dict)
    rematerialization_total_s_by_phase: dict[str, float] = field(default_factory=dict)
    rematerialization_count_by_node_id: dict[int, int] = field(default_factory=dict)
    current_cache_size: int = 0
    cache_evictions: int = 0
    total_reconstruction_depth: int = 0
    eviction_scan_count: int = 0
    eviction_nodes_scanned_count: int = 0
    eviction_total_s: float = 0.0
    eviction_payload_build_s: float = 0.0
    rematerialization_total_s: float = 0.0
    eviction_skipped_count_by_reason: dict[str, int] = field(default_factory=dict)
    delta_payload_attempt_count: int = 0
    delta_payload_success_count: int = 0
    delta_payload_fallback_count: int = 0
    delta_payload_fallback_count_by_reason: dict[str, int] = field(
        default_factory=dict
    )

    def skip(self, reason: str) -> None:
        """Record one skipped eviction attempt."""
        self.eviction_skipped_count_by_reason[reason] = (
            self.eviction_skipped_count_by_reason.get(reason, 0) + 1
        )

    def record_delta_payload_attempt(self) -> None:
        """Record one attempt to build a live delta eviction payload."""
        self.delta_payload_attempt_count += 1

    def record_delta_payload_success(self) -> None:
        """Record one successful live delta eviction payload build."""
        self.delta_payload_success_count += 1

    def record_delta_payload_fallback(self, reason: str) -> None:
        """Record one live delta ineligibility reason before anchor fallback."""
        self.delta_payload_fallback_count += 1
        self.delta_payload_fallback_count_by_reason[reason] = (
            self.delta_payload_fallback_count_by_reason.get(reason, 0) + 1
        )

    def record_rematerialization(
        self,
        *,
        node_id: int,
        phase: str,
        cache_hit: bool,
        elapsed_s: float,
    ) -> None:
        """Record one compact-state resolution under its diagnostic phase."""
        self.rematerialization_count_by_phase[phase] = (
            self.rematerialization_count_by_phase.get(phase, 0) + 1
        )
        self.rematerialization_total_s_by_phase[phase] = (
            self.rematerialization_total_s_by_phase.get(phase, 0.0) + elapsed_s
        )
        self.rematerialization_count_by_node_id[node_id] = (
            self.rematerialization_count_by_node_id.get(node_id, 0) + 1
        )
        if cache_hit:
            self.rematerialization_cache_hit_by_phase[phase] = (
                self.rematerialization_cache_hit_by_phase.get(phase, 0) + 1
            )
        else:
            self.rematerialization_cache_miss_by_phase[phase] = (
                self.rematerialization_cache_miss_by_phase.get(phase, 0) + 1
            )

    def snapshot(self) -> dict[str, object]:
        """Return a grep-friendly diagnostics payload."""
        average_reconstruction_depth = (
            0.0
            if self.rematerialization_count <= 0
            else self.total_reconstruction_depth / self.rematerialization_count
        )
        skipped_count = sum(self.eviction_skipped_count_by_reason.values())
        select_rematerialization_count_by_subphase = _phase_delta(
            {},
            self.rematerialization_count_by_phase,
            prefix="select",
        )
        select_rematerialization_cache_miss_by_subphase = _phase_delta(
            {},
            self.rematerialization_cache_miss_by_phase,
            prefix="select",
        )
        select_rematerialization_total_s_by_subphase = _phase_delta(
            {},
            self.rematerialization_total_s_by_phase,
            prefix="select",
        )
        return {
            "state_eviction_policy": self.state_eviction_policy,
            "state_eviction_payload_mode": self.state_eviction_payload_mode,
            "state_eviction_delta_chain_max_depth": (
                self.state_eviction_delta_chain_max_depth
            ),
            "eviction_attempt_count": self.eviction_attempt_count,
            "eviction_success_count": self.eviction_success_count,
            "eviction_skipped_count": skipped_count,
            "eviction_skipped_count_by_reason": dict(
                sorted(self.eviction_skipped_count_by_reason.items())
            ),
            "delta_payload_attempt_count": self.delta_payload_attempt_count,
            "delta_payload_success_count": self.delta_payload_success_count,
            "delta_payload_fallback_count": self.delta_payload_fallback_count,
            "delta_payload_fallback_count_by_reason": dict(
                sorted(self.delta_payload_fallback_count_by_reason.items())
            ),
            "evicted_materialized_state_count": self.evicted_materialized_state_count,
            "compact_payload_count": self.compact_payload_count,
            "anchor_payload_count": self.anchor_payload_count,
            "delta_payload_count": self.delta_payload_count,
            "rematerialization_count": self.rematerialization_count,
            "rematerialization_cache_hit": self.rematerialization_cache_hit,
            "rematerialization_cache_miss": self.rematerialization_cache_miss,
            "rematerialization_count_by_phase": dict(
                sorted(self.rematerialization_count_by_phase.items())
            ),
            "rematerialization_cache_hit_by_phase": dict(
                sorted(self.rematerialization_cache_hit_by_phase.items())
            ),
            "rematerialization_cache_miss_by_phase": dict(
                sorted(self.rematerialization_cache_miss_by_phase.items())
            ),
            "rematerialization_total_s_by_phase": dict(
                sorted(self.rematerialization_total_s_by_phase.items())
            ),
            "select_rematerialization_count_by_subphase": (
                select_rematerialization_count_by_subphase
            ),
            "select_rematerialization_cache_miss_by_subphase": (
                select_rematerialization_cache_miss_by_subphase
            ),
            "select_rematerialization_total_s_by_subphase": (
                select_rematerialization_total_s_by_subphase
            ),
            "top_rematerialized_node_ids": tuple(
                node_id
                for node_id, _count in sorted(
                    self.rematerialization_count_by_node_id.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:10]
            ),
            "current_cache_size": self.current_cache_size,
            "cache_evictions": self.cache_evictions,
            "average_reconstruction_depth": average_reconstruction_depth,
            "eviction_scan_count": self.eviction_scan_count,
            "eviction_nodes_scanned_count": self.eviction_nodes_scanned_count,
            "eviction_total_s": self.eviction_total_s,
            "eviction_payload_build_s": self.eviction_payload_build_s,
            "rematerialization_total_s": self.rematerialization_total_s,
        }


@dataclass(slots=True)
class _LiveCompactStateResolver:
    """In-RAM compact state resolver for experimental growth eviction.

    The decoded-state cache is deliberately bounded: cold evictions should avoid
    retaining every state, but repeated propagation/export touches must not rebuild
    large anchor payloads over and over.
    """

    state_codec: object
    state_payloads_by_node_id: dict[int, CheckpointNodeStatePayload] = field(
        default_factory=dict
    )
    payload_chain_depth_by_node_id: dict[int, int] = field(default_factory=dict)
    metrics: MorpionGrowthStateEvictionMetrics = field(
        default_factory=MorpionGrowthStateEvictionMetrics
    )
    cache_size: int = 10000
    current_phase: str = "unknown"
    _decoded_state_cache: OrderedDict[int, MorpionState] = field(
        default_factory=OrderedDict
    )
    _resolving_node_ids: set[int] = field(default_factory=set)

    @contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        """Temporarily attribute rematerializations to ``phase``."""
        previous_phase = self.current_phase
        self.current_phase = phase
        try:
            yield
        finally:
            self.current_phase = previous_phase

    def resolve(self, node_id: int) -> MorpionState:
        """Resolve one compact payload into a concrete Morpion state."""
        started_at = perf_counter()
        phase = self.current_phase
        self.metrics.rematerialization_count += 1
        cached_state = self._decoded_state_cache.get(node_id)
        if cached_state is not None:
            self.metrics.rematerialization_cache_hit += 1
            self._decoded_state_cache.move_to_end(node_id)
            self.metrics.current_cache_size = len(self._decoded_state_cache)
            elapsed_s = perf_counter() - started_at
            self.metrics.rematerialization_total_s += elapsed_s
            self.metrics.record_rematerialization(
                node_id=node_id,
                phase=phase,
                cache_hit=True,
                elapsed_s=elapsed_s,
            )
            return cached_state
        self.metrics.rematerialization_cache_miss += 1
        if node_id in self._resolving_node_ids:
            raise _live_compact_state_payload_cycle_error(node_id)
        self._resolving_node_ids.add(node_id)
        try:
            state = self._resolve_uncached(node_id, depth=1)
        finally:
            self._resolving_node_ids.discard(node_id)
            elapsed_s = perf_counter() - started_at
            self.metrics.rematerialization_total_s += elapsed_s
            self.metrics.record_rematerialization(
                node_id=node_id,
                phase=phase,
                cache_hit=False,
                elapsed_s=elapsed_s,
            )
        self._cache_state(node_id, state)
        return state

    def _resolve_uncached(self, node_id: int, *, depth: int) -> MorpionState:
        """Resolve one payload without checking the top-level decoded cache."""
        self.metrics.total_reconstruction_depth += depth
        payload = self.state_payloads_by_node_id[node_id]
        if isinstance(payload, AnchorCheckpointStatePayload):
            return cast("Any", self.state_codec).load_anchor_ref(payload.anchor_ref)
        if isinstance(payload, DeltaCheckpointStatePayload):
            parent_state = self.resolve(payload.state_parent_node_id)
            return cast("Any", self.state_codec).load_child_from_delta(
                parent_state=parent_state,
                delta_ref=payload.delta_ref,
                branch_from_parent=None,
            )
        raise KeyError(node_id)

    def store_payload(
        self,
        *,
        node_id: int,
        payload: CheckpointNodeStatePayload,
        chain_depth: int,
    ) -> None:
        """Store one live compact payload and its bounded delta-chain depth."""
        self.state_payloads_by_node_id[node_id] = payload
        self.payload_chain_depth_by_node_id[node_id] = chain_depth

    def payload_for_node_id_or_none(
        self,
        node_id: int,
    ) -> CheckpointNodeStatePayload | None:
        """Return one compact payload without resolving concrete state."""
        return self.state_payloads_by_node_id.get(node_id)

    def _cache_state(self, node_id: int, state: MorpionState) -> None:
        """Store one decoded state in the bounded LRU cache."""
        if self.cache_size <= 0:
            self.metrics.current_cache_size = 0
            return
        self._decoded_state_cache[node_id] = state
        self._decoded_state_cache.move_to_end(node_id)
        while len(self._decoded_state_cache) > self.cache_size:
            self._decoded_state_cache.popitem(last=False)
            self.metrics.cache_evictions += 1
        self.metrics.current_cache_size = len(self._decoded_state_cache)

    def summary(self, node_id: int) -> object | None:
        """Return optional state summary metadata for ``node_id``."""
        return self.state_payloads_by_node_id[node_id].state_summary


@dataclass(frozen=True, slots=True)
class MorpionRegressorMasterEvaluator(MorpionMasterEvaluator):
    """Anemone-compatible Morpion evaluator backed by a saved regressor bundle."""

    input_converter: MorpionStateToTensorConverter
    regressor: object

    @property
    def feature_converter(self) -> MorpionStateToTensorConverter:
        """Return the input converter under the legacy attribute name."""
        return self.input_converter

    def evaluate(self, state: object) -> Value:
        """Evaluate a Morpion state through the loaded regressor bundle."""
        over_event, terminal_value = self.over_detector.check_obvious_over_events(
            cast("Any", state)
        )
        if terminal_value is not None:
            return Value(
                score=terminal_value,
                certainty=Certainty.TERMINAL,
                over_event=over_event,
            )

        morpion_state = cast("MorpionState", state)
        tensor = self.input_converter.state_to_tensor(morpion_state)
        regressor = cast("Any", self.regressor)
        raw_output = regressor(tensor)
        score = float(raw_output.detach().cpu().reshape(-1)[0].item())
        return Value(
            score=score,
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )


def load_morpion_evaluator_from_model_bundle(
    model_bundle_path: str | Path,
) -> MorpionMasterEvaluator:
    """Load one saved Morpion bundle into the Anemone evaluator protocol."""
    model, model_args, _ = load_morpion_model_bundle(model_bundle_path)
    model.eval()
    over_detector = MorpionOverEventDetector()
    if model_args.model_kind == MORPION_GRAPH_MODEL_KIND:
        input_converter = MorpionGraphTokenConverter(
            dynamics=MorpionDynamics(),
            max_tokens=model_args.graph_max_tokens,
        )
    else:
        input_converter = MorpionFeatureTensorConverter(
            dynamics=MorpionDynamics(),
            feature_subset=model_args.feature_subset,
        )
    return MorpionRegressorMasterEvaluator(
        evaluator=MorpionStateEvaluator(),
        over=over_detector,
        over_detector=over_detector,
        input_converter=input_converter,
        regressor=model,
    )


class AnemoneMorpionSearchRunner(MorpionSearchRunner):
    """Concrete Morpion bootstrap runner backed by one live Anemone runtime."""

    def __init__(
        self,
        args: AnemoneMorpionSearchRunnerArgs | None = None,
    ) -> None:
        """Initialize the real runner with explicit or default runtime settings."""
        self._args = args if args is not None else AnemoneMorpionSearchRunnerArgs()
        self._runtime: object | None = None
        self._random_generator = Random(self._args.random_seed)
        self._dynamics = MorpionDynamics()
        self._state_codec = _ChipironMorpionStateCheckpointCodec(
            inner=_new_morpion_state_checkpoint_codec(profile_checkpoint=True),
            dynamics=self._dynamics,
            profile_checkpoint=True,
        )
        self._current_evaluator_bundle_path: Path | None = None
        self._last_applied_runtime_config = _runtime_config_from_search_args(
            self._args.search_args
        )
        self._state_eviction_metrics = MorpionGrowthStateEvictionMetrics(
            state_eviction_policy=_effective_growth_state_eviction_policy(
                self._args.growth_state_eviction_policy
            ),
            state_eviction_payload_mode=self._args.growth_state_eviction_payload_mode,
            state_eviction_delta_chain_max_depth=(
                self._args.growth_state_eviction_delta_chain_max_depth
            ),
        )
        self._live_compact_state_resolver = _LiveCompactStateResolver(
            state_codec=self._state_codec,
            metrics=self._state_eviction_metrics,
            cache_size=self._args.growth_state_rematerialization_cache_size,
        )
        self._growth_eviction_recent_node_ids: deque[int] = deque()
        self._growth_eviction_recent_node_id_set: set[int] = set()
        self._growth_eviction_selected_step_by_node_id: dict[int, int] = {}
        self._growth_eviction_scan_cursor: int = 0
        self._linoo_selection_table_artifact_path: Path | None = None
        self._linoo_selection_table_cycle_index: int | None = None
        self._linoo_selection_table_generation: int | None = None
        self._last_reevaluation_patch_apply_metrics: dict[str, object] | None = None
        self._latest_checkpoint_metrics: dict[str, object] | None = None
        self._latest_training_export_stats: dict[str, object] | None = None
        self._latest_training_export_profile: dict[str, object] | None = None

    def configure_linoo_selection_table_artifact(
        self,
        *,
        path: str | Path | None,
        cycle_index: int | None = None,
        generation: int | None = None,
    ) -> None:
        """Configure optional latest Linoo table persistence for growth steps."""
        self._linoo_selection_table_artifact_path = None if path is None else Path(path)
        self._linoo_selection_table_cycle_index = cycle_index
        self._linoo_selection_table_generation = generation

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
        *,
        reevaluate_tree: bool = False,
    ) -> None:
        """Load a persisted runtime or create a fresh one for Morpion bootstrap.

        Runtime reconfiguration currently applies by patching the live stopping
        criterion after create/restore. Rebinding checkpoint loads with different
        SearchArgs caused structural duplication on the restored tree, so the
        persisted-tree path keeps the base restore args stable and updates only
        the supported live runtime knobs afterward.
        """
        resolved_runtime_config = (
            self._last_applied_runtime_config
            if effective_runtime_config is None
            else effective_runtime_config
        )
        self._last_applied_runtime_config = resolved_runtime_config
        resolved_bundle_path = (
            None if model_bundle_path is None else Path(model_bundle_path)
        )
        self._reset_growth_state_eviction_runtime()
        LOGGER.info(
            "[search] selector=%s opening_type=%s opening_expansion=%s",
            _selector_family_name(self._args.search_args),
            _opening_type_name(self._args.search_args),
            _opening_expansion_kind_name(self._args.search_args),
        )
        _log_search_rollout_config(self._args.search_args)
        if tree_snapshot_path is None:
            LOGGER.info(
                "[runtime] create_start evaluator_bundle=%s",
                "none" if resolved_bundle_path is None else str(resolved_bundle_path),
            )
            started_at = time.perf_counter()
            self._runtime = self._create_fresh_runtime(
                resolved_bundle_path,
                search_args=self._args.search_args,
            )
            _apply_runtime_config_to_runtime(self._runtime, resolved_runtime_config)
            self._install_state_rematerialization_phase_hooks(self._runtime)
            elapsed_s = time.perf_counter() - started_at
            LOGGER.info("[runtime] create_done elapsed=%.3fs", elapsed_s)
            if resolved_bundle_path is not None:
                if not reevaluate_tree:
                    LOGGER.info("[reeval] skipped reason=fresh_runtime_attach")
                self._set_runtime_evaluator_from_bundle(
                    resolved_bundle_path,
                    reevaluate_tree=reevaluate_tree,
                )
            else:
                self._current_evaluator_bundle_path = resolved_bundle_path
            return

        LOGGER.info(
            "[runtime] restore_start checkpoint=%s evaluator_bundle=%s",
            str(tree_snapshot_path),
            "none" if resolved_bundle_path is None else str(resolved_bundle_path),
        )
        started_at = time.perf_counter()
        runtime = self._load_runtime_from_checkpoint(
            Path(tree_snapshot_path),
            search_args=self._args.search_args,
        )
        elapsed_s = time.perf_counter() - started_at
        self._runtime = runtime
        _apply_runtime_config_to_runtime(runtime, resolved_runtime_config)
        self._install_state_rematerialization_phase_hooks(runtime)
        LOGGER.info("[runtime] restore_done elapsed=%.3fs", elapsed_s)
        self._current_evaluator_bundle_path = None
        if resolved_bundle_path is not None:
            if not reevaluate_tree:
                LOGGER.info("[reeval] skipped reason=resume_restore")
            self._set_runtime_evaluator_from_bundle(
                resolved_bundle_path,
                reevaluate_tree=reevaluate_tree,
            )
        else:
            LOGGER.info("[runtime] evaluator_attach_skipped reason=no_bundle")

    def grow(self, max_growth_steps: int) -> None:
        """Advance the live runtime by up to ``max_growth_steps`` iterations."""
        runtime = self._require_runtime()
        initial_tree_size = _live_tree_node_count(runtime)
        LOGGER.info(
            "[growth] start max_steps=%s initial_tree_size=%s state_eviction_policy=%s state_eviction_recent_window=%s state_rematerialization_cache_size=%s state_eviction_scan_interval_steps=%s state_eviction_scan_node_limit=%s state_eviction_payload_mode=%s state_eviction_delta_chain_max_depth=%s",
            max_growth_steps,
            initial_tree_size,
            _effective_growth_state_eviction_policy(
                self._args.growth_state_eviction_policy
            ),
            self._args.growth_state_eviction_recent_window,
            self._args.growth_state_rematerialization_cache_size,
            self._args.growth_state_eviction_scan_interval_steps,
            self._args.growth_state_eviction_scan_node_limit,
            self._args.growth_state_eviction_payload_mode,
            self._args.growth_state_eviction_delta_chain_max_depth,
        )
        self._clear_linoo_selection_table_artifact()
        steps_executed = 0
        stop_reason = "max_steps_reached"
        for _step_index in range(max_growth_steps):
            if runtime.tree.root_node.tree_evaluation.has_exact_value():
                stop_reason = "exact_solution_found"
                break
            if not _runtime_can_step(runtime):
                stop_reason = _runtime_stop_reason(runtime)
                break
            rematerialization_count_by_phase_before = dict(
                self._state_eviction_metrics.rematerialization_count_by_phase
            )
            rematerialization_cache_miss_by_phase_before = dict(
                self._state_eviction_metrics.rematerialization_cache_miss_by_phase
            )
            rematerialization_total_s_by_phase_before = dict(
                self._state_eviction_metrics.rematerialization_total_s_by_phase
            )
            step_report = runtime.step()
            steps_executed += 1
            current_tree_size = _live_tree_node_count(runtime)
            tree = getattr(runtime, "tree", None)
            branch_count = getattr(tree, "branch_count", None)
            selected_node_id = None
            selected_depth = None
            selector_report = None
            if step_report is not None:
                selected_node_id = getattr(step_report, "selected_node_id", None)
                selected_depth = getattr(step_report, "selected_depth", None)
                selector_report = getattr(step_report, "selector_report", None)
                reported_nodes_after = getattr(step_report, "nodes_after", None)
                if isinstance(reported_nodes_after, int):
                    current_tree_size = reported_nodes_after
                reported_branch_count = getattr(step_report, "branch_count", None)
                if isinstance(reported_branch_count, int):
                    branch_count = reported_branch_count
            self._record_growth_selected_node(
                selected_node_id=selected_node_id,
                step=steps_executed,
            )
            self._maybe_evict_cold_nodes(
                runtime=runtime,
                step=steps_executed,
            )
            if not isinstance(selected_depth, int):
                node_selector = getattr(runtime, "node_selector", None)
                uniform_selector = getattr(node_selector, "base", node_selector)
                selected_depth = getattr(
                    uniform_selector, "current_depth_to_expand", None
                )
            selector_report_rows = (
                getattr(step_report, "selector_report_rows", None)
                if step_report is not None
                else None
            )
            if not isinstance(selector_report_rows, int):
                selector_report_rows = _selector_report_row_count(selector_report)
            selector_diagnostics = _selector_growth_diagnostic_fields(selector_report)
            LOGGER.info(
                "[growth-timing] step=%s total_s=%s select_s=%s limit_s=%s expand_s=%s evaluate_s=%s propagate_s=%s selector_total_s=%s selector_collect_s=%s selector_choose_depth_s=%s selector_heap_update_s=%s selector_choose_node_s=%s selector_report_s=%s rows=%s nodes_scanned=%s frontier_scanned=%s selected_depth_frontier=%s heap_registered=%s stale_skipped=%s selector_state_rebuilt=%s selector_nodes_incrementally_updated=%s selector_total_nodes_scanned=%s selector_frontier_nodes_scanned=%s",
                steps_executed,
                _format_optional_seconds(
                    getattr(step_report, "total_s", None)
                    if step_report is not None
                    else None
                ),
                _format_optional_seconds(
                    getattr(step_report, "select_s", None)
                    if step_report is not None
                    else None
                ),
                _format_optional_seconds(
                    getattr(step_report, "limit_s", None)
                    if step_report is not None
                    else None
                ),
                _format_optional_seconds(
                    getattr(step_report, "expand_s", None)
                    if step_report is not None
                    else None
                ),
                _format_optional_seconds(
                    getattr(step_report, "evaluate_s", None)
                    if step_report is not None
                    else None
                ),
                _format_optional_seconds(
                    getattr(step_report, "propagate_s", None)
                    if step_report is not None
                    else None
                ),
                _format_optional_seconds(getattr(selector_report, "total_s", None)),
                _format_optional_seconds(
                    getattr(selector_report, "collect_frontier_state_s", None)
                ),
                _format_optional_seconds(
                    getattr(selector_report, "choose_depth_s", None)
                ),
                _format_optional_seconds(
                    getattr(selector_report, "heap_update_s", None)
                ),
                _format_optional_seconds(
                    getattr(selector_report, "choose_node_s", None)
                ),
                _format_optional_seconds(
                    getattr(selector_report, "make_report_s", None)
                ),
                _format_optional_int_log(selector_report_rows),
                _format_optional_int_log(
                    getattr(selector_report, "total_nodes_scanned", None)
                ),
                _format_optional_int_log(
                    getattr(selector_report, "frontier_nodes_scanned", None)
                ),
                _format_optional_int_log(
                    getattr(selector_report, "selected_depth_frontier_count", None)
                ),
                _format_optional_int_log(
                    getattr(selector_report, "heap_candidates_registered", None)
                ),
                _format_optional_int_log(
                    getattr(selector_report, "stale_candidates_skipped", None)
                ),
                _metric_value(selector_diagnostics["selector_state_rebuilt"]),
                _metric_value(
                    selector_diagnostics["selector_nodes_incrementally_updated"]
                ),
                _metric_value(selector_diagnostics["selector_total_nodes_scanned"]),
                _metric_value(selector_diagnostics["selector_frontier_nodes_scanned"]),
            )
            selector_heap_details = _selector_heap_detail_fields(selector_report)
            rematerialization_count_by_phase = (
                self._state_eviction_metrics.rematerialization_count_by_phase
            )
            select_rematerialization_delta_by_phase = _phase_delta(
                rematerialization_count_by_phase_before,
                rematerialization_count_by_phase,
                prefix="select",
            )
            select_rematerialization_miss_delta_by_phase = _phase_delta(
                rematerialization_cache_miss_by_phase_before,
                self._state_eviction_metrics.rematerialization_cache_miss_by_phase,
                prefix="select",
            )
            select_rematerialization_s_delta_by_phase = _phase_delta(
                rematerialization_total_s_by_phase_before,
                self._state_eviction_metrics.rematerialization_total_s_by_phase,
                prefix="select",
            )
            select_rematerialization_count_total = sum(
                _phase_delta(
                    {}, rematerialization_count_by_phase, prefix="select"
                ).values()
            )
            LOGGER.info(
                "[selector-heap-detail] step=%s candidate_count=%s push_count=%s pop_count=%s stale_skip_count=%s signature_check_count=%s signature_recompute_count=%s version_mismatch_count=%s total_heap_entries=%s max_heap_size=%s depth_count=%s frontier_node_count_seen=%s select_rematerialization_count_total=%s select_rematerialization_count_delta=%s select_rematerialization_cache_miss_delta=%s select_rematerialization_total_s_delta=%s select_rematerialization_delta_by_phase=%r select_rematerialization_miss_delta_by_phase=%r select_rematerialization_s_delta_by_phase=%r",
                steps_executed,
                _metric_value(selector_heap_details["candidate_count"]),
                _metric_value(selector_heap_details["push_count"]),
                _metric_value(selector_heap_details["pop_count"]),
                _metric_value(selector_heap_details["stale_skip_count"]),
                _metric_value(selector_heap_details["signature_check_count"]),
                _metric_value(selector_heap_details["signature_recompute_count"]),
                _metric_value(selector_heap_details["version_mismatch_count"]),
                _metric_value(selector_heap_details["total_heap_entries"]),
                _metric_value(selector_heap_details["max_heap_size"]),
                _metric_value(selector_heap_details["depth_count"]),
                _metric_value(selector_heap_details["frontier_node_count_seen"]),
                select_rematerialization_count_total,
                sum(select_rematerialization_delta_by_phase.values()),
                sum(select_rematerialization_miss_delta_by_phase.values()),
                sum(select_rematerialization_s_delta_by_phase.values()),
                select_rematerialization_delta_by_phase,
                select_rematerialization_miss_delta_by_phase,
                select_rematerialization_s_delta_by_phase,
            )
            if step_report is not None:
                self._log_and_persist_linoo_selection_table(
                    step_report=step_report,
                    step=steps_executed,
                    selected_depth=selected_depth,
                    selected_node_id=selected_node_id,
                )
            _log_latest_rollout_report(runtime)
            LOGGER.info(
                "[growth] step=%s node_count=%s nodes_added=%s branch_count=%s selected_node_id=%s selected_depth=%s",
                steps_executed,
                current_tree_size,
                current_tree_size - initial_tree_size,
                branch_count if isinstance(branch_count, int) else "unknown",
                selected_node_id if isinstance(selected_node_id, int) else "unknown",
                selected_depth if isinstance(selected_depth, int) else "unknown",
            )
        final_tree_size = _live_tree_node_count(runtime)
        LOGGER.info(
            "[growth] done steps=%s nodes_added=%s final_size=%s stop_reason=%s",
            steps_executed,
            final_tree_size - initial_tree_size,
            final_tree_size,
            stop_reason,
        )
        LOGGER.info("[state-eviction] %s", self._format_state_eviction_metrics())

    def _reset_growth_state_eviction_runtime(self) -> None:
        """Reset live compact payload storage for the next loaded runtime."""
        self._state_eviction_metrics = MorpionGrowthStateEvictionMetrics(
            state_eviction_policy=_effective_growth_state_eviction_policy(
                self._args.growth_state_eviction_policy
            ),
            state_eviction_payload_mode=self._args.growth_state_eviction_payload_mode,
            state_eviction_delta_chain_max_depth=(
                self._args.growth_state_eviction_delta_chain_max_depth
            ),
        )
        self._live_compact_state_resolver = _LiveCompactStateResolver(
            state_codec=self._state_codec,
            metrics=self._state_eviction_metrics,
            cache_size=self._args.growth_state_rematerialization_cache_size,
        )
        self._growth_eviction_recent_node_ids = deque()
        self._growth_eviction_recent_node_id_set = set()
        self._growth_eviction_selected_step_by_node_id = {}
        self._growth_eviction_scan_cursor = 0

    @contextmanager
    def _state_rematerialization_phase(self, phase: str) -> Iterator[None]:
        """Attribute compact-state resolver activity to a diagnostic phase."""
        with self._live_compact_state_resolver.phase(phase):
            yield

    def _install_state_rematerialization_phase_hooks(self, runtime: object) -> None:
        """Wrap runtime step subphases to attribute resolver work by phase."""
        if bool(getattr(runtime, "_chipiron_rematerialization_phase_hooks", False)):
            return
        phase_by_method_name = {
            "_select_node_for_expansion": "select.total",
            "_expand_opening_instructions": "expand",
            "_evaluate_expansions": "evaluate",
            "_propagate_iteration_updates": "propagate",
        }
        for method_name, phase in phase_by_method_name.items():
            original_method = getattr(runtime, method_name, None)
            if not callable(original_method):
                continue

            def _phase_wrapped_method(
                *args: object,
                _original_method: object = original_method,
                _phase: str = phase,
                **kwargs: object,
            ) -> object:
                with self._state_rematerialization_phase(_phase):
                    return cast("Any", _original_method)(*args, **kwargs)

            setattr(runtime, method_name, _phase_wrapped_method)
        cast(
            "Any", runtime
        )._diagnostic_phase_context = self._state_rematerialization_phase
        self._install_selector_diagnostic_phase_context(
            getattr(runtime, "node_selector", None)
        )
        cast("Any", runtime)._chipiron_rematerialization_phase_hooks = True

    def _install_selector_diagnostic_phase_context(self, selector: object) -> None:
        """Attach optional diagnostic phase context hooks to selector objects."""
        if selector is None:
            return
        try:
            object.__setattr__(
                selector,
                "_diagnostic_phase_context",
                self._state_rematerialization_phase,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return
        base_selector = getattr(selector, "base", None)
        if base_selector is not None and base_selector is not selector:
            self._install_selector_diagnostic_phase_context(base_selector)

    def _record_growth_selected_node(
        self,
        *,
        selected_node_id: object,
        step: int,
    ) -> None:
        """Track recently selected nodes so propagation-hot states stay materialized."""
        if not isinstance(selected_node_id, int):
            return
        self._growth_eviction_selected_step_by_node_id[selected_node_id] = step
        recent_window = self._args.growth_state_eviction_recent_window
        if recent_window <= 0:
            return
        self._growth_eviction_recent_node_ids.append(selected_node_id)
        self._growth_eviction_recent_node_id_set.add(selected_node_id)
        while len(self._growth_eviction_recent_node_ids) > recent_window:
            expired_node_id = self._growth_eviction_recent_node_ids.popleft()
            if expired_node_id not in self._growth_eviction_recent_node_ids:
                self._growth_eviction_recent_node_id_set.discard(expired_node_id)

    def _maybe_evict_cold_nodes(
        self,
        *,
        runtime: object,
        step: int,
    ) -> None:
        """Batch-evict cold materialized states, never the hot selection."""
        policy = _effective_growth_state_eviction_policy(
            self._args.growth_state_eviction_policy
        )
        if policy == "none":
            return
        if policy not in {"cold_expanded", "frontier_cold"}:
            self._state_eviction_metrics.skip("unsupported_policy")
            return
        interval = self._args.growth_state_eviction_scan_interval_steps
        if interval <= 0:
            self._state_eviction_metrics.skip("scan_interval_disabled")
            return
        if step % interval != 0:
            return
        scan_limit = self._args.growth_state_eviction_scan_node_limit
        if scan_limit <= 0:
            self._state_eviction_metrics.skip("scan_node_limit_disabled")
            return
        started_at = perf_counter()
        self._state_eviction_metrics.eviction_scan_count += 1
        nodes_scanned = 0
        try:
            for node in self._iter_growth_eviction_scan_nodes(runtime):
                if nodes_scanned >= scan_limit:
                    break
                nodes_scanned += 1
                self._maybe_evict_one_cold_node(node, policy=policy)
        finally:
            self._state_eviction_metrics.eviction_nodes_scanned_count += nodes_scanned
            self._state_eviction_metrics.eviction_total_s += perf_counter() - started_at

    def _iter_growth_eviction_scan_nodes(self, runtime: object) -> Iterator[object]:
        """Yield candidate nodes for one bounded eviction scan.

        Scans rotate through tree order so a small scan limit does not keep
        revisiting the same early nodes forever.
        """
        scan_limit = self._args.growth_state_eviction_scan_node_limit
        if scan_limit <= 0:
            return iter(())
        all_nodes_in_tree_order = getattr(runtime, "_all_nodes_in_tree_order", None)
        if callable(all_nodes_in_tree_order):
            try:
                nodes = all_nodes_in_tree_order()
            except Exception:
                self._state_eviction_metrics.skip("scan_nodes_failed")
                return iter(())
        else:
            nodes = tuple(self.iter_profile_nodes())
        if not isinstance(nodes, Sequence):
            nodes = tuple(nodes)
        node_count = len(nodes)
        if node_count <= 0:
            return iter(())
        start = self._growth_eviction_scan_cursor % node_count
        count = min(scan_limit, node_count)
        self._growth_eviction_scan_cursor = (start + count) % node_count
        return (nodes[(start + offset) % node_count] for offset in range(count))

    def _state_eviction_skip_reason(
        self,
        *,
        node: object,
        node_id: object,
        policy: str,
    ) -> str | None:
        """Return why a node cannot be state-evicted, or ``None`` if eligible.

        ``cold_expanded`` preserves the C4c behavior: only cold nodes that have
        already generated all branches are eligible. ``frontier_cold`` is the
        C4d experiment: any cold materialized node may be evicted, including
        frontier nodes, as long as the recent-selection hot set does not protect
        it and the existing live compact resolver can rebuild it.
        """
        if policy == "none":
            return "policy_none"
        if policy not in {"cold_expanded", "frontier_cold"}:
            return "unsupported_policy"
        if not isinstance(node_id, int):
            return "missing_node_id"
        if node_id in self._growth_eviction_recent_node_id_set:
            return "recently_selected"
        raw_handle = getattr(node, "state_handle", None)
        if isinstance(raw_handle, CheckpointBackedStateHandle):
            return "already_checkpoint_backed"
        if not isinstance(raw_handle, MaterializedStateHandle):
            return "not_materialized_handle"
        tree_node = getattr(node, "tree_node", None)
        if tree_node is None:
            return "missing_tree_node"
        if policy == "cold_expanded" and not bool(
            getattr(node, "all_branches_generated", False)
        ):
            return "not_fully_expanded"
        return None

    def _maybe_evict_one_cold_node(self, node: object, *, policy: str) -> None:
        """Evict one eligible cold materialized node."""
        self._state_eviction_metrics.eviction_attempt_count += 1
        node_id = getattr(node, "id", None)
        skip_reason = self._state_eviction_skip_reason(
            node=node,
            node_id=node_id,
            policy=policy,
        )
        if skip_reason is not None:
            self._state_eviction_metrics.skip(skip_reason)
            return
        raw_handle = getattr(node, "state_handle", None)
        assert isinstance(node_id, int)
        assert isinstance(raw_handle, MaterializedStateHandle)
        state = raw_handle.state_
        payload_started_at = perf_counter()
        try:
            with self._state_rematerialization_phase("eviction_payload_build"):
                live_payload = self._build_live_eviction_payload(
                    node=node,
                    node_id=node_id,
                    state=state,
                )
        except Exception:  # pylint: disable=broad-exception-caught
            self._state_eviction_metrics.skip("payload_build_failed")
            return
        finally:
            self._state_eviction_metrics.eviction_payload_build_s += (
                perf_counter() - payload_started_at
            )
        tree_node = getattr(node, "tree_node", None)
        assert tree_node is not None
        self._live_compact_state_resolver.store_payload(
            node_id=node_id,
            payload=live_payload.payload,
            chain_depth=live_payload.chain_depth,
        )
        tree_node.state_handle_ = CheckpointBackedStateHandle(
            resolver=cast("Any", self._live_compact_state_resolver),
            node_id=node_id,
        )
        self._state_eviction_metrics.eviction_success_count += 1
        self._state_eviction_metrics.evicted_materialized_state_count += 1
        self._state_eviction_metrics.compact_payload_count = len(
            self._live_compact_state_resolver.state_payloads_by_node_id
        )
        if live_payload.kind == "delta":
            self._state_eviction_metrics.delta_payload_count += 1
        else:
            self._state_eviction_metrics.anchor_payload_count += 1

    def _build_live_eviction_payload(
        self,
        *,
        node: object,
        node_id: int,
        state: MorpionState,
    ) -> _LiveEvictionPayload:
        """Build a bounded live compact payload, preferring safe deltas."""
        state_summary = self._state_codec.dump_state_summary(state)
        if self._args.growth_state_eviction_payload_mode == "delta_when_safe":
            self._state_eviction_metrics.record_delta_payload_attempt()
            delta_payload = self._try_build_live_delta_eviction_payload(
                node=node,
                node_id=node_id,
                state=state,
                state_summary=state_summary,
            )
            if delta_payload is not None:
                self._state_eviction_metrics.record_delta_payload_success()
                return delta_payload

        anchor_ref = self._state_codec.dump_anchor_ref(state)
        return _LiveEvictionPayload(
            payload=AnchorCheckpointStatePayload(
                anchor_ref=anchor_ref,
                state_summary=state_summary,
            ),
            kind="anchor",
            chain_depth=0,
        )

    def _try_build_live_delta_eviction_payload(
        self,
        *,
        node: object,
        node_id: int,
        state: MorpionState,
        state_summary: object | None,
    ) -> _LiveEvictionPayload | None:
        """Return a parent-delta payload when every conservative precondition holds."""
        max_depth = self._args.growth_state_eviction_delta_chain_max_depth
        if max_depth <= 0:
            self._state_eviction_metrics.record_delta_payload_fallback(
                "delta_chain_depth_disabled"
            )
            return None
        parent_context = _single_parent_link_for_live_delta(node)
        if parent_context is None:
            self._state_eviction_metrics.record_delta_payload_fallback(
                "delta_parent_ambiguous"
            )
            return None
        resolver = self._live_compact_state_resolver
        if parent_context.parent_node_id not in resolver.state_payloads_by_node_id:
            self._state_eviction_metrics.record_delta_payload_fallback(
                "delta_parent_payload_missing"
            )
            return None
        parent_chain_depth = resolver.payload_chain_depth_by_node_id.get(
            parent_context.parent_node_id
        )
        if parent_chain_depth is None:
            self._state_eviction_metrics.record_delta_payload_fallback(
                "delta_parent_depth_unknown"
            )
            return None
        if parent_chain_depth >= max_depth:
            self._state_eviction_metrics.record_delta_payload_fallback(
                "delta_chain_depth_limit"
            )
            return None

        try:
            parent_state = cast("Any", parent_context.parent_node).state
            delta_ref = self._state_codec.dump_delta_from_parent(
                parent_state=parent_state,
                child_state=state,
                branch_from_parent=parent_context.branch_from_parent,
            )
            branch_payload = _dump_live_state_parent_branch_for_checkpoint(
                state_codec=self._state_codec,
                branch_from_parent=parent_context.branch_from_parent,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            self._state_eviction_metrics.record_delta_payload_fallback(
                "delta_payload_build_failed"
            )
            return None

        return _LiveEvictionPayload(
            payload=DeltaCheckpointStatePayload(
                state_parent_node_id=parent_context.parent_node_id,
                state_parent_branch=branch_payload,
                delta_ref=delta_ref,
                state_summary=state_summary,
            ),
            kind="delta",
            chain_depth=parent_chain_depth + 1,
        )

    def _format_state_eviction_metrics(self) -> str:
        """Format state eviction metrics as stable key=value log fields."""
        snapshot = self._state_eviction_metrics.snapshot()
        return " ".join(f"{key}={value!r}" for key, value in snapshot.items())

    def _log_and_persist_linoo_selection_table(
        self,
        *,
        step_report: object,
        step: int,
        selected_depth: object,
        selected_node_id: object,
    ) -> None:
        """Log and optionally persist the latest Linoo selector table."""
        selector_report = getattr(step_report, "selector_report", None)
        if selector_report is None:
            return
        row_count = _selector_report_row_count(selector_report)
        formatted_table: str | None = None
        format_elapsed_s: float | None = None
        log_elapsed_s: float | None = None
        if hasattr(selector_report, "format_depth_table"):
            format_started_at = time.perf_counter()
            formatted_table = selector_report.format_depth_table()
            format_elapsed_s = time.perf_counter() - format_started_at
            log_started_at = time.perf_counter()
            LOGGER.info(
                "[growth-selection-table] step=%s\n%s",
                step,
                formatted_table,
            )
            log_elapsed_s = time.perf_counter() - log_started_at
        LOGGER.info(
            "[growth-selection-table-timing] step=%s rows=%s format_s=%s log_s=%s",
            step,
            _format_optional_int_log(row_count),
            _format_optional_seconds(format_elapsed_s),
            _format_optional_seconds(log_elapsed_s),
        )
        table = linoo_selection_table_from_report(
            report=selector_report,
            updated_at_utc=_timestamp_utc_from_unix_s(time.time()),
            cycle_index=self._linoo_selection_table_cycle_index,
            generation=self._linoo_selection_table_generation,
            step=step,
            selected_depth=selected_depth if isinstance(selected_depth, int) else None,
            selected_node_id=(
                selected_node_id if isinstance(selected_node_id, int) else None
            ),
        )
        if table is None:
            return
        if self._linoo_selection_table_artifact_path is not None:
            save_linoo_selection_table(
                table,
                self._linoo_selection_table_artifact_path,
            )

    def _clear_linoo_selection_table_artifact(self) -> None:
        """Remove stale latest Linoo table state before a new growth batch."""
        artifact_path = self._linoo_selection_table_artifact_path
        if artifact_path is None:
            return
        try:
            artifact_path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            LOGGER.exception(
                "[growth-selection-table] stale_artifact_clear_failed path=%s",
                str(artifact_path),
            )

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Persist a training-grade snapshot from the live tree."""
        runtime = self._require_runtime()
        LOGGER.info(
            "[save] tree_export_start output=%s nodes=%s",
            str(output_path),
            _live_tree_node_count(runtime),
        )
        started_at = time.perf_counter()
        snapshot, _profile = self.build_training_tree_snapshot_payload()
        save_training_tree_snapshot(snapshot, output_path)
        elapsed_s = time.perf_counter() - started_at
        LOGGER.info(
            "[save] tree_export_done output=%s elapsed=%.3fs",
            str(output_path),
            elapsed_s,
        )

    def export_sharded_training_tree_snapshot(
        self,
        output_dir: str | Path,
        *,
        generation: int,
    ) -> Path:
        """Persist one additive sharded training export for the live tree."""
        runtime = self._require_runtime()
        ordered_nodes = runtime._all_nodes_in_tree_order()
        started_at = time.perf_counter()
        profile = MorpionTrainingExportProfile()

        def state_ref_dumper(state: object) -> object:
            return self._state_codec.dump_state_ref(cast("MorpionState", state))

        manifest_path, stats = save_morpion_sharded_training_tree_from_live_nodes(
            nodes=ordered_nodes,
            root_node_id=str(runtime.tree.root_node.id),
            output_dir=output_dir,
            generation=generation,
            state_ref_dumper=state_ref_dumper,
            direct_value_extractor=_value_to_scalar,
            backed_up_value_extractor=_value_to_scalar,
            profile=profile,
        )
        profile.payload_build_s = time.perf_counter() - started_at
        _log_sharded_training_export_stats(stats)
        _log_training_export_profile(profile)
        self._latest_training_export_stats = sharded_training_export_stats_to_dict(
            stats
        )
        self._latest_training_export_stats["output_path"] = str(manifest_path)
        self._latest_training_export_profile = training_export_profile_to_dict(profile)
        LOGGER.info(
            "[save] sharded_tree_export_done output=%s generation=%s nodes=%s rows=%s bytes=%s elapsed=%.3fs",
            str(manifest_path),
            generation,
            len(ordered_nodes),
            stats.rows_written,
            stats.bytes_written,
            time.perf_counter() - started_at,
        )
        return manifest_path

    def build_training_tree_snapshot_payload(
        self,
    ) -> tuple[TrainingTreeSnapshot, MorpionTrainingExportProfile]:
        """Build a training snapshot plus aggregate profiling for the live tree."""
        runtime = self._require_runtime()
        ordered_nodes = runtime._all_nodes_in_tree_order()
        profile = MorpionTrainingExportProfile()
        started_at = perf_counter()

        def state_ref_dumper(state: object) -> object:
            return self._state_codec.dump_state_ref(cast("MorpionState", state))

        snapshot = build_training_tree_snapshot(
            ordered_nodes,
            root_node_id=str(runtime.tree.root_node.id),
            state_ref_dumper=state_ref_dumper,
            direct_value_extractor=_value_to_scalar,
            backed_up_value_extractor=_value_to_scalar,
            profile=profile,
        )
        profile.payload_build_s = perf_counter() - started_at
        _log_training_export_profile(profile)
        return snapshot, profile

    def current_tree_size(self) -> int:
        """Return the number of nodes currently tracked by the live runtime."""
        runtime = self._require_runtime()
        return _live_tree_node_count(runtime)

    def iter_profile_nodes(self) -> Iterator[object]:
        """Yield live search nodes for memory profiling only."""
        runtime = self._runtime
        if runtime is None:
            return iter(())
        all_nodes_in_tree_order = getattr(runtime, "_all_nodes_in_tree_order", None)
        if callable(all_nodes_in_tree_order):
            try:
                return iter(all_nodes_in_tree_order())
            except Exception:
                return iter(())

        for attr_path in (
            ("node_store",),
            ("node_store", "nodes"),
            ("tree", "nodes"),
            ("search_tree", "nodes"),
            ("graph", "nodes"),
            ("_node_store",),
            ("_node_store", "nodes"),
            ("_tree", "nodes"),
            ("_search_tree", "nodes"),
            ("_nodes",),
        ):
            value: object = runtime
            found = True
            for attr_name in attr_path:
                if not hasattr(value, attr_name):
                    found = False
                    break
                value = getattr(value, attr_name)
            if not found:
                continue
            if isinstance(value, Mapping):
                return iter(value.values())
            if isinstance(value, Iterable) and not isinstance(
                value, str | bytes | bytearray
            ):
                return iter(value)
        return iter(())

    def profile_runtime_root(self) -> object | None:
        """Return the live Anemone runtime for recursive memory diagnostics."""
        return self._runtime

    def profile_selector(self) -> object | None:
        """Return the live selector root for memory diagnostics, if available."""
        runtime = self._runtime
        if runtime is None:
            return None
        return getattr(runtime, "node_selector", None)

    def profile_state_codec(self) -> object:
        """Return the checkpoint codec root used by lazy restored state handles."""
        return self._state_codec

    def profile_state_eviction_runtime(self) -> Mapping[str, object]:
        """Return experimental state-eviction counters for memory diagnostics."""
        snapshot = self._state_eviction_metrics.snapshot()
        snapshot.update(
            {
                "state_eviction_recent_window": (
                    self._args.growth_state_eviction_recent_window
                ),
                "state_rematerialization_cache_size": (
                    self._args.growth_state_rematerialization_cache_size
                ),
                "state_eviction_scan_interval_steps": (
                    self._args.growth_state_eviction_scan_interval_steps
                ),
                "state_eviction_scan_node_limit": (
                    self._args.growth_state_eviction_scan_node_limit
                ),
                "state_eviction_payload_mode": (
                    self._args.growth_state_eviction_payload_mode
                ),
                "state_eviction_delta_chain_max_depth": (
                    self._args.growth_state_eviction_delta_chain_max_depth
                ),
                "recent_selected_count": len(self._growth_eviction_recent_node_id_set),
                "selected_step_tracked_count": len(
                    self._growth_eviction_selected_step_by_node_id
                ),
                "state_eviction_scan_cursor": self._growth_eviction_scan_cursor,
                "state_resolver_current_phase": (
                    self._live_compact_state_resolver.current_phase
                ),
            }
        )
        return snapshot

    def latest_checkpoint_metrics(self) -> Mapping[str, object] | None:
        """Return the latest checkpoint save metrics for dashboard metadata."""
        if self._latest_checkpoint_metrics is None:
            return None
        return dict(self._latest_checkpoint_metrics)

    def latest_training_export_stats(self) -> Mapping[str, object] | None:
        """Return the latest sharded training-export write metrics."""
        if self._latest_training_export_stats is None:
            return None
        return dict(self._latest_training_export_stats)

    def latest_training_export_profile(self) -> Mapping[str, object] | None:
        """Return the latest training-export fast-path profile metrics."""
        if self._latest_training_export_profile is None:
            return None
        return dict(self._latest_training_export_profile)

    def iter_profile_branches(self) -> Iterator[object]:
        """Yield live branch or ordering objects for memory profiling only."""
        for node in self.iter_profile_nodes():
            branch_from_parent = getattr(node, "branch_from_parent", None)
            if branch_from_parent is not None:
                yield branch_from_parent

            iter_child_links = getattr(node, "iter_child_links", None)
            if callable(iter_child_links):
                try:
                    for branch, _child in iter_child_links():
                        yield branch
                    continue
                except Exception:
                    pass

            for attr_name in (
                "branches",
                "branches_children",
                "successors",
                "children",
                "moves_children",
                "parent_nodes",
            ):
                container = getattr(node, attr_name, None)
                if isinstance(container, Mapping):
                    yield from container
                    continue
                if isinstance(container, Iterable) and not isinstance(
                    container, str | bytes | bytearray
                ):
                    for branch in container:
                        yield branch

    def current_tree_branch_count(self) -> int | None:
        """Return the live tree branch count when Anemone exposes it."""
        runtime = self._require_runtime()
        branch_count = getattr(runtime.tree, "branch_count", None)
        return branch_count if isinstance(branch_count, int) else None

    def current_tree_status(self) -> MorpionBootstrapTreeStatus:
        """Return the best available live tree-monitoring status."""
        runtime = self._require_runtime()
        root_node = runtime.tree.root_node
        depth_node_counts = _runtime_depth_counts(runtime)
        depths_present = tuple(sorted(depth_node_counts))
        return MorpionBootstrapTreeStatus(
            num_nodes=_live_tree_node_count(runtime),
            num_expanded_nodes=_count_expanded_nodes(runtime),
            num_simulations=_safe_int_attr(root_node, "visit_count"),
            root_visit_count=_safe_int_attr(root_node, "visit_count"),
            min_depth_present=None if not depths_present else depths_present[0],
            max_depth_present=None if not depths_present else depths_present[-1],
            depth_node_counts=depth_node_counts,
        )

    def apply_reevaluation_patch(self, patch: MorpionReevaluationPatch) -> int:
        """Apply one reevaluation patch to the live runtime tree."""
        runtime = self._runtime
        if runtime is None:
            raise _uninitialized_reevaluation_patch_runtime_error()
        runtime = cast("Any", runtime)
        blend_alpha = self._last_applied_runtime_config.reevaluation_blend_alpha
        LOGGER.info(
            "[reevaluation-patch] runner_apply_start patch_id=%s rows=%s",
            patch.patch_id,
            len(patch.rows),
        )
        log_pipeline_memory(
            stage="reevaluation",
            generation=patch.tree_generation,
            event="before_patch_apply",
            rows=len(patch.rows),
            node_count=_optional_live_tree_node_count(runtime),
        )
        if blend_alpha >= 1.0:
            updates = tuple(
                NodeValueUpdate(
                    node_id=row.node_id,
                    direct_value=row.direct_value,
                    backed_up_value=row.backed_up_value,
                    is_exact=row.is_exact,
                    is_terminal=row.is_terminal,
                    metadata=row.metadata,
                )
                for row in patch.rows
            )
            result = runtime.apply_node_value_updates(
                updates,
                recompute_backups=True,
                allow_missing=True,
            )
            selector_invalidated: bool | None = None
        else:
            blend_metrics = _ReevaluationBlendMetrics()
            result, selector_invalidated = _apply_blended_reevaluation_patch(
                runtime=runtime,
                patch=patch,
                blend_alpha=blend_alpha,
                blend_metrics=blend_metrics,
            )
            LOGGER.info(
                "[reevaluation-blend] patch_id=%s alpha=%s count=%s "
                "avg_old=%s avg_new=%s avg_blended=%s selector_invalidated=%s",
                patch.patch_id,
                _metric_value(blend_alpha),
                blend_metrics.count,
                _metric_value(_blend_average(blend_metrics.old_sum, blend_metrics)),
                _metric_value(_blend_average(blend_metrics.new_sum, blend_metrics)),
                _metric_value(_blend_average(blend_metrics.blended_sum, blend_metrics)),
                _metric_value(selector_invalidated),
            )
        self._last_reevaluation_patch_apply_metrics = {
            "patch_id": patch.patch_id,
            "applied": result.applied_count,
            "missing": len(result.missing_node_ids),
            "recomputed": result.recomputed_count,
            "selector_invalidated": selector_invalidated,
        }
        LOGGER.info(
            "[reevaluation-patch] backup_refresh_done affected_nodes=%s ancestors_recomputed=%s selector_invalidated=%s",
            result.applied_count,
            result.recomputed_count,
            _metric_value(selector_invalidated),
        )
        LOGGER.info(
            "[reevaluation-patch] runner_apply_done "
            "patch_id=%s requested=%s applied=%s missing=%s recomputed=%s selector_invalidated=%s",
            patch.patch_id,
            result.requested_count,
            result.applied_count,
            len(result.missing_node_ids),
            result.recomputed_count,
            _metric_value(selector_invalidated),
        )
        log_pipeline_memory(
            stage="reevaluation",
            generation=patch.tree_generation,
            event="after_patch_apply",
            rows=result.applied_count,
            missing=len(result.missing_node_ids),
            recomputed=result.recomputed_count,
            node_count=_optional_live_tree_node_count(runtime),
        )
        return int(result.applied_count)

    @property
    def last_reevaluation_patch_apply_metrics(self) -> dict[str, object] | None:
        """Return the last detailed reevaluation patch apply metrics, if any."""
        if self._last_reevaluation_patch_apply_metrics is None:
            return None
        return dict(self._last_reevaluation_patch_apply_metrics)

    def current_runtime_config(self) -> MorpionBootstrapEffectiveRuntimeConfig:
        """Return the effective runtime config used to build the live runtime."""
        return self._last_applied_runtime_config

    def _create_fresh_runtime(
        self,
        model_bundle_path: Path | None,
        *,
        search_args: SearchArgs,
    ) -> object:
        """Create a fresh single-tree Morpion runtime with the selected evaluator."""
        evaluator = self._build_master_evaluator(model_bundle_path)
        return cast(
            "object",
            create_tree_and_value_exploration_with_tree_eval_factory(
                state_type=MorpionState,
                dynamics=self._dynamics,
                starting_state=self._dynamics.wrap_atomheart_state(initial_state()),
                args=search_args,
                random_generator=self._random_generator,
                master_state_evaluator=evaluator,
                state_representation_factory=None,
                node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
            ),
        )

    def _load_runtime_from_checkpoint(
        self,
        tree_snapshot_path: Path,
        *,
        search_args: SearchArgs,
    ) -> object:
        """Restore one live runtime from a persisted checkpoint JSON file."""
        LOGGER.info("[checkpoint] load_start path=%s", str(tree_snapshot_path))
        started_at = time.perf_counter()
        log_morpion_checkpoint_memory_phase(
            "before_runtime_restore",
            path=tree_snapshot_path,
        )
        restore_memory_logger = _restore_memory_logger_for_path(
            self._args,
            tree_snapshot_path,
        )
        if restore_memory_logger is not None:
            restore_memory_logger.log(
                "before_checkpoint_file_load",
                raw_checkpoint_referenced=False,
                typed_checkpoint_referenced=False,
            )
        if _is_sharded_runtime_checkpoint_path(tree_snapshot_path):
            return self._load_runtime_from_sharded_checkpoint(
                tree_snapshot_path,
                search_args=search_args,
                started_at=started_at,
                restore_memory_logger=restore_memory_logger,
            )
        cached_payload = _pop_cached_morpion_search_checkpoint_payload_for_restore(
            tree_snapshot_path
        )
        cache_state = "hit" if cached_payload is not None else "miss"
        if cached_payload is None:
            payload = load_morpion_search_checkpoint_payload(
                tree_snapshot_path,
                restore_memory_logger=restore_memory_logger,
            )
            bytes_loaded = tree_snapshot_path.stat().st_size
        else:
            payload, bytes_loaded = cached_payload
            del cached_payload
            if restore_memory_logger is not None:
                restore_memory_logger.log(
                    "after_typed_checkpoint_payload_build",
                    typed_payload=payload,
                    raw_checkpoint_referenced=False,
                    typed_checkpoint_referenced=True,
                    cache="hit",
                )
            LOGGER.info(
                "[checkpoint] candidate_reuse_for_restore path=%s",
                str(tree_snapshot_path),
            )
        node_count, anchor_count, delta_count = _checkpoint_node_counts(payload)
        restore_selector_state_fields = _checkpoint_selector_state_fields(
            payload,
            prefix="restore_checkpoint",
        )
        LOGGER.info(
            "[checkpoint] restore_selector_state path=%s restore_checkpoint_selector_state_present=%s restore_checkpoint_selector_state_type=%s restore_checkpoint_selector_state_version=%s",
            str(tree_snapshot_path),
            _metric_value(
                restore_selector_state_fields[
                    "restore_checkpoint_selector_state_present"
                ]
            ),
            _metric_value(
                restore_selector_state_fields["restore_checkpoint_selector_state_type"]
            ),
            _metric_value(
                restore_selector_state_fields[
                    "restore_checkpoint_selector_state_version"
                ]
            ),
        )
        if cache_state == "hit":
            _log_checkpoint_metrics(
                "payload_load",
                CheckpointIoMetrics(
                    path=str(tree_snapshot_path),
                    bytes=bytes_loaded,
                    total_s=0.0,
                    node_count=node_count,
                    anchor_count=anchor_count,
                    delta_count=delta_count,
                    cache="hit",
                ),
            )
        rss_before_mb = _current_rss_mb()
        runtime_started_at = time.perf_counter()
        runtime: object
        rss_after_rebuild_mb: float | None = None
        rss_after_release_mb: float | None = None
        try:
            runtime = cast(
                "object",
                load_search_from_checkpoint_payload(
                    payload,
                    state_codec=self._state_codec,
                    dynamics=self._dynamics,
                    args=search_args,
                    state_type=MorpionState,
                    master_state_value_evaluator=self._build_master_evaluator(None),
                    random_generator=self._random_generator,
                    state_representation_factory=None,
                    node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
                    restore_memory_phase_logger=(
                        None
                        if restore_memory_logger is None
                        else cast(
                            "RestoreMemoryPhaseLogger",
                            restore_memory_logger.callback,
                        )
                    ),
                ),
            )
            runtime_elapsed_s = time.perf_counter() - runtime_started_at
            rss_after_rebuild_mb = _current_rss_mb()
            log_morpion_checkpoint_memory_phase(
                "after_runtime_rebuild",
                path=tree_snapshot_path,
                nodes=node_count,
            )
        finally:
            del payload
            if restore_memory_logger is not None:
                restore_memory_logger.log(
                    "after_drop_raw_checkpoint_payload_if_applicable",
                    raw_checkpoint_referenced=False,
                    typed_checkpoint_referenced=False,
                )
            gc.collect()
            rss_after_release_mb = _current_rss_mb()
            if restore_memory_logger is not None:
                restore_memory_logger.log(
                    "after_gc",
                    raw_checkpoint_referenced=False,
                    typed_checkpoint_referenced=False,
                )
            log_morpion_checkpoint_memory_phase(
                "after_restore_payload_release",
                path=tree_snapshot_path,
                nodes=node_count,
            )
        if rss_after_rebuild_mb is None:
            rss_after_rebuild_mb = _current_rss_mb()
        if rss_after_release_mb is None:
            rss_after_release_mb = rss_after_rebuild_mb
        LOGGER.info(
            "[checkpoint] runtime_rebuild_done path=%s elapsed=%.3fs",
            str(tree_snapshot_path),
            runtime_elapsed_s,
        )
        elapsed_s = time.perf_counter() - started_at
        _log_checkpoint_metrics(
            "runtime_restore",
            CheckpointIoMetrics(
                path=str(tree_snapshot_path),
                runtime_rebuild_s=runtime_elapsed_s,
                total_s=elapsed_s,
                rss_before_mb=rss_before_mb,
                rss_after_mb=rss_after_release_mb,
                node_count=node_count,
                anchor_count=anchor_count,
                delta_count=delta_count,
                cache=cache_state,
            ),
        )
        LOGGER.info(
            "[checkpoint] load_done path=%s elapsed=%.3fs",
            str(tree_snapshot_path),
            elapsed_s,
        )
        return runtime

    def _load_runtime_from_sharded_checkpoint(
        self,
        tree_snapshot_path: Path,
        *,
        search_args: SearchArgs,
        started_at: float,
        restore_memory_logger: _RestoreMemoryLogger | None,
    ) -> object:
        """Restore one live runtime from an opt-in sharded checkpoint directory."""
        manifest = read_sharded_checkpoint_manifest(
            tree_snapshot_path / "manifest.json"
        )
        node_count = manifest.total_node_count
        branch_count = manifest.total_branch_count
        rss_before_mb = _current_rss_mb()
        runtime_started_at = time.perf_counter()
        runtime = cast(
            "object",
            load_search_from_sharded_checkpoint(
                tree_snapshot_path,
                state_codec=self._state_codec,
                dynamics=self._dynamics,
                args=search_args,
                state_type=MorpionState,
                master_state_value_evaluator=self._build_master_evaluator(None),
                random_generator=self._random_generator,
                state_representation_factory=None,
                node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
                restore_memory_phase_logger=(
                    None
                    if restore_memory_logger is None
                    else cast(
                        "RestoreMemoryPhaseLogger", restore_memory_logger.callback
                    )
                ),
            ),
        )
        runtime_elapsed_s = time.perf_counter() - runtime_started_at
        log_morpion_checkpoint_memory_phase(
            "after_runtime_rebuild",
            path=tree_snapshot_path,
            nodes=node_count,
        )
        if restore_memory_logger is not None:
            restore_memory_logger.log(
                "after_drop_raw_checkpoint_payload_if_applicable",
                raw_checkpoint_referenced=False,
                typed_checkpoint_referenced=False,
            )
        gc.collect()
        rss_after_release_mb = _current_rss_mb()
        if restore_memory_logger is not None:
            restore_memory_logger.log(
                "after_gc",
                raw_checkpoint_referenced=False,
                typed_checkpoint_referenced=False,
                node_count=node_count,
                branch_count=branch_count,
            )
        log_morpion_checkpoint_memory_phase(
            "after_restore_payload_release",
            path=tree_snapshot_path,
            nodes=node_count,
        )
        elapsed_s = time.perf_counter() - started_at
        _log_checkpoint_metrics(
            "runtime_restore",
            CheckpointIoMetrics(
                path=str(tree_snapshot_path),
                bytes=_checkpoint_artifact_bytes(tree_snapshot_path),
                runtime_rebuild_s=runtime_elapsed_s,
                total_s=elapsed_s,
                rss_before_mb=rss_before_mb,
                rss_after_mb=rss_after_release_mb,
                node_count=node_count,
                cache="skipped_sharded",
                runtime_checkpoint_format="sharded",
            ),
        )
        LOGGER.info(
            "[checkpoint] load_done path=%s elapsed=%.3fs format=sharded",
            str(tree_snapshot_path),
            elapsed_s,
        )
        return runtime

    def _build_master_evaluator(
        self,
        model_bundle_path: Path | None,
    ) -> MorpionMasterEvaluator:
        """Return the default or bundle-backed Morpion master evaluator."""
        if model_bundle_path is None:
            over_detector = MorpionOverEventDetector()
            return MorpionMasterEvaluator(
                evaluator=MorpionStateEvaluator(),
                over=over_detector,
                over_detector=over_detector,
            )
        return load_morpion_evaluator_from_model_bundle(model_bundle_path)

    def _set_runtime_evaluator_from_bundle(
        self,
        model_bundle_path: Path,
        *,
        reevaluate_tree: bool = True,
    ) -> None:
        """Load one bundle, install it into the live runtime, and optionally refresh."""
        runtime = self._require_runtime()
        LOGGER.info(
            "[runtime] evaluator_attach_start bundle=%s", str(model_bundle_path)
        )
        started_at = time.perf_counter()
        evaluator = load_morpion_evaluator_from_model_bundle(model_bundle_path)
        if reevaluate_tree:
            if not hasattr(runtime, "refresh_with_evaluator"):
                raise NotImplementedError(
                    "evaluator_update_policy='reevaluate_all' requested, but "
                    "AnemoneMorpionSearchRunner does not yet support full tree "
                    "reevaluation on restore."
                )
            LOGGER.info("[reeval] start bundle=%s", str(model_bundle_path))
            reeval_started_at = time.perf_counter()
            runtime.refresh_with_evaluator(
                evaluator,
                scope=self._args.reevaluation_scope,
            )
            elapsed_s = time.perf_counter() - reeval_started_at
            LOGGER.info(
                "[reeval] done elapsed=%.3fs",
                elapsed_s,
            )
        else:
            runtime.set_evaluator(evaluator)
            LOGGER.info("[runtime] evaluator bundle attached without reevaluation")
        attach_elapsed_s = time.perf_counter() - started_at
        LOGGER.info(
            "[runtime] evaluator_attach_done bundle=%s elapsed=%.3fs reevaluate_tree=%s",
            str(model_bundle_path),
            attach_elapsed_s,
            reevaluate_tree,
        )
        self._current_evaluator_bundle_path = model_bundle_path

    def _require_runtime(self) -> Any:
        """Return the live runtime or fail loudly if it has not been initialized."""
        if self._runtime is None:
            raise UninitializedMorpionSearchRunnerError
        return self._runtime

    def save_checkpoint(self, output_path: str | Path) -> None:
        """Persist the live Anemone runtime as a checkpoint file."""
        runtime = self._require_runtime()
        output = Path(output_path)

        LOGGER.info("[checkpoint] save_start path=%s", str(output))
        rss_before_mb = _current_rss_mb()
        save_started_at = time.perf_counter()
        payload_started_at = time.perf_counter()
        log_morpion_checkpoint_memory_phase(
            "before_checkpoint_save_payload_build",
            path=output,
            generation=_generation_from_checkpoint_path(output),
        )
        with self._state_rematerialization_phase("checkpoint"):
            payload = build_search_checkpoint_payload(
                runtime,
                state_codec=self._state_codec,
            )
        checkpoint_selector_state_fields = _checkpoint_selector_state_fields(
            payload,
            prefix="checkpoint",
        )
        payload_elapsed_s = time.perf_counter() - payload_started_at
        LOGGER.info(
            "[checkpoint] payload_build_done path=%s elapsed=%.3fs",
            str(output),
            payload_elapsed_s,
        )
        LOGGER.info(
            "[checkpoint] payload_selector_state path=%s checkpoint_selector_state_present=%s checkpoint_selector_state_type=%s checkpoint_selector_state_version=%s",
            str(output),
            _metric_value(
                checkpoint_selector_state_fields["checkpoint_selector_state_present"]
            ),
            _metric_value(
                checkpoint_selector_state_fields["checkpoint_selector_state_type"]
            ),
            _metric_value(
                checkpoint_selector_state_fields["checkpoint_selector_state_version"]
            ),
        )
        node_count, anchor_count, delta_count = _checkpoint_node_counts(payload)
        log_morpion_checkpoint_memory_phase(
            "after_checkpoint_save_payload_build",
            path=output,
            nodes=node_count,
            generation=_generation_from_checkpoint_path(output),
        )
        if self._args.runtime_checkpoint_format == "sharded":
            manifest = write_sharded_search_checkpoint(
                payload,
                output,
                layout="split",
            )
            checkpoint_bytes = _checkpoint_artifact_bytes(output)
            write_stats = None
        else:
            write_stats = write_checkpoint_json_payload(payload, output)
            checkpoint_bytes = write_stats.compressed_bytes
        log_morpion_checkpoint_memory_phase(
            "after_checkpoint_save_write",
            path=output,
            nodes=node_count,
            generation=_generation_from_checkpoint_path(output),
        )
        if write_stats is None:
            LOGGER.info(
                "[checkpoint] sharded_checkpoint_write_done path=%s shards=%s nodes=%s branches=%s bytes=%s",
                str(output),
                len(manifest.shards),
                manifest.total_node_count,
                _metric_value(manifest.total_branch_count),
                _metric_value(checkpoint_bytes),
            )
        else:
            if write_stats.jsonable_s is None:
                LOGGER.info(
                    "[checkpoint] payload_jsonable_skipped path=%s encoder=%s",
                    str(write_stats.output_path),
                    write_stats.encoder,
                )
            else:
                LOGGER.info(
                    "[checkpoint] payload_jsonable_done path=%s elapsed=%.3fs",
                    str(write_stats.output_path),
                    write_stats.jsonable_s,
                )
            LOGGER.info(
                "[checkpoint] checkpoint_write_done path=%s format=%s encoder=%s json_encode_s=%.3fs compress_s=%s write_s=%.3fs bytes=%s uncompressed_bytes=%s compression_ratio=%s",
                str(write_stats.output_path),
                write_stats.file_format,
                write_stats.encoder,
                write_stats.json_encode_s,
                _metric_value(write_stats.compress_s),
                write_stats.write_s,
                write_stats.compressed_bytes,
                write_stats.uncompressed_bytes,
                _metric_value(write_stats.compression_ratio),
            )
        elapsed_s = time.perf_counter() - save_started_at
        rss_after_mb = _current_rss_mb()
        checkpoint_metrics = CheckpointIoMetrics(
            path=str(output if write_stats is None else write_stats.output_path),
            bytes=checkpoint_bytes,
            file_format=(
                "sharded" if write_stats is None else str(write_stats.file_format)
            ),
            encoder=None if write_stats is None else write_stats.encoder,
            payload_build_s=payload_elapsed_s,
            jsonable_s=None if write_stats is None else write_stats.jsonable_s,
            json_encode_s=None if write_stats is None else write_stats.json_encode_s,
            compress_s=None if write_stats is None else write_stats.compress_s,
            write_s=None if write_stats is None else write_stats.write_s,
            total_s=elapsed_s,
            uncompressed_bytes=None
            if write_stats is None
            else write_stats.uncompressed_bytes,
            compression_ratio=None
            if write_stats is None
            else write_stats.compression_ratio,
            rss_before_mb=rss_before_mb,
            rss_after_mb=rss_after_mb,
            node_count=node_count,
            anchor_count=anchor_count,
            delta_count=delta_count,
            runtime_checkpoint_format=self._args.runtime_checkpoint_format,
        )
        _log_checkpoint_metrics(
            "save",
            checkpoint_metrics,
        )
        self._latest_checkpoint_metrics = checkpoint_io_metrics_to_dict(
            checkpoint_metrics
        )
        LOGGER.info(
            "[checkpoint] save_done path=%s elapsed=%.3fs",
            str(output if write_stats is None else write_stats.output_path),
            elapsed_s,
        )
        del payload
        gc.collect()
        log_morpion_checkpoint_memory_phase(
            "after_checkpoint_save_cleanup",
            path=output,
            nodes=node_count,
            generation=_generation_from_checkpoint_path(output),
        )


def load_morpion_search_checkpoint_payload(
    path: str | Path,
    *,
    restore_memory_logger: _RestoreMemoryLogger | None = None,
) -> SearchRuntimeCheckpointPayload:
    """Load a persisted search checkpoint payload and validate shape."""
    resolved_path = Path(path)
    LOGGER.info("[checkpoint] load_start path=%s", str(resolved_path))
    rss_before_mb = _current_rss_mb()
    log_morpion_checkpoint_memory_phase(
        "before_candidate_payload_load",
        path=resolved_path,
        generation=_generation_from_checkpoint_path(resolved_path),
    )
    started_at = time.perf_counter()
    try:
        raw_payload: object | None = None

        def log_read_phase(phase: str, metadata: Mapping[str, object]) -> None:
            if restore_memory_logger is None:
                return
            phase_raw_payload = metadata.get("raw_payload")
            log_metadata = {
                key: value for key, value in metadata.items() if key != "raw_payload"
            }
            restore_memory_logger.log(
                phase,
                raw_payload=(
                    phase_raw_payload if phase == "after_raw_json_decode" else None
                ),
                raw_checkpoint_referenced=phase == "after_raw_json_decode",
                typed_checkpoint_referenced=False,
                **log_metadata,
            )

        raw_payload, read_stats = load_checkpoint_json_payload(
            resolved_path,
            read_phase_logger=log_read_phase if restore_memory_logger else None,
        )
        LOGGER.info(
            "[checkpoint] json_load_done path=%s format=%s elapsed=%.3fs bytes=%s",
            str(resolved_path),
            read_stats.file_format,
            read_stats.json_load_s,
            read_stats.compressed_bytes,
        )
    except FileNotFoundError as exc:
        raise InvalidMorpionSearchCheckpointError(
            resolved_path,
            "file does not exist",
        ) from exc
    except Exception as exc:
        raise InvalidMorpionSearchCheckpointError(
            resolved_path,
            f"invalid checkpoint payload: {exc}",
        ) from exc

    try:
        payload_decode_started_at = time.perf_counter()
        normalized_payload = _normalize_search_checkpoint_payload_for_dacite(
            raw_payload
        )
        payload = from_dict(
            data_class=SearchRuntimeCheckpointPayload,
            data=normalized_payload,
            config=Config(cast=[tuple], check_types=False),
        )
        if restore_memory_logger is not None:
            restore_memory_logger.log(
                "after_typed_checkpoint_payload_build",
                raw_payload=raw_payload,
                typed_payload=payload,
                raw_checkpoint_referenced=True,
                typed_checkpoint_referenced=True,
            )
        payload_decode_elapsed_s = time.perf_counter() - payload_decode_started_at
        LOGGER.info(
            "[checkpoint] payload_decode_done path=%s elapsed=%.3fs",
            str(resolved_path),
            payload_decode_elapsed_s,
        )
        total_s = time.perf_counter() - started_at
        rss_after_mb = _current_rss_mb()
        node_count, anchor_count, delta_count = _checkpoint_node_counts(payload)
        log_morpion_checkpoint_memory_phase(
            "after_candidate_payload_load",
            path=resolved_path,
            nodes=node_count,
            generation=_generation_from_checkpoint_path(resolved_path),
        )
        _log_checkpoint_metrics(
            "payload_load",
            CheckpointIoMetrics(
                path=str(resolved_path),
                bytes=read_stats.compressed_bytes,
                file_format=read_stats.file_format,
                json_load_s=read_stats.json_load_s,
                payload_decode_s=payload_decode_elapsed_s,
                total_s=total_s,
                rss_before_mb=rss_before_mb,
                rss_after_mb=rss_after_mb,
                node_count=node_count,
                anchor_count=anchor_count,
                delta_count=delta_count,
                cache="miss",
            ),
        )
        del raw_payload
        del normalized_payload
        if restore_memory_logger is not None:
            restore_memory_logger.log(
                "after_drop_raw_checkpoint_payload_if_applicable",
                typed_payload=payload,
                raw_checkpoint_referenced=False,
                typed_checkpoint_referenced=True,
            )
    except Exception as exc:
        raise InvalidMorpionSearchCheckpointError(
            resolved_path,
            f"payload shape is invalid: {exc}",
        ) from exc
    else:
        return payload


def _generation_from_checkpoint_path(path: str | Path) -> int | None:
    """Return generation number from canonical checkpoint filenames when present."""
    stem_parts = Path(path).name.split(".")
    if not stem_parts:
        return None
    stem = stem_parts[0]
    if not stem.startswith("generation_"):
        return None
    try:
        return int(stem.removeprefix("generation_"))
    except ValueError:
        return None


def _mapping(data: object) -> dict[str, object] | None:
    """Return ``data`` as a mutable mapping when possible."""
    mapping = _string_key_mapping_or_none(data)
    if mapping is None:
        return None
    return dict(mapping)


def _string_key_mapping_or_none(data: object) -> Mapping[str, object] | None:
    """Return ``data`` as a string-keyed mapping when possible."""
    if not isinstance(data, dict):
        return None
    raw_mapping = cast("dict[object, object]", data)
    if not all(isinstance(key, str) for key in raw_mapping):
        return None
    return cast("Mapping[str, object]", raw_mapping)


def _normalize_search_checkpoint_payload_for_dacite(
    raw_payload: object,
) -> dict[str, object]:
    """Normalize union payload fields so dacite can rebuild checkpoint dataclasses."""
    normalized_payload = _mapping(raw_payload)
    if normalized_payload is None:
        raise _invalid_checkpoint_payload_mapping_error()

    normalized_selector_state = _mapping(normalized_payload.get("selector_state"))
    if normalized_selector_state is not None:
        normalized_payload["selector_state"] = from_dict(
            data_class=LinooSelectorCheckpointPayload,
            data=normalized_selector_state,
            config=Config(cast=[tuple], check_types=False),
        )

    normalized_tree = _mapping(normalized_payload.get("tree"))
    if normalized_tree is None:
        return normalized_payload

    raw_nodes = normalized_tree.get("nodes")
    if not isinstance(raw_nodes, list):
        return normalized_payload
    node_payloads = cast("list[object]", raw_nodes)

    normalized_tree["nodes"] = [
        _normalize_algorithm_node_payload_for_dacite(node_payload)
        for node_payload in node_payloads
    ]
    normalized_payload["tree"] = normalized_tree
    return normalized_payload


def _normalize_algorithm_node_payload_for_dacite(node_payload: object) -> object:
    """Normalize one algorithm-node payload before dacite reconstruction."""
    normalized_node_payload = _mapping(node_payload)
    if normalized_node_payload is None:
        return node_payload
    normalized_node_payload["state_payload"] = _checkpoint_state_payload_from_dict(
        normalized_node_payload.get("state_payload")
    )
    return normalized_node_payload


def _checkpoint_state_payload_from_dict(
    raw_state_payload: object,
) -> CheckpointNodeStatePayload | object:
    """Decode the explicit state-payload union used by Anemone checkpoints.

    Delta payloads carry their own state-parent edge fields; those are distinct
    from the graph/debug representative parent fields on algorithm nodes.
    """
    normalized_state_payload = _mapping(raw_state_payload)
    if normalized_state_payload is None:
        return raw_state_payload
    if "anchor_ref" in normalized_state_payload:
        return from_dict(
            data_class=AnchorCheckpointStatePayload,
            data=cast("Any", normalized_state_payload),
            config=Config(cast=[tuple], check_types=False),
        )
    if "delta_ref" in normalized_state_payload:
        return from_dict(
            data_class=DeltaCheckpointStatePayload,
            data=cast("Any", normalized_state_payload),
            config=Config(cast=[tuple], check_types=False),
        )
    return raw_state_payload


def apply_runtime_control_to_runner_args(
    runner_args: AnemoneMorpionSearchRunnerArgs,
    runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
) -> AnemoneMorpionSearchRunnerArgs:
    """Return runner args rebound to one effective runtime config.

    This helper is kept as the pure arg-transformation counterpart of the live
    runtime patching path used during checkpoint restore.
    """
    return replace(
        runner_args,
        search_args=_search_args_with_tree_branch_limit(
            runner_args.search_args,
            tree_branch_limit=runtime_config.tree_branch_limit,
        ),
    )


def _runtime_config_from_search_args(
    search_args: SearchArgs,
) -> MorpionBootstrapEffectiveRuntimeConfig:
    """Extract the supported effective runtime config from one SearchArgs object."""
    stopping_criterion = search_args.stopping_criterion
    if not isinstance(stopping_criterion, TreeBranchLimitArgs):
        raise TypeError(_TREE_BRANCH_LIMIT_ARGS_REQUIRED_MESSAGE)
    return MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=stopping_criterion.tree_branch_limit,
    )


def _search_args_with_tree_branch_limit(
    search_args: SearchArgs,
    *,
    tree_branch_limit: int,
) -> SearchArgs:
    """Return SearchArgs rebound to one explicit tree-branch limit."""
    stopping_criterion = search_args.stopping_criterion
    if not isinstance(stopping_criterion, TreeBranchLimitArgs):
        raise TypeError(_TREE_BRANCH_LIMIT_ARGS_REQUIRED_MESSAGE)
    return replace(
        search_args,
        stopping_criterion=replace(
            stopping_criterion,
            tree_branch_limit=tree_branch_limit,
        ),
    )


def _apply_runtime_config_to_runtime(
    runtime: object,
    runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
) -> None:
    """Apply the supported runtime config to one live runtime after create/restore."""
    stopping_criterion = getattr(runtime, "stopping_criterion", None)
    if not isinstance(stopping_criterion, TreeBranchLimit):
        raise TypeError(_LIVE_TREE_BRANCH_LIMIT_REQUIRED_MESSAGE)
    stopping_criterion.tree_branch_limit = runtime_config.tree_branch_limit


def _count_expanded_nodes(runtime: Any) -> int:
    """Count nodes that have already generated all branches in the live tree."""
    return sum(
        1
        for node in runtime._all_nodes_in_tree_order()
        if bool(getattr(node, "all_branches_generated", False))
    )


def _effective_growth_state_eviction_policy(policy: str) -> str:
    """Return the runtime semantics for a configured eviction policy."""
    if policy == "expanded":
        return "cold_expanded"
    return policy


def _single_parent_link_for_live_delta(node: object) -> _ParentDeltaContext | None:
    """Return the sole parent edge for live deltas, or ``None`` if ambiguous."""
    iter_parent_items = getattr(node, "iter_parent_items", None)
    if callable(iter_parent_items):
        try:
            parent_items = tuple(iter_parent_items())
        except Exception:  # pylint: disable=broad-exception-caught
            return None
    else:
        parent_nodes = getattr(node, "parent_nodes", None)
        if not isinstance(parent_nodes, Mapping):
            return None
        parent_items = tuple(parent_nodes.items())
    if len(parent_items) != 1:
        return None
    parent_node, branch_keys = parent_items[0]
    if not isinstance(branch_keys, set) or len(branch_keys) != 1:
        return None
    parent_node_id = getattr(parent_node, "id", None)
    if not isinstance(parent_node_id, int):
        return None
    return _ParentDeltaContext(
        parent_node=parent_node,
        parent_node_id=parent_node_id,
        branch_from_parent=next(iter(branch_keys)),
    )


def _dump_live_state_parent_branch_for_checkpoint(
    *,
    state_codec: object,
    branch_from_parent: object,
) -> object | None:
    """Return the optional compact branch payload stored beside a live delta."""
    hook = getattr(state_codec, "dump_state_parent_branch_for_checkpoint", None)
    if callable(hook):
        return hook(branch_from_parent)
    return branch_from_parent


def _selected_node_from_latest_expansions(
    runtime: object,
    *,
    selected_node_id: int,
) -> object | None:
    """Return the selected parent node from the latest expansion wave."""
    latest_tree_expansions = getattr(runtime, "latest_tree_expansions", None)
    if latest_tree_expansions is None:
        return None
    for bucket_name in (
        "expansions_with_node_creation",
        "expansions_without_node_creation",
    ):
        expansions = getattr(latest_tree_expansions, bucket_name, None)
        if not isinstance(expansions, Iterable):
            continue
        for expansion in expansions:
            parent_node = getattr(expansion, "parent_node", None)
            if getattr(parent_node, "id", None) == selected_node_id:
                return parent_node
    return None


def _runtime_depth_counts(runtime: Any) -> dict[int, int]:
    """Return live node counts grouped by relative tree depth."""
    tree = getattr(runtime, "tree", None)
    if tree is None:
        raise TypeError(_LIVE_TREE_REQUIRED_MESSAGE)
    descendants = getattr(tree, "descendants", None)
    if descendants is None:
        return {0: _live_tree_node_count(runtime)}

    root_depth = getattr(tree, "tree_root_tree_depth", 0)
    counts_by_depth: dict[int, int] = {0: 1}
    for absolute_depth in descendants:
        count_at_depth = getattr(
            descendants,
            "number_of_descendants_at_tree_depth",
            {},
        ).get(absolute_depth)
        if not isinstance(count_at_depth, int):
            count_at_depth = len(descendants[absolute_depth])
        counts_by_depth[int(absolute_depth) - int(root_depth)] = count_at_depth
    return counts_by_depth


def _apply_blended_reevaluation_patch(
    *,
    runtime: Any,
    patch: MorpionReevaluationPatch,
    blend_alpha: float,
    blend_metrics: _ReevaluationBlendMetrics,
) -> tuple[NodeValueUpdateResult, bool]:
    """Apply one smoothed patch using Anemone's existing single node lookup pass."""
    nodes_by_id = runtime._nodes_by_public_id()
    missing_node_ids = tuple(
        row.node_id for row in patch.rows if row.node_id not in nodes_by_id
    )
    applied_nodes: list[object] = []
    changed_nodes: list[object] = []
    for row in patch.rows:
        live_node = nodes_by_id.get(row.node_id)
        if live_node is None:
            continue
        update = NodeValueUpdate(
            node_id=row.node_id,
            direct_value=_reevaluation_patch_direct_value(
                row=row,
                live_node=live_node,
                blend_alpha=blend_alpha,
                blend_metrics=blend_metrics,
            ),
            backed_up_value=row.backed_up_value,
            is_exact=row.is_exact,
            is_terminal=row.is_terminal,
            metadata=row.metadata,
        )
        changed = bool(runtime._apply_node_value_update(node=live_node, update=update))
        applied_nodes.append(live_node)
        if changed:
            changed_nodes.append(live_node)

    recomputed_count = _recompute_after_blended_reevaluation(
        runtime=runtime,
        changed_nodes=changed_nodes,
    )
    selector_invalidated = False
    if changed_nodes:
        selector_invalidated = _invalidate_selector_cache_if_supported(runtime)
    return (
        NodeValueUpdateResult(
            requested_count=len(patch.rows),
            applied_count=len(applied_nodes),
            missing_node_ids=missing_node_ids,
            recomputed_count=recomputed_count,
        ),
        selector_invalidated,
    )


def _recompute_after_blended_reevaluation(
    *,
    runtime: Any,
    changed_nodes: list[object],
) -> int:
    """Refresh backups and exploration indices after local smoothed value writes."""
    if not changed_nodes:
        return 0
    tree_manager = runtime.tree_manager
    recomputed_nodes = (
        tree_manager.value_propagator.propagate_after_local_value_changes(changed_nodes)
    )
    tree_manager.refresh_exploration_indices(tree=runtime.tree)
    return len(recomputed_nodes)


def _reevaluation_patch_direct_value(
    *,
    row: MorpionReevaluationPatchRow,
    live_node: object | None,
    blend_alpha: float,
    blend_metrics: _ReevaluationBlendMetrics,
) -> float:
    """Return the direct value to write for one reevaluation patch row."""
    new_value = float(row.direct_value)
    if blend_alpha >= 1.0 or live_node is None:
        return new_value
    if _patch_row_is_authoritative(row) or _live_node_is_authoritative(live_node):
        return new_value

    old_value = _live_node_direct_value_score(live_node)
    if old_value is None:
        return new_value

    blended_value = ((1.0 - blend_alpha) * old_value) + (blend_alpha * new_value)
    blend_metrics.record(
        old_value=old_value,
        new_value=new_value,
        blended_value=blended_value,
    )
    return blended_value


def _patch_row_is_authoritative(row: object) -> bool:
    """Return whether one patch row should bypass smoothing."""
    return (
        getattr(row, "is_exact", None) is True
        or getattr(row, "is_terminal", None) is True
    )


def _live_node_is_authoritative(node: object) -> bool:
    """Return whether one live node has terminal or exact authoritative value state."""
    tree_evaluation = getattr(node, "tree_evaluation", None)
    has_exact_value = getattr(tree_evaluation, "has_exact_value", None)
    if callable(has_exact_value) and bool(has_exact_value()):
        return True

    state = getattr(node, "state", None)
    is_game_over = getattr(state, "is_game_over", None)
    if callable(is_game_over) and bool(is_game_over()):
        return True

    is_terminal = getattr(tree_evaluation, "is_terminal", None)
    return bool(is_terminal()) if callable(is_terminal) else False


def _live_node_direct_value_score(node: object) -> float | None:
    """Return one live node's direct-value score when present."""
    tree_evaluation = getattr(node, "tree_evaluation", None)
    direct_value = getattr(tree_evaluation, "direct_value", None)
    if direct_value is None:
        return None
    score = getattr(direct_value, "score", direct_value)
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return float(score)


def _blend_average(
    value_sum: float,
    metrics: _ReevaluationBlendMetrics,
) -> float | None:
    """Return one blend aggregate average, or None when no rows were blended."""
    if metrics.count == 0:
        return None
    return value_sum / metrics.count


def _selector_family_name(search_args: SearchArgs) -> str:
    """Return the effective selector family name for concise logging."""
    node_selector = search_args.node_selector
    base_selector = getattr(node_selector, "base", node_selector)
    selector_type = getattr(base_selector, "type", "unknown")
    return str(selector_type).lower()


def _runtime_can_step(runtime: Any) -> bool:
    """Return whether the live runtime can still perform a structural search step."""
    stopping_criterion = getattr(runtime, "stopping_criterion", None)
    should_we_continue = getattr(stopping_criterion, "should_we_continue", None)
    tree = getattr(runtime, "tree", None)
    if (
        callable(should_we_continue)
        and tree is not None
        and not bool(should_we_continue(tree=tree))
    ):
        return False

    node_selector = getattr(runtime, "node_selector", None)
    uniform_selector = getattr(node_selector, "base", node_selector)
    current_depth_to_expand = getattr(uniform_selector, "current_depth_to_expand", None)
    if not isinstance(current_depth_to_expand, int):
        return True
    if tree is None:
        return True
    tree_depth = tree.tree_root_tree_depth + current_depth_to_expand
    descendants = getattr(tree, "descendants", None)
    has_tree_depth = getattr(descendants, "has_tree_depth", None)
    if callable(has_tree_depth):
        return bool(has_tree_depth(tree_depth))
    if descendants is None:
        return True
    return tree_depth in descendants


def _runtime_stop_reason(runtime: Any) -> str:
    """Return a concise reason why the runtime cannot execute another step."""
    stopping_criterion = getattr(runtime, "stopping_criterion", None)
    should_we_continue = getattr(stopping_criterion, "should_we_continue", None)
    tree = getattr(runtime, "tree", None)
    if (
        callable(should_we_continue)
        and tree is not None
        and not bool(should_we_continue(tree=tree))
    ):
        branch_count = getattr(tree, "branch_count", None)
        tree_branch_limit = getattr(stopping_criterion, "tree_branch_limit", None)
        if isinstance(branch_count, int) and isinstance(tree_branch_limit, int):
            LOGGER.info(
                "[growth] stopping_criterion metric=%s limit=%s",
                branch_count,
                tree_branch_limit,
            )
        return "stopping_criterion_reached"

    node_selector = getattr(runtime, "node_selector", None)
    uniform_selector = getattr(node_selector, "base", node_selector)
    current_depth_to_expand = getattr(uniform_selector, "current_depth_to_expand", None)
    if not isinstance(current_depth_to_expand, int) or tree is None:
        return "runtime_cannot_step"
    tree_depth = tree.tree_root_tree_depth + current_depth_to_expand
    descendants = getattr(tree, "descendants", None)
    has_tree_depth = getattr(descendants, "has_tree_depth", None)
    if callable(has_tree_depth) and not bool(has_tree_depth(tree_depth)):
        return "runtime_cannot_step"
    if (
        descendants is not None
        and not callable(has_tree_depth)
        and tree_depth not in descendants
    ):
        return "runtime_cannot_step"
    return "runtime_cannot_step"


def _live_tree_node_count(runtime: Any) -> int:
    """Return the true live node count from the runtime tree bookkeeping."""
    tree = getattr(runtime, "tree", None)
    if tree is None:
        raise TypeError(_LIVE_TREE_REQUIRED_MESSAGE)
    nodes_count = getattr(tree, "nodes_count", None)
    if isinstance(nodes_count, int):
        return nodes_count
    descendants = getattr(tree, "descendants", None)
    get_count = getattr(descendants, "get_count", None)
    if callable(get_count):
        count = get_count()
        if isinstance(count, bool):
            return int(count)
        if isinstance(count, int | float | str):
            return int(count)
    return len(runtime._all_nodes_in_tree_order())


def _optional_live_tree_node_count(runtime: Any) -> int | None:
    """Return live node count when available without affecting caller behavior."""
    try:
        return _live_tree_node_count(runtime)
    except Exception:
        return None


def _safe_int_attr(node: Any, attribute_name: str) -> int | None:
    """Read one integer node attribute when the runtime exposes it."""
    value = getattr(node, attribute_name, None)
    return value if isinstance(value, int) else None


__all__ = [
    "AnemoneMorpionSearchRunner",
    "AnemoneMorpionSearchRunnerArgs",
    "InvalidMorpionSearchCheckpointError",
    "MorpionRegressorMasterEvaluator",
    "UninitializedMorpionSearchRunnerError",
    "apply_runtime_control_to_runner_args",
    "cache_morpion_search_checkpoint_payload_for_restore",
    "load_morpion_evaluator_from_model_bundle",
    "load_morpion_search_checkpoint_payload",
    "log_morpion_checkpoint_memory_phase",
    "restore_memory_logger_for_checkpoint_path",
]
