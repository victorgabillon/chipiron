"""Real Anemone-backed Morpion search runner for bootstrap cycles."""

from __future__ import annotations

import gc
import logging
import os
import resource
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    CheckpointNodeStatePayload,
    DeltaCheckpointStatePayload,
    LinooSelectorCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    build_search_checkpoint_payload,
    load_checkpoint_json_payload,
    load_search_from_checkpoint_payload,
    write_checkpoint_json_payload,
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

from .config import DEFAULT_MORPION_TREE_BRANCH_LIMIT, MorpionBootstrapRolloutConfig
from .control import MorpionBootstrapEffectiveRuntimeConfig
from .cycle_timing import timestamp_utc_from_unix_s as _timestamp_utc_from_unix_s
from .history import MorpionBootstrapTreeStatus
from .linoo_selection_table import (
    linoo_selection_table_from_report,
    save_linoo_selection_table,
)
from .search_runner_protocol import MorpionSearchRunner
from .sharded_training_export import (
    MorpionShardedTrainingExportStats,
    save_morpion_sharded_training_tree_from_live_nodes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .pipeline_artifacts import (
        MorpionReevaluationPatch,
        MorpionReevaluationPatchRow,
    )

LOGGER = logging.getLogger(__name__)


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
        # Profiling only for now: current Morpion training-export consumers decode
        # state_ref_payload via the anchor-only load_state_ref path, so reusable
        # checkpoint delta payloads are not yet drop-in compatible.
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
        "[sharded-training-export] generation=%s nodes=%s new_nodes=%s reused_nodes=%s",
        stats.generation,
        stats.node_count,
        stats.new_node_count,
        stats.reused_node_count,
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


@dataclass(frozen=True, slots=True)
class AnemoneMorpionSearchRunnerArgs:
    """Configuration for the real Anemone-backed Morpion runner."""

    search_args: SearchArgs = field(default_factory=_default_search_args)
    random_seed: int = 0
    reevaluation_scope: str = "leaves"


@dataclass(frozen=True, slots=True)
class MorpionRegressorMasterEvaluator(MorpionMasterEvaluator):
    """Anemone-compatible Morpion evaluator backed by a saved regressor bundle."""

    input_converter: object
    regressor: object

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
        converter = cast("Any", self.input_converter)
        tensor = cast("Any", converter.state_to_tensor(morpion_state))
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
        self._linoo_selection_table_artifact_path: Path | None = None
        self._linoo_selection_table_cycle_index: int | None = None
        self._linoo_selection_table_generation: int | None = None
        self._last_reevaluation_patch_apply_metrics: dict[str, object] | None = None

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
            "[growth] start max_steps=%s initial_tree_size=%s",
            max_growth_steps,
            initial_tree_size,
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
        )
        _log_sharded_training_export_stats(stats)
        LOGGER.info(
            "[save] sharded_tree_export_done output=%s generation=%s nodes=%s elapsed=%.3fs",
            str(manifest_path),
            generation,
            len(ordered_nodes),
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
        cached_payload = _pop_cached_morpion_search_checkpoint_payload_for_restore(
            tree_snapshot_path
        )
        cache_state = "hit" if cached_payload is not None else "miss"
        if cached_payload is None:
            payload = load_morpion_search_checkpoint_payload(tree_snapshot_path)
            bytes_loaded = tree_snapshot_path.stat().st_size
        else:
            payload, bytes_loaded = cached_payload
            del cached_payload
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
            gc.collect()
            rss_after_release_mb = _current_rss_mb()
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
        write_stats = write_checkpoint_json_payload(payload, output)
        log_morpion_checkpoint_memory_phase(
            "after_checkpoint_save_write",
            path=output,
            nodes=node_count,
            generation=_generation_from_checkpoint_path(output),
        )
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
        _log_checkpoint_metrics(
            "save",
            CheckpointIoMetrics(
                path=str(write_stats.output_path),
                bytes=write_stats.compressed_bytes,
                file_format=write_stats.file_format,
                encoder=write_stats.encoder,
                payload_build_s=payload_elapsed_s,
                jsonable_s=write_stats.jsonable_s,
                json_encode_s=write_stats.json_encode_s,
                compress_s=write_stats.compress_s,
                write_s=write_stats.write_s,
                total_s=elapsed_s,
                uncompressed_bytes=write_stats.uncompressed_bytes,
                compression_ratio=write_stats.compression_ratio,
                rss_before_mb=rss_before_mb,
                rss_after_mb=rss_after_mb,
                node_count=node_count,
                anchor_count=anchor_count,
                delta_count=delta_count,
            ),
        )
        LOGGER.info(
            "[checkpoint] save_done path=%s elapsed=%.3fs",
            str(write_stats.output_path),
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
        raw_payload, read_stats = load_checkpoint_json_payload(resolved_path)
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
]
