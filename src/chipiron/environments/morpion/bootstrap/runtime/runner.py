"""Real Anemone-backed Morpion search runner for bootstrap cycles."""

from __future__ import annotations

import gc
import logging
import os
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    CheckpointAtomPayload,
    DeltaCheckpointStatePayload,
    RestoreMemoryPhaseLogger,
    build_search_checkpoint_payload,
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
from anemone.value_updates import NodeValueUpdate, NodeValueUpdateResult
from atomheart.games.morpion import initial_state
from valanga.evaluations import Certainty, Value

from chipiron.environments.morpion.bootstrap.config import (
    DEFAULT_MORPION_TREE_BRANCH_LIMIT,
    MorpionBootstrapRolloutConfig,
)
from chipiron.environments.morpion.bootstrap.control import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.cycle_timing import (
    timestamp_utc_from_unix_s as _timestamp_utc_from_unix_s,
)
from chipiron.environments.morpion.bootstrap.history import MorpionBootstrapTreeStatus
from chipiron.environments.morpion.bootstrap.linoo_selection_table import (
    linoo_selection_table_from_report,
    save_linoo_selection_table,
)
from chipiron.environments.morpion.bootstrap.pipeline_memory import (
    log_pipeline_memory,
)
from chipiron.environments.morpion.bootstrap.search_runner_protocol import (
    MorpionSearchRunner,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    save_morpion_sharded_training_tree_from_live_nodes,
)
from chipiron.environments.morpion.players.evaluators.morpion_state_evaluator import (
    MorpionMasterEvaluator,
    MorpionOverEventDetector,
    MorpionStateEvaluator,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MorpionGraphTokenConverter,
    is_morpion_entity_token_transformer_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState

from . import checkpoint_io as _checkpoint_io
from .checkpoint_codec import (
    ChipironMorpionStateCheckpointCodec,
    InvalidMorpionSearchCheckpointError,
    generation_from_checkpoint_path,
    load_morpion_search_checkpoint_payload,
    new_morpion_state_checkpoint_codec,
)
from .checkpoint_io import (
    CheckpointIoMetrics,
    _checkpoint_artifact_bytes,
    _checkpoint_node_counts,
    _is_sharded_runtime_checkpoint_path,
    _log_checkpoint_metrics,
    _metric_value,
    _pop_cached_morpion_search_checkpoint_payload_for_restore,
    cache_morpion_search_checkpoint_payload_for_restore,
    checkpoint_io_metrics_to_dict,
)
from .restore_memory_logging import (
    RestoreMemoryLogger,
    current_rss_mb,
    log_morpion_checkpoint_memory_phase,
    restore_memory_logger_for_checkpoint_path,
)
from .rollout_logging import (
    _log_latest_rollout_report,
    _log_search_rollout_config,
    _opening_expansion_config_from_rollout,
    _opening_expansion_kind_name,
    _opening_type_name,
)
from .selection_logging import (
    checkpoint_selector_state_fields,
    format_linoo_selection_depth_table,
    format_mapping_metric,
    format_optional_int_log,
    format_selected_metric,
    linoo_depth_active_column,
    resolve_selected_int,
    selector_growth_diagnostic_fields,
    selector_heap_detail_fields,
    selector_report_row_count,
)
from .state_eviction import (
    MorpionGrowthStateEvictionMetrics,
    _dump_live_state_parent_branch_for_checkpoint,
    _effective_growth_state_eviction_policy,
    _LiveCompactStateResolver,
    _LiveEvictionPayload,
    _phase_delta,
    _single_parent_link_for_live_delta,
)
from .training_export_profile import (
    MorpionTrainingExportProfile,
    format_optional_seconds,
    format_optional_seconds_with_unit,
    log_sharded_training_export_stats,
    log_training_export_profile,
    sharded_training_export_stats_to_dict,
    training_export_profile_to_dict,
    value_to_scalar,
)

if TYPE_CHECKING:
    from anemone.checkpoints._protocols import CheckpointStateSummary

    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionReevaluationPatch,
        MorpionReevaluationPatchRow,
    )

LOGGER = logging.getLogger(__name__)
_GROWTH_STEP_SEPARATOR = (
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)
_VERBOSE_SELECTION_TABLE_ENV = "MORPION_VERBOSE_SELECTION_TABLE"


_TREE_BRANCH_LIMIT_ARGS_REQUIRED_MESSAGE = (
    "Morpion bootstrap runtime reconfiguration currently supports only "
    "TreeBranchLimitArgs stopping criteria."
)
_LIVE_TREE_BRANCH_LIMIT_REQUIRED_MESSAGE = (
    "Morpion bootstrap runtime reconfiguration currently supports only "
    "tree-branch-limit stopping criteria on the live runtime."
)
_LIVE_TREE_REQUIRED_MESSAGE = "Anemone runtime must expose a live tree."


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


def _env_flag_enabled(name: str) -> bool:
    """Return whether an operator-facing boolean env flag is enabled."""
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


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


def default_search_args(
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


_default_search_args = default_search_args


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


class MorpionStateToTensorConverter(Protocol):
    """Minimal interface shared by Morpion neural input converters."""

    def state_to_tensor(self, state: MorpionState) -> object:
        """Convert one Morpion state to the model input tensor."""
        ...


class _MorpionRegressor(Protocol):
    """Callable neural regressor loaded from a Morpion model bundle."""

    def __call__(self, tensor: object) -> Any:
        """Return the raw model output for one converted state."""


@dataclass(frozen=True, slots=True)
class AnemoneMorpionSearchRunnerArgs:
    """Configuration for the real Anemone-backed Morpion runner."""

    search_args: SearchArgs = field(default_factory=default_search_args)
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
        tensor = self.input_converter.state_to_tensor(  # pylint: disable=assignment-from-no-return
            morpion_state
        )
        regressor = cast(_MorpionRegressor, self.regressor)  # noqa: TC006
        raw_output = regressor(tensor)  # pylint: disable=not-callable
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
    input_converter: MorpionStateToTensorConverter
    if is_morpion_entity_token_transformer_model_kind(model_args.model_kind):
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
        self._state_codec = ChipironMorpionStateCheckpointCodec(
            inner=new_morpion_state_checkpoint_codec(profile_checkpoint=True),
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

    def apply_effective_runtime_config(
        self,
        runtime_config: MorpionBootstrapEffectiveRuntimeConfig,
    ) -> None:
        """Apply a supported runtime config to the loaded live runtime."""
        runtime = self._require_runtime()
        _apply_runtime_config_to_runtime(runtime, runtime_config)
        self._last_applied_runtime_config = runtime_config

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
                selector_report_rows = selector_report_row_count(selector_report)
            selector_diagnostics = selector_growth_diagnostic_fields(selector_report)
            LOGGER.debug(
                "[growth-timing] step=%s total_s=%s select_s=%s limit_s=%s expand_s=%s evaluate_s=%s propagate_s=%s selector_total_s=%s selector_collect_s=%s selector_choose_depth_s=%s selector_heap_update_s=%s selector_choose_node_s=%s selector_report_s=%s rows=%s nodes_scanned=%s frontier_scanned=%s selected_depth_frontier_count=%s heap_registered=%s stale_skipped=%s selector_state_rebuilt=%s selector_nodes_incrementally_updated=%s selector_total_nodes_scanned=%s selector_frontier_nodes_scanned=%s",
                steps_executed,
                format_optional_seconds(
                    getattr(step_report, "total_s", None)
                    if step_report is not None
                    else None
                ),
                format_optional_seconds(
                    getattr(step_report, "select_s", None)
                    if step_report is not None
                    else None
                ),
                format_optional_seconds(
                    getattr(step_report, "limit_s", None)
                    if step_report is not None
                    else None
                ),
                format_optional_seconds(
                    getattr(step_report, "expand_s", None)
                    if step_report is not None
                    else None
                ),
                format_optional_seconds(
                    getattr(step_report, "evaluate_s", None)
                    if step_report is not None
                    else None
                ),
                format_optional_seconds(
                    getattr(step_report, "propagate_s", None)
                    if step_report is not None
                    else None
                ),
                format_optional_seconds(getattr(selector_report, "total_s", None)),
                format_optional_seconds(
                    getattr(selector_report, "collect_frontier_state_s", None)
                ),
                format_optional_seconds(
                    getattr(selector_report, "choose_depth_s", None)
                ),
                format_optional_seconds(
                    getattr(selector_report, "heap_update_s", None)
                ),
                format_optional_seconds(
                    getattr(selector_report, "choose_node_s", None)
                ),
                format_optional_seconds(
                    getattr(selector_report, "make_report_s", None)
                ),
                format_optional_int_log(selector_report_rows),
                format_optional_int_log(
                    getattr(selector_report, "total_nodes_scanned", None)
                ),
                format_optional_int_log(
                    getattr(selector_report, "frontier_nodes_scanned", None)
                ),
                format_optional_int_log(
                    getattr(selector_report, "selected_depth_frontier_count", None)
                ),
                format_optional_int_log(
                    getattr(selector_report, "heap_candidates_registered", None)
                ),
                format_optional_int_log(
                    getattr(selector_report, "stale_candidates_skipped", None)
                ),
                _metric_value(selector_diagnostics["selector_state_rebuilt"]),
                _metric_value(
                    selector_diagnostics["selector_nodes_incrementally_updated"]
                ),
                _metric_value(selector_diagnostics["selector_total_nodes_scanned"]),
                _metric_value(selector_diagnostics["selector_frontier_nodes_scanned"]),
            )
            selector_heap_details = selector_heap_detail_fields(selector_report)
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
            LOGGER.debug(
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
            rollout_summary = _log_latest_rollout_report(
                runtime,
                step=steps_executed,
            )
            self._log_growth_step_summary(
                step=steps_executed,
                step_report=step_report,
                selector_report=selector_report,
                selector_report_rows=selector_report_rows,
                selector_diagnostics=selector_diagnostics,
                current_tree_size=current_tree_size,
                initial_tree_size=initial_tree_size,
                branch_count=branch_count,
                selected_node_id=selected_node_id,
                selected_depth=selected_depth,
                rollout_summary=rollout_summary,
            )
            if step_report is not None:
                self._log_and_persist_linoo_selection_table(
                    step_report=step_report,
                    step=steps_executed,
                    selected_depth=selected_depth,
                    selected_node_id=selected_node_id,
                )
            LOGGER.info(_GROWTH_STEP_SEPARATOR)
            LOGGER.debug(
                "[growth] step=%s node_count=%s nodes_added=%s branch_count=%s",
                steps_executed,
                current_tree_size,
                current_tree_size - initial_tree_size,
                branch_count if isinstance(branch_count, int) else "unknown",
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

    def _log_growth_step_summary(
        self,
        *,
        step: int,
        step_report: object | None,
        selector_report: object | None,
        selector_report_rows: int | None,
        selector_diagnostics: Mapping[str, object],
        current_tree_size: int,
        initial_tree_size: int,
        branch_count: object,
        selected_node_id: object,
        selected_depth: object,
        rollout_summary: object | None,
    ) -> None:
        """Emit the compact human-facing growth summary block."""
        selected_depth_frontier = getattr(
            selector_report,
            "selected_depth_frontier_count",
            None,
        )
        LOGGER.info(_GROWTH_STEP_SEPARATOR)
        LOGGER.info(
            "[growth-step] step=%s selected_depth=%s selected_node_id=%s "
            "mode=%s depth_policy=%s subpolicy=%s step_parity=%s",
            step,
            format_selected_metric(selected_depth),
            format_selected_metric(selected_node_id),
            _selector_family_name(self._args.search_args),
            _metric_value(getattr(selector_report, "depth_selection_policy", None)),
            _metric_value(getattr(selector_report, "depth_selection_subpolicy", None)),
            _metric_value(
                getattr(selector_report, "depth_selection_step_parity", None)
            ),
        )
        LOGGER.info(
            "[growth-step] step=%s tree nodes=%s branches=%s nodes_added_total=%s "
            "selected_depth_frontier_count=%s",
            step,
            current_tree_size,
            branch_count if isinstance(branch_count, int) else "unknown",
            current_tree_size - initial_tree_size,
            format_optional_int_log(selected_depth_frontier),
        )
        LOGGER.info(
            "[growth-step] step=%s timing total=%s select=%s expand=%s evaluate=%s "
            "propagate=%s",
            step,
            format_optional_seconds_with_unit(getattr(step_report, "total_s", None)),
            format_optional_seconds_with_unit(getattr(step_report, "select_s", None)),
            format_optional_seconds_with_unit(getattr(step_report, "expand_s", None)),
            format_optional_seconds_with_unit(getattr(step_report, "evaluate_s", None)),
            format_optional_seconds_with_unit(
                getattr(step_report, "propagate_s", None)
            ),
        )
        LOGGER.info(
            "[growth-step] step=%s selector rows=%s scanned_nodes=%s scanned_frontier=%s "
            "heap_candidates=%s stale_skipped=%s rebuilt=%s",
            step,
            format_optional_int_log(selector_report_rows),
            format_optional_int_log(
                getattr(selector_report, "total_nodes_scanned", None)
            ),
            format_optional_int_log(
                getattr(selector_report, "frontier_nodes_scanned", None)
            ),
            format_optional_int_log(
                getattr(selector_report, "heap_candidates_registered", None)
            ),
            format_optional_int_log(
                getattr(selector_report, "stale_candidates_skipped", None)
            ),
            _metric_value(selector_diagnostics["selector_state_rebuilt"]),
        )
        LOGGER.info(
            "[growth-step] step=%s rollout enabled=%s paths=%s total_edges=%s "
            "initial_edges=%s extra_edges=%s traversals=%s start_depth=%s "
            "end_depth=%s depth_delta=%s stops=%s",
            step,
            rollout_summary is not None,
            _metric_value(getattr(rollout_summary, "paths", None)),
            _metric_value(getattr(rollout_summary, "total_edges", None)),
            _metric_value(getattr(rollout_summary, "initial_edges", None)),
            _metric_value(getattr(rollout_summary, "extra_edges", None)),
            _metric_value(getattr(rollout_summary, "traversals", None)),
            _metric_value(getattr(rollout_summary, "start_depth", None)),
            _metric_value(getattr(rollout_summary, "end_depth", None)),
            _metric_value(getattr(rollout_summary, "depth_delta", None)),
            format_mapping_metric(getattr(rollout_summary, "stops", None)),
        )

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
        cast(  # pylint: disable=protected-access
            "Any", runtime
        )._diagnostic_phase_context = self._state_rematerialization_phase
        self._install_selector_diagnostic_phase_context(
            getattr(runtime, "node_selector", None)
        )
        cast("Any", runtime)._chipiron_rematerialization_phase_hooks = True  # pylint: disable=protected-access

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
        except (AttributeError, TypeError):
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
                raw_nodes = all_nodes_in_tree_order()
            except (AttributeError, RuntimeError, TypeError):
                self._state_eviction_metrics.skip("scan_nodes_failed")
                return iter(())
            if isinstance(raw_nodes, Sequence):
                nodes: Sequence[object] = raw_nodes
            elif isinstance(raw_nodes, Iterable):
                nodes = tuple(raw_nodes)
            else:
                self._state_eviction_metrics.skip("scan_nodes_failed")
                return iter(())
        else:
            nodes = tuple(self.iter_profile_nodes())
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
                    state=state,
                )
        except (AttributeError, RuntimeError, TypeError, ValueError):
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
        state: MorpionState,
    ) -> _LiveEvictionPayload:
        """Build a bounded live compact payload, preferring safe deltas."""
        state_summary = self._state_codec.dump_state_summary(state)
        if self._args.growth_state_eviction_payload_mode == "delta_when_safe":
            self._state_eviction_metrics.record_delta_payload_attempt()
            delta_payload = self._try_build_live_delta_eviction_payload(
                node=node,
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
                state_summary=cast("CheckpointStateSummary | None", state_summary),
            ),
            kind="anchor",
            chain_depth=0,
        )

    def _try_build_live_delta_eviction_payload(
        self,
        *,
        node: object,
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
        except (AttributeError, RuntimeError, TypeError, ValueError):
            self._state_eviction_metrics.record_delta_payload_fallback(
                "delta_payload_build_failed"
            )
            return None

        return _LiveEvictionPayload(
            payload=DeltaCheckpointStatePayload(
                state_parent_node_id=parent_context.parent_node_id,
                state_parent_branch=cast(
                    CheckpointAtomPayload | None,  # noqa: TC006
                    branch_payload,
                ),
                delta_ref=delta_ref,
                state_summary=cast("CheckpointStateSummary | None", state_summary),
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
        row_count = selector_report_row_count(selector_report)
        verbose_selection_table = _env_flag_enabled(_VERBOSE_SELECTION_TABLE_ENV)
        resolved_selected_depth = resolve_selected_int(
            explicit_value=selected_depth,
            report=selector_report,
            report_attribute="selected_depth",
        )
        resolved_selected_node_id = resolve_selected_int(
            explicit_value=selected_node_id,
            report=selector_report,
            report_attribute="selected_node_id",
        )
        LOGGER.debug(
            "[growth-selection] step=%s selected_depth=%s selected_node_id=%s "
            "selected_depth_frontier_count=%s probability=%s depth_policy=%s "
            "subpolicy=%s step_parity=%s",
            step,
            format_selected_metric(resolved_selected_depth),
            format_selected_metric(resolved_selected_node_id),
            format_optional_int_log(
                getattr(selector_report, "selected_depth_frontier_count", None)
            ),
            _metric_value(
                getattr(selector_report, "selected_depth_selection_probability", None)
            ),
            _metric_value(getattr(selector_report, "depth_selection_policy", None)),
            _metric_value(getattr(selector_report, "depth_selection_subpolicy", None)),
            _metric_value(
                getattr(selector_report, "depth_selection_step_parity", None)
            ),
        )
        formatted_table: str | None = None
        format_elapsed_s: float | None = None
        log_elapsed_s: float | None = None
        format_started_at = time.perf_counter()
        formatted_table = format_linoo_selection_depth_table(
            selector_report=selector_report,
            selected_depth=resolved_selected_depth,
        )
        format_elapsed_s = time.perf_counter() - format_started_at
        if formatted_table is not None:
            log_started_at = time.perf_counter()
            LOGGER.info(
                "[growth-selection-table] step=%s selected_depth=%s "
                "selected_node_id=%s depth_policy=%s subpolicy=%s step_parity=%s "
                "active_column=%s\n%s",
                step,
                format_selected_metric(resolved_selected_depth),
                format_selected_metric(resolved_selected_node_id),
                _metric_value(getattr(selector_report, "depth_selection_policy", None)),
                _metric_value(
                    getattr(selector_report, "depth_selection_subpolicy", None)
                ),
                _metric_value(
                    getattr(selector_report, "depth_selection_step_parity", None)
                ),
                linoo_depth_active_column(selector_report),
                formatted_table,
            )
            log_elapsed_s = time.perf_counter() - log_started_at
        timing_logger = LOGGER.info if verbose_selection_table else LOGGER.debug
        timing_logger(
            "[growth-selection-table-timing] step=%s rows=%s format_s=%s log_s=%s",
            step,
            format_optional_int_log(row_count),
            format_optional_seconds(format_elapsed_s),
            format_optional_seconds(log_elapsed_s),
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
        ordered_nodes = runtime._all_nodes_in_tree_order()  # pylint: disable=protected-access
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
            direct_value_extractor=value_to_scalar,
            backed_up_value_extractor=value_to_scalar,
            profile=profile,
        )
        profile.payload_build_s = time.perf_counter() - started_at
        log_sharded_training_export_stats(stats)
        log_training_export_profile(profile)
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
        ordered_nodes = runtime._all_nodes_in_tree_order()  # pylint: disable=protected-access
        profile = MorpionTrainingExportProfile()
        started_at = perf_counter()

        def state_ref_dumper(state: object) -> object:
            return self._state_codec.dump_state_ref(cast("MorpionState", state))

        snapshot = build_training_tree_snapshot(
            ordered_nodes,
            root_node_id=str(runtime.tree.root_node.id),
            state_ref_dumper=state_ref_dumper,
            direct_value_extractor=value_to_scalar,
            backed_up_value_extractor=value_to_scalar,
            profile=profile,
        )
        profile.payload_build_s = perf_counter() - started_at
        log_training_export_profile(profile)
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
                raw_nodes = all_nodes_in_tree_order()  # pylint: disable=not-callable
            except (AttributeError, RuntimeError, TypeError):
                return iter(())
            if isinstance(raw_nodes, Iterable):
                return iter(raw_nodes)
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
                    child_links = iter_child_links()
                except (AttributeError, RuntimeError, TypeError):
                    pass
                else:
                    if isinstance(child_links, Iterable):
                        for child_link in child_links:
                            if isinstance(child_link, tuple) and child_link:
                                yield child_link[0]
                        continue

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
                    yield from container

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
        restore_memory_logger = restore_memory_logger_for_checkpoint_path(
            tree_snapshot_path,
            enabled=self._args.restore_memory_profile,
            recursive_enabled=self._args.restore_memory_profile_recursive,
            recursive_max_objects=self._args.restore_memory_profile_recursive_max_objects,
            recursive_max_depth=self._args.restore_memory_profile_recursive_max_depth,
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
        restore_selector_state_fields = checkpoint_selector_state_fields(
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
        rss_before_mb = current_rss_mb()
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
                            RestoreMemoryPhaseLogger,  # noqa: TC006
                            restore_memory_logger.callback,
                        )
                    ),
                ),
            )
            runtime_elapsed_s = time.perf_counter() - runtime_started_at
            rss_after_rebuild_mb = current_rss_mb()
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
            rss_after_release_mb = current_rss_mb()
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
            rss_after_rebuild_mb = current_rss_mb()
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
        restore_memory_logger: RestoreMemoryLogger | None,
    ) -> object:
        """Restore one live runtime from an opt-in sharded checkpoint directory."""
        manifest = read_sharded_checkpoint_manifest(
            tree_snapshot_path / "manifest.json"
        )
        node_count = manifest.total_node_count
        branch_count = manifest.total_branch_count
        rss_before_mb = current_rss_mb()
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
                        RestoreMemoryPhaseLogger,  # noqa: TC006
                        restore_memory_logger.callback,
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
        rss_after_release_mb = current_rss_mb()
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
        rss_before_mb = current_rss_mb()
        save_started_at = time.perf_counter()
        payload_started_at = time.perf_counter()
        log_morpion_checkpoint_memory_phase(
            "before_checkpoint_save_payload_build",
            path=output,
            generation=generation_from_checkpoint_path(output),
        )
        with self._state_rematerialization_phase("checkpoint"):
            payload = build_search_checkpoint_payload(
                runtime,
                state_codec=self._state_codec,
            )
        checkpoint_selector_state = checkpoint_selector_state_fields(
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
                checkpoint_selector_state["checkpoint_selector_state_present"]
            ),
            _metric_value(checkpoint_selector_state["checkpoint_selector_state_type"]),
            _metric_value(
                checkpoint_selector_state["checkpoint_selector_state_version"]
            ),
        )
        node_count, anchor_count, delta_count = _checkpoint_node_counts(payload)
        log_morpion_checkpoint_memory_phase(
            "after_checkpoint_save_payload_build",
            path=output,
            nodes=node_count,
            generation=generation_from_checkpoint_path(output),
        )
        manifest = None
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
            generation=generation_from_checkpoint_path(output),
        )
        if write_stats is None:
            assert manifest is not None
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
        rss_after_mb = current_rss_mb()
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
            generation=generation_from_checkpoint_path(output),
        )


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
        for node in runtime._all_nodes_in_tree_order()  # pylint: disable=protected-access
        if bool(getattr(node, "all_branches_generated", False))
    )


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
    nodes_by_id = runtime._nodes_by_public_id()  # pylint: disable=protected-access
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
        changed = bool(
            runtime._apply_node_value_update(  # pylint: disable=protected-access
                node=live_node,
                update=update,
            )
        )
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
    return len(runtime._all_nodes_in_tree_order())  # pylint: disable=protected-access


def _optional_live_tree_node_count(runtime: Any) -> int | None:
    """Return live node count when available without affecting caller behavior."""
    try:
        return _live_tree_node_count(runtime)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _safe_int_attr(node: Any, attribute_name: str) -> int | None:
    """Read one integer node attribute when the runtime exposes it."""
    value = getattr(node, attribute_name, None)
    return value if isinstance(value, int) else None


def run_morpion_growth_search_once(
    *,
    runner_args: AnemoneMorpionSearchRunnerArgs | None = None,
    tree_snapshot_path: str | Path | None = None,
    model_bundle_path: str | Path | None = None,
    effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
    max_growth_steps: int,
    reevaluate_tree: bool = False,
) -> AnemoneMorpionSearchRunner:
    """Create/load a runner, grow it once, and return the live runner."""
    runner = AnemoneMorpionSearchRunner(runner_args)
    runner.load_or_create(
        tree_snapshot_path=tree_snapshot_path,
        model_bundle_path=model_bundle_path,
        effective_runtime_config=effective_runtime_config,
        reevaluate_tree=reevaluate_tree,
    )
    runner.grow(max_growth_steps)
    return runner


def __getattr__(name: str) -> object:
    """Expose selected runtime checkpoint internals kept in checkpoint_io."""
    if name == "_validated_checkpoint_payload_cache":
        return _checkpoint_io._validated_checkpoint_payload_cache.entry  # pylint: disable=protected-access
    raise AttributeError(name)


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
    "run_morpion_growth_search_once",
]
