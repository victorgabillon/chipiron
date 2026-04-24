"""Real Anemone-backed Morpion search runner for bootstrap cycles."""

from __future__ import annotations

import json
import inspect
import logging
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from random import Random
from typing import Any, cast

from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    CheckpointNodeStatePayload,
    DeltaCheckpointStatePayload,
    SearchRuntimeCheckpointPayload,
    build_search_checkpoint_payload,
    load_search_from_checkpoint_payload,
)
from anemone.factory import (
    SearchArgs,
    create_tree_and_value_exploration_with_tree_eval_factory,
)
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.linoo import LinooArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_evaluation.tree.single_agent.factory import (
    NodeMaxEvaluationFactory,
)
from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
from anemone.progress_monitor.progress_monitor import TreeBranchLimit
from anemone.recommender_rule.recommender_rule import AlmostEqualLogistic
from anemone.training_export import (
    build_training_tree_snapshot,
    save_training_tree_snapshot,
)
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
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState

from .bootstrap_loop import MorpionSearchRunner
from .config import DEFAULT_MORPION_TREE_BRANCH_LIMIT
from .control import MorpionBootstrapEffectiveRuntimeConfig
from .history import MorpionBootstrapTreeStatus

LOGGER = logging.getLogger(__name__)
_CHECKPOINT_PAYLOAD_CACHE: dict[
    Path, tuple[int, int, SearchRuntimeCheckpointPayload]
] = {}

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency for diagnostics only.
    psutil = None


def _get_process_rss_mb() -> float | None:
    """Return the current process RSS in MB when psutil or resource is available."""
    if psutil is not None:
        process = psutil.Process()
        rss_bytes = process.memory_info().rss
        return rss_bytes / (1024 * 1024)
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss_kb <= 0:
        return None
    return rss_kb / 1024


def _log_mem(logger: logging.Logger, tag: str) -> None:
    """Log the current process RSS in MB when available."""
    rss_mb = _get_process_rss_mb()
    if rss_mb is None:
        logger.info("[mem] %s rss_mb=unavailable", tag)
        return
    logger.info("[mem] %s rss_mb=%.3f", tag, rss_mb)


def _checkpoint_load_caller_summary() -> str:
    """Return a concise caller summary for checkpoint load diagnostics."""
    current_frame = inspect.currentframe()
    if current_frame is None:
        return "unknown"
    frame = current_frame.f_back
    this_module = __name__
    while frame is not None:
        module_name = frame.f_globals.get("__name__", "unknown")
        if module_name != this_module:
            return (
                f"{module_name}:{frame.f_code.co_name}:{frame.f_lineno}"
            )
        frame = frame.f_back
    return "unknown"


@contextmanager
def _instrument_checkpoint_runtime_rebuild() -> Any:
    """Temporarily wrap Anemone checkpoint restore phases with timing logs."""
    from anemone.checkpoints import load as checkpoint_load_module

    phase_names = {
        "assemble_search_runtime_dependencies": "dependency_assembly",
        "_validate_payload": "payload_validate",
        "_load_states": "state_restore",
        "_create_nodes": "node_create",
        "_link_nodes": "edge_link",
        "_restore_node_runtime_state": "node_runtime_restore",
        "_build_tree": "tree_build",
        "create_stopping_criterion": "stopping_criterion_create",
        "TreeExploration": "runtime_object_create",
        "_restore_runtime_state": "runtime_state_restore",
        "_restore_tree_expansions": "tree_expansions_restore",
        "_restore_inferred_depth_selector_state": "selector_state_restore",
    }
    original_values: dict[str, object] = {}

    def _make_wrapper(original: object, phase_name: str) -> Any:
        def _wrapped(*args: object, **kwargs: object) -> object:
            LOGGER.info("[runtime_rebuild] %s_start", phase_name)
            _log_mem(LOGGER, f"before runtime_rebuild.{phase_name}")
            started_at = time.perf_counter()
            result = cast("Any", original)(*args, **kwargs)
            elapsed_s = time.perf_counter() - started_at
            LOGGER.info(
                "[runtime_rebuild] %s_done elapsed=%.3fs",
                phase_name,
                elapsed_s,
            )
            _log_mem(LOGGER, f"after runtime_rebuild.{phase_name}")
            return result

        return _wrapped

    try:
        for attribute_name, phase_name in phase_names.items():
            if not hasattr(checkpoint_load_module, attribute_name):
                continue
            original_value = getattr(checkpoint_load_module, attribute_name)
            original_values[attribute_name] = original_value
            setattr(
                checkpoint_load_module,
                attribute_name,
                _make_wrapper(original_value, phase_name),
            )
        yield
    finally:
        for attribute_name, original_value in original_values.items():
            setattr(checkpoint_load_module, attribute_name, original_value)


def _default_search_args() -> SearchArgs:
    """Build the default Morpion tree-search args used by the bootstrap runner."""
    return SearchArgs(
        node_selector=ComposedNodeSelectorArgs(
            type="Composed",
            priority=NoPriorityCheckArgs(type="PriorityNoop"),
            base=LinooArgs(type=NodeSelectorType.LINOO),
        ),
        opening_type=OpeningType.ALL_CHILDREN,
        recommender_rule=AlmostEqualLogistic(
            type="almost_equal_logistic",
            temperature=1.0,
        ),
        stopping_criterion=TreeBranchLimitArgs(
            type="tree_branch_limit",
            tree_branch_limit=DEFAULT_MORPION_TREE_BRANCH_LIMIT,
        ),
    )


class UninitializedMorpionSearchRunnerError(RuntimeError):
    """Raised when a runner method requires a live runtime that does not exist."""

    def __init__(self) -> None:
        """Initialize the missing-runtime error."""
        super().__init__(
            "AnemoneMorpionSearchRunner has no live runtime. Call load_or_create() first."
        )


class InvalidMorpionSearchCheckpointError(ValueError):
    """Raised when a persisted Anemone checkpoint payload is invalid."""

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize the checkpoint validation error."""
        super().__init__(
            f"Invalid Morpion search checkpoint at {path!s}: {reason}"
        )


@dataclass(slots=True)
class _ChipironMorpionStateCheckpointCodec:
    """Minimal adapter from atomheart checkpoints to Chipiron Morpion states.

    Chipiron still runs its search on ``chipiron.environments.morpion.types.
    MorpionState`` while Atomheart's incremental codec operates on the lower
    layer ``atomheart.games.morpion.MorpionState``. This bridge keeps the
    active runtime on the new lower-layer checkpoint architecture without
    reimplementing restore mechanics locally.
    """

    inner: MorpionStateCheckpointCodec
    dynamics: MorpionDynamics

    def dump_state_ref(self, state: MorpionState) -> object:
        """Serialize one legacy state-ref payload through the atomheart codec."""
        return self.inner.dump_state_ref(state.to_atomheart_state())

    def load_state_ref(self, payload: object) -> MorpionState:
        """Restore one legacy state-ref payload into Chipiron state form."""
        return self.dynamics.wrap_atomheart_state(self.inner.load_state_ref(payload))

    def dump_anchor_ref(self, state: MorpionState) -> object:
        """Serialize one full anchor snapshot for the incremental checkpoint path."""
        return self.inner.dump_anchor_ref(state.to_atomheart_state())

    def dump_delta_from_parent(
        self,
        *,
        parent_state: MorpionState,
        child_state: MorpionState,
        branch_from_parent: object | None = None,
    ) -> object:
        """Serialize one child state as a delta from its parent."""
        return self.inner.dump_delta_from_parent(
            parent_state=parent_state.to_atomheart_state(),
            child_state=child_state.to_atomheart_state(),
            branch_from_parent=branch_from_parent,
        )

    def load_anchor_ref(self, payload: object) -> MorpionState:
        """Restore one anchor snapshot through Atomheart, then wrap it for Chipiron."""
        return self.dynamics.wrap_atomheart_state(self.inner.load_anchor_ref(payload))

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
                branch_from_parent=branch_from_parent,
            )
        )

    def dump_state_summary(self, state: MorpionState) -> object:
        """Serialize optional lightweight checkpoint summary metadata."""
        return self.inner.dump_state_summary(state.to_atomheart_state())


@dataclass(frozen=True, slots=True)
class AnemoneMorpionSearchRunnerArgs:
    """Configuration for the real Anemone-backed Morpion runner."""

    search_args: SearchArgs = field(default_factory=_default_search_args)
    random_seed: int = 0
    reevaluation_scope: str = "leaves"


@dataclass(frozen=True, slots=True)
class MorpionRegressorMasterEvaluator(MorpionMasterEvaluator):
    """Anemone-compatible Morpion evaluator backed by a saved regressor bundle."""

    feature_converter: MorpionFeatureTensorConverter
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
        tensor = self.feature_converter.state_to_tensor(morpion_state)
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
    return MorpionRegressorMasterEvaluator(
        evaluator=MorpionStateEvaluator(),
        over=over_detector,
        over_detector=over_detector,
        feature_converter=MorpionFeatureTensorConverter(
            dynamics=MorpionDynamics(),
            feature_subset=model_args.feature_subset,
        ),
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
            inner=MorpionStateCheckpointCodec(),
            dynamics=self._dynamics,
        )
        self._current_evaluator_bundle_path: Path | None = None
        self._last_applied_runtime_config = _runtime_config_from_search_args(
            self._args.search_args
        )

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
        effective_runtime_config: MorpionBootstrapEffectiveRuntimeConfig | None = None,
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
            "[search] selector=%s opening_type=%s",
            _selector_family_name(self._args.search_args),
            self._args.search_args.opening_type,
        )
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
                LOGGER.info("[reeval] skipped reason=fresh_runtime_attach")
                self._set_runtime_evaluator_from_bundle(
                    resolved_bundle_path,
                    reevaluate_tree=False,
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
            LOGGER.info("[reeval] skipped reason=resume_restore")
            self._set_runtime_evaluator_from_bundle(
                resolved_bundle_path,
                reevaluate_tree=False,
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
        steps_executed = 0
        stop_reason = "max_steps_reached"
        for step_index in range(max_growth_steps):
            if runtime.tree.root_node.tree_evaluation.has_exact_value():
                stop_reason = "exact_solution_found"
                break
            if not _runtime_can_step(runtime):
                stop_reason = _runtime_stop_reason(runtime)
                break
            runtime.step()
            steps_executed += 1
            current_tree_size = _live_tree_node_count(runtime)
            tree = getattr(runtime, "tree", None)
            branch_count = getattr(tree, "branch_count", None)
            node_selector = getattr(runtime, "node_selector", None)
            uniform_selector = getattr(node_selector, "base", node_selector)
            selected_depth = getattr(uniform_selector, "current_depth_to_expand", None)
            LOGGER.info(
                "[growth] step=%s node_count=%s nodes_added=%s branch_count=%s selected_depth=%s",
                steps_executed,
                current_tree_size,
                current_tree_size - initial_tree_size,
                branch_count if isinstance(branch_count, int) else "unknown",
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

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Persist a training-grade snapshot from the live tree."""
        runtime = self._require_runtime()
        ordered_nodes = runtime._all_nodes_in_tree_order()
        LOGGER.info(
            "[save] tree_export_start output=%s nodes=%s",
            str(output_path),
            len(ordered_nodes),
        )
        started_at = time.perf_counter()
        def _value_to_scalar(value):
            if value is None:
                return None
            # anemone Value object: has .score
            return getattr(value, "score", None)

        snapshot = build_training_tree_snapshot(
            ordered_nodes,
            root_node_id=str(runtime.tree.root_node.id),
            state_ref_dumper=self._state_codec.dump_state_ref,
            direct_value_extractor=_value_to_scalar,
            backed_up_value_extractor=_value_to_scalar,
        )
        save_training_tree_snapshot(snapshot, output_path)
        elapsed_s = time.perf_counter() - started_at
        LOGGER.info(
            "[save] tree_export_done output=%s elapsed=%.3fs",
            str(output_path),
            elapsed_s,
        )

    def current_tree_size(self) -> int:
        """Return the number of nodes currently tracked by the live runtime."""
        runtime = self._require_runtime()
        return _live_tree_node_count(runtime)

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
        return create_tree_and_value_exploration_with_tree_eval_factory(
            state_type=MorpionState,
            dynamics=self._dynamics,
            starting_state=self._dynamics.wrap_atomheart_state(initial_state()),
            args=search_args,
            random_generator=self._random_generator,
            master_state_evaluator=evaluator,
            state_representation_factory=None,
            node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        )

    def _load_runtime_from_checkpoint(
        self,
        tree_snapshot_path: Path,
        *,
        search_args: SearchArgs,
    ) -> object:
        """Restore one live runtime from a persisted checkpoint JSON file."""
        LOGGER.info("[checkpoint] load_start path=%s", str(tree_snapshot_path))
        _log_mem(LOGGER, "before total_load")
        started_at = time.perf_counter()
        LOGGER.info("[checkpoint] payload_read_start path=%s", str(tree_snapshot_path))
        _log_mem(LOGGER, "before payload_read")
        payload_started_at = time.perf_counter()
        payload = load_morpion_search_checkpoint_payload(tree_snapshot_path)
        payload_elapsed_s = time.perf_counter() - payload_started_at
        LOGGER.info(
            "[checkpoint] payload_read_done path=%s elapsed=%.3fs",
            str(tree_snapshot_path),
            payload_elapsed_s,
        )
        _log_mem(LOGGER, "after payload_read")
        LOGGER.info(
            "[checkpoint] runtime_rebuild_start path=%s",
            str(tree_snapshot_path),
        )
        _log_mem(LOGGER, "before runtime_rebuild")
        runtime_started_at = time.perf_counter()
        with _instrument_checkpoint_runtime_rebuild():
            runtime = load_search_from_checkpoint_payload(
                payload,
                state_codec=self._state_codec,
                dynamics=self._dynamics,
                args=search_args,
                state_type=MorpionState,
                master_state_value_evaluator=self._build_master_evaluator(None),
                random_generator=self._random_generator,
                state_representation_factory=None,
                node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
            )
        runtime_elapsed_s = time.perf_counter() - runtime_started_at
        LOGGER.info(
            "[checkpoint] runtime_rebuild_done path=%s elapsed=%.3fs",
            str(tree_snapshot_path),
            runtime_elapsed_s,
        )
        _log_mem(LOGGER, "after runtime_rebuild")
        elapsed_s = time.perf_counter() - started_at
        _log_mem(LOGGER, "after total_load")
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
        LOGGER.info("[runtime] evaluator_attach_start bundle=%s", str(model_bundle_path))
        started_at = time.perf_counter()
        evaluator = load_morpion_evaluator_from_model_bundle(model_bundle_path)
        if reevaluate_tree:
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
        """Persist the live Anemone runtime as a checkpoint JSON file."""
        runtime = self._require_runtime()
        output = Path(output_path)

        LOGGER.info("[checkpoint] save_start path=%s", str(output))
        _log_mem(LOGGER, "before total_save")
        save_started_at = time.perf_counter()
        LOGGER.info("[checkpoint] payload_build_start path=%s", str(output_path))
        _log_mem(LOGGER, "before payload_build")
        payload_started_at = time.perf_counter()
        payload = build_search_checkpoint_payload(
            runtime,
            state_codec=self._state_codec,
        )
        payload_elapsed_s = time.perf_counter() - payload_started_at
        LOGGER.info(
            "[checkpoint] payload_build_done path=%s elapsed=%.3fs",
            str(output),
            payload_elapsed_s,
        )
        _log_mem(LOGGER, "after payload_build")

        LOGGER.info("[checkpoint] asdict_start path=%s", str(output))
        _log_mem(LOGGER, "before asdict")
        asdict_started_at = time.perf_counter()
        payload_dict = asdict(payload)
        asdict_elapsed_s = time.perf_counter() - asdict_started_at
        LOGGER.info(
            "[checkpoint] asdict_done path=%s elapsed=%.3fs",
            str(output),
            asdict_elapsed_s,
        )
        _log_mem(LOGGER, "after asdict")

        output.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("[checkpoint] json_dump_start path=%s", str(output))
        _log_mem(LOGGER, "before json_dump")
        json_dump_started_at = time.perf_counter()
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(payload_dict, handle, indent=2, sort_keys=True)
        json_dump_elapsed_s = time.perf_counter() - json_dump_started_at
        bytes_written = output.stat().st_size
        LOGGER.info(
            "[checkpoint] json_dump_done path=%s elapsed=%.3fs bytes=%s",
            str(output),
            json_dump_elapsed_s,
            bytes_written,
        )
        _log_mem(LOGGER, "after json_dump")
        elapsed_s = time.perf_counter() - save_started_at
        _log_mem(LOGGER, "after total_save")
        LOGGER.info(
            "[checkpoint] save_done path=%s elapsed=%.3fs",
            str(output),
            elapsed_s,
        )


def load_morpion_search_checkpoint_payload(
    path: str | Path,
) -> SearchRuntimeCheckpointPayload:
    """Load a persisted search checkpoint payload from JSON and validate shape."""
    resolved_path = Path(path)
    caller_summary = _checkpoint_load_caller_summary()
    LOGGER.info(
        "[checkpoint] load_request path=%s caller=%s",
        str(resolved_path),
        caller_summary,
    )
    try:
        path_stat = resolved_path.stat()
    except FileNotFoundError as exc:
        _CHECKPOINT_PAYLOAD_CACHE.pop(resolved_path, None)
        raise InvalidMorpionSearchCheckpointError(
            resolved_path,
            "file does not exist",
        ) from exc

    cache_entry = _CHECKPOINT_PAYLOAD_CACHE.get(resolved_path)
    if cache_entry is not None:
        cached_size, cached_mtime_ns, cached_payload = cache_entry
        if (
            cached_size == path_stat.st_size
            and cached_mtime_ns == path_stat.st_mtime_ns
        ):
            LOGGER.info(
                "[checkpoint] payload_cache_hit path=%s caller=%s bytes=%s",
                str(resolved_path),
                caller_summary,
                path_stat.st_size,
            )
            _log_mem(LOGGER, "after payload_cache_hit")
            return cached_payload
    LOGGER.info(
        "[checkpoint] payload_cache_miss path=%s caller=%s bytes=%s",
        str(resolved_path),
        caller_summary,
        path_stat.st_size,
    )
    try:
        LOGGER.info("[checkpoint] json_read_start path=%s", str(resolved_path))
        _log_mem(LOGGER, "before json_read")
        json_read_started_at = time.perf_counter()
        with open(resolved_path, encoding="utf-8") as handle:
            raw_payload = json.load(handle)
        json_read_elapsed_s = time.perf_counter() - json_read_started_at
        LOGGER.info(
            "[checkpoint] json_read_done path=%s elapsed=%.3fs",
            str(resolved_path),
            json_read_elapsed_s,
        )
        _log_mem(LOGGER, "after json_read")
    except FileNotFoundError as exc:
        raise InvalidMorpionSearchCheckpointError(
            resolved_path,
            "file does not exist",
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidMorpionSearchCheckpointError(
            resolved_path,
            "invalid JSON",
        ) from exc

    try:
        LOGGER.info("[checkpoint] payload_decode_start path=%s", str(resolved_path))
        _log_mem(LOGGER, "before payload_decode")
        payload_decode_started_at = time.perf_counter()
        normalized_payload = _normalize_search_checkpoint_payload_for_dacite(raw_payload)
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
        _log_mem(LOGGER, "after payload_decode")
        _CHECKPOINT_PAYLOAD_CACHE[resolved_path] = (
            path_stat.st_size,
            path_stat.st_mtime_ns,
            payload,
        )
        return payload
    except Exception as exc:
        raise InvalidMorpionSearchCheckpointError(
            resolved_path,
            f"payload shape is invalid: {exc}",
        ) from exc


def _normalize_search_checkpoint_payload_for_dacite(
    raw_payload: object,
) -> object:
    """Normalize union payload fields so dacite can rebuild checkpoint dataclasses."""
    if not isinstance(raw_payload, dict):
        return raw_payload
    raw_tree = raw_payload.get("tree")
    if not isinstance(raw_tree, dict):
        return raw_payload
    raw_nodes = raw_tree.get("nodes")
    if not isinstance(raw_nodes, list):
        return raw_payload

    normalized_payload = dict(raw_payload)
    normalized_tree = dict(raw_tree)
    normalized_tree["nodes"] = [
        _normalize_algorithm_node_payload_for_dacite(node_payload)
        for node_payload in raw_nodes
    ]
    normalized_payload["tree"] = normalized_tree
    return normalized_payload


def _normalize_algorithm_node_payload_for_dacite(node_payload: object) -> object:
    """Normalize one algorithm-node payload before dacite reconstruction."""
    if not isinstance(node_payload, dict):
        return node_payload
    normalized_node_payload = dict(node_payload)
    normalized_node_payload["state_payload"] = _checkpoint_state_payload_from_dict(
        node_payload.get("state_payload")
    )
    return normalized_node_payload


def _checkpoint_state_payload_from_dict(
    raw_state_payload: object,
) -> CheckpointNodeStatePayload | object:
    """Decode the explicit state-payload union used by Anemone checkpoints."""
    if not isinstance(raw_state_payload, dict):
        return raw_state_payload
    if "anchor_ref" in raw_state_payload:
        return from_dict(
            data_class=AnchorCheckpointStatePayload,
            data=raw_state_payload,
            config=Config(cast=[tuple], check_types=False),
        )
    if "delta_ref" in raw_state_payload:
        return from_dict(
            data_class=DeltaCheckpointStatePayload,
            data=raw_state_payload,
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
        raise TypeError(
            "Morpion bootstrap runtime reconfiguration currently supports only "
            "TreeBranchLimitArgs stopping criteria."
        )
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
        raise TypeError(
            "Morpion bootstrap runtime reconfiguration currently supports only "
            "TreeBranchLimitArgs stopping criteria."
        )
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
        raise TypeError(
            "Morpion bootstrap runtime reconfiguration currently supports only "
            "tree-branch-limit stopping criteria on the live runtime."
        )
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
        raise TypeError("Anemone runtime must expose a live tree.")
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
    if callable(should_we_continue) and tree is not None:
        if not bool(should_we_continue(tree=tree)):
            return False

    node_selector = getattr(runtime, "node_selector", None)
    uniform_selector = getattr(node_selector, "base", node_selector)
    current_depth_to_expand = getattr(uniform_selector, "current_depth_to_expand", None)
    if not isinstance(current_depth_to_expand, int):
        return True
    tree_depth = tree.tree_root_tree_depth + current_depth_to_expand
    descendants = getattr(tree, "descendants", None)
    has_tree_depth = getattr(descendants, "has_tree_depth", None)
    if callable(has_tree_depth):
        return bool(has_tree_depth(tree_depth))
    return tree_depth in descendants


def _runtime_stop_reason(runtime: Any) -> str:
    """Return a concise reason why the runtime cannot execute another step."""
    stopping_criterion = getattr(runtime, "stopping_criterion", None)
    should_we_continue = getattr(stopping_criterion, "should_we_continue", None)
    tree = getattr(runtime, "tree", None)
    if callable(should_we_continue) and tree is not None:
        if not bool(should_we_continue(tree=tree)):
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
    if descendants is not None and not callable(has_tree_depth) and tree_depth not in descendants:
        return "runtime_cannot_step"
    return "runtime_cannot_step"


def _live_tree_node_count(runtime: Any) -> int:
    """Return the true live node count from the runtime tree bookkeeping."""
    tree = getattr(runtime, "tree", None)
    if tree is None:
        raise TypeError("Anemone runtime must expose a live tree.")
    nodes_count = getattr(tree, "nodes_count", None)
    if isinstance(nodes_count, int):
        return nodes_count
    descendants = getattr(tree, "descendants", None)
    get_count = getattr(descendants, "get_count", None)
    if callable(get_count):
        return int(get_count())
    return len(runtime._all_nodes_in_tree_order())


def _safe_int_attr(node: Any, attribute_name: str) -> int | None:
    """Read one integer node attribute when the runtime exposes it."""
    value = getattr(node, attribute_name, None)
    return value if isinstance(value, int) else None


__all__ = [
    "apply_runtime_control_to_runner_args",
    "AnemoneMorpionSearchRunner",
    "AnemoneMorpionSearchRunnerArgs",
    "InvalidMorpionSearchCheckpointError",
    "load_morpion_search_checkpoint_payload",
    "MorpionRegressorMasterEvaluator",
    "UninitializedMorpionSearchRunnerError",
    "load_morpion_evaluator_from_model_bundle",
]
