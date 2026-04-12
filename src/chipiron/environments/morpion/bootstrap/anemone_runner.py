"""Real Anemone-backed Morpion search runner for bootstrap cycles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from typing import Any, cast

from anemone.checkpoints import (
    SearchRuntimeCheckpointPayload,
    build_search_checkpoint_payload,
    load_search_from_checkpoint_payload,
)
from anemone.factory import (
    SearchArgs,
    create_tree_and_value_exploration_with_tree_eval_factory,
)
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.node_evaluation.tree.single_agent.factory import (
    NodeMaxEvaluationFactory,
)
from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
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
from .history import MorpionBootstrapTreeStatus


def _default_search_args() -> SearchArgs:
    """Build the default Morpion tree-search args used by the bootstrap runner."""
    return SearchArgs(
        node_selector=ComposedNodeSelectorArgs(
            type="Composed",
            priority=NoPriorityCheckArgs(type="PriorityNoop"),
            base=UniformArgs(type="Uniform"),
        ),
        opening_type=OpeningType.ALL_CHILDREN,
        recommender_rule=AlmostEqualLogistic(
            type="almost_equal_logistic",
            temperature=1.0,
        ),
        stopping_criterion=TreeBranchLimitArgs(
            type="tree_branch_limit",
            tree_branch_limit=128,
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
    """Bridge the atomheart Morpion checkpoint codec to Chipiron wrapper states."""

    inner: MorpionStateCheckpointCodec
    dynamics: MorpionDynamics

    def dump_state_ref(self, state: MorpionState) -> object:
        """Serialize one Chipiron Morpion state via the atomheart checkpoint codec."""
        return self.inner.dump_state_ref(state.to_atomheart_state())

    def load_state_ref(self, payload: object) -> MorpionState:
        """Restore one Chipiron Morpion state from a persisted checkpoint payload."""
        atomheart_state = self.inner.load_state_ref(payload)
        return self.dynamics.wrap_atomheart_state(atomheart_state)


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
    model, _, _ = load_morpion_model_bundle(model_bundle_path)
    model.eval()
    over_detector = MorpionOverEventDetector()
    return MorpionRegressorMasterEvaluator(
        evaluator=MorpionStateEvaluator(),
        over=over_detector,
        over_detector=over_detector,
        feature_converter=MorpionFeatureTensorConverter(dynamics=MorpionDynamics()),
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

    def load_or_create(
        self,
        tree_snapshot_path: str | Path | None,
        model_bundle_path: str | Path | None,
    ) -> None:
        """Load a persisted runtime or create a fresh one for Morpion bootstrap."""
        resolved_bundle_path = (
            None if model_bundle_path is None else Path(model_bundle_path)
        )
        if tree_snapshot_path is None:
            self._runtime = self._create_fresh_runtime(resolved_bundle_path)
            self._current_evaluator_bundle_path = resolved_bundle_path
            return

        runtime = self._load_runtime_from_checkpoint(Path(tree_snapshot_path))
        self._runtime = runtime
        self._current_evaluator_bundle_path = None
        if resolved_bundle_path is not None:
            self._set_runtime_evaluator_from_bundle(resolved_bundle_path)

    def grow(self, max_growth_steps: int) -> None:
        """Advance the live runtime by up to ``max_growth_steps`` iterations."""
        runtime = self._require_runtime()
        for _ in range(max_growth_steps):
            if runtime.tree.root_node.tree_evaluation.has_exact_value():
                break
            if not _runtime_can_step(runtime):
                break
            runtime.step()

    def export_training_tree_snapshot(self, output_path: str | Path) -> None:
        """Persist a training-grade snapshot from the live tree."""
        runtime = self._require_runtime()
        ordered_nodes = runtime._all_nodes_in_tree_order()
        snapshot = build_training_tree_snapshot(
            ordered_nodes,
            root_node_id=str(runtime.tree.root_node.id),
            state_ref_dumper=self._state_codec.dump_state_ref,
        )
        save_training_tree_snapshot(snapshot, output_path)

    def current_tree_size(self) -> int:
        """Return the number of nodes currently tracked by the live runtime."""
        runtime = self._require_runtime()
        return len(runtime.tree.descendants)

    def current_tree_status(self) -> MorpionBootstrapTreeStatus:
        """Return the best available live tree-monitoring status."""
        runtime = self._require_runtime()
        root_node = runtime.tree.root_node
        return MorpionBootstrapTreeStatus(
            num_nodes=len(runtime.tree.descendants),
            num_expanded_nodes=_count_expanded_nodes(runtime),
            num_simulations=_safe_int_attr(root_node, "visit_count"),
            root_visit_count=_safe_int_attr(root_node, "visit_count"),
        )

    def _create_fresh_runtime(
        self,
        model_bundle_path: Path | None,
    ) -> object:
        """Create a fresh single-tree Morpion runtime with the selected evaluator."""
        evaluator = self._build_master_evaluator(model_bundle_path)
        return create_tree_and_value_exploration_with_tree_eval_factory(
            state_type=MorpionState,
            dynamics=self._dynamics,
            starting_state=self._dynamics.wrap_atomheart_state(initial_state()),
            args=self._args.search_args,
            random_generator=self._random_generator,
            master_state_evaluator=evaluator,
            state_representation_factory=None,
            node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        )

    def _load_runtime_from_checkpoint(self, tree_snapshot_path: Path) -> object:
        """Restore one live runtime from a persisted checkpoint JSON file."""
        payload = _load_search_checkpoint_payload(tree_snapshot_path)
        return load_search_from_checkpoint_payload(
            payload,
            state_codec=self._state_codec,
            dynamics=self._dynamics,
            args=self._args.search_args,
            state_type=MorpionState,
            master_state_value_evaluator=self._build_master_evaluator(None),
            random_generator=self._random_generator,
            state_representation_factory=None,
            node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        )

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

    def _set_runtime_evaluator_from_bundle(self, model_bundle_path: Path) -> None:
        """Load one bundle, install it into the live runtime, and refresh leaves."""
        runtime = self._require_runtime()
        evaluator = load_morpion_evaluator_from_model_bundle(model_bundle_path)
        runtime.refresh_with_evaluator(
            evaluator,
            scope=self._args.reevaluation_scope,
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
        payload = build_search_checkpoint_payload(
            runtime,
            state_codec=self._state_codec,
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(asdict(payload), handle, indent=2, sort_keys=True)


def _load_search_checkpoint_payload(path: Path) -> SearchRuntimeCheckpointPayload:
    """Load a persisted search checkpoint payload from JSON and validate shape."""
    try:
        with open(path, encoding="utf-8") as handle:
            raw_payload = json.load(handle)
    except FileNotFoundError as exc:
        raise InvalidMorpionSearchCheckpointError(path, "file does not exist") from exc
    except json.JSONDecodeError as exc:
        raise InvalidMorpionSearchCheckpointError(path, "invalid JSON") from exc

    try:
        return from_dict(
            data_class=SearchRuntimeCheckpointPayload,
            data=raw_payload,
            config=Config(cast=[tuple], check_types=False),
        )
    except Exception as exc:
        raise InvalidMorpionSearchCheckpointError(
            path,
            f"payload shape is invalid: {exc}",
        ) from exc


def _count_expanded_nodes(runtime: Any) -> int:
    """Count nodes that have already generated all branches in the live tree."""
    return sum(
        1
        for node in runtime._all_nodes_in_tree_order()
        if bool(getattr(node, "generated_all_branches", False))
    )


def _runtime_can_step(runtime: Any) -> bool:
    """Return whether the runtime's selector depth exists in the current tree."""
    node_selector = getattr(runtime, "node_selector", None)
    uniform_selector = getattr(node_selector, "base", node_selector)
    current_depth_to_expand = getattr(uniform_selector, "current_depth_to_expand", None)
    if not isinstance(current_depth_to_expand, int):
        return True
    tree_depth = runtime.tree.tree_root_tree_depth + current_depth_to_expand
    return tree_depth in runtime.tree.descendants


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
    "load_morpion_evaluator_from_model_bundle",
]