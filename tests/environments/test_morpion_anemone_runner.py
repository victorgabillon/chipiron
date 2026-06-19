"""Tests for the real Anemone-backed Morpion bootstrap runner."""
# ruff: noqa: E402

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"
_MORPION_EVALUATORS_PACKAGE_ROOT = (
    _REPO_ROOT
    / "src"
    / "chipiron"
    / "environments"
    / "morpion"
    / "players"
    / "evaluators"
)

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "chipiron.environments.morpion.players.evaluators" not in sys.modules:
    _evaluators_stub = ModuleType("chipiron.environments.morpion.players.evaluators")
    _evaluators_stub.__path__ = [str(_MORPION_EVALUATORS_PACKAGE_ROOT)]
    sys.modules["chipiron.environments.morpion.players.evaluators"] = _evaluators_stub

if "atomheart" not in sys.modules:
    _atomheart_stub = ModuleType("atomheart")
    _atomheart_stub.__path__ = [str(_ATOMHEART_PACKAGE_ROOT)]
    sys.modules["atomheart"] = _atomheart_stub

if "anemone" not in sys.modules:
    _anemone_stub = ModuleType("anemone")
    _anemone_stub.__path__ = [str(_ANEMONE_PACKAGE_ROOT)]
    sys.modules["anemone"] = _anemone_stub

from anemone.checkpoints import (
    DEFAULT_CHECKPOINT_FILE_FORMAT,
    AlgorithmNodeCheckpointPayload,
    AnchorCheckpointStatePayload,
    DeltaCheckpointStatePayload,
    SearchRuntimeCheckpointPayload,
    TreeCheckpointPayload,
    checkpoint_file_suffix,
)
from anemone.checkpoints.state_handles import (
    CheckpointBackedStateHandle,
    CheckpointStateResolver,
)
from anemone.factory import SearchArgs
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.linoo import LinooArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
from anemone.recommender_rule.recommender_rule import AlmostEqualLogistic
from anemone.training_export import load_training_tree_snapshot
from anemone.tree_exploration import TreeGrowthStepReport
from anemone.tree_manager import OpeningExpansionKind, RolloutActionSelectorKind
from anemone.utils.logger import checkpoint_logger, set_checkpoint_logger_level
from anemone.value_updates import NodeValueUpdate, NodeValueUpdateResult

import chipiron.environments.morpion.bootstrap.anemone_runner as anemone_runner_module
from chipiron.environments.morpion.bootstrap import (
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    InvalidMorpionSearchCheckpointError,
    MorpionBootstrapArgs,
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    MorpionBootstrapPaths,
    MorpionBootstrapRolloutConfig,
    MorpionBootstrapRunState,
    MorpionBootstrapRuntimeControl,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    MorpionReevaluationPatch,
    MorpionReevaluationPatchRow,
    load_bootstrap_history,
    run_morpion_bootstrap_loop,
    save_bootstrap_control,
)
from chipiron.environments.morpion.bootstrap.cycle_runtime import (
    resolve_runtime_restore_path,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRegressorArgs,
    build_morpion_regressor,
    save_morpion_model_bundle,
)


def _make_model_bundle(output_dir: Path) -> Path:
    """Create one minimal valid Morpion bundle for evaluator-loading tests."""
    model_args = MorpionRegressorArgs(model_kind="linear")
    model = build_morpion_regressor(model_args)
    save_morpion_model_bundle(model, output_dir, model_args=model_args)
    return output_dir


def _runner_args_with_tree_branch_limit(
    tree_branch_limit: int,
) -> AnemoneMorpionSearchRunnerArgs:
    """Build runner args with an explicit branch budget for growth tests."""
    return AnemoneMorpionSearchRunnerArgs(
        search_args=SearchArgs(
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
                tree_branch_limit=tree_branch_limit,
            ),
        )
    )


def _multi_evaluator_config() -> MorpionEvaluatorsConfig:
    """Return one representative two-evaluator bootstrap config."""
    return MorpionEvaluatorsConfig(
        evaluators={
            "linear": MorpionEvaluatorSpec(
                name="linear",
                model_type="linear",
                hidden_sizes=None,
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            ),
            "mlp": MorpionEvaluatorSpec(
                name="mlp",
                model_type="mlp",
                hidden_sizes=(8, 4),
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            ),
        }
    )


class FakeAnemoneRuntime:
    """Tiny fake for live Anemone node-value update application."""

    def __init__(self, nodes: tuple[object, ...] = ()) -> None:
        """Initialize captured call fields."""
        self._nodes = nodes
        self.tree = object()
        self.tree_manager = _FakeTreeManager()
        self.received_updates: tuple[NodeValueUpdate, ...] | None = None
        self.recompute_backups: bool | None = None
        self.allow_missing: bool | None = None

    def _all_nodes_in_tree_order(self) -> list[object]:
        """Return fake live nodes for old-value lookup."""
        return list(self._nodes)

    def _nodes_by_public_id(self) -> dict[str, object]:
        """Return fake live nodes by public id."""
        return {str(getattr(node, "id")): node for node in self._nodes}

    def _apply_node_value_update(
        self,
        *,
        node: object,
        update: NodeValueUpdate,
    ) -> bool:
        """Capture one private value update for smoothed patch application."""
        del node
        current_updates = self.received_updates or ()
        self.received_updates = (*current_updates, update)
        return True

    def apply_node_value_updates(
        self,
        updates: object,
        *,
        recompute_backups: bool,
        allow_missing: bool,
    ) -> NodeValueUpdateResult:
        """Capture updates and return a representative partial-application result."""
        self.received_updates = tuple(updates)  # type: ignore[arg-type]
        self.recompute_backups = recompute_backups
        self.allow_missing = allow_missing
        requested_count = len(self.received_updates)
        missing_node_ids = ("missing-node",) if requested_count > 1 else ()
        return NodeValueUpdateResult(
            requested_count=requested_count,
            applied_count=1 if requested_count else 0,
            missing_node_ids=missing_node_ids,
            recomputed_count=3,
        )


class _FakeValuePropagator:
    """Tiny value propagator for private smoothed patch application."""

    def propagate_after_local_value_changes(
        self,
        changed_nodes: list[object],
    ) -> list[object]:
        """Return the changed nodes as the recomputation summary."""
        return list(changed_nodes)


class _FakeTreeManager:
    """Tiny tree manager for private smoothed patch application."""

    def __init__(self) -> None:
        self.value_propagator = _FakeValuePropagator()
        self.refresh_calls = 0

    def refresh_exploration_indices(self, *, tree: object) -> None:
        """Accept exploration-index refresh calls."""
        del tree
        self.refresh_calls += 1


def test_log_latest_rollout_report_includes_path_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rollout report logging includes aggregate, compact, and detail records."""
    caplog.set_level(logging.INFO)
    path_reports = (
        SimpleNamespace(
            start_node_id="n1",
            start_depth=1,
            end_node_id="n5",
            end_depth=5,
            initial_edge_count=1,
            extra_edge_count=2,
            traversal_count=2,
            total_edge_count=3,
            stop_reason="terminal",
            end_is_terminal=True,
            end_is_exact=True,
            end_was_created_node=True,
            end_was_existing_node=False,
            end_legal_action_count=None,
            end_openable_action_count=None,
            end_opened_action_count=None,
            end_non_opened_branch_count=0,
            no_legal_actions_but_not_terminal=False,
        ),
        SimpleNamespace(
            start_node_id="n2",
            start_depth=1,
            end_node_id="n2",
            end_depth=1,
            initial_edge_count=1,
            extra_edge_count=0,
            traversal_count=0,
            total_edge_count=1,
            stop_reason="action_selector_stop",
            end_is_terminal=False,
            end_is_exact=False,
            end_was_created_node=False,
            end_was_existing_node=True,
            end_legal_action_count=3,
            end_openable_action_count=2,
            end_opened_action_count=1,
            end_non_opened_branch_count=2,
            no_legal_actions_but_not_terminal=False,
        ),
    )
    report = SimpleNamespace(
        total_edge_count=4,
        initial_edge_count=2,
        extra_edge_count=2,
        traversal_count=2,
        stop_reason_counts={"terminal": 1, "action_selector_stop": 1},
        path_reports=path_reports,
    )
    runtime = SimpleNamespace(
        tree_manager=SimpleNamespace(latest_rollout_report=report)
    )

    anemone_runner_module._log_latest_rollout_report(runtime)

    assert "[rollout] total_edges=4 initial_edges=2" in caplog.text
    assert (
        "[rollout-lengths] count=2 total_lengths=[3, 1] extra_lengths=[2, 0]"
        in caplog.text
    )
    assert "stops={'terminal': 1, 'action_selector_stop': 1}" in caplog.text
    assert (
        "[rollout-detail] rollout_index=0 start_node_id=n1 start_depth=1 "
        "end_node_id=n5 end_depth=5 total_edges=3 initial_edges=1 "
        "extra_edges=2 traversals=2 stop_reason=terminal end_terminal=True "
        "end_exact=True"
    ) in caplog.text
    assert "end_created_node=True end_existing_node=False" in caplog.text
    assert (
        "end_legal_actions=3 end_openable_actions=2 end_opened_actions=1 "
        "end_non_opened_branches=2"
    ) in caplog.text


def test_log_latest_rollout_report_warns_on_no_legal_non_terminal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No-legal-actions paths that are not terminal emit a diagnostic warning."""
    caplog.set_level(logging.INFO)
    report = SimpleNamespace(
        total_edge_count=1,
        initial_edge_count=1,
        extra_edge_count=0,
        traversal_count=0,
        stop_reason_counts={"no_legal_actions": 1},
        path_reports=(
            SimpleNamespace(
                start_node_id="n1",
                start_depth=1,
                end_node_id="n1",
                end_depth=1,
                initial_edge_count=1,
                extra_edge_count=0,
                traversal_count=0,
                total_edge_count=1,
                stop_reason="no_legal_actions",
                end_is_terminal=False,
                end_is_exact=False,
                end_legal_action_count=0,
                end_openable_action_count=0,
                end_opened_action_count=0,
                end_non_opened_branch_count=0,
                no_legal_actions_but_not_terminal=True,
            ),
        ),
    )
    runtime = SimpleNamespace(
        tree_manager=SimpleNamespace(latest_rollout_report=report)
    )

    anemone_runner_module._log_latest_rollout_report(runtime)

    assert "no_legal_but_not_terminal=True" in caplog.text
    assert (
        "[rollout-warning] no_legal_actions_but_not_terminal rollout_index=0 "
        "end_node_id=n1 end_depth=1 end_legal_actions=0 "
        "end_non_opened_branches=0"
    ) in caplog.text


def test_log_latest_rollout_report_supports_legacy_report(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A report without path reports still logs aggregates without crashing."""
    caplog.set_level(logging.INFO)
    report = SimpleNamespace(
        total_edge_count=1,
        initial_edge_count=1,
        extra_edge_count=0,
        traversal_count=0,
        stop_reason_counts={"max_extra_steps": 1},
    )
    runtime = SimpleNamespace(
        tree_manager=SimpleNamespace(latest_rollout_report=report)
    )

    anemone_runner_module._log_latest_rollout_report(runtime)

    assert "[rollout] total_edges=1 initial_edges=1" in caplog.text
    assert "[rollout-lengths]" not in caplog.text
    assert "[rollout-detail]" not in caplog.text


class _FakeValue:
    """Tiny direct-value object with the Anemone ``score`` shape."""

    def __init__(self, score: float) -> None:
        self.score = score


class _FakeTreeEvaluation:
    """Tiny tree-evaluation object for smoothing tests."""

    def __init__(
        self,
        direct_value: float | None,
        *,
        exact: bool = False,
        terminal: bool = False,
    ) -> None:
        self.direct_value = None if direct_value is None else _FakeValue(direct_value)
        self._exact = exact
        self._terminal = terminal

    def has_exact_value(self) -> bool:
        return self._exact

    def is_terminal(self) -> bool:
        return self._terminal


class _FakeNode:
    """Tiny live node exposing the fields used by smoothing."""

    def __init__(
        self,
        node_id: str,
        direct_value: float | None,
        *,
        exact: bool = False,
        terminal: bool = False,
    ) -> None:
        self.id = node_id
        self.tree_evaluation = _FakeTreeEvaluation(
            direct_value,
            exact=exact,
            terminal=terminal,
        )
        self.state = SimpleNamespace(is_game_over=lambda: terminal)


class _SelectorInvalidationSpy:
    """Count selector invalidation requests from the blended patch path."""

    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate(self) -> None:
        self.invalidations += 1


class _NoopAwareFakeRuntime(FakeAnemoneRuntime):
    """Fake runtime that reports changes only when direct values actually differ."""

    def __init__(self, nodes: tuple[object, ...] = ()) -> None:
        super().__init__(nodes)
        self.node_selector = _SelectorInvalidationSpy()

    def _apply_node_value_update(
        self,
        *,
        node: object,
        update: NodeValueUpdate,
    ) -> bool:
        node_eval = getattr(node, "tree_evaluation")
        previous_direct_value = getattr(node_eval, "direct_value")
        previous_score = (
            None
            if previous_direct_value is None
            else getattr(previous_direct_value, "score", None)
        )
        changed = previous_score != update.direct_value
        node_eval.direct_value = _FakeValue(update.direct_value)
        return bool(changed)


def _make_reevaluation_patch() -> MorpionReevaluationPatch:
    """Build one representative reevaluation patch for runner adapter tests."""
    return MorpionReevaluationPatch(
        patch_id="patch-1",
        created_at_utc="2026-04-28T12:00:00Z",
        evaluator_generation=2,
        evaluator_name="default",
        model_bundle_path="models/generation_000002/default",
        rows=(
            MorpionReevaluationPatchRow(
                node_id="node-a",
                direct_value=1.25,
                backed_up_value=None,
                is_exact=False,
                is_terminal=False,
                metadata={"source": "test"},
            ),
            MorpionReevaluationPatchRow(
                node_id="node-b",
                direct_value=2.5,
                backed_up_value=2.75,
                is_exact=True,
                is_terminal=None,
                metadata={"source": "test"},
            ),
        ),
    )


def _patch_reported_losses(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loss_by_evaluator_name: dict[str, float],
) -> None:
    """Patch training so evaluator selection is deterministic while bundles still exist."""
    import chipiron.environments.morpion.bootstrap.cycle_training as cycle_training_module

    real_train = cycle_training_module.train_morpion_regressor

    def _patched_train(train_args: object) -> object:
        _model, metrics = real_train(train_args)
        evaluator_name = Path(str(train_args.output_dir)).name
        metrics["final_loss"] = loss_by_evaluator_name[evaluator_name]
        metrics["validation_loss"] = loss_by_evaluator_name[evaluator_name]
        return _model, metrics

    monkeypatch.setattr(
        cycle_training_module, "train_morpion_regressor", _patched_train
    )


def test_create_fresh_runtime_without_checkpoint() -> None:
    """The real runner should create and grow a fresh Morpion runtime."""
    runner = AnemoneMorpionSearchRunner()

    runner.load_or_create(None, None)

    initial_size = runner.current_tree_size()
    runner.grow(3)

    assert initial_size >= 1
    assert runner.current_tree_size() >= initial_size
    assert runner.current_tree_status().num_nodes == runner.current_tree_size()


def test_default_search_args_use_linoo_selector() -> None:
    """The default Morpion bootstrap runner should use the Linoo selector."""
    runner = AnemoneMorpionSearchRunner()

    node_selector = runner._args.search_args.node_selector

    assert isinstance(node_selector, ComposedNodeSelectorArgs)
    assert isinstance(node_selector.base, LinooArgs)
    assert node_selector.base.type == NodeSelectorType.LINOO
    assert (
        runner._args.search_args.opening_expansion.kind == OpeningExpansionKind.ONE_PLY
    )


def test_default_search_args_can_enable_rollout_after_opening() -> None:
    """Rollout config should map to Anemone's opening-expansion API."""
    search_args = anemone_runner_module._default_search_args(
        rollout=MorpionBootstrapRolloutConfig(enabled=True)
    )

    assert search_args.opening_expansion.kind == OpeningExpansionKind.ROLLOUT
    rollout = search_args.opening_expansion.rollout
    assert rollout.max_extra_steps is None
    assert rollout.action_selector_kind == RolloutActionSelectorKind(
        "random_legal_prefer_openable"
    )
    assert rollout.random_seed == 0
    assert rollout.stop_on_existing_node is False


def test_default_search_args_can_enable_traversing_rollout_selector() -> None:
    """Traversal-capable rollout selector strings should reach Anemone."""
    search_args = anemone_runner_module._default_search_args(
        rollout=MorpionBootstrapRolloutConfig(
            enabled=True,
            action_selector_kind="random_legal_prefer_openable",
        )
    )

    assert search_args.opening_expansion.kind == OpeningExpansionKind.ROLLOUT
    rollout = search_args.opening_expansion.rollout
    assert rollout.action_selector_kind == RolloutActionSelectorKind(
        "random_legal_prefer_openable"
    )


def test_disabled_rollout_preserves_one_ply_opening_expansion() -> None:
    """Disabled rollout should preserve Anemone's one-ply expansion default."""
    search_args = anemone_runner_module._default_search_args(
        rollout=MorpionBootstrapRolloutConfig(enabled=False)
    )

    assert search_args.opening_expansion.kind == OpeningExpansionKind.ONE_PLY


def test_runner_state_codec_exposes_incremental_checkpoint_protocol() -> None:
    """The runner should bridge Chipiron state to the new incremental codec API."""
    runner = AnemoneMorpionSearchRunner()
    state_codec = runner._state_codec

    assert hasattr(state_codec, "dump_anchor_ref")
    assert hasattr(state_codec, "dump_delta_from_parent")
    assert hasattr(state_codec, "load_anchor_ref")
    assert hasattr(state_codec, "load_child_from_delta")
    assert hasattr(state_codec, "dump_state_summary")
    assert not hasattr(state_codec, "begin_restore_session")
    assert not hasattr(state_codec, "finish_restore_session")


def test_apply_reevaluation_patch_converts_rows_to_anemone_updates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The runner hook should adapt patch rows into live Anemone updates."""
    runner = AnemoneMorpionSearchRunner()
    fake_runtime = FakeAnemoneRuntime()
    runner._runtime = fake_runtime
    patch = _make_reevaluation_patch()
    caplog.set_level(logging.INFO)

    applied = runner.apply_reevaluation_patch(patch)

    assert applied == 1
    assert fake_runtime.recompute_backups is True
    assert fake_runtime.allow_missing is True
    assert fake_runtime.received_updates is not None
    assert len(fake_runtime.received_updates) == 2

    first, second = fake_runtime.received_updates
    assert first.node_id == "node-a"
    assert first.direct_value == 1.25
    assert first.backed_up_value is None
    assert first.is_exact is False
    assert first.is_terminal is False
    assert first.metadata == {"source": "test"}

    assert second.node_id == "node-b"
    assert second.direct_value == 2.5
    assert second.backed_up_value == 2.75
    assert second.is_exact is True
    assert second.is_terminal is None
    assert (
        "[reevaluation-patch] runner_apply_start patch_id=patch-1 rows=2" in caplog.text
    )
    done_log = (
        "[reevaluation-patch] runner_apply_done "
        "patch_id=patch-1 requested=2 applied=1 missing=1 recomputed=3 "
        "selector_invalidated=none"
    )
    refresh_log = (
        "[reevaluation-patch] backup_refresh_done "
        "affected_nodes=1 ancestors_recomputed=3 selector_invalidated=none"
    )
    assert done_log in caplog.text
    assert refresh_log in caplog.text


def test_apply_reevaluation_patch_blends_direct_value_when_configured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Smoothing should blend old and new non-authoritative direct values."""
    runner = AnemoneMorpionSearchRunner()
    runner._last_applied_runtime_config = MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=4096,
        reevaluation_blend_alpha=0.2,
    )
    fake_runtime = FakeAnemoneRuntime(nodes=(_FakeNode("node-a", 10.0),))
    runner._runtime = fake_runtime
    patch = MorpionReevaluationPatch(
        patch_id="patch-blend",
        created_at_utc="2026-04-28T12:00:00Z",
        evaluator_generation=2,
        evaluator_name="default",
        model_bundle_path="models/generation_000002/default",
        rows=(
            MorpionReevaluationPatchRow(
                node_id="node-a",
                direct_value=0.0,
            ),
        ),
    )
    caplog.set_level(logging.INFO)

    runner.apply_reevaluation_patch(patch)

    assert fake_runtime.received_updates is not None
    assert fake_runtime.received_updates[0].direct_value == pytest.approx(8.0)
    assert "alpha=0.200000 count=1" in caplog.text
    assert "avg_old=10.000000" in caplog.text
    assert "avg_new=0.000000" in caplog.text
    assert "avg_blended=8.000000" in caplog.text
    assert "selector_invalidated=False" in caplog.text


def test_blended_reevaluation_patch_invalidates_selector_when_values_change() -> None:
    """Changed blended writes should invalidate selector caches once."""
    runtime = _NoopAwareFakeRuntime(nodes=(_FakeNode("node-a", 10.0),))
    patch = MorpionReevaluationPatch(
        patch_id="patch-blend-change",
        created_at_utc="2026-05-06T12:00:00Z",
        evaluator_generation=2,
        evaluator_name="default",
        model_bundle_path="models/generation_000002/default",
        rows=(MorpionReevaluationPatchRow(node_id="node-a", direct_value=0.0),),
    )

    result, selector_invalidated = (
        anemone_runner_module._apply_blended_reevaluation_patch(
            runtime=runtime,
            patch=patch,
            blend_alpha=0.2,
            blend_metrics=anemone_runner_module._ReevaluationBlendMetrics(),
        )
    )

    assert result.applied_count == 1
    assert result.recomputed_count == 1
    assert selector_invalidated is True
    assert runtime.node_selector.invalidations == 1
    assert runtime.tree_manager.refresh_calls == 1


def test_blended_reevaluation_patch_does_not_invalidate_selector_for_noop() -> None:
    """No-op blended writes should not invalidate selector caches."""
    runtime = _NoopAwareFakeRuntime(nodes=(_FakeNode("node-a", 10.0),))
    patch = MorpionReevaluationPatch(
        patch_id="patch-blend-noop",
        created_at_utc="2026-05-06T12:00:00Z",
        evaluator_generation=2,
        evaluator_name="default",
        model_bundle_path="models/generation_000002/default",
        rows=(MorpionReevaluationPatchRow(node_id="node-a", direct_value=10.0),),
    )

    result, selector_invalidated = (
        anemone_runner_module._apply_blended_reevaluation_patch(
            runtime=runtime,
            patch=patch,
            blend_alpha=0.2,
            blend_metrics=anemone_runner_module._ReevaluationBlendMetrics(),
        )
    )

    assert result.applied_count == 1
    assert result.recomputed_count == 0
    assert selector_invalidated is False
    assert runtime.node_selector.invalidations == 0
    assert runtime.tree_manager.refresh_calls == 0


def test_apply_reevaluation_patch_alpha_one_replaces_direct_value() -> None:
    """Default alpha should preserve exact replacement behavior."""
    runner = AnemoneMorpionSearchRunner()
    runner._last_applied_runtime_config = MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=4096,
        reevaluation_blend_alpha=1.0,
    )
    fake_runtime = FakeAnemoneRuntime(nodes=(_FakeNode("node-a", 10.0),))
    runner._runtime = fake_runtime
    patch = MorpionReevaluationPatch(
        patch_id="patch-replace",
        created_at_utc="2026-04-28T12:00:00Z",
        evaluator_generation=2,
        evaluator_name="default",
        model_bundle_path="models/generation_000002/default",
        rows=(MorpionReevaluationPatchRow(node_id="node-a", direct_value=0.0),),
    )

    runner.apply_reevaluation_patch(patch)

    assert fake_runtime.received_updates is not None
    assert fake_runtime.received_updates[0].direct_value == 0.0


def test_apply_reevaluation_patch_does_not_blend_terminal_rows() -> None:
    """Terminal patch rows should remain authoritative under smoothing."""
    runner = AnemoneMorpionSearchRunner()
    runner._last_applied_runtime_config = MorpionBootstrapEffectiveRuntimeConfig(
        tree_branch_limit=4096,
        reevaluation_blend_alpha=0.2,
    )
    fake_runtime = FakeAnemoneRuntime(nodes=(_FakeNode("node-a", 10.0),))
    runner._runtime = fake_runtime
    patch = MorpionReevaluationPatch(
        patch_id="patch-terminal",
        created_at_utc="2026-04-28T12:00:00Z",
        evaluator_generation=2,
        evaluator_name="default",
        model_bundle_path="models/generation_000002/default",
        rows=(
            MorpionReevaluationPatchRow(
                node_id="node-a",
                direct_value=0.0,
                is_terminal=True,
            ),
        ),
    )

    runner.apply_reevaluation_patch(patch)

    assert fake_runtime.received_updates is not None
    assert fake_runtime.received_updates[0].direct_value == 0.0


def test_apply_reevaluation_patch_requires_initialized_runtime() -> None:
    """The runner hook should fail clearly before a live runtime exists."""
    runner = AnemoneMorpionSearchRunner()

    with pytest.raises(
        RuntimeError,
        match=r"reevaluation patch.*runtime.*initialized|before.*initialized",
    ):
        runner.apply_reevaluation_patch(_make_reevaluation_patch())


def test_fresh_runtime_with_evaluator_bundle(tmp_path: Path) -> None:
    """The runner should create a fresh runtime with a saved Morpion evaluator."""
    bundle_path = _make_model_bundle(tmp_path / "bundle")
    runner = AnemoneMorpionSearchRunner()

    runner.load_or_create(None, bundle_path)
    runner.grow(2)

    assert runner.current_tree_size() >= 1


def test_load_or_create_logs_selector_family(caplog: pytest.LogCaptureFixture) -> None:
    """Runner startup logs should make the effective selector family explicit."""
    caplog.set_level(logging.INFO)
    runner = AnemoneMorpionSearchRunner()

    runner.load_or_create(None, None)

    assert "[search] selector=linoo opening_type=all_children" in caplog.text.lower()


def test_fresh_runtime_attach_with_bundle_skips_reevaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fresh runtime setup should attach the bundle without refreshing the tree."""
    bundle_path = _make_model_bundle(tmp_path / "bundle")
    refresh_calls: list[tuple[Path, bool]] = []
    real_set = anemone_runner_module.AnemoneMorpionSearchRunner._set_runtime_evaluator_from_bundle

    def _patched_set_runtime_evaluator_from_bundle(
        self: object,
        model_bundle_path: Path,
        *,
        reevaluate_tree: bool = True,
    ) -> None:
        refresh_calls.append((model_bundle_path, reevaluate_tree))
        real_set(self, model_bundle_path, reevaluate_tree=reevaluate_tree)

    monkeypatch.setattr(
        anemone_runner_module.AnemoneMorpionSearchRunner,
        "_set_runtime_evaluator_from_bundle",
        _patched_set_runtime_evaluator_from_bundle,
    )
    caplog.set_level(logging.INFO)

    runner = AnemoneMorpionSearchRunner()
    runner.load_or_create(None, bundle_path)

    assert refresh_calls == [(bundle_path, False)]
    assert "[reeval] skipped reason=fresh_runtime_attach" in caplog.text
    assert "[runtime] evaluator bundle attached without reevaluation" in caplog.text


def test_checkpoint_roundtrip_restores_and_continues_growth(tmp_path: Path) -> None:
    """The runner should restore a saved tree and continue the same runtime growth."""
    checkpoint_path = tmp_path / "tree_checkpoint.json"
    first_runner = AnemoneMorpionSearchRunner()
    first_runner.load_or_create(None, None)
    first_runner.grow(4)
    size_before_save = first_runner.current_tree_size()
    first_runner.save_checkpoint(checkpoint_path)

    second_runner = AnemoneMorpionSearchRunner()
    second_runner.load_or_create(checkpoint_path, None)
    restored_size = second_runner.current_tree_size()
    second_runner.grow(4)

    assert restored_size == size_before_save
    assert second_runner.current_tree_size() >= restored_size

    payload = anemone_runner_module.load_morpion_search_checkpoint_payload(
        checkpoint_path
    )
    assert all(
        isinstance(
            node_payload.state_payload,
            (AnchorCheckpointStatePayload, DeltaCheckpointStatePayload),
        )
        for node_payload in payload.tree.nodes
    )


def test_checkpoint_restore_does_not_retain_full_payload_graph(tmp_path: Path) -> None:
    """Runtime restore should not keep the decoded checkpoint DTO graph alive."""
    checkpoint_path = tmp_path / "tree_checkpoint.json"
    first_runner = AnemoneMorpionSearchRunner()
    first_runner.load_or_create(None, None)
    first_runner.grow(4)
    first_runner.save_checkpoint(checkpoint_path)
    gc.collect()

    second_runner = AnemoneMorpionSearchRunner()
    second_runner.load_or_create(checkpoint_path, None)
    restored_size = second_runner.current_tree_size()
    first_runner = None
    second_runner = None
    gc.collect()

    assert restored_size > 0
    search_count, tree_count, algo_count = _checkpoint_payload_type_counts()
    assert search_count == 0
    assert tree_count == 0
    assert algo_count == 0


def test_checkpoint_metrics_logs_for_save_load_and_restore(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checkpoint save/load should emit stable parseable metrics summary logs."""
    checkpoint_path = tmp_path / "tree_checkpoint.json"
    caplog.set_level(logging.INFO)

    first_runner = AnemoneMorpionSearchRunner()
    first_runner.load_or_create(None, None)
    first_runner.grow(3)
    first_runner.save_checkpoint(checkpoint_path)

    second_runner = AnemoneMorpionSearchRunner()
    second_runner.load_or_create(checkpoint_path, None)

    metrics_lines = [
        record.getMessage()
        for record in caplog.records
        if "[checkpoint-metrics]" in record.getMessage()
    ]
    emitted_output = capsys.readouterr().out
    assert any("operation=save" in line for line in metrics_lines)
    assert any("operation=payload_load" in line for line in metrics_lines)
    assert any("operation=runtime_restore" in line for line in metrics_lines)
    assert any("bytes=" in line for line in metrics_lines)
    assert any("format=" in line for line in metrics_lines)
    assert any("encoder=" in line for line in metrics_lines)
    assert any("nodes=" in line for line in metrics_lines)
    assert any("anchors=" in line for line in metrics_lines)
    assert any("deltas=" in line for line in metrics_lines)
    assert any("jsonable_s=" in line for line in metrics_lines)
    assert any("json_encode_s=" in line for line in metrics_lines)
    assert any("compress_s=" in line for line in metrics_lines)
    assert any("write_s=" in line for line in metrics_lines)
    assert emitted_output == "" or "[checkpoint-profile]" in emitted_output
    assert "checkpoint_selector_state_present=" in caplog.text
    assert "restore_checkpoint_selector_state_present=" in caplog.text


def test_checkpoint_roundtrip_supports_default_compressed_format(
    tmp_path: Path,
) -> None:
    """The runner should save and restore the preferred compressed checkpoint format."""
    checkpoint_path = tmp_path / (
        f"tree_checkpoint{checkpoint_file_suffix(DEFAULT_CHECKPOINT_FILE_FORMAT)}"
    )
    first_runner = AnemoneMorpionSearchRunner()
    first_runner.load_or_create(None, None)
    first_runner.grow(4)
    first_runner.save_checkpoint(checkpoint_path)

    second_runner = AnemoneMorpionSearchRunner()
    second_runner.load_or_create(checkpoint_path, None)

    assert checkpoint_path.is_file()
    assert second_runner.current_tree_size() >= 1


def test_checkpoint_validation_payload_is_reused_for_immediate_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Candidate validation should pass its decoded payload to runtime restore."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    checkpoint_path = paths.runtime_checkpoint_path_for_generation(1)
    checkpoint_path.write_text("{}", encoding="utf-8")
    payload = SearchRuntimeCheckpointPayload(
        evaluator_version=1,
        tree=TreeCheckpointPayload(root_node_id=0),
    )
    load_calls: list[Path] = []

    def _fake_payload_loader(path: str | Path) -> SearchRuntimeCheckpointPayload:
        load_calls.append(Path(path))
        return payload

    fake_runtime = SimpleNamespace()
    captured_payload: dict[str, SearchRuntimeCheckpointPayload] = {}

    def _fake_runtime_restore(
        payload_to_restore: SearchRuntimeCheckpointPayload,
        **kwargs: object,
    ) -> object:
        del kwargs
        captured_payload["payload"] = payload_to_restore
        return fake_runtime

    monkeypatch.setattr(
        anemone_runner_module,
        "load_morpion_search_checkpoint_payload",
        _fake_payload_loader,
    )
    monkeypatch.setattr(
        anemone_runner_module,
        "load_search_from_checkpoint_payload",
        _fake_runtime_restore,
    )
    monkeypatch.setattr(
        anemone_runner_module.AnemoneMorpionSearchRunner,
        "_build_master_evaluator",
        lambda self, model_bundle_path: object(),
    )
    run_state = MorpionBootstrapRunState(
        generation=1,
        cycle_index=0,
        latest_tree_snapshot_path=None,
        latest_rows_path=None,
        latest_model_bundle_paths=None,
        active_evaluator_name=None,
        tree_size_at_last_save=0,
        last_save_unix_s=None,
        latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint_path),
    )
    caplog.set_level(logging.INFO)

    resolved_path = resolve_runtime_restore_path(paths=paths, run_state=run_state)
    assert resolved_path == checkpoint_path
    runner = AnemoneMorpionSearchRunner()
    restored_runtime = runner._load_runtime_from_checkpoint(
        resolved_path,
        search_args=AnemoneMorpionSearchRunnerArgs().search_args,
    )

    assert restored_runtime is fake_runtime
    assert captured_payload["payload"] is payload
    assert load_calls == [checkpoint_path]
    assert anemone_runner_module._validated_checkpoint_payload_cache is None
    assert "[checkpoint] candidate_reuse_for_restore" in caplog.text
    assert "operation=payload_load" in caplog.text
    assert "cache=hit" in caplog.text
    assert "[memory] phase=before_candidate_validation" in caplog.text
    assert "[memory] phase=after_candidate_validation" in caplog.text
    assert "[memory] phase=before_runtime_restore" in caplog.text
    assert "[memory] phase=after_runtime_rebuild" in caplog.text
    assert "[memory] phase=after_restore_payload_release" in caplog.text


def test_checkpoint_restore_phase_logs_are_suppressed_by_default(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Detailed checkpoint restore phases should stay out of normal INFO logs."""
    checkpoint_path = tmp_path / "tree_checkpoint.json"
    caplog.set_level(logging.INFO)

    first_runner = AnemoneMorpionSearchRunner()
    first_runner.load_or_create(None, None)
    first_runner.grow(2)
    first_runner.save_checkpoint(checkpoint_path)

    second_runner = AnemoneMorpionSearchRunner()
    second_runner.load_or_create(checkpoint_path, None)

    assert "[checkpoint-restore]" not in caplog.text


def test_checkpoint_restore_phase_logs_can_be_reenabled(
    tmp_path: Path,
) -> None:
    """Checkpoint debug logging should re-enable detailed restore phase records."""
    checkpoint_path = tmp_path / "tree_checkpoint.json"
    messages: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    previous_level = checkpoint_logger.level
    handler = _ListHandler(level=logging.DEBUG)
    checkpoint_logger.addHandler(handler)
    set_checkpoint_logger_level(logging.DEBUG)
    try:
        first_runner = AnemoneMorpionSearchRunner()
        first_runner.load_or_create(None, None)
        first_runner.grow(2)
        first_runner.save_checkpoint(checkpoint_path)

        second_runner = AnemoneMorpionSearchRunner()
        second_runner.load_or_create(checkpoint_path, None)
    finally:
        checkpoint_logger.removeHandler(handler)
        set_checkpoint_logger_level(previous_level)

    assert any(
        "[checkpoint-restore] phase=validate_payload status=start" in message
        for message in messages
    )
    assert any(
        "[checkpoint-restore] phase=restore_selector_state status=done" in message
        for message in messages
    )


def _checkpoint_payload_type_counts() -> tuple[int, int, int]:
    payload_types = (
        SearchRuntimeCheckpointPayload,
        TreeCheckpointPayload,
        AlgorithmNodeCheckpointPayload,
    )
    return tuple(
        sum(1 for obj in gc.get_objects() if isinstance(obj, payload_type))
        for payload_type in payload_types
    )


def test_checkpoint_roundtrip_continues_growth_when_branch_budget_remains(
    tmp_path: Path,
) -> None:
    """Restored runtimes should keep growing when the branch budget is not exhausted."""
    checkpoint_path = tmp_path / "tree_checkpoint.json"
    runner_args = _runner_args_with_tree_branch_limit(4096)
    first_runner = AnemoneMorpionSearchRunner(runner_args)
    first_runner.load_or_create(
        None,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=4096),
    )
    first_runner.grow(1)
    size_before_save = first_runner.current_tree_size()
    first_runner.save_checkpoint(checkpoint_path)

    second_runner = AnemoneMorpionSearchRunner(runner_args)
    second_runner.load_or_create(
        checkpoint_path,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=4096),
    )
    restored_size = second_runner.current_tree_size()
    second_runner.grow(1)

    assert restored_size == size_before_save
    assert second_runner.current_tree_size() > restored_size


def test_current_tree_status_reports_live_node_and_expanded_counts() -> None:
    """Status helpers should reflect the real runtime tree bookkeeping."""
    runner = AnemoneMorpionSearchRunner(_runner_args_with_tree_branch_limit(4096))
    runner.load_or_create(
        None,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=4096),
    )
    runner.grow(1)
    runtime = runner._require_runtime()

    expected_num_nodes = runtime.tree.nodes_count
    expected_num_expanded_nodes = sum(
        1
        for node in runtime._all_nodes_in_tree_order()
        if getattr(node, "all_branches_generated", False)
    )
    status = runner.current_tree_status()

    assert runner.current_tree_size() == expected_num_nodes
    assert status.num_nodes == expected_num_nodes
    assert status.num_expanded_nodes == expected_num_expanded_nodes


def test_current_tree_status_reports_depth_counts() -> None:
    """Tree status should include compact per-depth counts from descendants."""
    runner = AnemoneMorpionSearchRunner(_runner_args_with_tree_branch_limit(4096))
    runner.load_or_create(
        None,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=4096),
    )
    runner.grow(2)
    status = runner.current_tree_status()

    assert status.min_depth_present == 0
    assert status.max_depth_present is not None
    assert status.depth_node_counts[0] == 1
    assert sum(status.depth_node_counts.values()) == status.num_nodes


def test_runtime_step_returns_selected_node_growth_report() -> None:
    """Anemone runtime steps should report the selected node and depth."""
    runner = AnemoneMorpionSearchRunner(_runner_args_with_tree_branch_limit(4096))
    runner.load_or_create(
        None,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=4096),
    )
    runtime = runner._require_runtime()
    nodes_before = runtime.tree.nodes_count

    report = runtime.step()

    assert isinstance(report, TreeGrowthStepReport)
    assert report.selected_node_id is not None
    assert report.selected_depth is not None
    assert report.nodes_before == nodes_before
    assert report.nodes_after >= report.nodes_before
    assert report.nodes_added == report.nodes_after - report.nodes_before
    assert report.select_s is not None and report.select_s >= 0.0
    assert report.limit_s is not None and report.limit_s >= 0.0
    assert report.expand_s is not None and report.expand_s >= 0.0
    assert report.evaluate_s is not None and report.evaluate_s >= 0.0
    assert report.propagate_s is not None and report.propagate_s >= 0.0
    assert report.total_s is not None and report.total_s >= 0.0
    assert report.selector_report_rows is not None and report.selector_report_rows >= 1


def test_runner_growth_logs_selected_node_id_and_depth(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Growth logs should surface selected-node observability from Anemone."""
    caplog.set_level(logging.INFO)
    runner = AnemoneMorpionSearchRunner(_runner_args_with_tree_branch_limit(4096))
    runner.load_or_create(
        None,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=4096),
    )

    runner.grow(1)

    assert "selected_node_id=" in caplog.text
    assert "selected_depth=" in caplog.text
    assert "selected_depth=unknown" not in caplog.text
    assert "[growth-timing] step=1" in caplog.text
    assert "selector_total_s=" in caplog.text
    assert "selector_state_rebuilt=" in caplog.text
    assert "selector_nodes_incrementally_updated=" in caplog.text
    assert "selector_total_nodes_scanned=" in caplog.text
    assert "selector_frontier_nodes_scanned=" in caplog.text
    assert "[growth-selection-table] step=1" in caplog.text
    assert "[growth-selection-table-timing] step=1" in caplog.text
    assert (
        "depth total opened frontier terminal exact uncached_terminal "
        "non_openable index selected"
    ) in caplog.text


def test_runner_growth_writes_latest_linoo_selection_table(tmp_path: Path) -> None:
    """Growth should persist the latest structured Linoo depth table."""
    artifact_path = tmp_path / "pipeline" / "latest_linoo_selection_table.json"
    runner = AnemoneMorpionSearchRunner(_runner_args_with_tree_branch_limit(4096))
    runner.configure_linoo_selection_table_artifact(
        path=artifact_path,
        cycle_index=3,
        generation=5,
    )
    runner.load_or_create(
        None,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=4096),
    )

    runner.grow(1)

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    runtime = runner._require_runtime()
    step_report = runtime.latest_step_report
    assert step_report is not None
    selector_report = step_report.selector_report
    assert selector_report is not None

    assert payload["cycle_index"] == 3
    assert payload["generation"] == 5
    assert payload["step"] == 1
    assert payload["selected_depth"] == selector_report.selected_depth
    assert payload["selected_node_id"] == selector_report.selected_node_id
    assert [row["depth"] for row in payload["rows"]] == [
        row.depth for row in selector_report.depth_rows
    ]
    assert any(row["selected"] for row in payload["rows"])
    assert sum(1 for row in payload["rows"] if row["selected"]) == 1
    for row in payload["rows"]:
        assert row["index"] == row["opened"] * (row["depth"] + 1)


def test_runner_growth_falls_back_safely_when_step_report_is_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Runner logs should remain safe when runtimes do not return a step report."""

    class _FakeEvaluation:
        def has_exact_value(self) -> bool:
            return False

    class _FakeRootNode:
        tree_evaluation = _FakeEvaluation()

    class _FakeTree:
        root_node = _FakeRootNode()
        nodes_count = 3
        branch_count = 12

    class _FakeRuntime:
        tree = _FakeTree()

        def step(self) -> None:
            return None

    caplog.set_level(logging.INFO)
    runner = AnemoneMorpionSearchRunner()
    runner._runtime = _FakeRuntime()

    runner.grow(1)

    assert "selected_node_id=unknown selected_depth=unknown" in caplog.text
    assert "[growth-timing] step=1 total_s=unknown" in caplog.text


def test_runner_growth_logs_unknown_timing_fields_without_crashing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Growth timing logs should tolerate partial fake reports."""

    class _FakeEvaluation:
        def has_exact_value(self) -> bool:
            return False

    class _FakeRootNode:
        tree_evaluation = _FakeEvaluation()

    class _FakeTree:
        root_node = _FakeRootNode()
        nodes_count = 3
        branch_count = 12

    class _FakeRuntime:
        tree = _FakeTree()

        def step(self) -> object:
            return SimpleNamespace(
                selected_node_id=5,
                selected_depth=2,
                selector_report=SimpleNamespace(depth_rows=()),
            )

    caplog.set_level(logging.INFO)
    runner = AnemoneMorpionSearchRunner()
    runner._runtime = _FakeRuntime()

    runner.grow(1)

    assert "[growth-timing] step=1 total_s=unknown" in caplog.text
    assert "rows=unknown" not in caplog.text
    assert (
        "[growth-selection-table-timing] step=1 rows=0 format_s=unknown log_s=unknown"
        in caplog.text
    )


def test_selector_growth_diagnostic_fields_extract_required_values() -> None:
    """Growth log helpers should expose the stable selector diagnostic names."""
    selector_report = SimpleNamespace(
        state_rebuilt=False,
        nodes_incrementally_updated=2,
        total_nodes_scanned=2,
        frontier_nodes_scanned=10,
    )

    fields = anemone_runner_module._selector_growth_diagnostic_fields(selector_report)

    assert fields["selector_state_rebuilt"] is False
    assert fields["selector_nodes_incrementally_updated"] == 2
    assert fields["selector_total_nodes_scanned"] == 2
    assert fields["selector_frontier_nodes_scanned"] == 10


def test_selector_growth_diagnostic_fields_tolerate_missing_values() -> None:
    """Growth log helpers should leave missing selector fields unset."""
    fields = anemone_runner_module._selector_growth_diagnostic_fields(SimpleNamespace())

    assert fields["selector_state_rebuilt"] is None
    assert fields["selector_nodes_incrementally_updated"] is None
    assert fields["selector_total_nodes_scanned"] is None
    assert fields["selector_frontier_nodes_scanned"] is None


def test_checkpoint_selector_state_fields_report_presence() -> None:
    """Checkpoint selector-state helpers should expose presence and metadata."""
    payload = SimpleNamespace(selector_state=SimpleNamespace(type="linoo", version=1))

    fields = anemone_runner_module._checkpoint_selector_state_fields(
        payload,
        prefix="checkpoint",
    )

    assert fields["checkpoint_selector_state_present"] is True
    assert fields["checkpoint_selector_state_type"] == "linoo"
    assert fields["checkpoint_selector_state_version"] == 1


def test_checkpoint_selector_state_fields_handle_absence() -> None:
    """Checkpoint selector-state helpers should stay safe when absent."""
    fields = anemone_runner_module._checkpoint_selector_state_fields(
        SimpleNamespace(selector_state=None),
        prefix="restore_checkpoint",
    )

    assert fields["restore_checkpoint_selector_state_present"] is False
    assert fields["restore_checkpoint_selector_state_type"] is None
    assert fields["restore_checkpoint_selector_state_version"] is None


def test_restore_with_evaluator_bundle_skips_reevaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Restore with a selected evaluator bundle should attach without refreshing."""
    checkpoint_path = tmp_path / "tree_checkpoint.json"
    bundle_path = _make_model_bundle(tmp_path / "bundle")
    first_runner = AnemoneMorpionSearchRunner()
    first_runner.load_or_create(None, None)
    first_runner.grow(3)
    first_runner.save_checkpoint(checkpoint_path)

    reevaluation_calls: list[tuple[Path, bool]] = []
    real_refresh = anemone_runner_module.AnemoneMorpionSearchRunner._set_runtime_evaluator_from_bundle

    def _patched_set_runtime_evaluator_from_bundle(
        self: object,
        model_bundle_path: Path,
        *,
        reevaluate_tree: bool = True,
    ) -> None:
        reevaluation_calls.append((model_bundle_path, reevaluate_tree))
        real_refresh(self, model_bundle_path, reevaluate_tree=reevaluate_tree)

    monkeypatch.setattr(
        anemone_runner_module.AnemoneMorpionSearchRunner,
        "_set_runtime_evaluator_from_bundle",
        _patched_set_runtime_evaluator_from_bundle,
    )
    caplog.set_level(logging.INFO)

    second_runner = AnemoneMorpionSearchRunner()
    second_runner.load_or_create(checkpoint_path, bundle_path)
    second_runner.grow(2)

    assert reevaluation_calls == [(bundle_path, False)]
    assert "[reeval] skipped reason=resume_restore" in caplog.text
    assert "[runtime] evaluator bundle attached without reevaluation" in caplog.text
    assert second_runner.current_tree_size() >= 1


def test_restore_with_evaluator_bundle_reevaluates_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit reevaluate-all should refresh existing nodes on restore."""
    checkpoint_path = tmp_path / "tree_checkpoint.json"
    bundle_path = _make_model_bundle(tmp_path / "bundle")
    first_runner = AnemoneMorpionSearchRunner()
    first_runner.load_or_create(None, None)
    first_runner.grow(3)
    first_runner.save_checkpoint(checkpoint_path)

    refresh_calls: list[str] = []
    real_set = anemone_runner_module.AnemoneMorpionSearchRunner._set_runtime_evaluator_from_bundle

    def _patched_set_runtime_evaluator_from_bundle(
        self: object,
        model_bundle_path: Path,
        *,
        reevaluate_tree: bool = True,
    ) -> None:
        refresh_calls.append(f"{model_bundle_path}:{reevaluate_tree}")
        real_set(self, model_bundle_path, reevaluate_tree=reevaluate_tree)

    monkeypatch.setattr(
        anemone_runner_module.AnemoneMorpionSearchRunner,
        "_set_runtime_evaluator_from_bundle",
        _patched_set_runtime_evaluator_from_bundle,
    )
    caplog.set_level(logging.INFO)

    second_runner = AnemoneMorpionSearchRunner()
    second_runner.load_or_create(checkpoint_path, bundle_path, reevaluate_tree=True)

    assert refresh_calls == [f"{bundle_path}:True"]
    assert "[reeval] start bundle=" in caplog.text
    assert "[runtime] evaluator bundle attached without reevaluation" not in caplog.text


def test_load_or_create_reevaluate_all_fails_loudly_when_runtime_lacks_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reevaluate-all should never be silently downgraded when unsupported."""
    bundle_path = _make_model_bundle(tmp_path / "bundle")
    runner = AnemoneMorpionSearchRunner()

    class _RuntimeWithoutRefresh:
        def set_evaluator(self, evaluator: object) -> None:
            _ = evaluator

    monkeypatch.setattr(
        runner,
        "_require_runtime",
        lambda: _RuntimeWithoutRefresh(),
    )

    with pytest.raises(
        NotImplementedError, match="does not yet support full tree reevaluation"
    ):
        runner._set_runtime_evaluator_from_bundle(bundle_path, reevaluate_tree=True)


def test_explicit_evaluator_change_still_reevaluates_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit evaluator changes should still refresh the configured tree scope."""
    bundle_path = _make_model_bundle(tmp_path / "bundle")
    runner = AnemoneMorpionSearchRunner()
    runtime = Mock()
    runner._runtime = runtime
    loaded_evaluator = object()
    monkeypatch.setattr(
        anemone_runner_module,
        "load_morpion_evaluator_from_model_bundle",
        lambda path: loaded_evaluator,
    )
    caplog.set_level(logging.INFO)

    runner._set_runtime_evaluator_from_bundle(bundle_path, reevaluate_tree=True)

    runtime.refresh_with_evaluator.assert_called_once_with(
        loaded_evaluator,
        scope=runner._args.reevaluation_scope,
    )
    runtime.set_evaluator.assert_not_called()
    assert "[reeval] start bundle=" in caplog.text
    assert "[reeval] done elapsed=" in caplog.text


def test_attach_evaluator_without_reevaluation_updates_runtime_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Attach-only evaluator updates should avoid full tree reevaluation."""
    bundle_path = _make_model_bundle(tmp_path / "bundle")
    runner = AnemoneMorpionSearchRunner()
    runtime = Mock()
    runner._runtime = runtime
    loaded_evaluator = object()
    monkeypatch.setattr(
        anemone_runner_module,
        "load_morpion_evaluator_from_model_bundle",
        lambda path: loaded_evaluator,
    )
    caplog.set_level(logging.INFO)

    runner._set_runtime_evaluator_from_bundle(bundle_path, reevaluate_tree=False)

    runtime.set_evaluator.assert_called_once_with(loaded_evaluator)
    runtime.refresh_with_evaluator.assert_not_called()
    assert "[runtime] evaluator bundle attached without reevaluation" in caplog.text


def test_export_training_snapshot_from_real_runner(tmp_path: Path) -> None:
    """The real runner should export a training snapshot loadable by existing code."""
    runner = AnemoneMorpionSearchRunner()
    snapshot_path = tmp_path / "training_snapshot.json"
    runner.load_or_create(None, None)
    runner.grow(3)

    runner.export_training_tree_snapshot(snapshot_path)

    snapshot = load_training_tree_snapshot(snapshot_path)
    assert snapshot_path.is_file()
    assert snapshot.root_node_id is not None
    assert len(snapshot.nodes) >= 1


def test_training_export_profile_logging_includes_state_and_reuse_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Aggregate training-export profile logs should expose the new split metrics."""
    profile = anemone_runner_module.MorpionTrainingExportProfile(
        node_count=11,
        state_ref_count=10,
        payload_build_s=12.5,
        node_traversal_s=12.0,
        state_ref_serialization_s=10.5,
        node_payload_total_s=11.9,
        node_metadata_total_s=0.5,
        node_value_total_s=0.6,
        node_children_total_s=0.4,
        node_state_access_total_s=10.0,
        state_ref_conversion_total_s=0.5,
        checkpoint_backed_state_handles=9,
        reusable_checkpoint_payloads=9,
        plain_or_materialized_states=2,
        state_access_calls=10,
    )
    caplog.set_level(logging.INFO)

    anemone_runner_module._log_training_export_profile(profile)

    assert "[training-export-profile]" in caplog.text
    assert "node_state_access_total_s=" in caplog.text
    assert "state_ref_conversion_total_s=" in caplog.text
    assert "reusable_checkpoint_payloads=" in caplog.text
    assert "[training-export-profile-rates]" in caplog.text
    assert "state_ref_avg_ms=" in caplog.text


def test_training_export_handle_classification_does_not_resolve_state() -> None:
    """Raw-handle classification must not touch ``node.state`` during export."""

    class _NeverResolveNode:
        def __init__(self, state_handle: object) -> None:
            self.state_handle = state_handle

        @property
        def state(self) -> object:
            raise AssertionError(
                "node.state should not be resolved during classification"
            )

    resolver = CheckpointStateResolver(
        state_codec=Mock(),
        state_payloads_by_node_id={
            7: AnchorCheckpointStatePayload(anchor_ref={"anchor": 1})
        },
    )
    node = _NeverResolveNode(
        CheckpointBackedStateHandle(
            resolver=resolver,
            node_id=7,
        )
    )
    profile = anemone_runner_module.MorpionTrainingExportProfile()

    profile.observe_state_handle(node)

    assert profile.checkpoint_backed_state_handles == 1
    assert profile.reusable_checkpoint_payloads == 1
    assert profile.plain_or_materialized_states == 0
    assert profile.state_access_calls == 0


def test_sharded_training_export_logging_includes_reuse_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sharded export logs should expose new-versus-reused node counts."""
    stats = anemone_runner_module.MorpionShardedTrainingExportStats(
        generation=135,
        node_count=227716,
        new_node_count=1307,
    )
    caplog.set_level(logging.INFO)

    anemone_runner_module._log_sharded_training_export_stats(stats)

    assert "[sharded-training-export]" in caplog.text
    assert "generation=135" in caplog.text
    assert "nodes=227716" in caplog.text
    assert "new_nodes=1307" in caplog.text
    assert "reused_nodes=226409" in caplog.text


def test_invalid_model_bundle_path_fails_loudly(tmp_path: Path) -> None:
    """Missing evaluator bundles should fail loudly instead of falling back."""
    runner = AnemoneMorpionSearchRunner()

    with pytest.raises(FileNotFoundError):
        runner.load_or_create(None, tmp_path / "missing_bundle")


def test_invalid_checkpoint_path_fails_loudly(tmp_path: Path) -> None:
    """Missing checkpoints should fail loudly instead of silently resetting the tree."""
    runner = AnemoneMorpionSearchRunner()

    with pytest.raises(InvalidMorpionSearchCheckpointError):
        runner.load_or_create(tmp_path / "missing_checkpoint.json", None)


def test_apply_runtime_control_to_runner_args_updates_tree_branch_limit() -> None:
    """Runner args rebinding helper should update TreeBranchLimitArgs cleanly."""
    runner_args = AnemoneMorpionSearchRunnerArgs()

    rebound_args = anemone_runner_module.apply_runtime_control_to_runner_args(
        runner_args,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=64),
    )

    assert rebound_args.search_args.stopping_criterion.tree_branch_limit == 64
    assert runner_args.search_args.stopping_criterion.tree_branch_limit == 128


def test_bootstrap_loop_works_with_real_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bootstrap loop should create, save, restore, and continue one real tree."""
    _patch_reported_losses(
        monkeypatch,
        loss_by_evaluator_name={"linear": 0.1, "mlp": 0.2},
    )
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.0,
        save_after_seconds=0.0,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        evaluators_config=_multi_evaluator_config(),
    )
    runner = AnemoneMorpionSearchRunner()

    first_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    first_tree_path = MorpionBootstrapPaths.from_work_dir(
        tmp_path
    ).resolve_work_dir_path(first_state.latest_tree_snapshot_path)
    assert first_tree_path is not None and first_tree_path.is_file()
    first_saved_tree_size = first_state.tree_size_at_last_save

    second_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)

    assert second_state.generation == 2
    assert second_state.cycle_index == 1
    assert second_state.tree_size_at_last_save >= first_saved_tree_size
    assert second_state.active_evaluator_name == "linear"
    assert MorpionBootstrapPaths.from_work_dir(tmp_path).run_state_path.is_file()


def test_bootstrap_loop_reapplies_runtime_branch_limit_between_cycles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real runner should stop cleanly when a lowered branch limit is exhausted."""
    _patch_reported_losses(
        monkeypatch,
        loss_by_evaluator_name={"linear": 0.1, "mlp": 0.2},
    )
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.0,
        save_after_seconds=0.0,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        tree_branch_limit=128,
        evaluators_config=_multi_evaluator_config(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = AnemoneMorpionSearchRunner()

    first_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    first_saved_tree_size = first_state.tree_size_at_last_save
    save_bootstrap_control(
        MorpionBootstrapControl(
            runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=64)
        ),
        paths.control_path,
    )

    second_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    history = load_bootstrap_history(paths.history_jsonl_path)

    assert second_state.generation == 1
    assert second_state.tree_size_at_last_save == first_saved_tree_size
    assert runner.current_runtime_config().tree_branch_limit == 64
    assert second_state.metadata["growth_status"] == "growth_budget_already_exhausted"
    assert second_state.metadata["checkpoint_skipped_reason"] == (
        "no_growth_and_limit_reached"
    )
    assert second_state.metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] == {
        "reevaluation_blend_alpha": 1.0,
        "tree_branch_limit": 64,
    }
    assert history[-1].metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] == {
        "reevaluation_blend_alpha": 1.0,
        "tree_branch_limit": 64,
    }


def test_bootstrap_loop_allows_runtime_branch_limit_widening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Widening tree_branch_limit should let an existing tree continue growing."""
    _patch_reported_losses(
        monkeypatch,
        loss_by_evaluator_name={"linear": 0.1, "mlp": 0.2},
    )
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        max_growth_steps_per_cycle=5,
        save_after_tree_growth_factor=1.0,
        save_after_seconds=0.0,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        tree_branch_limit=128,
        evaluators_config=_multi_evaluator_config(),
    )
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    runner = AnemoneMorpionSearchRunner()

    run_morpion_bootstrap_loop(args, runner, max_cycles=1)
    save_bootstrap_control(
        MorpionBootstrapControl(
            runtime=MorpionBootstrapRuntimeControl(tree_branch_limit=256)
        ),
        paths.control_path,
    )

    second_state = run_morpion_bootstrap_loop(args, runner, max_cycles=1)

    assert runner.current_runtime_config().tree_branch_limit == 256
    assert second_state.metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] == {
        "reevaluation_blend_alpha": 1.0,
        "tree_branch_limit": 256,
    }
