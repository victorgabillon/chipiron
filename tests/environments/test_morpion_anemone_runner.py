"""Tests for the real Anemone-backed Morpion bootstrap runner."""
# ruff: noqa: E402

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType
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

from anemone.training_export import load_training_tree_snapshot
from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    DeltaCheckpointStatePayload,
)
from anemone.factory import SearchArgs
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.linoo import LinooArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
from anemone.recommender_rule.recommender_rule import AlmostEqualLogistic

import chipiron.environments.morpion.bootstrap.anemone_runner as anemone_runner_module
from chipiron.environments.morpion.bootstrap import (
    BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY,
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    MorpionBootstrapControl,
    MorpionBootstrapEffectiveRuntimeConfig,
    InvalidMorpionSearchCheckpointError,
    MorpionBootstrapArgs,
    MorpionBootstrapPaths,
    MorpionBootstrapRuntimeControl,
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
    UnsupportedMorpionRuntimeReconfigurationError,
    load_bootstrap_history,
    run_morpion_bootstrap_loop,
    save_bootstrap_control,
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


def _patch_reported_losses(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loss_by_evaluator_name: dict[str, float],
) -> None:
    """Patch training so evaluator selection is deterministic while bundles still exist."""
    import chipiron.environments.morpion.bootstrap.bootstrap_loop as bootstrap_loop_module

    real_train = bootstrap_loop_module.train_morpion_regressor

    def _patched_train(train_args: object) -> object:
        _model, metrics = real_train(train_args)
        evaluator_name = Path(str(getattr(train_args, "output_dir"))).name
        metrics["final_loss"] = loss_by_evaluator_name[evaluator_name]
        return _model, metrics

    monkeypatch.setattr(
        bootstrap_loop_module, "train_morpion_regressor", _patched_train
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
    """The real runner should restore the same tree under a changed branch limit."""
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

    assert second_state.generation == 2
    assert second_state.tree_size_at_last_save >= first_saved_tree_size
    assert runner.current_runtime_config().tree_branch_limit == 64
    assert second_state.metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] == {
        "tree_branch_limit": 64
    }
    assert history[-1].metadata[BOOTSTRAP_EFFECTIVE_RUNTIME_METADATA_KEY] == {
        "tree_branch_limit": 64
    }


def test_bootstrap_loop_rejects_runtime_branch_limit_widening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Widening tree_branch_limit should fail loudly instead of silently resetting."""
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

    with pytest.raises(UnsupportedMorpionRuntimeReconfigurationError):
        run_morpion_bootstrap_loop(args, runner, max_cycles=1)
