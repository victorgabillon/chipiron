"""Value-level certainty through real Morpion reevaluation and Dataset paths."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, get_args

import pytest
import torch
from anemone.checkpoints import serialize_value
from anemone.node_evaluation.common import canonical_value
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.node_selector.linoo.errors import LinooSelectionError
from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
from anemone.value_updates import NodeValueUpdate
from atomheart.games.morpion import MorpionDynamics as AtomDynamics
from atomheart.games.morpion import initial_state
from valanga.evaluations import Certainty

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapArgs,
    MorpionReevaluationPatch,
    load_reevaluation_patch,
    save_reevaluation_patch,
)
from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
from chipiron.environments.morpion.players.evaluators.morpion_state_evaluator import (
    MorpionOverEventDetector,
    MorpionStateEvaluator,
)
from chipiron.environments.morpion.players.evaluators.neural_evaluator import (
    MorpionRegressorMasterEvaluator,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def endgame() -> tuple[Any, Any]:
    """Legally reach a one-action position and its terminal child, without search."""
    dynamics = AtomDynamics()
    state = initial_state()
    previous = state
    for _ in range(200):
        actions = dynamics.all_legal_actions(state)
        if not actions:
            assert len(dynamics.all_legal_actions(previous)) == 1
            return previous, state
        previous = state
        state = dynamics.step(state, actions[0]).next_state
    pytest.fail("deterministic fixture did not terminate")


def make_runner(
    monkeypatch: pytest.MonkeyPatch, *, state: Any = None, alpha: float = 1.0
) -> Any:
    """Use the real factory, node evaluation, proof policy, propagator and selector."""
    args = runner_module.AnemoneMorpionSearchRunnerArgs(
        search_args=replace(
            runner_module.default_search_args(),
            stopping_criterion=TreeBranchLimitArgs(
                type="tree_branch_limit",
                tree_branch_limit=128 if state is not None else 1,
            ),
        ),
        random_seed=4,
    )
    runner = runner_module.AnemoneMorpionSearchRunner(args)
    runner._last_applied_runtime_config = replace(
        runner._last_applied_runtime_config, reevaluation_blend_alpha=alpha
    )
    with monkeypatch.context() as context:
        if state is not None:
            context.setattr(runner_module, "initial_state", lambda: state)
        runner.load_or_create(None, None)
    runner.grow(1)
    assert all(
        isinstance(n.tree_evaluation, NodeMaxEvaluation)
        for n in runner.iter_profile_nodes()
    )
    return runner


def neural_evaluator(score: float) -> MorpionRegressorMasterEvaluator:
    """Actual neural evaluator and feature conversion, with deterministic weights."""
    dynamics = MorpionDynamics()
    converter = MorpionFeatureTensorConverter(dynamics)
    model = torch.nn.Linear(converter.input_dim, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(score)
    detector = MorpionOverEventDetector(dynamics)
    return MorpionRegressorMasterEvaluator(
        evaluator=MorpionStateEvaluator(dynamics),
        over=detector,
        over_detector=detector,
        input_converter=converter,
        regressor=model,
    )


def model_patch(
    runner: Any, monkeypatch: pytest.MonkeyPatch, score: float, tmp_path: Path
) -> MorpionReevaluationPatch:
    """Produce, save and reload the built-in adapter's actual model patch rows."""
    snapshot, _ = runner.build_training_tree_snapshot_payload()
    evaluator = neural_evaluator(score)
    monkeypatch.setattr(
        worker, "load_morpion_evaluator_from_model_bundle", lambda _: evaluator
    )
    adapter = worker.MorpionActiveModelNodeReevaluationEvaluator(tmp_path / "model")
    rows = adapter.evaluate_patch_rows(
        snapshot, tuple(n.node_id for n in snapshot.nodes)
    )
    patch = MorpionReevaluationPatch(
        patch_id="certainty-test",
        created_at_utc="2026-10-20T00:00:00Z",
        evaluator_generation=2,
        evaluator_name="deterministic-neural",
        model_bundle_path="model",
        tree_generation=1,
        rows=rows,
    )
    path = tmp_path / "patch.json"
    save_reevaluation_patch(patch, path)
    assert load_reevaluation_patch(path) == patch
    return load_reevaluation_patch(path)


def values(runner: Any) -> dict[str, Any]:
    """Capture all value payloads and their actual effective source, without flags."""
    return {
        str(n.id): {
            "direct": serialize_value(n.tree_evaluation.direct_value),
            "tree": serialize_value(n.tree_evaluation.tree_value)
            if n.tree_evaluation.tree_value is not None
            else None,
            "effective": serialize_value(n.tree_evaluation.effective_value),
            "source": n.tree_evaluation.effective_value_source.value,
        }
        for n in runner.iter_profile_nodes()
    }


@pytest.mark.parametrize("alpha", [1.0, 0.25])
@pytest.mark.parametrize("prediction", [35.0, 42.0])
def test_model_prediction_keeps_its_own_certainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endgame: tuple[Any, Any],
    alpha: float,
    prediction: float,
) -> None:
    """A real child proof must not certify a different (or equal) neural score."""
    runner = make_runner(monkeypatch, state=endgame[0], alpha=alpha)
    root, terminal = tuple(runner.iter_profile_nodes())
    evaluation = root.tree_evaluation
    assert evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert evaluation.tree_value.certainty == Certainty.FORCED
    assert evaluation.tree_value.score == 35
    assert root.tree_node.all_branches_generated and evaluation.has_exact_value()
    before_tree = evaluation.tree_value
    before_terminal = terminal.tree_evaluation.direct_value
    patch = model_patch(runner, monkeypatch, prediction, tmp_path)
    assert runner.apply_reevaluation_patch(patch) == 2
    # Assert the actual live value first: this fails on the unmodified producer.
    assert evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert patch.rows[0].is_exact is False
    assert (
        evaluation.direct_value.score == prediction
    )  # live exactness bypasses blending
    assert evaluation.tree_value == before_tree
    assert evaluation.effective_value == before_tree
    assert evaluation.effective_value_source.value == "tree_child"
    assert terminal.tree_evaluation.direct_value == before_terminal
    assert terminal.tree_evaluation.direct_value.certainty == Certainty.TERMINAL
    assert patch.rows[1].metadata["source"] == "terminal_existing_value"
    assert runner.last_reevaluation_patch_apply_metrics["recomputed"] == 0
    expected = values(runner)
    checkpoint = tmp_path / "checkpoint.json"
    runner.save_checkpoint(checkpoint)
    restored = runner_module.AnemoneMorpionSearchRunner(runner._args)
    restored.load_or_create(checkpoint, None)
    assert values(restored) == expected
    assert restored._random_generator.getstate() == runner._random_generator.getstate()
    before_count = runner.current_tree_size()
    runner.grow(1)
    restored.grow(1)
    assert runner.current_tree_size() == restored.current_tree_size() == before_count
    assert values(restored) == values(runner)


@pytest.mark.parametrize("alpha", [1.0, 0.25])
def test_partial_estimated_tree_still_updates_and_blends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alpha: float,
) -> None:
    """Ordinary partial nodes retain the existing weighted direct update rule."""
    runner = make_runner(monkeypatch, alpha=alpha)
    root = next(iter(runner.iter_profile_nodes()))
    assert not root.tree_node.all_branches_generated
    evaluation = root.tree_evaluation
    assert evaluation.tree_value.certainty == Certainty.ESTIMATE
    old = evaluation.direct_value.score
    patch = model_patch(runner, monkeypatch, 10, tmp_path)
    patch = replace(patch, rows=(patch.rows[0],))
    assert runner.apply_reevaluation_patch(patch) == len(patch.rows)
    assert evaluation.direct_value.score == pytest.approx(
        (1 - alpha) * old + alpha * 10
    )
    assert evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert evaluation.tree_value.certainty == Certainty.ESTIMATE
    assert evaluation.effective_value == evaluation.direct_value
    assert evaluation.effective_value_source.value == "direct_self"


@pytest.mark.parametrize("alpha", [1.0, 0.25])
def test_model_does_not_replace_explicit_direct_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alpha: float,
) -> None:
    """An API-valid external proof is preserved when an estimated model arrives."""
    runner = make_runner(monkeypatch, alpha=alpha)
    root = next(iter(runner.iter_profile_nodes()))
    proof = canonical_value.make_forced_value(score=9)
    root.tree_evaluation.direct_value = proof
    assert root.tree_evaluation.has_exact_value()
    patch = model_patch(runner, monkeypatch, 42, tmp_path)
    assert runner.apply_reevaluation_patch(replace(patch, rows=(patch.rows[0],))) == 0
    assert root.tree_evaluation.direct_value == proof
    assert root.tree_evaluation.effective_value == proof
    assert runner.last_reevaluation_patch_apply_metrics["skipped_exact_node_ids"] == (
        str(root.id),
    )
    checkpoint = tmp_path / "direct-proof.json"
    runner.save_checkpoint(checkpoint)
    restored = runner_module.AnemoneMorpionSearchRunner(runner._args)
    restored.load_or_create(checkpoint, None)
    assert values(restored) == values(runner)


@pytest.mark.parametrize("alpha", [1.0, 0.25])
@pytest.mark.parametrize("direct_score", [2.0, 42.0])
def test_partial_tree_proof_is_protected_independently_of_effective_exactness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alpha: float,
    direct_score: float,
) -> None:
    """Explicit API-supplied tree proof, distinct from a normal partial max backup."""
    runner = make_runner(monkeypatch, alpha=alpha)
    root = next(iter(runner.iter_profile_nodes()))
    runtime = runner._runtime
    runtime.apply_node_value_updates(
        (
            NodeValueUpdate(
                str(root.id), direct_value=3, backed_up_value=3, is_exact=True
            ),
        ),
        recompute_backups=False,
    )
    runtime.apply_node_value_updates(
        (NodeValueUpdate(str(root.id), direct_value=direct_score, is_exact=False),),
        recompute_backups=False,
    )
    assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
    assert root.tree_evaluation.has_exact_value() is (direct_score < 3)
    before = values(runner)
    patch = model_patch(runner, monkeypatch, 10, tmp_path)
    assert runner.apply_reevaluation_patch(replace(patch, rows=(patch.rows[0],))) == 0
    assert values(runner) == before


@pytest.mark.parametrize("alpha", [1.0, 0.25])
def test_explicit_authoritative_external_patch_retains_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alpha: float,
) -> None:
    """The generic external update contract continues accepting certified values."""
    runner = make_runner(monkeypatch, alpha=alpha)
    patch = model_patch(runner, monkeypatch, 10, tmp_path)
    certified = replace(
        patch.rows[0],
        direct_value=9,
        is_exact=True,
        metadata={"source": "external_certificate"},
    )
    runner.apply_reevaluation_patch(replace(patch, rows=(certified,)))
    evaluation = next(iter(runner.iter_profile_nodes())).tree_evaluation
    assert evaluation.direct_value.certainty == Certainty.FORCED
    assert evaluation.direct_value.score == 9


def test_standard_evaluators_and_explicit_switch(
    monkeypatch: pytest.MonkeyPatch,
    endgame: tuple[Any, Any],
) -> None:
    """Normal evaluators never force nonterminal scores, including model switching."""
    runner = make_runner(monkeypatch, state=endgame[0])
    root, terminal = tuple(runner.iter_profile_nodes())
    neural = neural_evaluator(42)
    for state, certainty in (
        (root.state, Certainty.ESTIMATE),
        (terminal.state, Certainty.TERMINAL),
    ):
        assert MorpionStateEvaluator().evaluate(state).certainty == certainty
        assert (
            runner._runtime.tree_manager.node_evaluator.master_state_value_evaluator.evaluate(
                state
            ).certainty
            == certainty
        )
        assert neural.evaluate(state).certainty == certainty
    before = values(runner)
    runner._runtime.set_evaluator(neural)
    assert values(runner) == before
    runner._runtime.reevaluate_nodes(list(runner.iter_profile_nodes()))
    assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert root.tree_evaluation.direct_value.score == 42
    assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
    assert terminal.tree_evaluation.direct_value.certainty == Certainty.TERMINAL


def test_proof_conditions_and_selector_remain_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    endgame: tuple[Any, Any],
) -> None:
    """All generated branches and all exact children remain necessary for proof."""
    runner = make_runner(monkeypatch, state=endgame[0])
    root, terminal = tuple(runner.iter_profile_nodes())
    evaluation = root.tree_evaluation
    branch = next(iter(root.branches_children))
    terminal_value = terminal.tree_evaluation.direct_value

    def backup() -> None:
        evaluation.backup_from_children({branch}, {branch})

    assert evaluation.tree_value.certainty == Certainty.FORCED
    root.tree_node.all_branches_generated = False
    backup()
    assert evaluation.tree_value.certainty == Certainty.ESTIMATE
    root.tree_node.all_branches_generated = True
    terminal.tree_evaluation.direct_value = canonical_value.make_estimate_value(
        score=35
    )
    backup()
    assert evaluation.tree_value.certainty == Certainty.ESTIMATE
    terminal.tree_evaluation.direct_value = terminal_value
    backup()
    assert evaluation.tree_value.certainty == Certainty.FORCED
    assert not evaluation.is_terminal()
    runtime = runner._runtime
    selector = runtime.node_selector
    assert selector.base._classify_node(root) == "opened"
    assert selector.base._classify_node(terminal) == "terminal"
    with pytest.raises(LinooSelectionError, match="unopened unresolved"):
        selector.choose_node_and_branch_to_open(
            runtime.tree, runtime.latest_tree_expansions
        )


@pytest.mark.parametrize("alpha", [1.0, 0.25])
def test_estimated_child_update_recomputes_parent_without_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alpha: float,
) -> None:
    """Real ancestor propagation associates the updated scalar with ESTIMATE."""
    runner = make_runner(monkeypatch, alpha=alpha)
    root, child = tuple(runner.iter_profile_nodes())
    old = child.tree_evaluation.direct_value.score
    patch = model_patch(runner, monkeypatch, 10, tmp_path)
    applied = runner.apply_reevaluation_patch(replace(patch, rows=(patch.rows[1],)))
    assert applied == 1
    expected = (1 - alpha) * old + alpha * 10
    assert root.tree_evaluation.tree_value.score == pytest.approx(expected)
    assert root.tree_evaluation.tree_value.certainty == Certainty.ESTIMATE
    assert child.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert runner.last_reevaluation_patch_apply_metrics["recomputed"] == 1
    assert runner._runtime.node_selector.base._classify_node(root) == "frontier"


@pytest.mark.parametrize("complete", [False, True])
def test_equal_score_candidates_keep_the_selected_certainty(
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
) -> None:
    """Use actual NodeMaxEvaluation for ties and the full-opening override."""
    runner = make_runner(monkeypatch)
    node = next(iter(runner.iter_profile_nodes()))
    evaluation = node.tree_evaluation
    node.tree_node.all_branches_generated = complete
    evaluation.direct_value = canonical_value.make_forced_value(score=3)
    evaluation.tree_value = canonical_value.make_estimate_value(score=3)
    assert evaluation.effective_value.certainty == (
        Certainty.ESTIMATE if complete else Certainty.FORCED
    )
    assert evaluation.effective_value_source.value == (
        "tree_child" if complete else "direct_self"
    )


def test_snapshot_fallback_keeps_its_legacy_authority_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endgame: tuple[Any, Any],
) -> None:
    """Expose the v1 fallback's ambiguity; do not silently redefine its API here."""
    runner = make_runner(monkeypatch, state=endgame[0])
    snapshot, _ = runner.build_training_tree_snapshot_payload()
    patch = model_patch(runner, monkeypatch, 42, tmp_path)
    rows = worker.snapshot_values_to_patch_rows(snapshot, ("0", "1"))
    assert rows[0].is_exact is True
    assert rows[0].direct_value == 34.01 and rows[0].backed_up_value == 35
    runner.apply_reevaluation_patch(replace(patch, rows=rows))
    root = next(iter(runner.iter_profile_nodes()))
    assert root.tree_evaluation.direct_value.certainty == Certainty.FORCED
    assert root.tree_evaluation.effective_value.score == 35


@pytest.mark.parametrize("alpha", [1.0, 0.25])
def test_streaming_window_model_values_match_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endgame: tuple[Any, Any],
    alpha: float,
) -> None:
    """Run against the isolated streaming integration when that API is available."""
    if not hasattr(
        worker.MorpionActiveModelNodeReevaluationEvaluator, "evaluate_node_window"
    ):
        pytest.skip("streaming API belongs to the separately pinned PR")
    runner = make_runner(monkeypatch, state=endgame[0], alpha=alpha)
    snapshot, _ = runner.build_training_tree_snapshot_payload()
    legacy = model_patch(runner, monkeypatch, 42, tmp_path)
    adapter = worker.MorpionActiveModelNodeReevaluationEvaluator(tmp_path / "model")
    assert adapter.evaluate_node_window(snapshot.nodes) == legacy.rows
    runner.apply_reevaluation_patch(legacy)
    root, terminal = tuple(runner.iter_profile_nodes())
    assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert root.tree_evaluation.effective_value.score == 35
    assert terminal.tree_evaluation.direct_value.certainty == Certainty.TERMINAL


def test_streaming_worker_actual_proof_matches_flat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endgame: tuple[Any, Any],
) -> None:
    """Exercise real bounded source reconstruction, publication and live application."""
    if not hasattr(
        worker.MorpionActiveModelNodeReevaluationEvaluator, "evaluate_node_window"
    ):
        pytest.skip("streaming API belongs to the separately pinned PR")
    from chipiron.environments.morpion.bootstrap import (
        MorpionBootstrapPaths,
        load_reevaluation_cursor,
    )
    from tests.environments.test_morpion_bootstrap_reevaluation_worker import (
        _artifact_pipeline_args,
        _write_active_model,
        _write_manifest_with_sharded_snapshot,
        _write_manifest_with_snapshot,
    )

    observed = []
    for mode in ("flat", "sharded_full", "sharded_indexed"):
        runner = make_runner(monkeypatch, state=endgame[0])
        snapshot, _ = runner.build_training_tree_snapshot_payload()
        paths = MorpionBootstrapPaths.from_work_dir(tmp_path / mode)
        paths.ensure_directories()
        _write_active_model(paths)
        writer = (
            _write_manifest_with_snapshot
            if mode == "flat"
            else _write_manifest_with_sharded_snapshot
        )
        writer(paths, generation=7, snapshot=snapshot)
        adapter = worker.MorpionActiveModelNodeReevaluationEvaluator(tmp_path / "model")
        adapter._evaluator = neural_evaluator(42)
        monkeypatch.setattr(
            worker,
            "build_active_model_reevaluation_evaluator",
            lambda adapter=adapter, **_: adapter,
        )
        result = worker.run_morpion_reevaluation_worker_once(
            _artifact_pipeline_args(paths.work_dir),
            max_nodes_per_patch=2,
            now_unix_s=123456.0,
            patch_id="same-identity",
            use_streaming_sharded_reader=mode == "sharded_indexed",
            reader_cache_dir=paths.work_dir / "reader-cache",
        )
        assert result.patch_written
        patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
        assert patch.rows[0].direct_value == 42 and patch.rows[0].is_exact is False
        assert patch.rows[1].direct_value == 35 and patch.rows[1].is_terminal is True
        assert runner.apply_reevaluation_patch(patch) == 2
        root, terminal = tuple(runner.iter_profile_nodes())
        assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
        assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
        assert terminal.tree_evaluation.direct_value.certainty == Certainty.TERMINAL
        observed.append((
            patch,
            load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path),
            values(runner),
        ))
    assert observed[0] == observed[1] == observed[2]


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
@pytest.mark.parametrize("use_backup", [True, False])
@pytest.mark.parametrize("prediction", [None, 42.0])
def test_dataset_actual_proof_target_certainty_mismatch_is_retained_for_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endgame: tuple[Any, Any],
    policy: PvFamilyTargetPolicy,
    use_backup: bool,
    prediction: float | None,
) -> None:
    """Characterize v1 scientific policy without changing its rows or confidence."""
    runner = make_runner(monkeypatch, state=endgame[0])
    if prediction is not None:
        runner.apply_reevaluation_patch(
            model_patch(runner, monkeypatch, prediction, tmp_path)
        )
    snapshot, _ = runner.build_training_tree_snapshot_payload()
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        require_exact_or_terminal=True,
        use_backed_up_value=use_backup,
        dataset_family_target_policy=policy,
    )
    rows = extract_rows_from_training_snapshot(
        args=args, snapshot=snapshot, generation=1
    )
    root_row = rows.rows[0]
    assert root_row.is_exact and not root_row.is_terminal
    # Normal production-reachable node: effective/tree exact 35, direct estimated 34.01.
    direct = 34.01 if prediction is None else prediction
    assert root_row.metadata["raw_target"] == (35 if use_backup else direct)
    exact_then = policy.startswith("pv_exact_then_")
    family_max = max(direct, 35)
    assert root_row.target_value == (
        35 if use_backup else family_max if exact_then else direct
    )
    terminal_row = rows.rows[1]
    assert terminal_row.is_terminal and terminal_row.is_exact
    assert terminal_row.target_value == (
        family_max if not use_backup and exact_then else 35
    )


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
@pytest.mark.parametrize("use_backup", [True, False])
@pytest.mark.parametrize("exact_only", [True, False])
def test_synthetic_direct_exact_tree_estimate_dataset_consequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: PvFamilyTargetPolicy,
    use_backup: bool,
    exact_only: bool,
) -> None:
    """Actual NodeMaxEvaluation/API-valid fixture; no claim of normal reachability."""
    runner = make_runner(monkeypatch)
    root, child = tuple(runner.iter_profile_nodes())
    root.tree_evaluation.direct_value = canonical_value.make_forced_value(score=9)
    root.tree_evaluation.tree_value = canonical_value.make_estimate_value(score=3)
    child.tree_evaluation.direct_value = canonical_value.make_estimate_value(score=2)
    child.tree_evaluation.tree_value = canonical_value.make_estimate_value(score=2)
    assert root.tree_evaluation.has_exact_value()
    snapshot, _ = runner.build_training_tree_snapshot_payload()
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        dataset_family_target_policy=policy,
        require_exact_or_terminal=exact_only,
        use_backed_up_value=use_backup,
    )
    rows = extract_rows_from_training_snapshot(
        args=args, snapshot=snapshot, generation=1
    )
    row = rows.rows[0]
    target = 3 if use_backup else 9
    assert row.is_exact and row.metadata["raw_target"] == target
    assert row.target_value == target
    assert row.metadata["family_exact_target"] == target
    assert row.metadata["family_target_rule"] == (
        "pv_exact_family"
        if policy.startswith("pv_exact_then_")
        else "hard_exact_anchor"
    )
    if exact_only:
        assert len(rows.rows) == 1
    else:
        peer = rows.rows[1]
        expected_peer = {
            "none": 2,
            "pv_mean_prediction": 5.5,
            "pv_min_prediction": 2,
            "pv_blend_mean_prediction": 2.875,
            "pv_blend_min_prediction": 2,
        }.get(policy, target)
        assert peer.target_value == expected_peer
        assert not peer.is_exact
        assert peer.metadata["family_exact_target"] == target
        assert peer.metadata["family_has_exact_or_terminal"] is True
