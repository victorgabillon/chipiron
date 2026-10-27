"""Real scientific Dataset/Training receipts preserve independent candidate chains."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.run_state import (
        MorpionBootstrapRunState,
    )

import pytest
import torch

from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import (
    MorpionBootstrapRolloutConfig,
    MorpionBootstrapSearchConfig,
    bootstrap_config_from_args,
)
from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    publication_records,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_growth_stage,
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
    default_search_args,
)
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_experiment,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRegressorArgs,
    build_morpion_regressor,
    save_morpion_model_bundle,
)
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)


def setup_continual(
    root: Path,
    *,
    automatic: bool = False,
    extra_steps: int = 2,
    proof: bool = False,
    maximum: int = 5000,
    training_cadence: int = 1,
) -> tuple[MorpionBootstrapArgs, MorpionBootstrapPaths]:
    """Start at the original single root with an explicit zero ESTIMATE evaluator."""
    args = MorpionBootstrapArgs(
        work_dir=root,
        pipeline_mode="artifact_pipeline",
        runtime_checkpoint_format="sharded",
        training_export_mode="sharded",
        save_after_seconds=0.0,
        tree_branch_limit=1000000,
        dataset_confidence_policy="selected_value_v2",
        training_device="cpu",
        validation_fraction=0.25,
        validation_seed=17,
        skip_evaluator_diagnostics=True,
        search=MorpionBootstrapSearchConfig(
            rollout=MorpionBootstrapRolloutConfig(
                enabled=True, max_extra_steps=64, random_seed=23
            )
        ),
        evaluators_config=MorpionEvaluatorsConfig(
            evaluators={
                name: MorpionEvaluatorSpec(
                    name=name,
                    model_type="linear",
                    hidden_sizes=None,
                    num_epochs=1,
                    batch_size=4,
                    learning_rate=rate,
                )
                for name, rate in [("linear_a", 0.001), ("linear_b", 0.003)]
            }
        ),
    )
    args = replace(
        args,
        max_growth_steps_per_cycle=2,
        max_rows=128,
        use_backed_up_value=True,
        continual_learning=MorpionContinualLearningPolicy(
            automatic_activation=automatic,
            max_tree_nodes=maximum,
            seed=1 if proof else 0,
            training_every_generations=training_cadence,
        ),
        search=replace(
            args.search,
            rollout=replace(args.search.rollout, max_extra_steps=extra_steps),
        ),
    )
    if proof:
        args = replace(
            args,
            validation_fraction=0.5,
            evaluators_config=MorpionEvaluatorsConfig(
                evaluators={
                    name: replace(
                        spec,
                        feature_subset_name="moves_only",
                        feature_names=("moves",),
                        num_epochs=10,
                        batch_size=1,
                        learning_rate=0.01,
                    )
                    for name, spec in args.resolved_evaluators_config().evaluators.items()
                }
            ),
        )
    seed = root.parent / "zero"
    model_args = MorpionRegressorArgs(
        model_kind="linear", feature_subset_name="zero_moves", feature_names=("moves",)
    )
    model = build_morpion_regressor(model_args)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    save_morpion_model_bundle(
        model,
        seed,
        model_args=model_args,
        metadata={"purpose": "initial_zero_search_evaluator", "certainty": "ESTIMATE"},
    )
    initialize_scientific_experiment(
        args, bootstrap_config_from_args(args), seed_model=seed, initial_nodes=1
    )
    return args, MorpionBootstrapPaths.from_work_dir(root)


def grow(args: MorpionBootstrapArgs) -> MorpionBootstrapRunState:
    """Use the actual four-worker Growth entrypoint and complete scientific exports."""
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    state = load_bootstrap_run_state(paths.run_state_path)
    producer = publication_records(state.metadata)[-1]["scientific_identity"][
        "producer"
    ]["identity"]
    runner = AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            runtime_checkpoint_format="sharded",
            random_seed=4,
            search_args=default_search_args(rollout=args.search.rollout),
        )
    )
    return run_pipeline_growth_stage(
        args,
        runner,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
        scientific_v2_producer=producer,
        closed_scientific_snapshot=True,
    )


@pytest.mark.fast
def test_three_accepted_rounds_and_idempotent_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W1→W2→W3 for every candidate; receipts rather than active-model paths win."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths = setup_continual(tmp_path / "workspace")
    previous = None
    for round_number in range(1, 4):
        state = grow(args)
        run_dataset(args, generation=state.generation)
        result = run_pipeline_training_stage(args, generation=state.generation)
        assert result.training_status == "done"
        state = load_bootstrap_run_state(paths.run_state_path)
        reference = publication_records(state.metadata)[-1][training.ACCEPTED]
        proposal = training._proposal(paths, reference)
        assert proposal["admission"]["round_number"] == round_number
        assert set(proposal["models"]) == set(
            args.resolved_evaluators_config().evaluators
        )
        for name in proposal["models"]:
            parent = proposal["admission"]["parents"][name]
            if previous is None:
                assert parent is None
            else:
                assert parent["accepted_training"] == previous
        authority = paths.run_state_path.read_bytes()
        with monkeypatch.context() as retry_patch:
            retry_patch.setattr(
                training,
                "_compare_incumbent",
                lambda *a, **k: pytest.fail("Retry trained twice"),
            )
            assert (
                run_pipeline_training_stage(args, generation=state.generation) == result
            )
            assert paths.run_state_path.read_bytes() == authority
        previous = reference


@pytest.mark.fast
def test_automatic_activation_preserves_proofs_and_recovers_exact_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Honest held-out proof targets justify two real transitions; no forced winner."""
    from valanga.evaluations import Certainty

    from chipiron.environments.morpion.bootstrap import model_activation as activation
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        committed_evaluator,
    )
    from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
    from tests.environments.test_morpion_value_certainty import endgame

    local_leaderboard(monkeypatch, tmp_path)
    # This explicit semantic fixture is a reachable endgame. The fresh Greek-cross
    # experiments are tested separately; it is not presented as their initial tree.
    position, _ = endgame.__wrapped__()
    with monkeypatch.context() as context:
        context.setattr(runner_module, "initial_state", lambda: position)
        args, paths = setup_continual(
            tmp_path / "workspace", automatic=True, proof=True
        )
    for _ in range(3):
        state = grow(args)
        run_dataset(args, generation=state.generation)
        run_pipeline_training_stage(args, generation=state.generation)
    # The third Growth boundary has activated the first two justified decisions.
    state = load_bootstrap_run_state(paths.run_state_path)
    transitions = [
        r
        for r in publication_records(state.metadata)
        if "activation_request_sha256" in r
    ]
    assert len(transitions) >= 2
    for record in transitions:
        completion = activation.activation_completion(paths, state, record)
        assert completion["refresh"]
    active, bundle = committed_evaluator(paths.work_dir, state)
    restored, effective = activation._runner(paths, state)
    try:
        restored.load_or_create(
            paths.work_dir / state.latest_runtime_checkpoint_path, bundle, effective
        )
        root = restored._require_runtime().tree.root_node
        assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
        assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
        assert root.tree_evaluation.effective_value.score == 35
        restored.validate_model_refresh_certificates()
    finally:
        restored.discard_uncommitted_runtime()
    # Simulate death after durable request creation, before the atomic transition.
    with monkeypatch.context() as context:
        context.setattr(
            activation,
            "activate_model",
            lambda *a: (_ for _ in ()).throw(RuntimeError("interrupted")),
        )
        with pytest.raises(RuntimeError, match="interrupted"):
            activation.activate_selected_round_automatically(paths)
    activation.activate_selected_round_automatically(paths)
    recovered = load_bootstrap_run_state(paths.run_state_path)
    assert committed_evaluator(paths.work_dir, recovered)[0] != active
    authority = paths.run_state_path.read_bytes()
    activation.activate_selected_round_automatically(paths)
    assert paths.run_state_path.read_bytes() == authority


@pytest.mark.fast
def test_cross_generation_attempt_cannot_skip_a_newly_accepted_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct concurrent stage invocations cannot accept divergent candidate successors."""
    local_leaderboard(monkeypatch, tmp_path)
    # Automatic workers honor this cadence. Explicit stage calls below simulate
    # two admitted generations, exercising acceptance independently of scheduling.
    args, paths = setup_continual(tmp_path / "workspace", training_cadence=10)
    first = grow(args)
    run_dataset(args, generation=first.generation)
    accept = training.accept_training
    winning_authority = None

    def overtake(original_paths: MorpionBootstrapPaths, reference: dict) -> object:
        nonlocal winning_authority
        with monkeypatch.context() as context:
            context.setattr(training, "accept_training", accept)
            newer = grow(args)
            run_dataset(args, generation=newer.generation)
            run_pipeline_training_stage(args, generation=newer.generation)
        winning_authority = paths.run_state_path.read_bytes()
        return accept(original_paths, reference)

    monkeypatch.setattr(training, "accept_training", overtake)
    with pytest.raises(ValueError, match="Stale Training parents"):
        run_pipeline_training_stage(args, generation=first.generation)
    assert paths.run_state_path.read_bytes() == winning_authority
    records = publication_records(
        load_bootstrap_run_state(paths.run_state_path).metadata
    )
    assert training.ACCEPTED not in next(
        r for r in records if r["generation"] == first.generation
    )
    assert training.ACCEPTED in records[-1]


@pytest.mark.fast
def test_completed_node_budget_never_restarts_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap is a durable scientific stop, including a zero-step final boundary."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths = setup_continual(tmp_path / "workspace", maximum=5)
    state = grow(args)
    assert state.tree_size_at_last_save <= 5
    assert state.metadata["experiment_termination"]["maximum_unique_nodes"] == 5
    authority = paths.run_state_path.read_bytes()
    assert grow(args) == load_bootstrap_run_state(paths.run_state_path)
    assert paths.run_state_path.read_bytes() == authority


def test_automatic_request_renews_after_real_m1_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted automatic transition keeps the selection and both exact requests."""
    from chipiron.environments.morpion.bootstrap import model_activation as activation
    from chipiron.environments.morpion.bootstrap.runtime.runner import (
        AnemoneMorpionSearchRunner,
    )
    from tests.environments.test_morpion_model_activation_edges import propose

    local_leaderboard(monkeypatch, tmp_path)
    args, paths = setup_continual(
        tmp_path / "workspace", automatic=True, extra_steps=64, proof=True
    )
    state = grow(args)
    run_dataset(args, generation=state.generation)
    run_pipeline_training_stage(args, generation=state.generation)
    assert propose(args).patch_written
    with monkeypatch.context() as context:

        def interrupted(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("interrupted_after_drain")

        context.setattr(
            AnemoneMorpionSearchRunner, "refresh_model_estimates", interrupted
        )
        with pytest.raises(RuntimeError, match="interrupted_after_drain"):
            activation.activate_selected_round_automatically(paths)
    drained = load_bootstrap_run_state(paths.run_state_path)
    assert drained.generation == state.generation + 1
    # A real learner can finish another round on the drained M1 head before
    # recovery. Its comparison against M1 must never authorize replacement of M2.
    run_dataset(args, generation=drained.generation)
    run_pipeline_training_stage(args, generation=drained.generation)
    directory = paths.pipeline_dir / "automatic_activation"
    [original] = list(directory.glob("*.json"))
    original_bytes = original.read_bytes()
    activation.activate_selected_round_automatically(paths)
    recovered = load_bootstrap_run_state(paths.run_state_path)
    assert recovered.generation == drained.generation + 1
    assert len(list(directory.glob("*.json"))) == 2
    assert original.read_bytes() == original_bytes
    authority = paths.run_state_path.read_bytes()
    activation.activate_selected_round_automatically(paths)
    assert paths.run_state_path.read_bytes() == authority
