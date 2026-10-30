"""Scoped CPU threads preserve defaults and repeat accepted scientific lineages."""

import argparse
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import pytest
import torch

from chipiron.environments.morpion.bootstrap import candidate_resources, cycle_training
from chipiron.environments.morpion.bootstrap import training_publication as publication
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.candidate_job import (
    CandidateJob,
    run_candidate,
)
from chipiron.environments.morpion.bootstrap.candidate_threading import (
    candidate_threads,
    validate_thread_provenance,
)
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    bootstrap_config_sha256,
)
from chipiron.environments.morpion.bootstrap.continual_operator import (
    add_continual_arguments,
    prepare_continual_experiment,
)
from chipiron.environments.morpion.bootstrap.control import MorpionBootstrapControl
from chipiron.environments.morpion.bootstrap.evaluator_config import (
    MorpionEvaluatorsConfig,
    MorpionEvaluatorSpec,
)
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    transformer_v1_morpion_evaluator_spec,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    file_digest,
    publication_records,
    read_object,
)
from chipiron.environments.morpion.bootstrap.launcher import launcher_args_from_cli
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.continuation import (
    MorpionTrainingContinuation,
)
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_candidate_execution import assert_state_equal
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_continual_training import rows_file

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def cpu_scope(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Bound tests independently of host capacity and restore their ambient setting."""
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", "1")
    monkeypatch.setenv("CHIPIRON_CANDIDATE_CPU_BUDGET", "4")
    monkeypatch.setenv("CHIPIRON_CANDIDATE_RESERVED_CPUS", "2")
    monkeypatch.setattr(candidate_resources, "cpu_capacity", lambda: 8.0)
    try:
        yield
    finally:
        torch.set_num_threads(original)


@pytest.mark.fast
def test_historical_identities_and_explicit_policy() -> None:
    """Omission keeps both pre-I3 genesis hashes and the old policy shape."""
    policy = MorpionContinualLearningPolicy()
    old = asdict(policy)
    del old["transformer_cpu_threads"]
    assert policy.to_metadata() == old
    for configured, expected in (
        (None, "92495cfd87ae654770948bbd2d036acb1c55fea332fb628e965747fc94721169"),
        (policy, "b878ebf83b6de50656372f0a7bcf1d15fe71546f9e19faeb5ad40fccbd05827d"),
    ):
        args = MorpionBootstrapArgs(
            work_dir="/tmp/fixed-i3-identity",
            training_device="cpu",
            continual_learning=configured,
            evaluator_family_preset=CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
        )
        assert bootstrap_config_sha256(bootstrap_config_from_args(args)) == expected
    parser = argparse.ArgumentParser()
    add_continual_arguments(parser)
    assert parser.parse_args([]).transformer_cpu_threads is None
    assert (
        parser.parse_args(["--transformer-cpu-threads", "2"]).transformer_cpu_threads
        == 2
    )
    hashes = {
        bootstrap_config_sha256(
            bootstrap_config_from_args(
                replace(
                    args,
                    continual_learning=replace(policy, transformer_cpu_threads=threads),
                )
            )
        )
        for threads in (None, 1, 2)
    }
    assert len(hashes) == 3


@pytest.mark.fast
@pytest.mark.parametrize("value", [0, 3, 4, True, 2.0, "2"])
def test_invalid_thread_option(value: object) -> None:
    """No automatic tuning or truthy coercion of scientific execution settings."""
    with pytest.raises(ValueError, match="transformer_cpu_threads"):
        MorpionContinualLearningPolicy(transformer_cpu_threads=value)


@pytest.mark.fast
@pytest.mark.parametrize("error", [None, RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("previous", [1, 3])
def test_scope_restores_after_success_or_failure(
    error: type[BaseException] | None,
    previous: int,
) -> None:
    """Unwinding all ordinary/cancellation exits restores the Training process only."""
    interop = torch.get_num_interop_threads()
    torch.set_num_threads(previous)
    try:
        with candidate_threads(2, name="transformer_v1", device="cpu") as evidence:
            assert torch.get_num_threads() == 2
            if error is not None:
                raise error  # noqa: TRY301 - exercise unwinding the actual scope
    except BaseException as exc:
        assert type(exc) is error
    assert torch.get_num_threads() == previous
    assert torch.get_num_interop_threads() == interop
    assert evidence["actual_intra_op_threads"] == 2
    assert evidence["restored_intra_op_threads"] == previous


@pytest.mark.fast
def test_omitted_option_does_not_set_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy execution retains its ambient setting without invoking the thread setter."""
    with monkeypatch.context() as patch:
        patch.setattr(
            torch, "set_num_threads", lambda *_: pytest.fail("Default changed threads")
        )
        with candidate_threads(None, name="linear", device="auto") as evidence:
            assert evidence == {}


@pytest.mark.fast
@pytest.mark.parametrize("device,workers", [("auto", "1"), ("cuda", "1"), ("cpu", "2")])
def test_configuration_rejects_unsupported_execution(
    monkeypatch: pytest.MonkeyPatch, device: str, workers: str
) -> None:
    """Even a parallel plan that would resolve to one worker is not admitted."""
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", workers)
    args = MorpionBootstrapArgs(
        work_dir="unused",
        training_device=device,
        continual_learning=MorpionContinualLearningPolicy(transformer_cpu_threads=2),
        evaluator_family_preset=CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    )
    with pytest.raises(ValueError, match="explicit CPU and serial"):
        publication.configuration(
            args, args.resolved_evaluators_config(), MorpionBootstrapControl()
        )
    with pytest.raises(ValueError, match="explicit CPU and serial"):
        candidate_resources.resolve_candidate_plan(9, args.continual_learning, device)


@pytest.mark.fast
@pytest.mark.parametrize("capacity,budget", [(3.9, "4"), (8.0, "1")])
def test_capacity_refused_before_training(
    monkeypatch: pytest.MonkeyPatch, capacity: float, budget: str
) -> None:
    """Both quota/affinity capacity and the explicit budget preserve pipeline reserves."""
    monkeypatch.setattr(candidate_resources, "cpu_capacity", lambda: capacity)
    monkeypatch.setenv("CHIPIRON_CANDIDATE_CPU_BUDGET", budget)
    policy = MorpionContinualLearningPolicy(transformer_cpu_threads=2)
    with pytest.raises(ValueError, match="insufficient"):
        candidate_resources.resolve_candidate_plan(9, policy, "cpu")
    with (
        pytest.raises(ValueError, match="insufficient"),
        candidate_threads(2, name="transformer_v1", device="cpu"),
    ):
        pytest.fail("Insufficient capacity reached training")
    assert torch.get_num_threads() == 1


@pytest.mark.fast
def test_provenance_cannot_be_dropped_or_changed() -> None:
    """Both publication and candidate-parent loading enforce the recorded policy."""
    good = {
        "cpu_threading": {"requested_intra_op_threads": 2, "actual_intra_op_threads": 2}
    }
    validate_thread_provenance(good, 2)
    for metadata, requested in (
        ({}, 2),
        (good, None),
        (good, 1),
        (
            {
                "cpu_threading": {
                    "requested_intra_op_threads": 2,
                    "actual_intra_op_threads": 1,
                }
            },
            2,
        ),
    ):
        with pytest.raises(ValueError, match="CPU thread"):
            validate_thread_provenance(metadata, requested)


@pytest.mark.fast
@pytest.mark.parametrize(
    "device,preset",
    [("auto", CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET), ("cpu", None)],
)
def test_preparation_rejects_invalid_mode_before_writing(
    tmp_path: Path, device: str, preset: str | None
) -> None:
    """The CLI must not create an unusable experiment for an unsupported thread mode."""
    args = MorpionBootstrapArgs(
        work_dir=tmp_path / "workspace",
        training_device=device,
        evaluator_family_preset=preset,
    )
    with pytest.raises(ValueError, match="Transformer CPU threads require"):
        prepare_continual_experiment(
            args, MorpionContinualLearningPolicy(transformer_cpu_threads=2)
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.fast
def test_launcher_explicit_cpu_thread_option() -> None:
    """The documented preparation flags reach the actual immutable experiment policy."""
    base = [
        "--scientific",
        "--scientific-operation",
        "prepare-continuous",
        "--work-dir",
        "/unused/new",
    ]
    legacy = launcher_args_from_cli(base)
    assert legacy.bootstrap_args.training_device == "auto"
    assert legacy.continual_policy.transformer_cpu_threads is None
    explicit = launcher_args_from_cli([
        *base,
        "--training-device",
        "cpu",
        "--transformer-cpu-threads",
        "2",
        "--evaluator-family",
        CANONICAL_8_LINEAR_MLP_TRANSFORMER_V1_PRESET,
    ])
    assert explicit.bootstrap_args.training_device == "cpu"
    assert explicit.continual_policy.transformer_cpu_threads == 2
    assert (
        "transformer_v1"
        in explicit.bootstrap_args.resolved_evaluators_config().evaluators
    )


def evaluators() -> MorpionEvaluatorsConfig:
    """Keep the canonical transformer recipe and two independent cheap competitors."""
    return MorpionEvaluatorsConfig(
        evaluators={
            "linear_a": MorpionEvaluatorSpec(
                name="linear_a",
                model_type="linear",
                num_epochs=5,
                hidden_sizes=None,
                batch_size=64,
                learning_rate=0.001,
            ),
            "transformer_v1": transformer_v1_morpion_evaluator_spec(),
            "linear_b": MorpionEvaluatorSpec(
                name="linear_b",
                model_type="linear",
                num_epochs=5,
                learning_rate=0.003,
                hidden_sizes=None,
                batch_size=64,
            ),
        }
    )


def test_real_transformer_optimizer_load_and_failed_attempt_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspect Adam before its first update; an interrupted attempt cannot change its parent."""
    rows = rows_file(tmp_path)

    def job(output: str, parent: Path | None = None) -> CandidateJob:
        continuation = MorpionTrainingContinuation(
            "transformer_v1",
            711,
            parent_bundle=None if parent is None else str(parent),
            parent_bundle_sha256=None if parent is None else artifact_digest(parent),
            parent_weights_sha256=None
            if parent is None
            else file_digest(parent / "param.pt"),
        )
        args = cycle_training.morpion_training_args_from_evaluator_spec(
            spec=transformer_v1_morpion_evaluator_spec(),
            dataset_file=rows,
            output_dir=tmp_path / output,
            shuffle=True,
            validation_fraction=0.2,
            validation_seed=0,
            device="cpu",
            continuation=continuation,
        )
        return CandidateJob(
            "transformer_v1",
            MorpionStreamingTrainingArgs(args, row_chunk_size=5),
            cpu_threads=2,
        )

    run_candidate(job("parent"))
    parent = tmp_path / "parent"
    identity = artifact_digest(parent)
    expected = torch.load(parent / "optimizer.pt", weights_only=True)
    step = torch.optim.Adam.step
    visited = []

    def interrupted(
        optimizer: torch.optim.Adam, *args: object, **kwargs: object
    ) -> object:
        assert torch.get_num_threads() == 2
        assert_state_equal(optimizer.state_dict(), expected)
        message = "injected first-update interruption"
        raise RuntimeError(message)

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", interrupted)
        with pytest.raises(RuntimeError, match="first-update interruption"):
            run_candidate(job("failed", parent))
    assert torch.get_num_threads() == 1
    assert artifact_digest(parent) == identity

    def checked(optimizer: torch.optim.Adam, *args: object, **kwargs: object) -> object:
        if not visited:
            assert_state_equal(optimizer.state_dict(), expected)
        visited.append(True)
        return step(optimizer, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", checked)
        run_candidate(job("retry", parent))
    run_candidate(job("replay", parent))
    assert visited
    assert torch.get_num_threads() == 1
    assert artifact_digest(parent) == identity
    for filename in ("param.pt", "optimizer.pt"):
        assert_state_equal(
            torch.load(tmp_path / "retry" / filename, weights_only=True),
            torch.load(tmp_path / "replay" / filename, weights_only=True),
        )
    for setting in (None, 1):
        with pytest.raises(ValueError, match="CPU thread"):
            run_candidate(
                replace(job(f"changed-{setting}", parent), cpu_threads=setting)
            )
        assert not (tmp_path / f"changed-{setting}").exists()


def test_publication_refuses_missing_thread_evidence_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed model without actual-thread evidence cannot become an accepted round."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        metadata,
    )

    local_leaderboard(monkeypatch, tmp_path)
    args, paths = setup_continual(
        tmp_path / "workspace",
        memory_floor_mb=1,
        evaluators=evaluators(),
        transformer_cpu_threads=2,
    )
    state = grow(args)
    run_dataset(args, generation=state.generation)
    authority = paths.run_state_path.read_bytes()
    update = metadata.update_saved_manifest_metadata

    def drop_evidence(output: Path, value: dict) -> None:
        update(
            output, {key: item for key, item in value.items() if key != "cpu_threading"}
        )

    with monkeypatch.context() as patch:
        patch.setattr(metadata, "update_saved_manifest_metadata", drop_evidence)
        with pytest.raises(ValueError, match="CPU thread provenance"):
            run_pipeline_training_stage(args, generation=state.generation)
    assert torch.get_num_threads() == 1
    assert paths.run_state_path.read_bytes() == authority
    result = run_pipeline_training_stage(args, generation=state.generation)
    assert result.training_status == "done"


@pytest.mark.parametrize("threads", [None, 1, 2])
def test_repeatable_three_accepted_rounds_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threads: int | None
) -> None:
    """Replay every model/Adam/RNG lineage, including losers, and receipt-only retries."""
    local_leaderboard(monkeypatch, tmp_path)
    histories = []
    observed = []
    trainer = cycle_training.train_morpion_regressor_streaming

    def observe(args: object) -> object:
        observed.append((
            args.training_args.continuation.candidate_name,
            torch.get_num_threads(),
        ))
        return trainer(args)

    monkeypatch.setattr(cycle_training, "train_morpion_regressor_streaming", observe)
    for replica in range(2):
        args, paths = setup_continual(
            tmp_path / str(replica) / "workspace",
            memory_floor_mb=1,
            evaluators=evaluators(),
            transformer_cpu_threads=threads,
        )
        previous = None
        history = []
        for _ in range(3):
            state = grow(args)
            run_dataset(args, generation=state.generation)
            result = run_pipeline_training_stage(args, generation=state.generation)
            assert torch.get_num_threads() == 1
            accepted = load_bootstrap_run_state(paths.run_state_path)
            reference = publication_records(accepted.metadata)[-1][publication.ACCEPTED]
            proposal = publication._proposal(paths, reference)
            current = {}
            for name, model in proposal["models"].items():
                bundle = paths.work_dir / model["path"]
                metadata = read_object(bundle / "morpion_manifest.json")["metadata"]
                continuation = metadata["continuation"]
                parent = proposal["admission"]["parents"][name]
                if previous is None:
                    assert parent is None
                    assert not continuation["optimizer_state_loaded"]
                else:
                    assert parent["accepted_training"] == previous
                    assert continuation["optimizer_state_loaded"]
                    assert (
                        continuation["starting_optimizer_step_count"]
                        == history[-1]["models"][name]["metadata"]["continuation"][
                            "final_optimizer_step_count"
                        ]
                    )
                    assert (
                        continuation["starting_weights_sha256"]
                        == history[-1]["models"][name]["metadata"]["continuation"][
                            "final_weights_sha256"
                        ]
                    )
                validate_thread_provenance(
                    metadata, threads if name == "transformer_v1" else None
                )
                current[name] = {"bundle": bundle, "metadata": metadata}
            losses = proposal["status"]["metadata"]["continual_selection"][
                "candidate_validation_losses"
            ]
            selected = min(losses, key=lambda name: (losses[name], name))
            assert proposal["status"]["selected_evaluator_name"] == selected
            authority = paths.run_state_path.read_bytes()
            calls = len(observed)
            assert (
                run_pipeline_training_stage(args, generation=state.generation) == result
            )
            assert len(observed) == calls
            assert paths.run_state_path.read_bytes() == authority
            changed = replace(
                args,
                continual_learning=replace(
                    args.continual_learning,
                    transformer_cpu_threads=1 if threads == 2 else 2,
                ),
            )
            with pytest.raises(ValueError, match="policy changed"):
                run_pipeline_training_stage(changed, generation=state.generation)
            assert paths.run_state_path.read_bytes() == authority
            history.append({"models": current, "selected": selected, "losses": losses})
            previous = reference
        histories.append(history)
    assert all(
        count == ((threads or 1) if name == "transformer_v1" else 1)
        for name, count in observed
    )
    for left, right in zip(*histories, strict=True):
        assert left["selected"] == right["selected"]
        assert left["losses"] == right["losses"]
        for name, a in left["models"].items():
            b = right["models"][name]
            for filename in ("param.pt", "optimizer.pt"):
                assert_state_equal(
                    torch.load(a["bundle"] / filename, weights_only=True),
                    torch.load(b["bundle"] / filename, weights_only=True),
                )
            assert a["metadata"]["state_split"] == b["metadata"]["state_split"]
            for key in (
                "rng_before_initialization",
                "rng_before_training",
                "rng_after_training",
                "optimizer_steps",
                "training_examples_seen",
            ):
                assert (
                    a["metadata"]["continuation"][key]
                    == b["metadata"]["continuation"][key]
                )
