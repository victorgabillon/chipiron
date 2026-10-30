"""The canonical scientific launcher uses admitted coordination for real rounds."""

import json
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.bootstrap import attempts
from chipiron.environments.morpion.bootstrap import (
    cycle_training,
)
from chipiron.environments.morpion.bootstrap import (
    model_activation as activation,
)
from chipiron.environments.morpion.bootstrap import (
    training_publication as training,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    committed_evaluator,
    file_digest,
    publication_records,
    read_object,
)
from chipiron.environments.morpion.bootstrap.launcher import (
    launcher_args_from_cli,
    run_morpion_bootstrap_experiment,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard
from tests.environments.test_morpion_continual_publication import setup_continual
from tests.environments.test_morpion_value_certainty import endgame as endgame

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionPipelineTrainingStatusArtifact,
    )


@pytest.mark.fast
@pytest.mark.parametrize("candidate_workers", [1, 2])
def test_launcher_two_rounds_use_engine_and_preserve_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_workers: int,
    endgame: tuple[Any, Any],
) -> None:
    """Real targets, losing parents, activation and next search cross the new core."""
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", str(candidate_workers))
    monkeypatch.setenv("CHIPIRON_CANDIDATE_RESERVED_CPUS", "1")
    local_leaderboard(monkeypatch, tmp_path)
    position, _ = endgame
    with monkeypatch.context() as genesis:
        genesis.setattr(runner_module, "initial_state", lambda: position)
        _, paths = setup_continual(
            tmp_path / "workspace", proof=True, memory_floor_mb=1
        )

    engine = attempts.run_attempt
    admit = training.admit_training
    configure = training.configuration
    restore = runner_module.AnemoneMorpionSearchRunner.load_or_create
    grow = runner_module.AnemoneMorpionSearchRunner.grow
    refresh = runner_module.AnemoneMorpionSearchRunner.refresh_model_estimates
    events: list[str] = []
    engine_calls = 0
    refresh_calls = 0
    config_started: list[float] = []
    admission_started: list[float] = []
    publication_started: list[float] = []
    computed: list[MorpionPipelineTrainingStatusArtifact] = []
    loaded_models: list[str] = []
    growth_steps: list[int] = []

    def observed_configuration(*args: Any, **kwargs: Any) -> Any:
        config_started.append(time.time())
        events.append("configuration")
        return configure(*args, **kwargs)

    @contextmanager
    def observed_admission(*args: Any, **kwargs: Any) -> Iterator[Any]:
        admission_started.append(time.time())
        events.append("admission")
        try:
            with admit(*args, **kwargs) as admitted:
                yield admitted
        finally:
            events.append("release")

    def observed_engine(**kwargs: Any) -> Any:
        nonlocal engine_calls
        engine_calls += 1
        compute, publish, accept = (
            kwargs[key] for key in ("compute", "publish", "accept")
        )

        def observed_compute(attempt: Any) -> Any:
            events.append("compute")
            status = compute(attempt)
            computed.append(status)
            return status

        def observed_publish(attempt: Any, status: Any) -> Any:
            events.append("publish")
            publication_started.append(time.time())
            return publish(attempt, status)

        def observed_accept(reference: Any) -> Any:
            events.append("accept")
            return accept(reference)

        return engine(
            admission=kwargs["admission"],
            compute=observed_compute,
            publish=observed_publish,
            accept=observed_accept,
        )

    def observed_restore(self: Any, *args: Any, **kwargs: Any) -> Any:
        loaded_models.append(artifact_digest(args[1]))
        return restore(self, *args, **kwargs)

    def observed_refresh(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal refresh_calls
        refresh_calls += 1
        return refresh(self, *args, **kwargs)

    def observed_growth(self: Any, steps: int) -> None:
        growth_steps.append(steps)
        grow(self, steps)

    monkeypatch.setattr(training, "configuration", observed_configuration)
    monkeypatch.setattr(training, "admit_training", observed_admission)
    monkeypatch.setattr(attempts, "run_attempt", observed_engine)
    monkeypatch.setattr(
        runner_module.AnemoneMorpionSearchRunner, "grow", observed_growth
    )
    monkeypatch.setattr(
        runner_module.AnemoneMorpionSearchRunner, "load_or_create", observed_restore
    )
    monkeypatch.setattr(
        runner_module.AnemoneMorpionSearchRunner,
        "refresh_model_estimates",
        observed_refresh,
    )

    def run(*options: str) -> Any:
        return run_morpion_bootstrap_experiment(
            launcher_args_from_cli([
                "--scientific",
                "--work-dir",
                str(paths.work_dir),
                "--max-growth-steps-per-cycle",
                "2",
                *options,
            ])
        )

    previous: dict[str, Any] | None = None
    previous_reference: dict[str, Any] | None = None
    for number in range(1, 3):
        before = load_bootstrap_run_state(paths.run_state_path)
        active = committed_evaluator(paths.work_dir, before)[0]
        assert active is not None
        load_count = len(loaded_models)
        growth_count = len(growth_steps)
        run("--pipeline-stage", "growth", "--max-cycles", "1")
        state = load_bootstrap_run_state(paths.run_state_path)
        # A completed tiny tree may reuse the fresh snapshot from activation.
        assert state.generation >= before.generation
        assert growth_steps[growth_count:] == [2]
        assert loaded_models[load_count:] == [active["sha256"]]
        generation = state.generation
        authority = paths.run_state_path.read_bytes()
        # Activation publications retain the existing exemption from this barrier.
        if "activation_request_sha256" not in publication_records(state.metadata)[-1]:
            run("--pipeline-stage", "growth", "--max-cycles", "1")
            assert paths.run_state_path.read_bytes() == authority
        run("--pipeline-stage", "dataset_worker")
        events.clear()
        run("--pipeline-stage", "training_worker", "--skip-evaluator-diagnostics")
        assert events == [
            "configuration",
            "admission",
            "compute",
            "publish",
            "accept",
            "release",
        ]
        assert len(computed) == number
        state = load_bootstrap_run_state(paths.run_state_path)
        assert committed_evaluator(paths.work_dir, state)[0] == active
        reference = publication_records(state.metadata)[-1][training.ACCEPTED]
        proposal = training._proposal(paths, reference)
        assert proposal["admission"]["round_number"] == number
        assert list(proposal["models"]) == ["linear_a", "linear_b"]
        metadata = proposal["status"]["metadata"]
        assert metadata == computed[-1].metadata
        observation = metadata["round_performance"]
        assert (
            observation["started_unix_s"] <= config_started[-1] <= admission_started[-1]
        )
        assert (
            admission_started[-1]
            <= observation["finished_unix_s"]
            <= publication_started[-1]
        )
        if candidate_workers > 1:
            assert metadata["candidate_execution"]["children_reaped"]
            assert metadata["candidate_execution"]["completed"] == [
                "linear_a",
                "linear_b",
            ]
        for name, model in proposal["models"].items():
            parent = proposal["admission"]["parents"][name]
            if previous is None:
                assert parent is None
            else:
                assert parent["accepted_training"] == previous_reference
                assert parent["model"] == previous["models"][name]
                bundle = paths.work_dir / model["path"]
                continuation = read_object(bundle / "morpion_manifest.json")[
                    "metadata"
                ]["continuation"]
                parent_bundle = paths.work_dir / previous["models"][name]["path"]
                parent_evidence = read_object(parent_bundle / "morpion_manifest.json")[
                    "metadata"
                ]["continuation"]
                assert continuation["parent_weights_sha256"] == file_digest(
                    parent_bundle / "param.pt"
                )
                assert (
                    continuation["starting_weights_sha256"]
                    == parent_evidence["final_weights_sha256"]
                )
                assert (
                    continuation["starting_optimizer_step_count"]
                    == parent_evidence["final_optimizer_step_count"]
                )
                assert continuation["optimizer_state_loaded"]
        if previous is not None:
            loser = next(
                name
                for name in previous["models"]
                if name != previous["status"]["selected_evaluator_name"]
            )
            assert (
                proposal["admission"]["parents"][loser]["model"]
                == previous["models"][loser]
            )

        authority = paths.run_state_path.read_bytes()
        events.clear()
        with monkeypatch.context() as retry:
            retry.setattr(
                cycle_training,
                "train_and_select_evaluators_streaming",
                lambda *a, **k: pytest.fail("Accepted retry retrained"),
            )
            run(
                "--pipeline-stage", "training", "--pipeline-generation", str(generation)
            )
        assert events == ["configuration", "admission", "release"]
        assert len(computed) == number
        assert paths.run_state_path.read_bytes() == authority
        assert not paths.pipeline_training_claim_path_for_generation(
            generation
        ).exists()

        # Reevaluation only proposes; the existing owned activation drains the patch.
        run(
            "--pipeline-stage",
            "reevaluation",
            "--reevaluation-max-nodes-per-patch",
            "2",
        )
        assert paths.pipeline_reevaluation_patch_path.is_file()
        assert (
            committed_evaluator(
                paths.work_dir, load_bootstrap_run_state(paths.run_state_path)
            )[0]
            == active
        )
        request_path = tmp_path / f"activation-{number}.json"
        run(
            "--scientific-operation",
            "prepare-activation",
            "--pipeline-generation",
            str(generation),
            "--activation-request",
            str(request_path),
        )
        assert metadata["continual_selection"]["activation_eligible"]
        run(
            "--scientific-operation",
            "activate",
            "--activation-request",
            str(request_path),
        )
        activated = load_bootstrap_run_state(paths.run_state_path)
        assert (
            committed_evaluator(paths.work_dir, activated)[0]
            == json.loads(request_path.read_text())["candidate"]
        )
        assert refresh_calls == number
        assert not paths.pipeline_reevaluation_patch_path.exists()
        authority = paths.run_state_path.read_bytes()
        paths.pipeline_active_model_path.unlink()
        with monkeypatch.context() as retry:
            retry.setattr(
                activation,
                "_runner",
                lambda *a: pytest.fail("Committed retry constructed a runtime"),
            )
            run(
                "--scientific-operation",
                "activate",
                "--activation-request",
                str(request_path),
            )
        assert paths.run_state_path.read_bytes() == authority
        assert paths.pipeline_active_model_path.is_file()
        assert refresh_calls == number
        previous, previous_reference = proposal, reference

    assert engine_calls == 4
    active = committed_evaluator(
        paths.work_dir, load_bootstrap_run_state(paths.run_state_path)
    )[0]
    assert active is not None
    load_count = len(loaded_models)
    run("--pipeline-stage", "growth", "--max-cycles", "1")
    assert loaded_models[load_count:] == [active["sha256"]]
    assert (
        committed_evaluator(
            paths.work_dir, load_bootstrap_run_state(paths.run_state_path)
        )[0]
        == active
    )
