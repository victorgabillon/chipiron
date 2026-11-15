"""Admission pressure retains real admitted Training and its original authority."""

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

import pytest

from chipiron.environments.morpion.bootstrap import (
    continuous_publication as publication,
)
from chipiron.environments.morpion.bootstrap import continuous_resources as resources
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.continuous_contract import Policy
from chipiron.environments.morpion.bootstrap.managed_sessions import begin_owned
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
    run_next_pipeline_dataset_stage_once,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_continuous_runtime import candidate_family
from tests.environments.test_morpion_continuous_runtime import continuous as continuous
from tests.environments.test_morpion_learning_schedule import monitored as monitored


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
def test_capacity_returns_after_real_training_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred Dataset neither leaks claims nor rewrites Training pins/clocks/debt."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        evaluators=candidate_family(),
        learning_schedule=Policy(),
        strict_execution=True,
        managed=True,
        workspace_mb=2048,
        scratch_mb=1024,
    )
    begin_owned(
        paths,
        load_bootstrap_config(paths.bootstrap_config_path),
        "42",
        time.monotonic(),
        60,
    )
    first = grow(args).generation
    run_dataset(args, generation=first)
    original = training.admit_training
    pending = []

    @contextmanager
    def during_training(*pos: Any, **kw: Any) -> Iterator[Any]:
        with original(*pos, **kw) as attempt:
            pending.append(grow(args).generation)
            before = paths.run_state_path.read_bytes()
            state = load_bootstrap_run_state(paths.run_state_path)
            frame = publication.load_frame(paths, state)
            assert frame.state.cohort is not None
            assert frame.state.pending_source is not None
            # Repeated scheduled invocations remain finite and retain all authority.
            for _ in range(3):
                result = run_next_pipeline_dataset_stage_once(
                    args, use_streaming_sharded_reader=True, allow_scientific_v2=True
                )
                assert result.reason == "resource_backpressure" and not result.ran_stage
                assert paths.run_state_path.read_bytes() == before
                assert not paths.pipeline_dataset_claim_path_for_generation(
                    pending[0]
                ).exists()
            yield attempt

    monkeypatch.setattr(training, "admit_training", during_training)
    assert run_pipeline_training_stage(args, generation=first).training_status == "done"
    accepted = load_bootstrap_run_state(paths.run_state_path)
    assert publication.load_frame(paths, accepted).state.cohort is None
    result = run_next_pipeline_dataset_stage_once(
        args, use_streaming_sharded_reader=True, allow_scientific_v2=True
    )
    assert result.ran_stage and result.generation == pending[0]
    final = publication.load_frame(
        paths, load_bootstrap_run_state(paths.run_state_path)
    )
    assert final.state.last_accepted_round == 1 and final.state.ready is not None


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
def test_corrupt_accounting_is_not_deferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupted record still fails, rather than appearing as idle capacity pressure."""
    from tests.environments.test_morpion_continuous_runtime import fixture

    args, paths = fixture(tmp_path / "run")
    grow(args)

    def corrupted(*_args: Any) -> tuple[int, int]:
        message = "Invalid accounting record"
        raise ValueError(message)

    monkeypatch.setattr(resources, "outstanding", corrupted)
    with pytest.raises(ValueError, match="Invalid accounting record"):
        run_next_pipeline_dataset_stage_once(
            args, use_streaming_sharded_reader=True, allow_scientific_v2=True
        )
    assert (
        publication.load_frame(
            paths, load_bootstrap_run_state(paths.run_state_path)
        ).state.dataset_job
        is None
    )
