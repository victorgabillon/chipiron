"""Only initial promotion ownership contention defers a Training dispatch."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    continuous_publication,
    pipeline_orchestrator,
    promotion,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.continuous_contract import Policy
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GrowthOwnershipBlockingIOError,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.managed_sessions import begin_owned
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.pipeline_claims import (
    load_active_pipeline_stage_claim,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_continuous_runtime import candidate_family, fixture
from tests.environments.test_morpion_continuous_runtime import continuous as continuous
from tests.environments.test_morpion_dataset_admission_contention import _wait
from tests.environments.test_morpion_learning_schedule import monitored as monitored

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize(
    "activation",
    ["owned_cold_full", "owned_cold_model_only_v1"],
    ids=["full", "model_only"],
)
def test_real_growth_training_overlap_reloads_and_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, activation: Any
) -> None:
    """Real Growth flock, persistent Training process, then real accepted learning."""
    monkeypatch.setenv("MORPION_WORKER_OWNERSHIP_WAIT_SECONDS", "15")
    args, paths = setup_continual(
        tmp_path / "work",
        automatic=True,
        memory_floor_mb=1,
        evaluators=candidate_family(),
        learning_schedule=Policy(activation=activation),
        strict_execution=True,
        managed=True,
        target_source="effective",
    )
    begin_owned(
        paths,
        load_bootstrap_config(paths.bootstrap_config_path),
        "42",
        time.monotonic(),
        180,
    )
    markers = tmp_path / "signals"
    markers.mkdir()
    children: list[subprocess.Popen[Any]] = []
    started = time.monotonic()
    with (
        (tmp_path / "growth.log").open("w") as growth_log,
        (tmp_path / "training.log").open("w") as training_log,
    ):
        try:
            for stage, log in (
                ("growth", growth_log),
                ("training_worker", training_log),
            ):
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "tests.environments.dataset_admission_workload",
                        str(paths.work_dir),
                        str(markers),
                        stage,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                children.append(child)
                if stage == "growth":
                    _wait(markers / "growth-owned", child)
                    before = paths.run_state_path.read_bytes()
                    old_generation = load_bootstrap_run_state(
                        paths.run_state_path
                    ).generation
            _wait(markers / "training-contended", children[1])
            assert all(c.poll() is None for c in children)
            assert paths.run_state_path.read_bytes() == before
            state = load_bootstrap_run_state(paths.run_state_path)
            frame = continuous_publication.load_frame(paths, state)
            assert frame.state.cohort is None and frame.state.reserved_promotion is None
            for generation in (old_generation, old_generation + 1):
                assert (
                    load_active_pipeline_stage_claim(
                        paths.pipeline_training_claim_path_for_generation(generation)
                    )
                    is None
                )
            assert json.loads((markers / "training-results.jsonl").read_text()) == {
                "stage": "training",
                "generation": None,
                "ran_stage": False,
                "reason": "ownership_contended",
            }
            (markers / "release-growth").write_text("release")
            assert children[0].wait(timeout=60) == 0
            generation = load_bootstrap_run_state(paths.run_state_path).generation
            assert generation > old_generation
            run_dataset(args, generation=generation)
            ready_state = load_bootstrap_run_state(paths.run_state_path)
            ready = continuous_publication.load_frame(paths, ready_state).state.ready
            assert ready is not None
            assert children[1].poll() is None
            (markers / "resume-training").write_text("resume")
            assert children[1].wait(timeout=60) == 0, "See retained Training log"
        finally:
            (markers / "release-growth").touch()
            (markers / "resume-training").touch()
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
    results = [
        json.loads(line)
        for line in (markers / "training-results.jsonl").read_text().splitlines()
    ]
    assert len(results) == 2
    assert results[1] == {
        "stage": "training",
        "generation": generation,
        "ran_stage": True,
        "reason": None,
    }
    state = load_bootstrap_run_state(paths.run_state_path)
    records = publication_records(state.metadata)
    accepted = [r for r in records if "accepted_training" in r]
    assert len(accepted) == 1 and accepted[0]["generation"] == generation
    frame = continuous_publication.load_frame(paths, state)
    assert frame.state.cohort is None and frame.state.last_accepted_round == 1
    assert frame.state.pending_promotion is not None
    assert (
        load_active_pipeline_stage_claim(
            paths.pipeline_training_claim_path_for_generation(generation)
        )
        is None
    )
    # The accepted generation and ready population did not exist at first dispatch.
    # Real acceptance after that dispatch proves a fresh admission, not a cached job.
    (tmp_path / "training-overlap-receipt.json").write_text(
        json.dumps(
            {
                "activation": activation,
                "wait_seconds": 15,
                "elapsed_s": time.monotonic() - started,
                "old_generation": old_generation,
                "accepted_generation": generation,
                "ready_before_retry": ready.to_metadata(),
                "accepted_training": accepted,
                "results": results,
                "worker_exit_codes": [c.returncode for c in children],
                "scope": "real persistent workers and flock; managed fixture ancestry; controlled Growth pause",
            },
            indent=2,
        )
    )


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("error_type", [BlockingIOError, ValueError])
def test_initial_unrelated_error_remains_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """Do not turn generic I/O or invalid state into no-work."""
    args, paths = fixture(tmp_path / "work")
    before = paths.run_state_path.read_bytes()

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        message = "initial unrelated failure"
        raise error_type(message)

    monkeypatch.setattr(promotion, "growth_receipt_ownership", fail)
    with pytest.raises(error_type, match="initial unrelated failure"):
        pipeline_orchestrator.run_next_pipeline_training_stage_once(args)
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize(
    "error_type", [GrowthOwnershipBlockingIOError, BlockingIOError, ValueError]
)
def test_error_inside_initial_ownership_remains_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """Even the typed error is fatal once the initial ownership was acquired."""
    args, paths = fixture(tmp_path / "work")
    before = paths.run_state_path.read_bytes()

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        message = "inside acquired ownership"
        raise error_type(message)

    monkeypatch.setattr(promotion, "load_bootstrap_run_state", fail)
    with pytest.raises(error_type, match="inside acquired ownership"):
        pipeline_orchestrator.run_next_pipeline_training_stage_once(args)
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("failure_entry", [2, 3], ids=["acceptance", "cleanup"])
def test_comparison_reacquisition_failure_remains_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_entry: int
) -> None:
    """Actual reserved comparison is admitted before injecting later lock failure."""
    args, paths = fixture(tmp_path / "work")
    generation = grow(args).generation
    run_dataset(args, generation=generation)
    stages.run_pipeline_training_stage(args, generation=generation)
    ownership = promotion.growth_receipt_ownership
    entries = 0

    @contextmanager
    def controlled(*positional: Any, **keywords: Any) -> Iterator[None]:
        nonlocal entries
        entries += 1
        if entries == failure_entry:
            state = load_bootstrap_run_state(paths.run_state_path)
            record = promotion.reservation(paths, state)
            assert record is not None
            assert record.payload.phase == (
                "COMPARING" if failure_entry == 2 else "VERIFIED"
            )
            assert (
                load_active_pipeline_stage_claim(
                    paths.pipeline_training_claim_path_for_generation(generation)
                )
                is not None
            )
            message = "after comparison admission"
            raise GrowthOwnershipBlockingIOError(message)
        with ownership(*positional, **keywords):
            yield

    monkeypatch.setattr(promotion, "growth_receipt_ownership", controlled)
    with pytest.raises(
        GrowthOwnershipBlockingIOError, match="after comparison admission"
    ) as caught:
        pipeline_orchestrator.run_next_pipeline_training_stage_once(args)
    assert type(caught.value) is GrowthOwnershipBlockingIOError
    assert entries == 3
