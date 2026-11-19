"""Initial Dataset contention defers; real owned work and other failures do not."""

import json
import os
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    continuous_publication,
    pipeline_orchestrator,
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
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_continuous_runtime import continuous as continuous
from tests.environments.test_morpion_continuous_runtime import candidate_family, fixture
from tests.environments.test_morpion_learning_schedule import monitored as monitored


if TYPE_CHECKING:
    from pathlib import Path


def _wait(path: Path, process: subprocess.Popen[Any], timeout: float = 40) -> None:
    """Bound fixture coordination; a child failure is never retried."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert process.poll() is None, f"Worker exited before {path.name}; see logs"
        assert time.monotonic() < deadline, f"Timed out waiting for {path}"
        time.sleep(0.02)


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize(
    "activation",
    ["owned_cold_full", "owned_cold_model_only_v1"],
    ids=["full", "model_only"],
)
def test_real_growth_dataset_overlap_reloads_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, activation: Any
) -> None:
    """Two real process/session leaders, real flock, persistent poll and accepted rows.

    The pytest parent supplies the existing monitored fixture ancestry. Only the
    Growth pause and Dataset observation are hooks; no scientific result is mocked.
    """
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
        120,
    )
    markers = tmp_path / "signals"
    markers.mkdir()
    children: list[subprocess.Popen[Any]] = []
    started = time.monotonic()
    before = b""
    before_state = load_bootstrap_run_state(paths.run_state_path)
    with (
        (tmp_path / "growth.log").open("w") as growth_log,
        (tmp_path / "dataset.log").open("w") as dataset_log,
    ):
        try:
            for stage, log in (("growth", growth_log), ("dataset_worker", dataset_log)):
                process = subprocess.Popen(
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
                children.append(process)
                if stage == "growth":
                    _wait(markers / "growth-owned", process)
                    before = paths.run_state_path.read_bytes()
                    before_state = load_bootstrap_run_state(paths.run_state_path)
            _wait(markers / "dataset-contended", children[1])
            assert all(child.poll() is None for child in children)
            assert paths.run_state_path.read_bytes() == before
            for generation in (before_state.generation, before_state.generation + 1):
                assert (
                    load_active_pipeline_stage_claim(
                        paths.pipeline_dataset_claim_path_for_generation(generation)
                    )
                    is None
                )
            first = json.loads(
                (markers / "dataset-results.jsonl").read_text().splitlines()[0]
            )
            assert first == {
                "stage": "dataset",
                "generation": None,
                "ran_stage": False,
                "reason": "ownership_contended",
            }
            (markers / "release-growth").write_text("release")
            for child in children:
                assert child.wait(timeout=60) == 0, "See retained child logs"
        finally:
            (markers / "release-growth").touch()
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
    results = [
        json.loads(line)
        for line in (markers / "dataset-results.jsonl").read_text().splitlines()
    ]
    assert len(results) == 3
    completed = [row for row in results if row["ran_stage"]]
    assert len(completed) == 1
    state = load_bootstrap_run_state(paths.run_state_path)
    assert state.generation > before_state.generation
    assert completed[0]["generation"] == state.generation
    records = publication_records(state.metadata)
    accepted = [record for record in records if "accepted_dataset" in record]
    assert len(accepted) == 1 and accepted[0]["generation"] == state.generation
    frame = continuous_publication.load_frame(paths, state)
    assert frame.state.ready is not None and frame.state.dataset_job is None
    assert (
        load_active_pipeline_stage_claim(
            paths.pipeline_dataset_claim_path_for_generation(state.generation)
        )
        is None
    )
    assert not any("accepted_training" in record for record in records)
    (tmp_path / "overlap-receipt.json").write_text(
        json.dumps(
            {
                "activation": activation,
                "wait_seconds": 15,
                "elapsed_s": time.monotonic() - started,
                "results": results,
                "old_generation": before_state.generation,
                "new_generation": state.generation,
                "accepted_datasets": len(accepted),
                "worker_exit_codes": [child.returncode for child in children],
                "scope": "real workers; fixture ancestry; controlled Growth pause",
            },
            indent=2,
        )
    )


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize(
    "error_type", [BlockingIOError, ValueError], ids=["blocking", "value"]
)
def test_initial_unexpected_error_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """Only the ownership-specific subtype is eligible for initial no-work."""
    args, paths = fixture(tmp_path / "work")
    before = paths.run_state_path.read_bytes()

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise error_type("unexpected reconciliation failure")

    monkeypatch.setattr(continuous_publication, "reconcile_freshness", fail)
    with pytest.raises(error_type, match="unexpected reconciliation failure"):
        pipeline_orchestrator.run_next_pipeline_dataset_stage_once(args)
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize(
    "error_type",
    [GrowthOwnershipBlockingIOError, BlockingIOError, ValueError],
    ids=["typed", "blocking", "value"],
)
def test_after_dataset_admission_failure_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """Real admission precedes the injected publication fault; never report no-work."""
    args, paths = fixture(tmp_path / "work")
    generation = grow(args).generation

    def fail(*_args: Any, **_kwargs: Any) -> None:
        state = load_bootstrap_run_state(paths.run_state_path)
        frame = continuous_publication.load_frame(paths, state)
        assert frame.state.dataset_job is not None
        assert (
            load_active_pipeline_stage_claim(
                paths.pipeline_dataset_claim_path_for_generation(generation)
            )
            is not None
        )
        raise error_type("after real dataset admission")

    monkeypatch.setattr(stages, "publish_dataset", fail)
    with pytest.raises(error_type, match="after real dataset admission"):
        pipeline_orchestrator.run_next_pipeline_dataset_stage_once(
            args,
            use_streaming_sharded_reader=True,
            allow_scientific_v2=True,
        )
    state = load_bootstrap_run_state(paths.run_state_path)
    assert not any(
        "accepted_dataset" in record for record in publication_records(state.metadata)
    )
