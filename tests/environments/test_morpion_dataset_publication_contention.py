"""A completed Dataset waits within its original job budget, without re-extraction."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import continuous_publication
from chipiron.environments.morpion.bootstrap import (
    generation_publication_io as ownership,
)
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.continuous_contract import Policy
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GenerationPublicationError,
    GrowthOwnershipBlockingIOError,
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.managed_sessions import begin_owned
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
def test_real_growth_overlap_accepts_same_completed_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, activation: Any
) -> None:
    """Actual persistent workers, real Growth flock >15s and one immutable proposal."""
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
    generation = grow(args).generation
    markers = tmp_path / "signals"
    markers.mkdir()
    children: list[subprocess.Popen[Any]] = []
    with (
        (tmp_path / "dataset.log").open("w") as dataset_log,
        (tmp_path / "growth.log").open("w") as growth_log,
    ):
        try:
            for stage, log, marker in (
                ("dataset_worker", dataset_log, "published.jsonl"),
                ("growth", growth_log, "growth-owned"),
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
                        "publication",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                children.append(child)
                _wait(markers / marker, child)
            _wait(markers / "acceptance-waiting", children[0])
            before = load_bootstrap_run_state(paths.run_state_path)
            job = continuous_publication.load_frame(paths, before).state.dataset_job
            claim_path = paths.pipeline_dataset_claim_path_for_generation(generation)
            claim = claim_path.read_bytes()
            reference_bytes = (markers / "published.jsonl").read_bytes()
            start = time.monotonic()
            while time.monotonic() - start < 16:
                assert all(child.poll() is None for child in children), "See child logs"
                time.sleep(0.05)
            held_seconds = time.monotonic() - start
            # Still admitted, with exactly the same claim, budget and proposal.
            waiting = load_bootstrap_run_state(paths.run_state_path)
            assert (
                continuous_publication.load_frame(paths, waiting).state.dataset_job
                == job
            )
            assert claim_path.read_bytes() == claim
            assert (markers / "published.jsonl").read_bytes() == reference_bytes
            assert not any(
                "accepted_dataset" in r for r in publication_records(waiting.metadata)
            )
            (markers / "release-growth").touch()
            for child in reversed(children):
                assert child.wait(timeout=60) == 0, "See child logs"
        finally:
            (markers / "release-growth").touch()
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
    after = load_bootstrap_run_state(paths.run_state_path)
    assert after.generation > before.generation
    accepted = [
        r for r in publication_records(after.metadata) if "accepted_dataset" in r
    ]
    assert len(accepted) == 1 and accepted[0]["generation"] == generation
    references = [
        json.loads(line)
        for line in (markers / "published.jsonl").read_text().splitlines()
    ]
    assert references == [accepted[0]["accepted_dataset"]]
    results = [
        json.loads(line)
        for line in (markers / "dataset-results.jsonl").read_text().splitlines()
    ]
    assert len(results) == 1 and results[0]["ran_stage"]
    assert results[0]["generation"] == generation
    frame = continuous_publication.load_frame(paths, after)
    assert frame.state.ready is not None and frame.state.dataset_job is None
    assert load_active_pipeline_stage_claim(claim_path) is None
    actual_wait = json.loads((markers / "acceptance-wait.json").read_text())
    assert actual_wait["wait_s"] > 15
    (tmp_path / "publication-overlap-receipt.json").write_text(
        json.dumps(
            {
                "activation": activation,
                "held_seconds_after_proposal": held_seconds,
                "acceptance_wait": actual_wait,
                "source_generation": generation,
                "final_generation": after.generation,
                "accepted_reference": references[0],
                "original_job": job.to_metadata() if job is not None else None,
                "results": results,
                "worker_exit_codes": [child.returncode for child in children],
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("deadline_offset", [-1, 0.1])
def test_original_deadline_is_not_renewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deadline_offset: float
) -> None:
    """A real conflicting flock cannot extend the supplied absolute deadline."""
    monkeypatch.setenv("MORPION_WORKER_OWNERSHIP_WAIT_SECONDS", "15")
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.pipeline_dir.mkdir(parents=True)
    with (paths.pipeline_dir / ".growth_patch_consumer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        with (
            pytest.raises(GrowthOwnershipBlockingIOError),
            growth_receipt_ownership(
                paths, wait=True, wait_until=start + deadline_offset
            ),
        ):
            pytest.fail("Expired publication entered its owned body")
        assert time.monotonic() - start < 2


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True])
def test_invalid_deadline_refused(tmp_path: Path, deadline: float) -> None:
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    with (
        pytest.raises(
            GenerationPublicationError, match="Invalid ownership wait deadline"
        ),
        growth_receipt_ownership(paths, wait=True, wait_until=deadline),
    ):
        pytest.fail("Invalid deadline entered its owned body")


def test_owned_body_failure_is_not_retried(tmp_path: Path) -> None:
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    failure = GrowthOwnershipBlockingIOError("inside owned body")
    with (
        pytest.raises(GrowthOwnershipBlockingIOError) as caught,
        growth_receipt_ownership(paths, wait=True, wait_until=time.monotonic() + 10),
    ):
        raise failure
    assert caught.value is failure
    with growth_receipt_ownership(paths):
        pass


@pytest.mark.usefixtures("continuous")
def test_acceptance_rechecks_claim_after_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing the pre-wait check never permits acceptance with a replaced claim."""
    args, paths = fixture(tmp_path / "work", target_source="effective")
    generation = grow(args).generation
    claim_path = paths.pipeline_dataset_claim_path_for_generation(generation)
    original = ownership.growth_receipt_ownership
    changed = False

    @contextmanager
    def replace_before_owned_body(*args: Any, **kwargs: Any) -> Iterator[None]:
        nonlocal changed
        with original(*args, **kwargs):
            if kwargs.get("wait_until") is not None:
                claim = json.loads(claim_path.read_text())
                claim["claim_id"] = "replacement-claim"
                claim_path.write_text(json.dumps(claim))
                changed = True
            yield

    monkeypatch.setattr(
        ownership, "growth_receipt_ownership", replace_before_owned_body
    )
    with pytest.raises(ValueError, match="Job claim was replaced or removed"):
        run_dataset(args, generation=generation)
    assert changed
    assert json.loads(claim_path.read_text())["claim_id"] == "replacement-claim"
    state = load_bootstrap_run_state(paths.run_state_path)
    assert not any("accepted_dataset" in r for r in publication_records(state.metadata))
