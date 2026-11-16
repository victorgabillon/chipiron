"""Real exclusive contention around admitted Training and comparison readers."""

import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    continuous_publication as publication,
)
from chipiron.environments.morpion.bootstrap import prepared_dataset as prepared
from chipiron.environments.morpion.bootstrap import prepared_dataset_storage as storage
from chipiron.environments.morpion.bootstrap import promotion
from chipiron.environments.morpion.bootstrap.continuous_contract import (
    ControlRecord,
    JobRecord,
    record_ref,
)
from chipiron.environments.morpion.bootstrap.continuous_inputs import has_consumed
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    growth_receipt_ownership,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    shared_preparation,
)
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow
from tests.environments.test_morpion_continuous_runtime import continuous as continuous
from tests.environments.test_morpion_continuous_runtime import fixture
from tests.environments.test_morpion_learning_schedule import monitored as monitored

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextmanager
def holder(
    lock: Path, growth: Path, *, seconds: float = 0.2, interrupt: int = 0
) -> Iterator[None]:
    """Own a real kernel lock; probe Growth while the parent is waiting, then release."""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import fcntl, os, signal, sys, time
with open(sys.argv[1], 'a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('held', flush=True)
    sys.stdin.readline()
    time.sleep(.1)
    with open(sys.argv[2], 'a+') as growth:
        fcntl.flock(growth, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('growth-free', flush=True)
    if int(sys.argv[4]):
        os.kill(os.getppid(), int(sys.argv[4]))
    time.sleep(float(sys.argv[3]))
""",
            str(lock),
            str(growth),
            str(seconds),
            str(interrupt),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None and process.stdin is not None
        assert process.stdout.readline().strip() == "held"
        process.stdin.write("wait\n")
        process.stdin.flush()
        yield
        output, errors = process.communicate(timeout=5)
        assert process.returncode == 0, errors
        assert "growth-free" in output
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def lock_path(paths: Any, ready: Any, kind: str) -> Path:
    """Locate the stable or native exclusive lock for this exact Ready."""
    root = storage.directory(paths, ready.payload.accepted_dataset.descriptor_sha256)
    return root / (
        ".lease" if kind == "stable" else ".rows.jsonl.shared-preparation-v1/.lock"
    )


def short_clock(paths: Any, job: JobRecord) -> JobRecord:
    """Admit a shorter immutable fixture clock without changing the configured budget."""
    budget = publication.read_record(paths, job.payload.budget, job.context)
    assert isinstance(budget, ControlRecord)
    clock = publication.read_ref(paths, budget.payload.clock_evidence)
    proof = publication.put_proof(
        paths,
        {
            **clock,
            "deadline": time.monotonic() + 0.35,
            "utc_deadline": time.time() + 0.35,
        },
    )
    budget = replace(budget, payload=replace(budget.payload, clock_evidence=proof))
    publication.put_record(paths, budget)
    job = replace(job, payload=replace(job.payload, budget=record_ref(budget)))
    publication.put_record(paths, job)
    state = load_bootstrap_run_state(paths.run_state_path)
    frame = publication.load_frame(paths, state)
    frame = replace(frame.state, cohort=record_ref(job))
    save_bootstrap_run_state(
        replace(
            state, metadata={**state.metadata, publication.KEY: frame.to_metadata()}
        ),
        paths.run_state_path,
        durable=True,
    )
    return job


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("stage", ["TRAIN", "COMPARE"])
@pytest.mark.parametrize("kind", ["stable", "native"])
def test_reader_waits_without_growth_and_keeps_original_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, kind: str
) -> None:
    """Real Training and comparison finish after stable or native contention."""
    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    run_dataset(args, generation=generation)
    if stage == "COMPARE":
        run_pipeline_training_stage(args, generation=generation)
    original = prepared.borrow_continuous
    visited = []

    @contextmanager
    def contended(p: Any, a: Any, job: Any, ready: Any) -> Iterator[Any]:
        assert job.payload.stage == stage
        before = paths.run_state_path.read_bytes()
        deadline = prepared._continuous_reader_deadline(paths, job)
        budget_bytes = (paths.work_dir / job.payload.budget.path).read_bytes()
        with holder(
            lock_path(paths, ready, kind),
            paths.pipeline_dir / ".growth_patch_consumer.lock",
        ):
            started = time.monotonic()
            with original(p, a, job, ready) as inputs:
                assert time.monotonic() - started >= 0.25
                assert prepared._continuous_reader_deadline(paths, job) == deadline
                assert (
                    paths.work_dir / job.payload.budget.path
                ).read_bytes() == budget_bytes
                assert has_consumed(
                    paths, load_bootstrap_run_state(paths.run_state_path), ready, job
                )
                visited.append(stage)
                yield inputs
        assert (
            before != paths.run_state_path.read_bytes()
        )  # only real consumption/acceptance

    monkeypatch.setattr(prepared, "borrow_continuous", contended)
    if stage == "TRAIN":
        assert (
            run_pipeline_training_stage(args, generation=generation).training_status
            == "done"
        )
    else:
        assert (
            promotion.service(paths, args, ttl_seconds=60, owner="fixture") is not None
        )
    assert visited == [stage]


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("kind", ["stable", "native"])
def test_expiry_does_not_consume_or_renew(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """An exhausted original allowance leaves the admitted inputs untouched."""
    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    run_dataset(args, generation=generation)
    original = prepared.borrow_continuous
    observed = []

    @contextmanager
    def expired(p: Any, a: Any, job: Any, ready: Any) -> Iterator[Any]:
        job = short_clock(paths, job)
        before = paths.run_state_path.read_bytes()
        with holder(
            lock_path(paths, ready, kind),
            paths.pipeline_dir / ".growth_patch_consumer.lock",
            seconds=0.5,
        ):
            started = time.monotonic()
            with (
                pytest.raises((BlockingIOError, TimeoutError)),
                original(p, a, job, ready),
            ):
                pytest.fail("expired reader yielded inputs")
            assert 0.15 < time.monotonic() - started < 1.5
        assert paths.run_state_path.read_bytes() == before
        assert not has_consumed(
            paths, load_bootstrap_run_state(paths.run_state_path), ready, job
        )
        observed.append(True)
        message = "fixture expiry observed"
        raise RuntimeError(message)
        yield  # pragma: no cover

    monkeypatch.setattr(prepared, "borrow_continuous", expired)
    with pytest.raises(RuntimeError, match="fixture expiry observed"):
        run_pipeline_training_stage(args, generation=generation)
    assert observed == [True]


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("kind", ["stable", "native"])
@pytest.mark.parametrize("change", ["job", "dataset", "launch"])
@pytest.mark.parametrize("stage", ["TRAIN", "COMPARE"])
def test_authority_rechecked_after_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, change: str, stage: str
) -> None:
    """Replacement at either lease boundary cannot authorize stale consumption."""
    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    run_dataset(args, generation=generation)
    if stage == "COMPARE":
        run_pipeline_training_stage(args, generation=generation)
    original_borrow = prepared.borrow_continuous
    real_lease = (
        storage.lease if kind == "stable" else shared_preparation.prepare_shared_inputs
    )
    observed = []

    @contextmanager
    def borrow(p: Any, a: Any, job: Any, ready: Any) -> Iterator[Any]:
        @contextmanager
        def replaced(*pos: Any, **kw: Any) -> Iterator[Any]:
            with (
                holder(
                    lock_path(paths, ready, kind),
                    paths.pipeline_dir / ".growth_patch_consumer.lock",
                ),
                real_lease(*pos, **kw) as value,
            ):
                with growth_receipt_ownership(paths):
                    state = load_bootstrap_run_state(paths.run_state_path)
                    data = state.metadata
                    if change == "job":
                        frame = publication.load_frame(paths, state)
                        data = {
                            **data,
                            publication.KEY: replace(
                                frame.state,
                                **(
                                    {"cohort": None}
                                    if stage == "TRAIN"
                                    else {"reserved_promotion": None}
                                ),
                            ).to_metadata(),
                        }
                    elif change == "dataset":
                        key = ready.payload.accepted_dataset.descriptor_sha256
                        entries = storage.entries(state)
                        entries[key] = {**entries[key], "phase": "SUSPENDED"}
                        state = replace(
                            state, metadata={**state.metadata, storage.KEY: entries}
                        )
                        data = state.metadata
                    else:
                        from chipiron.environments.morpion.bootstrap.managed_sessions import (
                            KEY,
                        )

                        data = {**data, KEY: {**data[KEY], "active": None}}
                    save_bootstrap_run_state(
                        replace(state, metadata=data),
                        paths.run_state_path,
                        durable=True,
                    )
                yield value

        with monkeypatch.context() as patch:
            patch.setattr(
                storage if kind == "stable" else shared_preparation,
                "lease" if kind == "stable" else "prepare_shared_inputs",
                replaced,
            )
            with pytest.raises(ValueError), original_borrow(p, a, job, ready):
                pytest.fail("replaced reader yielded inputs")
        assert not has_consumed(
            paths, load_bootstrap_run_state(paths.run_state_path), ready, job
        )
        observed.append(True)
        message = "fixture replacement observed"
        raise RuntimeError(message)
        yield  # pragma: no cover

    monkeypatch.setattr(prepared, "borrow_continuous", borrow)
    with pytest.raises(RuntimeError, match="fixture replacement observed"):
        if stage == "TRAIN":
            run_pipeline_training_stage(args, generation=generation)
        else:
            promotion.service(paths, args, ttl_seconds=60, owner="fixture")
    assert observed == [True]


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("kind", ["stable", "native"])
@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT])
def test_shutdown_interrupts_wait_and_releases_leases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, number: int
) -> None:
    """Native worker signal handlers unwind both locks before supervisor cleanup."""
    from chipiron.environments.morpion.bootstrap.persistent_worker import _interrupt

    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    run_dataset(args, generation=generation)
    original = prepared.borrow_continuous

    @contextmanager
    def interrupted(p: Any, a: Any, job: Any, ready: Any) -> Iterator[Any]:
        before = paths.run_state_path.read_bytes()
        previous = signal.signal(number, _interrupt)
        try:
            with holder(
                lock_path(paths, ready, kind),
                paths.pipeline_dir / ".growth_patch_consumer.lock",
                seconds=10,
                interrupt=number,
            ):
                with original(p, a, job, ready):
                    pytest.fail("interrupted reader yielded inputs")
                yield  # pragma: no cover
        except SystemExit as exc:
            assert exc.code == 128 + number
            assert paths.run_state_path.read_bytes() == before
            assert not has_consumed(
                paths, load_bootstrap_run_state(paths.run_state_path), ready, job
            )
            with storage.lease(paths, ready.payload.accepted_dataset.descriptor_sha256):
                pass
            raise
        finally:
            signal.signal(number, previous)

    monkeypatch.setattr(prepared, "borrow_continuous", interrupted)
    with pytest.raises(SystemExit) as error:
        run_pipeline_training_stage(args, generation=generation)
    assert error.value.code == 128 + number


@pytest.mark.fast
@pytest.mark.parametrize(
    ("job_end", "consumed", "limit", "expected"),
    [
        (1040, 890, 60, 1010),
        (1003, 890, 60, 1003),
        (1040, 890, 2, 1002),
        (1040, 900, 60, None),
        (1040, 0, float("nan"), None),
        (1040, 0, True, None),
    ],
)
def test_launch_and_original_clock_bounds(
    monkeypatch: pytest.MonkeyPatch,
    job_end: float,
    consumed: float,
    limit: Any,
    expected: float | None,
) -> None:
    """Consumed session time and supervisor caps constrain an immutable job clock."""
    from types import SimpleNamespace

    from chipiron.environments.morpion.bootstrap import (
        continuous_lifecycle,
        managed_sessions,
    )

    budget = SimpleNamespace(clock_evidence=None)
    job = SimpleNamespace(payload=SimpleNamespace(managed_launch=None))
    launch = {
        "started_monotonic_s": 1000,
        "authorized_seconds": 900,
        "consumed_seconds": consumed,
        "supervisor_limit_seconds": limit,
    }
    monkeypatch.setattr(continuous_lifecycle, "remaining", lambda *args: budget)
    monkeypatch.setattr(publication, "read_ref", lambda *args: {})
    monkeypatch.setattr(prepared, "_remaining", lambda *args: job_end)
    monkeypatch.setattr(managed_sessions, "_read", lambda *args: launch)
    monkeypatch.setattr(prepared, "time", SimpleNamespace(monotonic=lambda: 1000))
    if expected is None:
        with pytest.raises(ValueError):
            prepared._continuous_reader_deadline(None, job)
    else:
        assert prepared._continuous_reader_deadline(None, job) == expected
        monkeypatch.setattr(prepared, "time", SimpleNamespace(monotonic=lambda: 1001))
        assert prepared._continuous_reader_deadline(None, job) == expected


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
def test_expired_launch_refuses_unexpired_job_before_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live admission cannot borrow after the immutable managed launch expires."""
    from types import SimpleNamespace

    from chipiron.environments.morpion.bootstrap import managed_sessions
    from chipiron.environments.morpion.bootstrap.continuous_lifecycle import remaining

    args, paths = fixture(tmp_path / "run")
    generation = grow(args).generation
    run_dataset(args, generation=generation)
    original = prepared.borrow_continuous
    observed = []

    @contextmanager
    def expired(p: Any, a: Any, job: Any, ready: Any) -> Iterator[Any]:
        budget = remaining(paths, job)
        clock = publication.read_ref(paths, budget.clock_evidence)
        launch = managed_sessions._read(
            paths,
            publication.read_ref(paths, job.payload.managed_launch),
            managed_sessions._reader(),
        )
        end = launch["started_monotonic_s"] + launch["supervisor_limit_seconds"]
        assert end + 1 < clock["deadline"]
        before = paths.run_state_path.read_bytes()
        clock_bytes = (paths.work_dir / budget.clock_evidence.path).read_bytes()
        offset = end + 1 - time.monotonic()
        with monkeypatch.context() as expired_time:
            expired_time.setattr(
                prepared,
                "time",
                SimpleNamespace(
                    monotonic=lambda: time.monotonic() + offset,
                    time=lambda: time.time() + offset,
                ),
            )
            expired_time.setattr(
                storage,
                "lease",
                lambda *a, **k: pytest.fail("expired launch attempted a lease"),
            )
            # The job's own original clock is still valid at this later time.
            publication.authenticate_job(
                paths, load_bootstrap_run_state(paths.run_state_path), job
            )
            with (
                pytest.raises(
                    ValueError, match="Continuous reader allowance exhausted"
                ),
                original(p, a, job, ready),
            ):
                pytest.fail("expired launch yielded inputs")
        assert paths.run_state_path.read_bytes() == before
        assert (paths.work_dir / budget.clock_evidence.path).read_bytes() == clock_bytes
        observed.append(True)
        message = "fixture managed expiry observed"
        raise RuntimeError(message)
        yield  # pragma: no cover

    monkeypatch.setattr(prepared, "borrow_continuous", expired)
    with pytest.raises(RuntimeError, match="fixture managed expiry observed"):
        run_pipeline_training_stage(args, generation=generation)
    assert observed == [True]
