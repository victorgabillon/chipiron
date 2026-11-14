"""Closed lifecycle commands retain immutable work; clocks and output limits fail closed."""

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from chipiron.environments.morpion.bootstrap.continuous_contract import (
    ContractError,
    Debt,
    JobRecord,
    record_ref,
)
from chipiron.environments.morpion.bootstrap.continuous_transitions import (
    Admission,
    Discard,
    DropReady,
    Frame,
    Retry,
    Suspend,
    transition,
)
from chipiron.utils.bounded_output import output_allowance, output_file
from tests.environments.test_morpion_continuous_contract import (
    boundary,
    changed,
    extraction,
    initialize,
    ready_frame,
    ref,
    source,
)


def test_stop_suspend_resume_preserves_original_work() -> None:
    """Close before suspension and retain all admitted bindings across a new claim."""
    frame = initialize("a" * 64, "b" * 64)
    job = extraction(frame, source())
    frame = Frame(
        frame.context,
        replace(
            frame.state, dataset_job=record_ref(job), learning_debt=Debt("source", 11)
        ),
    )
    records = {record_ref(job): job}
    with pytest.raises(ContractError, match="Close admission"):
        transition(frame, boundary(frame), Suspend(), records)
    frame = changed(transition(frame, boundary(frame), Admission(False), records)).frame
    result = changed(transition(frame, boundary(frame), Suspend(), records))
    suspended = result.records[0]
    assert isinstance(suspended, JobRecord)
    assert replace(suspended.payload, status="ADMITTED") == job.payload
    frame = result.frame
    records[record_ref(suspended)] = suspended
    frame = changed(transition(frame, boundary(frame), Admission(True), records)).frame
    retry = JobRecord(
        frame.context,
        replace(
            suspended.payload,
            status="ADMITTED",
            claim=ref("new-claim"),
            managed_launch=ref("new-launch"),
            budget=ref("same-clock-new-launch"),
        ),
    )
    result = changed(
        transition(frame, boundary(frame), Retry(suspended, retry), records)
    )
    assert result.frame.state.dataset_job == record_ref(retry)
    assert result.frame.state.last_accepted_round == 0
    assert result.frame.state.learning_debt == frame.state.learning_debt
    for field, value in (("source", source(15, 12)), ("id", "other-job")):
        tampered = JobRecord(frame.context, replace(retry.payload, **{field: value}))
        with pytest.raises(ContractError, match="retargeted"):
            transition(frame, boundary(frame), Retry(suspended, tampered), records)


def test_only_ineligible_unclaimed_ready_can_be_disposed() -> None:
    """Keep accepted Dataset identity and debt when old teachers become ineligible."""
    frame, ready = ready_frame()
    records = {record_ref(ready): ready}
    with pytest.raises(ContractError, match="Eligible Ready"):
        transition(frame, boundary(frame), DropReady(ready), records)
    frame = Frame(frame.context, replace(frame.state, incumbent_epoch=2))
    result = changed(transition(frame, boundary(frame), DropReady(ready), records))
    assert result.frame.state.ready is None
    assert result.frame.state.learning_debt == frame.state.learning_debt
    assert result.frame.state.last_consumed_generation == 0


def test_discard_requires_closed_admission() -> None:
    """Never turn an active slot into an implicit acceptance or implicit fork."""
    frame, ready = ready_frame()
    records = {record_ref(ready): ready}
    with pytest.raises(ContractError, match="closed admission"):
        transition(frame, boundary(frame), Discard(), records)
    frame = changed(transition(frame, boundary(frame), Admission(False), records)).frame
    result = changed(transition(frame, boundary(frame), Discard(), records))
    assert result.frame.state.ready is None
    assert result.frame.state.learning_debt is None
    assert result.frame.state.last_accepted_round == 0
    assert not result.frame.state.admission_open


def test_private_output_limit_checks_before_bytes_are_written(tmp_path: Path) -> None:
    """Bound text, binary and seek rewrites together, while leaving other paths alone."""
    root = tmp_path / "attempt"
    root.mkdir()
    with output_allowance(root, 10):
        with output_file(root / "first", "wb") as stream:
            assert stream.write(b"123456") == 6
            stream.seek(0)
            assert stream.write(b"xy") == 2
            with pytest.raises(OSError, match="resource_backpressure"):
                stream.write(b"abc")
        with output_file(root / "second", encoding="utf-8") as stream:
            stream.write("é")
        with output_file(tmp_path / "other", "wb") as stream:
            stream.write(b"outside-reservation")
    assert (root / "first").read_bytes() == b"xy3456"
    assert (root / "second").read_bytes() == "é".encode()
    with output_file(root / "legacy", "wb") as stream:
        stream.write(b"legacy-no-new-cap")


@pytest.mark.parametrize(
    "slot",
    [
        "pending_source",
        "dataset_job",
        "ready",
        "cohort",
        "reserved_promotion",
        "pending_promotion",
    ],
)
@pytest.mark.parametrize("owner", ["dead", "live", "unknown"])
def test_six_slot_stop_and_owner_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slot: str, owner: str
) -> None:
    """Close every slot before reap; only positive OS quiescence permits suspension."""
    import json
    import os
    import subprocess
    import sys

    from chipiron.environments.morpion.bootstrap import (
        continuous_lifecycle as lifecycle,
    )
    from chipiron.environments.morpion.bootstrap import (
        continuous_publication as publication,
    )
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.continuous_contract import (
        ControlRecord,
        Population,
        Reservation,
    )
    from chipiron.environments.morpion.bootstrap.learning_worker import (
        require_quiescent,
    )
    from chipiron.environments.morpion.bootstrap.managed_sessions import _local_identity
    from tests.environments.test_morpion_continuous_contract import train_job
    from tests.environments.test_morpion_continuous_publication import opportunity

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    current = _local_identity()
    current.update(
        session=os.getsid(0),
        monitor_pid=os.getpid(),
        monitor_start_ticks=current["start_ticks"],
    )
    previous = dict(current)
    if owner == "dead":
        code = "import os,json; from chipiron.environments.morpion.bootstrap.managed_sessions import _local_identity; d=_local_identity(); d.update(session=os.getsid(0), monitor_pid=os.getpid(), monitor_start_ticks=d['start_ticks']); print(json.dumps(d))"
        child = subprocess.run(
            [sys.executable, "-c", code],
            start_new_session=True,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        previous = json.loads(child.stdout)
    elif owner == "unknown":
        previous["host"] = "unknown-host"
    frame, ready = ready_frame()
    claim = publication.put_proof(paths, {"metadata": {"learning_worker": previous}})
    job = train_job(frame, ready)
    job = replace(job, payload=replace(job.payload, claim=claim))
    offer = opportunity(frame)
    population = ControlRecord(
        frame.context,
        Population(
            ready.payload.accepted_dataset,
            ready.payload.rows.sha256,
            ready.payload.split.sha256,
            2,
            ref("recipe"),
        ),
    )
    compared = replace(
        job,
        payload=replace(
            job.payload, stage="COMPARE", reserved_id=record_ref(offer).sha256
        ),
    )
    reserved = ControlRecord(
        frame.context,
        Reservation(
            record_ref(offer),
            ref("incumbent"),
            record_ref(population),
            record_ref(compared),
            None,
            None,
            None,
            "COMPARING",
        ),
    )
    dataset = extraction(frame, ready.payload.source)
    dataset = replace(dataset, payload=replace(dataset.payload, claim=claim))
    records = {
        record_ref(r): r
        for r in (ready, job, offer, population, compared, reserved, dataset)
    }
    for record in records.values():
        publication.put_record(paths, record)
    values = dict.fromkeys((
        "pending_source",
        "dataset_job",
        "ready",
        "cohort",
        "reserved_promotion",
        "pending_promotion",
    ))
    values[slot] = {
        "pending_source": ready.payload.source,
        "dataset_job": record_ref(dataset),
        "ready": record_ref(ready),
        "cohort": record_ref(job),
        "reserved_promotion": record_ref(reserved),
        "pending_promotion": record_ref(offer),
    }[slot]
    frame = Frame(
        frame.context,
        replace(
            frame.state,
            **values,
            learning_debt=Debt("learn", 11),
            promotion_debt=Debt("promote", 11),
        ),
    )
    closed = changed(
        transition(frame, boundary(frame), Admission(False), records)
    ).frame
    assert not closed.state.admission_open
    assert (
        replace(closed.state, admission_open=True, revision=frame.state.revision)
        == frame.state
    )
    monkeypatch.setattr(publication, "load_frame", lambda *args: closed)

    def suspend() -> None:
        # Native shutdown must prove the prior owner quiescent even for slots
        # without active jobs; the facade additionally authenticates admitted jobs.
        require_quiescent(previous, current)
        lifecycle.authenticate(paths, None, Suspend())
        result = changed(transition(closed, boundary(closed), Suspend(), records))
        assert result.frame.state.learning_debt == closed.state.learning_debt
        assert result.frame.state.promotion_debt == closed.state.promotion_debt
        assert (
            result.frame.state.last_accepted_round == closed.state.last_accepted_round
        )

    if owner == "dead":
        suspend()
    else:
        with pytest.raises(ValueError, match=r"quiescent|Unknown"):
            suspend()


@pytest.mark.parametrize(
    "damage", ["none", "expired", "contradictory", "reboot", "namespace", "nonfinite"]
)
def test_original_clock_cannot_be_renewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Explicit counters distinguish exhausted, foreign and contradictory original allowances."""
    from chipiron.environments.morpion.bootstrap import (
        continuous_lifecycle as lifecycle,
    )
    from chipiron.environments.morpion.bootstrap import (
        continuous_publication as publication,
    )
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.continuous_contract import (
        Budget,
        ControlRecord,
    )
    from chipiron.environments.morpion.bootstrap.managed_sessions import _local_identity

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    identity = _local_identity()
    clock = {"owner": identity, "deadline": 110, "utc_deadline": 1010}
    if damage == "expired":
        clock["deadline"] = 90
    elif damage == "contradictory":
        clock["utc_deadline"] = 1050
    elif damage == "reboot":
        clock["owner"] = {**identity, "boot": "another-boot"}
    elif damage == "namespace":
        clock["owner"] = {**identity, "pid_namespace": "pid:[1]"}
    elif damage == "nonfinite":
        clock["deadline"] = "not-a-clock"
    frame, ready = ready_frame()
    job = extraction(frame, ready.payload.source)
    budget = ControlRecord(
        frame.context,
        Budget(
            ref("launch"),
            60,
            0.0,
            publication.put_proof(paths, clock),
            12,
            34,
            "c" * 64,
        ),
    )
    job = replace(
        job, payload=replace(job.payload, budget=publication.put_record(paths, budget))
    )
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: 100)
    monkeypatch.setattr(lifecycle.time, "time", lambda: 1000)
    if damage == "none":
        assert lifecycle.remaining(paths, job) == budget.payload
    else:
        with pytest.raises(ValueError):
            lifecycle.remaining(paths, job)
