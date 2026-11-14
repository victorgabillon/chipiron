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
