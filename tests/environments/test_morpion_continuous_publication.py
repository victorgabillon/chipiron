"""Continuous immutable controls, reserved execution and durable object boundaries."""

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.continuous_contract import (
    Accepted,
    Budget,
    Comparison,
    ContractError,
    ControlRecord,
    Debt,
    Disposition,
    Opportunity,
    Population,
    Reservation,
    Retirement,
    decode_record,
    record_ref,
)
from chipiron.environments.morpion.bootstrap.continuous_publication import (
    put_record,
    read_record,
)
from chipiron.environments.morpion.bootstrap.continuous_transitions import (
    AcceptCohort,
    AdmitCohort,
    Frame,
    NoWork,
    ReservePromotion,
    ResolvePromotion,
    StartComparison,
    VerifyComparison,
    admit_cohort,
    transition,
)
from tests.environments.test_morpion_continuous_contract import (
    boundary,
    changed,
    ready_frame,
    ref,
    train_job,
)

if TYPE_CHECKING:
    from pathlib import Path


def opportunity(frame: Frame, number: int = 1) -> ControlRecord:
    """Bind illustrative accepted identities; IO acceptance is tested separately."""
    return ControlRecord(
        frame.context,
        Opportunity(
            f"promotion-{number}",
            Accepted(f"training-{number}", "a" * 64),
            14 + number - 1,
            number,
            "losing-then-winning",
            ref("candidate"),
            Accepted("dataset", "1" * 64),
            Debt(f"promotion-{number}", 11),
        ),
    )


def test_control_envelopes_are_exact_and_detached() -> None:
    """Round-trip every control envelope without accepting unknown or missing keys."""
    frame, ready = ready_frame()
    accepted = ready.payload.accepted_dataset
    payloads = [
        Budget(ref("launch"), 60, 1.0, ref("clock"), 12, 34, "c" * 64),
        opportunity(frame).payload,
        Population(accepted, "b" * 64, "c" * 64, 2, ref("recipe")),
        Reservation(
            ref("opportunity"),
            ref("incumbent"),
            ref("population"),
            None,
            None,
            None,
            None,
            "RESERVED",
        ),
        Comparison(
            "r1",
            ref("claim"),
            ref("candidate"),
            ref("incumbent"),
            ref("population"),
            1.0,
            1.0,
            "retain",
            "incumbent_on_equal_else_candidate_name",
        ),
        Disposition(
            ready.payload.source, "ineligible_for_training", None, Debt("d", 1)
        ),
        Retirement(
            ref("ready"), ("pipeline/cache/flat.pt",), "unreferenced_operational_cache"
        ),
    ]
    for payload in payloads:
        record = ControlRecord(frame.context, payload)
        encoded = record.to_metadata()
        assert decode_record(encoded) == record
        encoded["payload"]["unknown"] = 0
        with pytest.raises(ContractError):
            decode_record(encoded)
        assert "unknown" not in record.to_metadata()["payload"]
        encoded = record.to_metadata()
        encoded["schema"] = "unreviewed_v1"
        with pytest.raises(ContractError):
            decode_record(encoded)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -1.0])
def test_invalid_job_allowances(bad: float) -> None:
    """Reject nonfinite, boolean or negative consumed allowances."""
    with pytest.raises(ContractError):
        Budget(ref("launch"), 60, bad, ref("clock"), 0, 0, "b" * 64)


def test_comparison_ties_and_phase_proofs() -> None:
    """Require incumbent retention on equality and complete phase evidence."""
    tied = Comparison(
        "r",
        ref(),
        ref(),
        ref(),
        ref(),
        1.0,
        1.0,
        "retain",
        "incumbent_on_equal_else_candidate_name",
    )
    with pytest.raises(ContractError):
        replace(tied, decision="promote")
    with pytest.raises(ContractError):
        replace(tied, incumbent_loss=float("nan"))
    reserved = Reservation(ref(), ref(), ref(), None, None, None, None, "RESERVED")
    for phase in ("COMPARING", "VERIFIED", "CUTOVER", "UNKNOWN"):
        with pytest.raises(ContractError):
            replace(reserved, phase=phase)


def test_fsynced_objects_do_not_grant_state_authority(tmp_path: Path) -> None:
    """Publish exact bytes idempotently, rejecting corruption and foreign context."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    frame, _ = ready_frame()
    record = opportunity(frame)
    reference = put_record(paths, record)
    assert reference == record_ref(record)
    assert put_record(paths, record) == reference
    assert read_record(paths, reference, frame.context) == record
    assert not paths.run_state_path.exists()
    assert not list((tmp_path / reference.path).parent.glob(".pending-*"))
    with pytest.raises(ContractError):
        read_record(
            paths, reference, replace(frame.context, workspace_epoch_sha256="f" * 64)
        )
    (tmp_path / reference.path).write_text("{}")
    with pytest.raises(ContractError):
        put_record(paths, record)
    with pytest.raises(ContractError):
        read_record(paths, reference, frame.context)


def test_symlinked_objects_refuse_without_touching_target(tmp_path: Path) -> None:
    """Refuse a contained-looking path through a symlinked namespace."""
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path / "workspace")
    paths.work_dir.mkdir()
    target = tmp_path / "external"
    target.mkdir()
    (paths.work_dir / "pipeline").symlink_to(target, target_is_directory=True)
    frame, _ = ready_frame()
    with pytest.raises(ValueError):
        put_record(paths, opportunity(frame))
    assert list(target.iterdir()) == []


def test_reserved_comparison_precedes_new_cohort_and_retains_tie() -> None:
    """Keep the reserved candidate stable and release executor only after verification."""
    frame, ready = ready_frame()
    pending = opportunity(frame)
    assert isinstance(pending.payload, Opportunity)
    frame = Frame(
        frame.context,
        replace(
            frame.state,
            pending_promotion=record_ref(pending),
            promotion_debt=pending.payload.debt,
        ),
    )
    reserved = ControlRecord(
        frame.context,
        Reservation(
            record_ref(pending),
            ref("incumbent"),
            ref("population"),
            None,
            None,
            None,
            None,
            "RESERVED",
        ),
    )
    records = {record_ref(ready): ready, record_ref(pending): pending}
    frame = changed(
        transition(frame, boundary(frame), ReservePromotion(reserved), records)
    ).frame
    records[record_ref(reserved)] = reserved
    training = train_job(frame, ready)
    assert transition(
        frame, boundary(frame), AdmitCohort(training, ready, 2), records
    ) == NoWork("learner_executor")
    comparison_job = replace(
        training,
        payload=replace(
            training.payload, stage="COMPARE", reserved_id=record_ref(pending).sha256
        ),
    )
    update = changed(
        transition(
            frame, boundary(frame), StartComparison(reserved, comparison_job), records
        )
    )
    frame = update.frame
    records.update((record_ref(r), r) for r in update.records)
    comparing = records[frame.state.reserved_promotion]
    assert isinstance(comparing, ControlRecord)
    comparison = ControlRecord(
        frame.context,
        Comparison(
            record_ref(pending).sha256,
            comparison_job.payload.claim,
            pending.payload.candidate,
            ref("incumbent"),
            ref("population"),
            1.0,
            1.0,
            "retain",
            "incumbent_on_equal_else_candidate_name",
        ),
    )
    with pytest.raises(ContractError):
        wrong = replace(
            comparison, payload=replace(comparison.payload, population=ref("other"))
        )
        transition(frame, boundary(frame), VerifyComparison(comparing, wrong), records)
    update = changed(
        transition(
            frame, boundary(frame), VerifyComparison(comparing, comparison), records
        )
    )
    frame = update.frame
    records.update((record_ref(r), r) for r in update.records)
    verified = records[frame.state.reserved_promotion]
    assert isinstance(verified, ControlRecord)
    result = changed(
        transition(frame, boundary(frame), AdmitCohort(training, ready, 2), records)
    )
    assert result.frame.state.cohort == record_ref(training)
    retained = changed(
        transition(
            frame,
            boundary(frame),
            ResolvePromotion(verified, comparison, False),
            records,
        )
    )
    assert retained.frame.state.reserved_promotion is None
    assert retained.frame.state.incumbent_epoch == frame.state.incumbent_epoch


def test_cohort_acceptance_does_not_recheck_incumbent_epoch() -> None:
    """Accept only the original cohort and never reset its round twice."""
    frame, ready = ready_frame()
    job = train_job(frame, ready)
    frame = changed(
        admit_cohort(frame, boundary(frame), ready, job, validation_count=2)
    ).frame
    frame = Frame(frame.context, replace(frame.state, incumbent_epoch=9))
    action = AcceptCohort(job, Accepted("learned", "c" * 64), None, None)
    records = {record_ref(job): job}
    result = changed(transition(frame, boundary(frame), action, records))
    assert result.frame.state.last_accepted_round == 1
    assert result.frame.state.cohort is None
    assert result.frame.state.learning_debt is None
    with pytest.raises(ContractError):
        transition(result.frame, boundary(result.frame), action, records)
