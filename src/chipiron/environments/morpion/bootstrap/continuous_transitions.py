"""Pure first-path proposals for the unwired continuous contract.

No function grants admission authority. Callers must authenticate every supplied
record/ref, current committed cycle, source, claim, validation population, parent
map and resource reservation under the existing owner. Persist records first,
then save the returned State once with related accepted/publication references.
Displacements require durable receipts in that same save. No IO, clock, lease,
claim takeover, Training acceptance or promotion transition is implemented here.
"""

from dataclasses import dataclass, replace
from typing import Literal

from .continuous_contract import (
    Accepted,
    Context,
    Debt,
    JobRecord,
    Policy,
    ReadyRecord,
    Ref,
    Source,
    State,
    integer,
    record_ref,
    require,
)


@dataclass(frozen=True, slots=True)
class Frame:
    """State bound to its caller-authenticated workspace context, not a new wire."""

    context: Context
    state: State

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        require(
            isinstance(self.context, Context) and isinstance(self.state, State),
            "Invalid state frame.",
        )
        require(
            self.state.policy_sha256 == self.context.policy_sha256 == Policy().digest(),
            "Policy/context mismatch.",
        )


@dataclass(frozen=True, slots=True)
class Boundary:
    """Explicit expected context/revision and authenticated ordinary cycle."""

    context: Context
    revision: int
    cycle: int

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        require(isinstance(self.context, Context), "Invalid expected context.")
        integer(self.revision)
        integer(self.cycle)


@dataclass(frozen=True, slots=True)
class NoWork:
    """Bounded refusal with no mutation or automatic recovery implied."""

    reason: Literal[
        "suspended",
        "no_input",
        "dataset_slot_full",
        "training_slot_full",
        "source_age",
        "teacher_lag",
        "generation_not_new",
        "learning_debt",
        "promotion_debt",
        "freshness_disposition_required",
        "empty_validation_population",
    ]


@dataclass(frozen=True, slots=True)
class Displacement:
    """Required receipt inputs; these are not best-effort telemetry."""

    reason: Literal["coalesced_before_admission", "ready_superseded_unconsumed"]
    subject: Source | Ref
    successor: Source | Ref
    debt: Debt


@dataclass(frozen=True, slots=True)
class Change:
    """Proposed state plus immutable records/receipt obligation; not a commit."""

    frame: Frame
    records: tuple[JobRecord | ReadyRecord, ...] = ()
    displacement: Displacement | None = None


def initialize(workspace_epoch_sha256: str, config_sha256: str) -> Frame:
    """Create fresh empty state/context without enabling any runtime policy."""
    context = Context(workspace_epoch_sha256, config_sha256, Policy().digest())
    return Frame(
        context,
        State(
            1,
            0,
            context.policy_sha256,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            True,
        ),
    )


def _check(frame: Frame, boundary: Boundary) -> None:
    require(
        frame.context == boundary.context, "Workspace/configuration context changed."
    )
    require(frame.state.revision == boundary.revision, "Stale expected revision.")
    for debt in (frame.state.learning_debt, frame.state.promotion_debt):
        require(
            debt is None or debt.anchor_cycle <= boundary.cycle,
            "Debt anchor is in the future.",
        )


def _record(
    frame: Frame, ref: Ref | None, record: JobRecord | ReadyRecord | None
) -> None:
    if ref is None:
        require(record is None, "Unexpected resolved slot record.")
    else:
        require(record is not None, "Missing resolved slot record.")
        assert record is not None
        require(
            record.context == frame.context and record_ref(record) == ref,
            "Slot record/context mismatch.",
        )


def _eligible(frame: Frame, boundary: Boundary, source: Source) -> NoWork | None:
    require(
        source.cycle <= boundary.cycle
        and source.incumbent_epoch <= frame.state.incumbent_epoch,
        "Source clock/epoch is in the future.",
    )
    if source.generation <= frame.state.last_consumed_generation:
        return NoWork("generation_not_new")
    if boundary.cycle - source.cycle > Policy().max_source_age_cycles:
        return NoWork("source_age")
    if (
        frame.state.incumbent_epoch - source.incumbent_epoch
        > Policy().max_teacher_epoch_lag
    ):
        return NoWork("teacher_lag")
    return None


def _changed(
    frame: Frame,
    state: State,
    *records: JobRecord | ReadyRecord,
    displacement: Displacement | None = None,
) -> Change:
    return Change(
        Frame(frame.context, replace(state, revision=frame.state.revision + 1)),
        records,
        displacement,
    )


def growth_wait(
    frame: Frame,
    boundary: Boundary,
    *,
    dataset: JobRecord | None,
    ready: ReadyRecord | None,
) -> NoWork | None:
    """Gate a *next* ordinary cycle; completion at the frozen head is still allowed."""
    _check(frame, boundary)
    _record(frame, frame.state.dataset_job, dataset)
    _record(frame, frame.state.ready, ready)
    if not frame.state.admission_open:
        return NoWork("suspended")
    for debt, limit, refusal in (
        (frame.state.learning_debt, 8, NoWork("learning_debt")),
        (frame.state.promotion_debt, 2, NoWork("promotion_debt")),
    ):
        if debt is not None and boundary.cycle - debt.anchor_cycle >= limit:
            return refusal
    for record in (dataset, ready):
        if record is not None:
            source = record.payload.source
            require(source.cycle <= boundary.cycle, "Source clock is in the future.")
            if boundary.cycle - source.cycle >= 8:
                return NoWork("source_age")
    return None


def offer_source(
    frame: Frame,
    boundary: Boundary,
    source: Source,
    *,
    dataset: JobRecord | None,
    ready: ReadyRecord | None,
    sealed_head: Source | None = None,
) -> Change | NoWork:
    """Coalesce only pending work; a gate repair must name the authenticated head.

    At a gate, stale ready/admitted disposal is deliberately unsupported. Return
    a finite no-work outcome until a later owned disposition slice handles it.
    """
    gate = growth_wait(frame, boundary, dataset=dataset, ready=ready)
    if not frame.state.admission_open:
        return NoWork("suspended")
    refusal = _eligible(frame, boundary, source)
    if refusal is not None:
        return refusal
    old = frame.state.pending_source
    previous = [
        s
        for s in (
            old,
            None if dataset is None else dataset.payload.source,
            None if ready is None else ready.payload.source,
        )
        if s is not None
    ]
    for prior in previous:
        if source.generation <= prior.generation:
            return NoWork("generation_not_new")
        require(
            source.cycle >= prior.cycle
            and source.incumbent_epoch >= prior.incumbent_epoch,
            "Newer source regressed its clock/epoch.",
        )
    if gate is not None:
        if sealed_head != source or dataset is not None or ready is not None:
            return NoWork("freshness_disposition_required")
        if old is not None and _eligible(frame, boundary, old) is None:
            return NoWork("learning_debt")
    else:
        require(sealed_head is None, "Freshness repair requires a frozen-head gate.")
    debt = frame.state.learning_debt or Debt(source.publication_id, source.cycle)
    displaced = (
        None
        if old is None
        else Displacement("coalesced_before_admission", old, source, debt)
    )
    return _changed(
        frame,
        replace(frame.state, pending_source=source, learning_debt=debt),
        displacement=displaced,
    )


def admit_dataset(frame: Frame, boundary: Boundary, job: JobRecord) -> Change | NoWork:
    """Pin the selected source in EXTRACT without changing existing work."""
    _check(frame, boundary)
    state = frame.state
    if not state.admission_open:
        return NoWork("suspended")
    if state.dataset_job is not None:
        return NoWork("dataset_slot_full")
    if state.pending_source is None:
        return NoWork("no_input")
    require(
        job.context == frame.context
        and job.payload.stage == "EXTRACT"
        and job.payload.status == "ADMITTED",
        "Expected a matching admitted extraction job.",
    )
    require(
        job.payload.source == state.pending_source,
        "Dataset admission retargeted pending source.",
    )
    refusal = _eligible(frame, boundary, job.payload.source)
    if refusal is not None:
        return refusal
    return _changed(
        frame, replace(state, pending_source=None, dataset_job=record_ref(job)), job
    )


def bind_dataset(
    frame: Frame, boundary: Boundary, job: JobRecord, accepted: Accepted
) -> Change | NoWork:
    """Propose acceptance+PREPARE together; caller joins the accepted source ref."""
    _check(frame, boundary)
    _record(frame, frame.state.dataset_job, job)
    if not frame.state.admission_open:
        return NoWork("suspended")
    require(
        job.payload.stage == "EXTRACT" and job.payload.status == "ADMITTED",
        "Only admitted extraction can bind accepted data.",
    )
    prepared = JobRecord(
        frame.context, replace(job.payload, stage="PREPARE", accepted_dataset=accepted)
    )
    return _changed(
        frame, replace(frame.state, dataset_job=record_ref(prepared)), prepared
    )


def publish_ready(
    frame: Frame,
    boundary: Boundary,
    job: JobRecord,
    ready: ReadyRecord,
    *,
    previous_ready: ReadyRecord | None,
) -> Change | NoWork:
    """Propose complete readiness after the caller has closed the producer lease."""
    _check(frame, boundary)
    _record(frame, frame.state.dataset_job, job)
    _record(frame, frame.state.ready, previous_ready)
    if not frame.state.admission_open:
        return NoWork("suspended")
    require(
        job.payload.stage == "PREPARE" and job.payload.status == "ADMITTED",
        "Only admitted preparation can publish readiness.",
    )
    require(
        ready.context == frame.context
        and (
            ready.payload.source,
            ready.payload.accepted_dataset,
            ready.payload.producer_job_id,
        )
        == (job.payload.source, job.payload.accepted_dataset, job.payload.id),
        "Ready/source/accepted/job binding differs.",
    )
    displaced = None
    if previous_ready is not None:
        require(
            ready.payload.source.generation > previous_ready.payload.source.generation,
            "Ready publication would regress source generation.",
        )
        assert frame.state.learning_debt is not None
        displaced = Displacement(
            "ready_superseded_unconsumed",
            record_ref(previous_ready),
            record_ref(ready),
            frame.state.learning_debt,
        )
    return _changed(
        frame,
        replace(frame.state, dataset_job=None, ready=record_ref(ready)),
        ready,
        displacement=displaced,
    )


def admit_cohort(
    frame: Frame,
    boundary: Boundary,
    ready: ReadyRecord,
    job: JobRecord,
    *,
    validation_count: int,
) -> Change | NoWork:
    """Bind parents at Training admission, never during CPU preparation.

    validation_count must come from an authenticated common held-out split, not
    a caller guess. Its proof, claim/parent freshness and executor admission are
    IO-publisher obligations. Empty validation is explicit no-work, not training.
    """
    _check(frame, boundary)
    integer(validation_count)
    state = frame.state
    if not state.admission_open:
        return NoWork("suspended")
    if state.cohort is not None:
        return NoWork("training_slot_full")
    if state.ready is None:
        return NoWork("no_input")
    _record(frame, state.ready, ready)
    require(
        job.context == frame.context
        and job.payload.stage == "TRAIN"
        and job.payload.status == "ADMITTED",
        "Expected an admitted Training job.",
    )
    require(
        (job.payload.source, job.payload.accepted_dataset, job.payload.ready)
        == (ready.payload.source, ready.payload.accepted_dataset, state.ready),
        "Training input binding differs.",
    )
    require(
        job.payload.round_number == state.last_accepted_round + 1,
        "Training round is not the next accepted round.",
    )
    refusal = _eligible(frame, boundary, ready.payload.source)
    if refusal is not None:
        return refusal
    if validation_count == 0:
        return NoWork("empty_validation_population")
    # The pending slot may never become an already-consumed generation.
    require(
        state.pending_source is None
        or state.pending_source.generation > ready.payload.source.generation,
        "Pending source is not newer than consumed data.",
    )
    return _changed(
        frame,
        replace(
            state,
            ready=None,
            cohort=record_ref(job),
            last_consumed_generation=ready.payload.source.generation,
        ),
        job,
    )
