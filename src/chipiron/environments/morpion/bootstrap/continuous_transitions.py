"""Pure closed transition proposals for bounded continuous coordination.

No function grants admission authority. Callers must authenticate every supplied
record/ref, current committed cycle, source, claim, validation population, parent
map and resource reservation under the existing owner. Persist records first,
then save the returned State once with related accepted/publication references.
Displacements require durable receipts in that same save. No IO, clock, lease,
or claim takeover is performed here.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, assert_never

from .continuous_contract import (
    Accepted,
    Comparison,
    Context,
    ControlRecord,
    Debt,
    JobRecord,
    Opportunity,
    Policy,
    Population,
    ReadyRecord,
    Record,
    Ref,
    Reservation,
    Source,
    State,
    integer,
    record_ref,
    require,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


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
        "learner_executor",
        "resource_backpressure",
    ]


@dataclass(frozen=True, slots=True)
class Displacement:
    """Required receipt inputs; these are not best-effort telemetry."""

    reason: Literal[
        "coalesced_before_admission",
        "ready_superseded_unconsumed",
        "promotion_coalesced",
    ]
    subject: Source | Ref
    successor: Source | Ref
    debt: Debt


@dataclass(frozen=True, slots=True)
class Change:
    """Proposed state plus immutable records/receipt obligation; not a commit."""

    frame: Frame
    records: tuple[Record, ...] = ()
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


def _record(frame: Frame, ref: Ref | None, record: Record | None) -> None:
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
    *records: Record,
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


@dataclass(frozen=True, slots=True)
class Offer:
    """Offer a freshly authenticated source, optionally repairing a frozen head."""

    source: Source
    sealed_head: Source | None = None


@dataclass(frozen=True, slots=True)
class AdmitDataset:
    """Reserve extraction using an authenticated job and resource allowance."""

    job: JobRecord


@dataclass(frozen=True, slots=True)
class BindDataset:
    """Join accepted Dataset and PREPARE in the caller's single save."""

    job: JobRecord
    accepted: Accepted


@dataclass(frozen=True, slots=True)
class PublishReady:
    """Install a verified native completion after producer ownership closes."""

    job: JobRecord
    ready: ReadyRecord


@dataclass(frozen=True, slots=True)
class AdmitCohort:
    """Freeze current own parents and transfer an unconsumed ready input."""

    job: JobRecord
    ready: ReadyRecord
    validation_count: int


@dataclass(frozen=True, slots=True)
class AcceptCohort:
    """Join complete accepted lineage with the next outstanding debt/opportunity."""

    job: JobRecord
    accepted: Accepted
    opportunity: ControlRecord | None
    remaining_debt: Debt | None


@dataclass(frozen=True, slots=True)
class ReservePromotion:
    """Pin the oldest pending opportunity; no newer acceptance can replace it."""

    reservation: ControlRecord


@dataclass(frozen=True, slots=True)
class StartComparison:
    """Bind a serial learner job to an existing reservation."""

    reservation: ControlRecord
    job: JobRecord


@dataclass(frozen=True, slots=True)
class VerifyComparison:
    """Attach a fenced common-population result."""

    reservation: ControlRecord
    comparison: ControlRecord


@dataclass(frozen=True, slots=True)
class ResolvePromotion:
    """Clear a verified retain or join the canonical activation commit."""

    reservation: ControlRecord
    comparison: ControlRecord
    promoted: bool


@dataclass(frozen=True, slots=True)
class Recompare:
    """Retain the original opportunity/debt while replacing a stale incumbent proof."""

    reservation: ControlRecord
    incumbent: Ref


@dataclass(frozen=True, slots=True)
class Admission:
    """Close before reap, or reopen after authenticated managed reconciliation."""

    opened: bool


@dataclass(frozen=True, slots=True)
class Suspend:
    """Preserve every pin while marking positively quiescent jobs suspended."""


@dataclass(frozen=True, slots=True)
class Retry:
    """Replace only a dead job's fencing claim/session, retaining its work."""

    previous: JobRecord
    job: JobRecord


@dataclass(frozen=True, slots=True)
class Discard:
    """Explicitly settle unclaimed and proven-dead incomplete work."""


@dataclass(frozen=True, slots=True)
class DropReady:
    """Remove only an unconsumed ineligible input, preserving scientific history."""

    ready: ReadyRecord


@dataclass(frozen=True, slots=True)
class BeginCutover:
    """Bind the current-head activation request to the verified reservation."""

    reservation: ControlRecord
    request: Ref


type Transition = (
    Offer
    | AdmitDataset
    | BindDataset
    | PublishReady
    | AdmitCohort
    | AcceptCohort
    | ReservePromotion
    | StartComparison
    | VerifyComparison
    | ResolvePromotion
    | Recompare
    | Admission
    | Suspend
    | Retry
    | Discard
    | DropReady
    | BeginCutover
)


def _slot[T: Record](
    frame: Frame, ref: Ref | None, records: Mapping[Ref, Record], kind: type[T]
) -> T | None:
    if ref is None:
        return None
    value = records.get(ref)
    _record(frame, ref, value)
    require(isinstance(value, kind), "Incorrect slot envelope type.")
    assert isinstance(value, kind)
    return value


def _reservation(record: ControlRecord) -> Reservation:
    require(isinstance(record.payload, Reservation), "Expected reservation envelope.")
    assert isinstance(record.payload, Reservation)
    return record.payload


def comparison_owed(frame: Frame, records: Mapping[Ref, Record]) -> bool:
    """Give a reserved or unfinished comparison priority over new Training."""
    record = _slot(frame, frame.state.reserved_promotion, records, ControlRecord)
    return record is not None and _reservation(record).phase in {
        "RESERVED",
        "COMPARING",
    }


def transition(
    frame: Frame, boundary: Boundary, action: Transition, records: Mapping[Ref, Record]
) -> Change | NoWork:
    """Apply only a closed command; all IO/authentication belongs to the publisher."""
    _check(frame, boundary)
    dataset = _slot(frame, frame.state.dataset_job, records, JobRecord)
    ready = _slot(frame, frame.state.ready, records, ReadyRecord)
    if isinstance(action, Admission):
        require(
            type(action.opened) is bool,  # pylint: disable=unidiomatic-typecheck
            "Invalid admission gate.",
        )
        return _changed(frame, replace(frame.state, admission_open=action.opened))
    if isinstance(action, Suspend):
        require(not frame.state.admission_open, "Close admission before suspension.")
        state = frame.state
        changes: list[Record] = []
        for name in ("dataset_job", "cohort"):
            suspending_job = _slot(frame, getattr(state, name), records, JobRecord)
            if suspending_job is not None:
                suspended = JobRecord(
                    frame.context, replace(suspending_job.payload, status="SUSPENDED")
                )
                state = (
                    replace(state, dataset_job=record_ref(suspended))
                    if name == "dataset_job"
                    else replace(state, cohort=record_ref(suspended))
                )
                changes.append(suspended)
        reserved = _slot(frame, state.reserved_promotion, records, ControlRecord)
        if reserved is not None:
            value = _reservation(reserved)
            suspending_job = _slot(frame, value.comparison_job, records, JobRecord)
            if suspending_job is not None and value.phase == "COMPARING":
                suspended = JobRecord(
                    frame.context, replace(suspending_job.payload, status="SUSPENDED")
                )
                updated = ControlRecord(
                    frame.context, replace(value, comparison_job=record_ref(suspended))
                )
                state = replace(state, reserved_promotion=record_ref(updated))
                changes.extend((suspended, updated))
        return _changed(frame, state, *changes)
    if isinstance(action, Discard):
        require(
            not frame.state.admission_open, "Settled discard requires closed admission."
        )
        reserved = _slot(frame, frame.state.reserved_promotion, records, ControlRecord)
        require(
            reserved is None or _reservation(reserved).phase != "CUTOVER",
            "Unresolved cold cutover cannot be discarded.",
        )
        return _changed(
            frame,
            replace(
                frame.state,
                pending_source=None,
                dataset_job=None,
                ready=None,
                cohort=None,
                reserved_promotion=None,
                pending_promotion=None,
                learning_debt=None,
                promotion_debt=None,
            ),
        )
    if isinstance(action, Retry):
        old, new = action.previous.payload, action.job.payload
        require(
            frame.state.admission_open
            and action.job.context == action.previous.context == frame.context,
            "Retry requires open matching authority.",
        )
        require(
            replace(
                new,
                claim=old.claim,
                managed_launch=old.managed_launch,
                budget=old.budget,
                status=old.status,
            )
            == old
            and new.status == "ADMITTED"
            and new.claim != old.claim,
            "Retry retargeted immutable admitted work.",
        )
        state = frame.state
        if old.stage in {"EXTRACT", "PREPARE"}:
            _record(frame, state.dataset_job, action.previous)
            state = replace(state, dataset_job=record_ref(action.job))
        elif old.stage == "TRAIN":
            _record(frame, state.cohort, action.previous)
            state = replace(state, cohort=record_ref(action.job))
        else:
            reserved = _slot(frame, state.reserved_promotion, records, ControlRecord)
            require(reserved is not None, "Missing comparison reservation.")
            assert reserved is not None
            value = _reservation(reserved)
            require(value.phase == "COMPARING", "Comparison retry is not pending.")
            _record(frame, value.comparison_job, action.previous)
            updated = ControlRecord(
                frame.context, replace(value, comparison_job=record_ref(action.job))
            )
            return _changed(
                frame,
                replace(state, reserved_promotion=record_ref(updated)),
                action.job,
                updated,
            )
        return _changed(frame, state, action.job)
    if isinstance(action, DropReady):
        _record(frame, frame.state.ready, action.ready)
        require(
            _eligible(frame, boundary, action.ready.payload.source) is not None,
            "Eligible Ready cannot be discarded for freshness.",
        )
        return _changed(frame, replace(frame.state, ready=None))
    if isinstance(action, BeginCutover):
        _record(frame, frame.state.reserved_promotion, action.reservation)
        value = _reservation(action.reservation)
        require(
            value.phase in {"VERIFIED", "CUTOVER"} and value.comparison is not None,
            "Unverified cutover.",
        )
        updated = ControlRecord(
            frame.context,
            replace(
                value,
                phase="CUTOVER",
                activation_request=action.request,
                previous=record_ref(action.reservation)
                if value.phase == "VERIFIED"
                else value.previous,
            ),
        )
        return _changed(
            frame, replace(frame.state, reserved_promotion=record_ref(updated)), updated
        )
    if isinstance(action, Offer):
        return offer_source(
            frame,
            boundary,
            action.source,
            dataset=dataset,
            ready=ready,
            sealed_head=action.sealed_head,
        )
    if isinstance(action, AdmitDataset):
        return admit_dataset(frame, boundary, action.job)
    if isinstance(action, BindDataset):
        return bind_dataset(frame, boundary, action.job, action.accepted)
    if isinstance(action, PublishReady):
        return publish_ready(
            frame, boundary, action.job, action.ready, previous_ready=ready
        )
    if isinstance(action, AdmitCohort):
        if comparison_owed(frame, records) or (
            frame.state.reserved_promotion is None
            and frame.state.pending_promotion is not None
        ):
            return NoWork("learner_executor")
        return admit_cohort(
            frame,
            boundary,
            action.ready,
            action.job,
            validation_count=action.validation_count,
        )
    if not frame.state.admission_open:
        return NoWork("suspended")
    if isinstance(action, AcceptCohort):
        return _accept_cohort(frame, boundary, action, records, dataset, ready)
    if isinstance(action, ReservePromotion):
        require(
            frame.state.reserved_promotion is None,
            "Cannot replace a reserved promotion.",
        )
        value = _reservation(action.reservation)
        require(
            action.reservation.context == frame.context
            and value.phase == "RESERVED"
            and value.opportunity == frame.state.pending_promotion
            and value.previous is None,
            "Reservation does not bind oldest pending opportunity.",
        )
        require(frame.state.pending_promotion is not None, "No pending opportunity.")
        return _changed(
            frame,
            replace(
                frame.state,
                reserved_promotion=record_ref(action.reservation),
                pending_promotion=None,
            ),
            action.reservation,
        )
    if isinstance(action, Recompare):
        _record(frame, frame.state.reserved_promotion, action.reservation)
        value = _reservation(action.reservation)
        require(
            value.phase == "VERIFIED" and action.incumbent != value.incumbent,
            "Only stale verified comparisons may recompare.",
        )
        updated = ControlRecord(
            frame.context,
            replace(
                value,
                phase="RESERVED",
                incumbent=action.incumbent,
                comparison_job=None,
                comparison=None,
                previous=record_ref(action.reservation),
            ),
        )
        return _changed(
            frame, replace(frame.state, reserved_promotion=record_ref(updated)), updated
        )
    if isinstance(action, (StartComparison, VerifyComparison, ResolvePromotion)):
        _record(frame, frame.state.reserved_promotion, action.reservation)
        value = _reservation(action.reservation)
        if isinstance(action, StartComparison):
            require(frame.state.cohort is None, "Training owns the serial learner.")
            require(value.phase == "RESERVED", "Comparison already admitted.")
            job = action.job.payload
            require(
                action.job.context == frame.context
                and job.stage == "COMPARE"
                and job.status == "ADMITTED"
                and job.reserved_id == value.opportunity.sha256,
                "Comparison job reservation differs.",
            )
            opportunity = _slot(frame, value.opportunity, records, ControlRecord)
            population = _slot(frame, value.population, records, ControlRecord)
            require(
                opportunity is not None
                and isinstance(opportunity.payload, Opportunity)
                and population is not None
                and isinstance(population.payload, Population),
                "Missing comparison population roots.",
            )
            assert opportunity is not None and isinstance(
                opportunity.payload, Opportunity
            )
            assert population is not None and isinstance(population.payload, Population)
            require(
                job.accepted_dataset
                == opportunity.payload.accepted_dataset
                == population.payload.accepted_dataset
                and job.round_number == opportunity.payload.round_number
                and job.source.generation == opportunity.payload.generation,
                "Comparison job retargeted reserved inputs.",
            )
            updated = ControlRecord(
                frame.context,
                replace(
                    value,
                    phase="COMPARING",
                    comparison_job=record_ref(action.job),
                    previous=record_ref(action.reservation),
                ),
            )
            return _changed(
                frame,
                replace(frame.state, reserved_promotion=record_ref(updated)),
                action.job,
                updated,
            )
        require(
            isinstance(action.comparison.payload, Comparison),
            "Expected comparison result.",
        )
        comparison = action.comparison.payload
        assert isinstance(comparison, Comparison)
        require(
            action.comparison.context == frame.context
            and comparison.reservation_id == value.opportunity.sha256
            and comparison.incumbent == value.incumbent
            and comparison.population == value.population,
            "Comparison reservation/population differs.",
        )
        opportunity = _slot(frame, value.opportunity, records, ControlRecord)
        require(
            opportunity is not None and isinstance(opportunity.payload, Opportunity),
            "Missing reserved opportunity.",
        )
        assert opportunity is not None and isinstance(opportunity.payload, Opportunity)
        require(
            comparison.candidate == opportunity.payload.candidate,
            "Comparison candidate differs.",
        )
        if isinstance(action, VerifyComparison):
            require(value.phase == "COMPARING", "Comparison was not admitted.")
            job_record = _slot(frame, value.comparison_job, records, JobRecord)
            require(
                job_record is not None and comparison.claim == job_record.payload.claim,
                "Comparison claim differs.",
            )
            updated = ControlRecord(
                frame.context,
                replace(
                    value,
                    phase="VERIFIED",
                    comparison=record_ref(action.comparison),
                    previous=record_ref(action.reservation),
                ),
            )
            return _changed(
                frame,
                replace(frame.state, reserved_promotion=record_ref(updated)),
                action.comparison,
                updated,
            )
        require(
            value.phase in {"VERIFIED", "CUTOVER"}
            and value.comparison == record_ref(action.comparison),
            "Unverified promotion resolution.",
        )
        require(
            isinstance(action.promoted, bool)
            and action.promoted == (comparison.decision == "promote"),
            "Resolution disagrees with comparison.",
        )
        pending = _slot(frame, frame.state.pending_promotion, records, ControlRecord)
        require(
            pending is None or isinstance(pending.payload, Opportunity),
            "Invalid pending opportunity.",
        )
        assert pending is None or isinstance(pending.payload, Opportunity)
        return _changed(
            frame,
            replace(
                frame.state,
                reserved_promotion=None,
                promotion_debt=None if pending is None else _opportunity(pending).debt,
                incumbent_epoch=frame.state.incumbent_epoch + int(action.promoted),
            ),
        )

    assert_never(action)


def _accept_cohort(
    frame: Frame,
    boundary: Boundary,
    action: AcceptCohort,
    records: Mapping[Ref, Record],
    dataset: JobRecord | None,
    ready: ReadyRecord | None,
) -> Change:
    _record(frame, frame.state.cohort, action.job)
    job = action.job.payload
    require(
        job.stage == "TRAIN"
        and job.status == "ADMITTED"
        and job.round_number == frame.state.last_accepted_round + 1,
        "Wrong accepted cohort.",
    )
    remaining = [
        s
        for s in (
            frame.state.pending_source,
            None if dataset is None else dataset.payload.source,
            None if ready is None else ready.payload.source,
        )
        if s is not None
    ]
    require(
        bool(remaining) == (action.remaining_debt is not None),
        "Outstanding learning debt missing or fabricated.",
    )
    if action.remaining_debt is not None:
        require(
            action.remaining_debt.anchor_cycle
            <= min(s.cycle for s in remaining)
            <= boundary.cycle,
            "Outstanding debt moved past its source.",
        )
    assert job.round_number is not None
    state = replace(
        frame.state,
        cohort=None,
        last_accepted_round=job.round_number,
        learning_debt=action.remaining_debt,
    )
    if action.opportunity is None:
        return _changed(frame, state)
    opportunity = action.opportunity.payload
    require(
        action.opportunity.context == frame.context
        and isinstance(opportunity, Opportunity),
        "Invalid accepted opportunity.",
    )
    assert isinstance(opportunity, Opportunity)
    require(
        opportunity.accepted_training == action.accepted
        and opportunity.accepted_dataset == job.accepted_dataset
        and opportunity.generation == job.source.generation
        and opportunity.round_number == job.round_number,
        "Opportunity retargeted accepted cohort.",
    )
    old = _slot(frame, state.pending_promotion, records, ControlRecord)
    require(
        old is None or isinstance(old.payload, Opportunity),
        "Invalid pending opportunity.",
    )
    assert old is None or isinstance(old.payload, Opportunity)
    expected_debt = (
        Debt(opportunity.id, boundary.cycle) if old is None else _opportunity(old).debt
    )
    require(opportunity.debt == expected_debt, "Promotion replacement moved debt.")
    if old is not None:
        require(
            opportunity.round_number > _opportunity(old).round_number,
            "Promotion round regressed.",
        )
    return _changed(
        frame,
        replace(
            state,
            pending_promotion=record_ref(action.opportunity),
            promotion_debt=state.promotion_debt or expected_debt,
        ),
        action.opportunity,
        displacement=None
        if old is None
        else Displacement(
            "promotion_coalesced",
            record_ref(old),
            record_ref(action.opportunity),
            expected_debt,
        ),
    )


def _opportunity(record: ControlRecord) -> Opportunity:
    require(isinstance(record.payload, Opportunity), "Expected opportunity envelope.")
    assert isinstance(record.payload, Opportunity)
    return record.payload
