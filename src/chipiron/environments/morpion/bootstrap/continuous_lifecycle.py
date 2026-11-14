"""Owned lifecycle reconciliation for the six continuous slots.

No process is killed here. Native managed shutdown proves quiescence; immutable
claim evidence then fences retries without replacing source, parents or clocks.
"""

import math
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from . import continuous_publication as publication
from . import continuous_transitions as transitions
from .continuous_contract import (
    Budget,
    ControlRecord,
    Disposition,
    JobRecord,
    Reservation,
    record_ref,
    require,
)
from .continuous_dataset import boundary
from .generation_publication_io import growth_receipt_ownership, read_object
from .run_state import load_bootstrap_run_state, save_bootstrap_run_state

if TYPE_CHECKING:
    from .bootstrap_paths import MorpionBootstrapPaths
    from .pipeline_artifacts import MorpionPipelineStageClaim
    from .run_state import MorpionBootstrapRunState


def remaining(paths: MorpionBootstrapPaths, job: JobRecord) -> Budget:
    """Validate both original clocks in the monitor as well as worker processes."""
    from .managed_sessions import _local_identity

    record = publication.read_record(paths, job.payload.budget, job.context)
    require(
        isinstance(record, ControlRecord) and isinstance(record.payload, Budget),
        "Missing original job allowance.",
    )
    assert isinstance(record, ControlRecord) and isinstance(record.payload, Budget)
    budget = record.payload
    clock = publication.read_ref(paths, budget.clock_evidence)
    local = _local_identity()
    require(
        all(clock["owner"][k] == local[k] for k in ("host", "boot", "pid_namespace")),
        "Original job clock belongs to another host/boot/namespace.",
    )
    end, wall = clock["deadline"], clock["utc_deadline"]
    require(
        all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (end, wall)
        ),
        "Invalid original job clocks.",
    )
    left = end - time.monotonic()
    require(
        0 < left <= budget.max_job_seconds and abs(left - (wall - time.time())) < 5,
        "Original job allowance exhausted or clocks disagree.",
    )
    return budget


def dead(paths: MorpionBootstrapPaths, job: JobRecord) -> None:
    """TTL and a missing claim file never establish worker death."""
    from .learning_worker import require_quiescent
    from .managed_sessions import _local_identity

    claim = publication.read_ref(paths, job.payload.claim)
    previous = claim.get("metadata", {}).get("learning_worker")
    require(isinstance(previous, dict), "Missing immutable worker proof.")
    assert isinstance(previous, dict)
    require_quiescent(previous, {**previous, **_local_identity()})


def authenticate(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    action: transitions.Transition,
) -> None:
    """Authenticate lifecycle commands before the sole publication facade applies."""
    frame = publication.load_frame(paths, state)
    records = publication.slot_records(paths, frame)
    active_refs = {frame.state.dataset_job, frame.state.cohort}
    if frame.state.reserved_promotion is not None:
        reserved = records[frame.state.reserved_promotion]
        assert isinstance(reserved.payload, Reservation)
        if reserved.payload.phase == "COMPARING":
            active_refs.add(reserved.payload.comparison_job)
    jobs = [
        r
        for ref, r in records.items()
        if ref in active_refs and isinstance(r, JobRecord)
    ]
    if isinstance(action, transitions.Admission) and action.opened:
        for job in jobs:
            dead(paths, job)
            remaining(paths, job)
            publication.authenticate_source(paths, state, job.payload.source)
    if isinstance(action, (transitions.Suspend, transitions.Discard)):
        for job in jobs:
            dead(paths, job)
    if isinstance(action, transitions.Retry):
        dead(paths, action.previous)
        old = remaining(paths, action.previous)
        new = remaining(paths, action.job)
        require(
            replace(
                new, launch=old.launch, consumed_job_seconds=old.consumed_job_seconds
            )
            == old
            and new.consumed_job_seconds >= old.consumed_job_seconds,
            "Retry renewed the original allowance.",
        )
        publication.authenticate_job(paths, state, action.job)


def apply(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    action: transitions.Transition,
) -> MorpionBootstrapRunState:
    """Propose one lifecycle transition for the caller's existing durable save."""
    updated = publication.apply_owned(paths, state, boundary(paths, state), action)
    require(
        not isinstance(updated, transitions.NoWork), "Lifecycle transition refused."
    )
    assert not isinstance(updated, transitions.NoWork)
    return updated


def close_admission(paths: MorpionBootstrapPaths) -> None:
    """Close before native reaping, using a nonblocking Growth acquisition."""
    if not publication.enabled(load_bootstrap_run_state(paths.run_state_path)):
        return
    with growth_receipt_ownership(paths, create=False):
        state = load_bootstrap_run_state(paths.run_state_path)
        if (
            publication.enabled(state)
            and publication.load_frame(paths, state).state.admission_open
        ):
            state = apply(paths, state, transitions.Admission(False))
            save_bootstrap_run_state(state, paths.run_state_path, durable=True)


def suspend_owned(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> MorpionBootstrapRunState:
    """After positive native reap, keep unresolved pins in the managed close save."""
    if not publication.enabled(state):
        return state
    # Closing and suspension are one transition here if shutdown was interrupted.
    # The native monitor normally commits the close before it acknowledges TERM.
    frame = publication.load_frame(paths, state)
    require(
        not frame.state.admission_open,
        "Continuous admission was not closed before reap.",
    )
    return apply(paths, state, transitions.Suspend())


def resume_owned(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> MorpionBootstrapRunState:
    """Reconcile every original job before a new managed launch can spawn workers."""
    if (
        not publication.enabled(state)
        or publication.load_frame(paths, state).state.admission_open
    ):
        return state
    return apply(paths, state, transitions.Admission(True))


def retry_owned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    previous: JobRecord,
    claim: MorpionPipelineStageClaim,
) -> tuple[MorpionBootstrapRunState, JobRecord]:
    """Rebind a proven-dead job to the new claim, retaining its original clock."""
    from .managed_sessions import KEY as MANAGED

    old = remaining(paths, previous)
    clock = publication.read_ref(paths, old.clock_evidence)
    launch = publication.put_proof(paths, state.metadata[MANAGED]["active"])
    budget = publication.put_record(
        paths,
        ControlRecord(
            previous.context,
            replace(
                old,
                launch=launch,
                consumed_job_seconds=max(
                    old.consumed_job_seconds,
                    old.max_job_seconds - (clock["deadline"] - time.monotonic()),
                ),
            ),
        ),
    )
    claim_path = (
        paths.pipeline_dataset_claim_path_for_generation(claim.generation)
        if claim.stage == "dataset"
        else paths.pipeline_training_claim_path_for_generation(claim.generation)
    )
    job = JobRecord(
        previous.context,
        replace(
            previous.payload,
            claim=publication.put_proof(paths, read_object(claim_path)),
            managed_launch=launch,
            budget=budget,
            status="ADMITTED",
        ),
    )
    return apply(paths, state, transitions.Retry(previous, job)), job


def require_settled(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> None:
    """Refuse to import pending work, debt, cache authority or cutover."""
    if not publication.enabled(state):
        return
    frame = publication.load_frame(paths, state)
    publication.slot_records(paths, frame)
    require(
        all(
            getattr(frame.state, k) is None
            for k in (
                "pending_source",
                "dataset_job",
                "ready",
                "cohort",
                "reserved_promotion",
                "pending_promotion",
                "learning_debt",
                "promotion_debt",
            )
        ),
        "Unsettled continuous slots/debts prevent fork.",
    )


def settled_discard(paths: MorpionBootstrapPaths) -> None:
    """Explicit operator action; never invoked automatically by stop or fork."""
    with growth_receipt_ownership(paths, create=False):
        state = load_bootstrap_run_state(paths.run_state_path)
        frame = publication.load_frame(paths, state)
        records = publication.slot_records(paths, frame)
        subjects = [
            r
            for r in (
                frame.state.pending_source,
                frame.state.dataset_job,
                frame.state.ready,
                frame.state.cohort,
                frame.state.pending_promotion,
            )
            if r is not None
        ]
        if frame.state.reserved_promotion is not None:
            reservation = records[frame.state.reserved_promotion]
            assert isinstance(reservation.payload, Reservation)
            subjects.append(reservation.payload.opportunity)
        state = apply(paths, state, transitions.Discard())
        for subject in subjects:
            state = publication.attach_disposition(
                paths,
                state,
                Disposition(
                    subject, "settled_discard", None, frame.state.learning_debt
                ),
            )
        save_bootstrap_run_state(state, paths.run_state_path, durable=True)
        # The durable discard precedes deletion. Only the exact proven-dead
        # attempts are moved out of publication namespaces; scientific accepts
        # and compact failure evidence remain untouched.
        import os

        from .continuous_contract import Ref
        from .generation_publication_io import sync_artifact
        from .worker_result_io import publication_record

        for job in records.values():
            if not isinstance(job, JobRecord) or job.payload.stage not in {
                "EXTRACT",
                "TRAIN",
            }:
                continue
            release_dead_claim(paths, job)
            source = publication_record(state, job.payload.source.generation)
            accepted = source.get(
                "accepted_training"
                if job.payload.stage == "TRAIN"
                else "accepted_dataset"
            )
            namespace = (
                "training_publications_v1"
                if job.payload.stage == "TRAIN"
                else "dataset_publications_v1"
            )
            for raw in source.get("continuous_admissions", []):
                attempt = publication.read_ref(paths, Ref.from_metadata(raw))
                original = publication.read_record(
                    paths,
                    Ref.from_metadata(attempt["continuous_admission"]),
                    frame.context,
                )
                assert isinstance(original, JobRecord)
                if original.payload.id != job.payload.id or (
                    accepted is not None
                    and accepted["attempt_id"] == attempt["attempt_id"]
                ):
                    continue
                for area in ("staging", "objects"):
                    directory = (
                        paths.work_dir / namespace / area / attempt["attempt_id"]
                    )
                    if directory.exists():
                        destination = (
                            paths.pipeline_dir
                            / "continuous_v1/discarded"
                            / namespace
                            / area
                            / attempt["attempt_id"]
                        )
                        require(
                            not directory.is_symlink() and not destination.exists(),
                            "Conflicting discard destination.",
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.rename(directory, destination)
                        sync_artifact(destination, paths.work_dir)


def remember_admission(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    admission: dict[str, Any],
) -> MorpionBootstrapRunState:
    """Keep the exact retry recipe in its source receipt, before the single save."""
    from .generation_publication_io import PUBLICATION_KEY

    proof = publication.put_proof(paths, admission).to_metadata()
    data = state.metadata[PUBLICATION_KEY]
    records = [
        {**r, "continuous_admissions": [*r.get("continuous_admissions", []), proof]}
        if r["generation"] == admission["generation"]
        else r
        for r in data["generations"]
    ]
    return replace(
        state,
        metadata={**state.metadata, PUBLICATION_KEY: {**data, "generations": records}},
    )


def previous_admission(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, job: JobRecord
) -> dict[str, Any]:
    """Resolve the last immutable attempt belonging to this exact job claim."""
    from .continuous_contract import Ref
    from .worker_result_io import publication_record

    for ref in reversed(
        publication_record(state, job.payload.source.generation).get(
            "continuous_admissions", []
        )
    ):
        value = publication.read_ref(paths, Ref.from_metadata(ref))
        if value.get("continuous_admission") == record_ref(job).to_metadata():
            return value
        original = publication.read_record(
            paths, Ref.from_metadata(value["continuous_admission"]), job.context
        )
        # Suspension changes only the status and hence the envelope digest.
        if (
            isinstance(original, JobRecord)
            and replace(original.payload, status=job.payload.status) == job.payload
        ):
            return value
    message = "Missing original immutable continuous admission."
    raise ValueError(message)


def release_dead_claim(paths: MorpionBootstrapPaths, job: JobRecord) -> None:
    """Release only the exact proven-dead claim, even if its TTL has not expired."""
    from .pipeline_claims import release_owned_pipeline_stage_claim

    dead(paths, job)
    claim = publication.read_ref(paths, job.payload.claim)
    path = (
        paths.pipeline_dataset_claim_path_for_generation(job.payload.source.generation)
        if job.payload.stage in {"EXTRACT", "PREPARE"}
        else paths.pipeline_training_claim_path_for_generation(
            job.payload.source.generation
        )
    )
    if path.exists():
        require(read_object(path) == claim, "A different claim owns this job.")
        release_owned_pipeline_stage_claim(claim_path=path, claim_id=claim["claim_id"])


def recover_preparation(paths: MorpionBootstrapPaths) -> None:
    """Fence a dead producer before restarting its already accepted Dataset."""
    from . import prepared_dataset_storage as storage
    from .persistent_worker_state import monitored_identity
    from .pipeline_claims import claim_pipeline_stage

    with growth_receipt_ownership(paths, wait=True):
        state = load_bootstrap_run_state(paths.run_state_path)
        if not publication.enabled(state):
            return
        frame = publication.load_frame(paths, state)
        if frame.state.dataset_job is None or not frame.state.admission_open:
            return
        job = publication.read_record(paths, frame.state.dataset_job, frame.context)
        assert isinstance(job, JobRecord)
        if job.payload.stage != "PREPARE":
            return
        previous = publication.read_ref(paths, job.payload.claim)
        if previous["metadata"]["learning_worker"] == monitored_identity():
            publication.authenticate_job(paths, state, job)
            return
        budget = remaining(paths, job)
        release_dead_claim(paths, job)
        clock = publication.read_ref(paths, budget.clock_evidence)
        claim = claim_pipeline_stage(
            generation=job.payload.source.generation,
            stage="dataset",
            claim_path=paths.pipeline_dataset_claim_path_for_generation(
                job.payload.source.generation
            ),
            ttl_seconds=clock["deadline"] - time.monotonic(),
            owner=None,
            metadata={"learning_worker": monitored_identity()},
        )
        state, retried = retry_owned(paths, state, job, claim)
        assert retried.payload.accepted_dataset is not None
        data = storage.entries(state)
        key = retried.payload.accepted_dataset.descriptor_sha256
        record = data[key]
        data[key] = {
            **record,
            "dataset_claim_id": claim.claim_id,
            "owner": monitored_identity(),
            "phase": "PREPARING",
        }
        save_bootstrap_run_state(
            storage.updated(state, data), paths.run_state_path, durable=True
        )


def prepare_dispatch(paths: MorpionBootstrapPaths, *, training: bool) -> None:
    """Make only a proven-dead admitted job discoverable by the existing worker."""
    with growth_receipt_ownership(paths, wait=True):
        state = load_bootstrap_run_state(paths.run_state_path)
        frame = publication.load_frame(paths, state)
        ref = frame.state.cohort if training else frame.state.dataset_job
        if ref is not None and frame.state.admission_open:
            job = publication.read_record(paths, ref, frame.context)
            assert isinstance(job, JobRecord)
            remaining(paths, job)
            release_dead_claim(paths, job)
