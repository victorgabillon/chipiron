"""Join Dataset admission and native preparation to continuous authority.

These owned operations return a proposed run state. Their callers retain the
existing Growth lease and perform the one durable scientific acceptance save.
"""

from typing import TYPE_CHECKING, Any

from . import continuous_publication as publication
from . import continuous_transitions as transitions
from . import prepared_dataset_storage as storage
from .continuous_contract import (
    Accepted,
    Budget,
    ControlRecord,
    Job,
    JobRecord,
    Ready,
    ReadyRecord,
    Ref,
    record_ref,
)
from .generation_publication_io import object_digest, require

if TYPE_CHECKING:
    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .pipeline_artifacts import MorpionPipelineStageClaim
    from .run_state import MorpionBootstrapRunState


def boundary(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> transitions.Boundary:
    """Use the current owned revision and committed ordinary-cycle clock."""
    frame = publication.load_frame(paths, state)
    return transitions.Boundary(frame.context, frame.state.revision, state.cycle_index)


def job_owned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    args: MorpionBootstrapArgs,
    claim: MorpionPipelineStageClaim,
    identifier: str,
) -> JobRecord:
    """Capture the original monitored Dataset dispatch clock and immutable claim."""
    from .managed_sessions import KEY as MANAGED
    from .persistent_worker_state import monitored_identity
    from .prepared_dataset_intent import _deadline

    frame = publication.load_frame(paths, state)
    require(args.continual_learning is not None, "Missing finite continuous budgets.")
    assert args.continual_learning is not None
    policy = args.continual_learning
    owner = monitored_identity()
    end, wall_end = _deadline(paths, args, claim, owner)
    clock = publication.put_proof(
        paths, {"owner": owner, "deadline": end, "utc_deadline": wall_end}
    )
    launch = publication.put_proof(paths, state.metadata[MANAGED]["active"])
    budget = publication.put_record(
        paths,
        ControlRecord(
            frame.context,
            Budget(
                launch,
                policy.worker_max_seconds,
                0.0,
                clock,
                policy.max_workspace_mb * 2**20,
                policy.max_scratch_mb * 2**20,
                object_digest(policy.to_metadata()),
            ),
        ),
    )
    return JobRecord(
        frame.context,
        Job(
            identifier,
            "EXTRACT",
            publication.source(paths, state, claim.generation),
            publication.reference(
                paths,
                paths.pipeline_dataset_claim_path_for_generation(claim.generation),
            ),
            launch,
            budget,
            "ADMITTED",
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )


def admit_owned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    args: MorpionBootstrapArgs,
    admission: dict[str, Any],
    claim: MorpionPipelineStageClaim,
) -> tuple[MorpionBootstrapRunState, dict[str, Any]]:
    """Select only the offered source; claims alone never grant extraction authority."""
    require(
        len(storage.entries(state)) < 2, "Continuous Dataset resource_backpressure."
    )
    job = job_owned(paths, state, args, claim, admission["attempt_id"])
    proposed = publication.apply_owned(
        paths, state, boundary(paths, state), transitions.AdmitDataset(job)
    )
    require(
        not isinstance(proposed, transitions.NoWork),
        "Continuous Dataset admission refused.",
    )
    assert not isinstance(proposed, transitions.NoWork)
    return proposed, {
        **admission,
        "schema_version": 2,
        "continuous_admission": record_ref(job).to_metadata(),
    }


def admitted(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    admission: dict[str, Any],
) -> JobRecord:
    """Resolve the original EXTRACT evidence, including on accepted projection retry."""
    frame = publication.load_frame(paths, state)
    job = publication.read_record(
        paths, Ref.from_metadata(admission["continuous_admission"]), frame.context
    )
    require(isinstance(job, JobRecord), "Missing Dataset admission job.")
    assert isinstance(job, JobRecord)
    require(
        job.payload.stage == "EXTRACT"
        and job.payload.id == admission["attempt_id"]
        and job.payload.source.generation == admission["generation"],
        "Dataset admission binding changed.",
    )
    publication.authenticate_source(paths, state, job.payload.source)
    return job


def accept_owned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    admission: dict[str, Any],
    accepted: dict[str, Any],
) -> MorpionBootstrapRunState:
    """Bind PREPARE and accepted rows in the same scientific save."""
    job = admitted(paths, state, admission)
    operational = storage.entries(state).get(accepted["descriptor_sha256"])
    require(operational is not None, "Continuous preparation reservation is required.")
    assert operational is not None
    budget_record = publication.read_record(paths, job.payload.budget, job.context)
    assert isinstance(budget_record, ControlRecord) and isinstance(
        budget_record.payload, Budget
    )
    clock = publication.read_ref(paths, budget_record.payload.clock_evidence)
    # Acceptance may never recapture or extend the original dispatch deadline.
    operational = {
        **operational,
        "deadline": clock["deadline"],
        "utc_deadline": clock["utc_deadline"],
    }
    state = storage.updated(
        state, {**storage.entries(state), accepted["descriptor_sha256"]: operational}
    )
    result = publication.apply_owned(
        paths,
        state,
        boundary(paths, state),
        transitions.BindDataset(job, Accepted.from_metadata(accepted)),
    )
    require(
        not isinstance(result, transitions.NoWork),
        "Continuous Dataset completion suspended.",
    )
    assert not isinstance(result, transitions.NoWork)
    return result


def ready_record(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    operational: dict[str, Any],
    job: JobRecord,
) -> ReadyRecord:
    """Map verified native evidence; no legacy annotation is inferred."""
    from .dataset_publication import accepted_dataset_identity

    identity = accepted_dataset_identity(paths, state, job.payload.source.generation)
    value = storage.ready_descriptor(paths, operational)
    require(
        identity["reference"]
        == operational["accepted"]
        == job.payload.accepted_dataset.to_metadata()
        if job.payload.accepted_dataset is not None
        else False,
        "Ready Dataset identity differs.",
    )
    require(value["context"] == job.context.to_metadata(), "Ready context differs.")
    return ReadyRecord(
        job.context,
        Ready(
            Accepted.from_metadata(operational["accepted"]),
            job.payload.source,
            Ref.from_metadata(operational["rows"]),
            Ref.from_metadata(value["rows"]),
            object_digest(value["recipe"]),
            Ref.from_metadata(value["native"]),
            Ref.from_metadata(value["files"]["split.json"]),
            tuple(
                (name, Ref.from_metadata(value["files"][name]))
                for name in sorted(storage.FILES)
            ),
            job.payload.id,
        ),
    )


def ready_owned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    operational: dict[str, Any],
) -> MorpionBootstrapRunState:
    """Clear the producer and publish Ready only after native ownership has closed."""
    frame = publication.load_frame(paths, state)
    require(
        frame.state.dataset_job is not None, "Missing authoritative preparation job."
    )
    assert frame.state.dataset_job is not None
    job = publication.read_record(paths, frame.state.dataset_job, frame.context)
    require(
        isinstance(job, JobRecord) and job.payload.stage == "PREPARE",
        "Invalid preparation slot.",
    )
    assert isinstance(job, JobRecord)
    ready = ready_record(paths, state, operational, job)
    result = publication.apply_owned(
        paths, state, boundary(paths, state), transitions.PublishReady(job, ready)
    )
    require(
        not isinstance(result, transitions.NoWork),
        "Continuous Ready publication suspended.",
    )
    assert not isinstance(result, transitions.NoWork)
    return result
