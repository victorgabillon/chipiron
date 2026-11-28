"""Continuous receipt IO under the existing Growth owner and single state save.

No scheduler, lock, cache, or second journal lives here. Immutable descriptors are
published before the caller saves the returned run state. An orphan descriptor
or claim file never grants admission. Public policy dispatch is wired separately.
"""

import json
import os
import stat
import time
import uuid
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from . import continuous_transitions as transitions
from .continuous_contract import (
    Budget,
    Context,
    ControlRecord,
    Debt,
    Disposition,
    JobRecord,
    Policy,
    Record,
    Ref,
    Source,
    State,
    decode_record,
    encode_record,
    record_ref,
    require,
)
from .derived.provenance import inside
from .generation_publication_io import (
    file_digest,
    object_digest,
    publication_records,
    read_committed_descriptor,
    read_object,
    sync_directory,
)
from .run_state import load_bootstrap_run_state
from .worker_result_io import publication_record, workspace_epoch

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState

KEY = "continuous_pipeline_v1"
MAX_RECORD_BYTES = 2**20


def cycle(state: MorpionBootstrapRunState) -> int:
    """Translate the native zero-based cycle index; fresh genesis is index -1."""
    require(
        type(state.cycle_index) is int  # pylint: disable=unidiomatic-typecheck
        and state.cycle_index >= -1,
        "Invalid native ordinary-cycle index.",
    )
    return state.cycle_index + 1


def enabled(state: MorpionBootstrapRunState) -> bool:
    """Recognize only explicit context4; absent state is corruption, not genesis."""
    marker = state.metadata.get("continual_learning")
    active = isinstance(marker, dict) and marker.get("schema_version") == 4
    require(active == (KEY in state.metadata), "Continuous context/state mismatch.")
    return active


def context(paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState) -> Context:
    """Authenticate canonical persisted policy/configuration and real workspace."""
    from .config import bootstrap_config_sha256, load_bootstrap_config

    config = load_bootstrap_config(paths.bootstrap_config_path)
    policy = Policy.from_metadata(config.metadata.get("learning_schedule"))
    digest = bootstrap_config_sha256(config)
    require(
        digest == state.metadata.get("bootstrap_config_hash"),
        "Continuous configuration changed.",
    )
    require(
        state.metadata.get("continual_learning")
        == {"schema_version": 4, "learning_schedule_sha256": policy.digest()},
        "Continuous genesis marker changed.",
    )
    return Context(object_digest(workspace_epoch(state)), digest, policy.digest())


def load_frame(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> transitions.Frame:
    """Decode authenticated authority, never reconstruct it from advisory files."""
    require(enabled(state), "Continuous authority is unavailable.")
    return transitions.Frame(
        context(paths, state), State.from_metadata(state.metadata[KEY])
    )


def reference(paths: MorpionBootstrapPaths, path: Path) -> Ref:
    """Bind an existing contained regular file, without creating new authority."""
    relative = paths.relative_to_work_dir(path)
    checked = inside(paths.work_dir, relative)
    require(stat.S_ISREG(checked.lstat().st_mode), "Expected regular evidence file.")
    return Ref(relative, file_digest(checked), checked.stat().st_size)


def read_ref(paths: MorpionBootstrapPaths, ref: Ref) -> dict[str, Any]:
    """Authenticate bounded descriptor bytes and reject symlinks."""
    require(ref.bytes <= MAX_RECORD_BYTES, "Continuous descriptor exceeds bound.")
    path = inside(paths.work_dir, ref.path)
    require(reference(paths, path) == ref, "Continuous evidence bytes changed.")
    return read_object(path)


def read_record(paths: MorpionBootstrapPaths, ref: Ref, expected: Context) -> Record:
    """Resolve a canonical immutable envelope and its workspace binding."""
    record = decode_record(read_ref(paths, ref))
    require(
        record.context == expected and record_ref(record) == ref,
        "Continuous envelope identity changed.",
    )
    return record


def put_record(paths: MorpionBootstrapPaths, record: Record) -> Ref:
    """Fsync private bytes then install without overwriting an existing object."""
    ref = record_ref(record)
    body = encode_record(record)
    require(len(body) <= MAX_RECORD_BYTES, "Continuous descriptor exceeds bound.")
    final = paths.work_dir / ref.path
    require(
        not any(p.is_symlink() for p in (final, *final.parents)),
        "Symlinked continuous publication path.",
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        require(reference(paths, final) == ref, "Conflicting continuous object.")
        return ref
    temporary = final.parent / (".pending-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, final)
        except FileExistsError:
            require(reference(paths, final) == ref, "Conflicting continuous object.")
        sync_directory(final.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return ref


def put_proof(paths: MorpionBootstrapPaths, value: dict[str, Any]) -> Ref:
    """Freeze existing native proof shapes without giving them selection authority."""
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    require(len(body) <= MAX_RECORD_BYTES, "Continuous proof exceeds bound.")
    digest = object_digest(value)
    ref = Ref(f"pipeline/continuous_v1/proofs/{digest}.json", digest, len(body))
    path = paths.work_dir / ref.path
    require(
        not any(p.is_symlink() for p in (path, *path.parents)), "Symlinked proof path."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.parent / (".pending-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            with suppress(FileExistsError):
                os.link(temporary, path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    require(reference(paths, path) == ref, "Conflicting continuous proof.")
    return ref


def prepare_publication_owned(
    paths: MorpionBootstrapPaths,
    previous: MorpionBootstrapRunState | None,
    proposed: MorpionBootstrapRunState,
    activation: dict[str, Any] | None,
    *,
    captured_frame: transitions.Frame | None = None,
) -> tuple[MorpionBootstrapRunState, dict[str, int] | None]:
    """Bind source clocks using owned authority or its captured ordinary frame.

    Only ordinary private preparation may use the captured frame. Acceptance
    rechecks current authority; activation and genesis remain entirely owned.
    """
    require(
        captured_frame is None or (previous is not None and activation is None),
        "Only ordinary Growth may prepare from captured continuous authority.",
    )
    marker = proposed.metadata.get("continual_learning", {})
    if marker.get("schema_version") != 4:
        require(KEY not in proposed.metadata, "Continuous state lacks fresh context4.")
        return proposed, None
    if previous is None:
        require(
            KEY not in proposed.metadata,
            "Cannot adopt previous continuous authority at genesis.",
        )
        expected = context(paths, proposed)
        frame = transitions.initialize(
            expected.workspace_epoch_sha256,
            expected.config_sha256,
            policy=Policy.from_metadata(
                read_object(paths.bootstrap_config_path)["metadata"][
                    "learning_schedule"
                ]
            ),
        )
        proposed = replace(
            proposed, metadata={**proposed.metadata, KEY: frame.state.to_metadata()}
        )
    else:
        frame = captured_frame or load_frame(paths, previous)
        require(
            frame.state.to_metadata() == previous.metadata.get(KEY)
            and frame.context.config_sha256
            == previous.metadata.get("bootstrap_config_hash"),
            "Captured continuous authority differs from its computation source.",
        )
        require(
            proposed.metadata.get(KEY) == previous.metadata.get(KEY),
            "Publication changed continuous authority before fencing.",
        )
        require(
            cycle(proposed) in (cycle(previous), cycle(previous) + 1),
            "Nonsequential continuous cycle.",
        )
        if cycle(proposed) > cycle(previous):
            records = slot_records(paths, frame)
            dataset = (
                records.get(frame.state.dataset_job)
                if frame.state.dataset_job
                else None
            )
            ready = records.get(frame.state.ready) if frame.state.ready else None
            from .continuous_contract import ReadyRecord

            require(
                dataset is None or isinstance(dataset, JobRecord),
                "Invalid Dataset slot.",
            )
            require(
                ready is None or isinstance(ready, ReadyRecord), "Invalid Ready slot."
            )
            assert dataset is None or isinstance(dataset, JobRecord)
            assert ready is None or isinstance(ready, ReadyRecord)
            gate = transitions.growth_wait(
                frame,
                transitions.Boundary(
                    frame.context, frame.state.revision, cycle(previous)
                ),
                dataset=dataset,
                ready=ready,
            )
            require(
                gate is None, "Growth would cross a continuous debt/freshness gate."
            )
        if activation is not None:
            # Deferred runtime import: both modules are initialized before this call.
            from .promotion import resolve_owned  # pylint: disable=cyclic-import

            proposed = resolve_owned(
                paths, proposed, promoted=True, request=activation["request"]
            )
            frame = load_frame(paths, proposed)
    return proposed, {
        "cycle": cycle(proposed),
        "incumbent_epoch": frame.state.incumbent_epoch,
    }


def validate_growth_authority(
    paths: MorpionBootstrapPaths,
    captured: MorpionBootstrapRunState,
    latest: MorpionBootstrapRunState,
) -> None:
    """Authenticate current worker slots without restoring captured scheduling state."""
    from .continuous_contract import ReadyRecord, Retirement

    if not enabled(captured):
        require(not enabled(latest), "Growth continuous protocol changed.")
        return
    before, current = load_frame(paths, captured), load_frame(paths, latest)
    require(
        before.context == current.context
        and before.state.incumbent_epoch == current.state.incumbent_epoch
        and current.state.revision >= before.state.revision
        and current.state.last_accepted_round >= before.state.last_accepted_round
        and current.state.last_consumed_generation
        >= before.state.last_consumed_generation,
        "Growth continuous prerequisite changed or worker authority moved backward.",
    )
    require(
        current.state == before.state or current.state.revision > before.state.revision,
        "Changed continuous authority did not advance its revision.",
    )
    slot_records(paths, current)
    for old, new in zip(
        publication_records(captured.metadata),
        publication_records(latest.metadata),
        strict=True,
    ):
        for value in new.get("continuous_admissions", [])[
            len(old.get("continuous_admissions", [])) :
        ]:
            admission = read_ref(paths, Ref.from_metadata(value))
            job = read_record(
                paths,
                Ref.from_metadata(admission["continuous_admission"]),
                current.context,
            )
            require(
                isinstance(job, JobRecord)
                and job.payload.source == source(paths, latest, new["generation"])
                and admission["generation"] == new["generation"],
                "Growth worker admission belongs to another source.",
            )
        for value in new.get("continuous_dispositions", [])[
            len(old.get("continuous_dispositions", [])) :
        ]:
            record = read_record(paths, Ref.from_metadata(value), current.context)
            require(
                isinstance(record, ControlRecord)
                and isinstance(record.payload, Disposition)
                and disposition_generation(paths, current, record.payload)
                == new["generation"],
                "Growth worker disposition has no authenticated proof for its generation.",
            )
        if "continuous_ready" in new and "continuous_ready" not in old:
            ready = read_record(
                paths, Ref.from_metadata(new["continuous_ready"]), current.context
            )
            require(
                isinstance(ready, ReadyRecord)
                and ready.payload.source == source(paths, latest, new["generation"])
                and ready.payload.accepted_dataset.to_metadata()
                == new.get("accepted_dataset"),
                "Growth Ready proof lost its accepted scientific input.",
            )
        for value in new.get("continuous_consumed", [])[
            len(old.get("continuous_consumed", [])) :
        ]:
            proof = read_ref(paths, Ref.from_metadata(value))
            job = read_record(paths, Ref.from_metadata(proof["job"]), current.context)
            require(
                set(proof) == {"job_id", "job", "ready"}
                and isinstance(job, JobRecord)
                and job.payload.id == proof["job_id"]
                and job.payload.ready == Ref.from_metadata(new["continuous_ready"])
                and proof["ready"] == new["continuous_ready"]
                and job.payload.source == source(paths, latest, new["generation"]),
                "Growth consumed-input proof lost its admitted job or Ready.",
            )
        if "continuous_retirement" in new and "continuous_retirement" not in old:
            retired = read_record(
                paths, Ref.from_metadata(new["continuous_retirement"]), current.context
            )
            require(
                isinstance(retired, ControlRecord)
                and isinstance(retired.payload, Retirement)
                and retired.payload.ready == Ref.from_metadata(new["continuous_ready"]),
                "Growth retirement proof lost its Ready identity.",
            )


def on_publication_owned(
    paths: MorpionBootstrapPaths,
    previous: MorpionBootstrapRunState | None,
    proposed: MorpionBootstrapRunState,
) -> MorpionBootstrapRunState:
    """Join an eligible ordinary offer to the caller's generation publication save."""
    if not enabled(proposed) or previous is None or cycle(proposed) == cycle(previous):
        return proposed
    from .config import load_bootstrap_config
    from .continuous_contract import ReadyRecord

    # Deferred runtime import: both modules are initialized before this call.
    from .learning_policy import policy_from_metadata  # pylint: disable=cyclic-import

    policy = policy_from_metadata(
        load_bootstrap_config(paths.bootstrap_config_path).metadata
    )
    require(policy is not None, "Missing continuous learning policy.")
    assert policy is not None
    if (
        proposed.generation <= 1
        or proposed.generation % policy.dataset_every_generations
    ):
        return proposed
    frame = load_frame(paths, proposed)
    records = slot_records(paths, frame)
    dataset = records.get(frame.state.dataset_job) if frame.state.dataset_job else None
    ready = records.get(frame.state.ready) if frame.state.ready else None
    assert dataset is None or isinstance(dataset, JobRecord)
    assert ready is None or isinstance(ready, ReadyRecord)
    change = transitions.offer_source(
        frame,
        transitions.Boundary(frame.context, frame.state.revision, cycle(proposed)),
        source(paths, proposed, proposed.generation),
        dataset=dataset,
        ready=ready,
    )
    if isinstance(change, transitions.NoWork):
        return proposed
    result = replace(
        proposed, metadata={**proposed.metadata, KEY: change.frame.state.to_metadata()}
    )
    if change.displacement is not None:
        d = change.displacement
        result = attach_disposition(
            paths, result, Disposition(d.subject, d.reason, d.successor, d.debt)
        )
    return result


def source(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, generation: int
) -> Source:
    """Resolve authenticated fresh annotations; never zero-fill historical data."""
    record = publication_record(state, generation)
    annotation = record.get("continuous_source")
    require(
        isinstance(annotation, dict)
        and set(annotation) == {"cycle", "incumbent_epoch"},
        "Missing continuous source annotation.",
    )
    assert isinstance(annotation, dict)
    result = Source(
        record["id"],
        generation,
        record["descriptor_sha256"],
        annotation["cycle"],
        annotation["incumbent_epoch"],
    )
    authenticate_source(paths, state, result)
    return result


def authenticate_source(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, value: Source
) -> dict[str, Any]:
    """Bind source clocks to the actual immutable scientific publication seal."""
    record = publication_record(state, value.generation)
    annotation = {"cycle": value.cycle, "incumbent_epoch": value.incumbent_epoch}
    require(
        (record["id"], record["descriptor_sha256"], record.get("continuous_source"))
        == (value.publication_id, value.descriptor_sha256, annotation),
        "Continuous source identity differs.",
    )
    descriptor = read_committed_descriptor(paths.work_dir, record)
    require(
        descriptor.get("continuous_source") == annotation
        and descriptor.get("scientific_identity") == record["scientific_identity"]
        and descriptor["bootstrap_config_hash"]
        == state.metadata["bootstrap_config_hash"],
        "Continuous source seal differs.",
    )
    frame = load_frame(paths, state)
    require(
        value.cycle <= cycle(state)
        and value.incumbent_epoch <= frame.state.incumbent_epoch,
        "Continuous source is from the future.",
    )
    return record


def slot_records(
    paths: MorpionBootstrapPaths, frame: transitions.Frame
) -> dict[Ref, Record]:
    """Resolve the bounded live slot closure, excluding historical back-links."""
    from .continuous_contract import Reservation

    result: dict[Ref, Record] = {}
    pending = [
        r
        for r in (
            frame.state.dataset_job,
            frame.state.ready,
            frame.state.cohort,
            frame.state.reserved_promotion,
            frame.state.pending_promotion,
        )
        if r is not None
    ]
    while pending:
        ref = pending.pop()
        if ref in result:
            continue
        require(len(result) < 20, "Continuous live closure exceeds slot bound.")
        record = read_record(paths, ref, frame.context)
        result[ref] = record
        if isinstance(record, JobRecord) and record.payload.ready is not None:
            pending.append(record.payload.ready)
        if isinstance(record, ControlRecord) and isinstance(
            record.payload, Reservation
        ):
            value = record.payload
            pending.extend(
                r
                for r in (
                    value.opportunity,
                    value.population,
                    value.comparison_job,
                    value.comparison,
                )
                if r is not None
            )
    return result


def prepared_roots(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> frozenset[str]:
    """Keep every admitted reader and pending/reserved comparison population live."""
    from .continuous_contract import Opportunity, Population, ReadyRecord

    frame = load_frame(paths, state)
    roots: set[str] = set()
    for record in slot_records(paths, frame).values():
        payload = record.payload
        if isinstance(record, (JobRecord, ReadyRecord)):
            accepted = record.payload.accepted_dataset
            if accepted is not None:
                roots.add(accepted.descriptor_sha256)
        elif isinstance(payload, (Opportunity, Population)):
            roots.add(payload.accepted_dataset.descriptor_sha256)
    return frozenset(roots)


def dispatch_generation(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, *, training: bool
) -> int | None:
    """Project only eligible authoritative work into existing worker discovery."""
    from .continuous_contract import ReadyRecord
    from .prepared_dataset_storage import entries

    frame = load_frame(paths, state)
    if not frame.state.admission_open:
        return None
    records = slot_records(paths, frame)
    candidate: Source | None
    active = frame.state.cohort if training else frame.state.dataset_job
    if active is not None:
        job = records[active]
        assert isinstance(job, JobRecord)
        return job.payload.source.generation
    if training:
        if (
            frame.state.cohort is not None
            or transitions.comparison_owed(frame, records)
            or frame.state.ready is None
        ):
            return None
        ready = records[frame.state.ready]
        assert isinstance(ready, ReadyRecord)
        candidate = ready.payload.source
    else:
        if frame.state.dataset_job is not None or len(entries(state)) == 2:
            return None
        candidate = frame.state.pending_source
    if candidate is None:
        return None
    policy = Policy()
    if (
        cycle(state) - candidate.cycle > policy.max_source_age_cycles
        or frame.state.incumbent_epoch - candidate.incumbent_epoch
        > policy.max_teacher_epoch_lag
    ):
        return None
    authenticate_source(paths, state, candidate)
    return candidate.generation


def authenticate_job(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, job: JobRecord
) -> None:
    """Require live native claim/session and the original unexpired clock proof."""
    from .managed_sessions import KEY as MANAGED
    from .managed_sessions import require_managed_execution
    from .persistent_worker_state import monitored_identity

    frame = load_frame(paths, state)
    require(
        job.context == frame.context and job.payload.status == "ADMITTED",
        "Inactive or foreign job.",
    )
    authenticate_source(paths, state, job.payload.source)
    require_managed_execution(paths, state)
    expected_claim = (
        paths.pipeline_dataset_claim_path_for_generation(job.payload.source.generation)
        if job.payload.stage in {"EXTRACT", "PREPARE"}
        else paths.pipeline_training_claim_path_for_generation(
            job.payload.source.generation
        )
    )
    claim = read_ref(paths, job.payload.claim)
    require(read_object(expected_claim) == claim, "Job claim was replaced or removed.")
    owner = monitored_identity()
    require(
        claim.get("generation") == job.payload.source.generation
        and claim.get("stage")
        == ("dataset" if job.payload.stage in {"EXTRACT", "PREPARE"} else "training")
        and claim.get("metadata", {}).get("learning_worker") == owner,
        "Job claim/worker differs.",
    )
    require(
        datetime.fromisoformat(claim["expires_at_utc"]).timestamp() > time.time(),
        "Job claim expired.",
    )
    launch = read_ref(paths, job.payload.managed_launch)
    require(launch == state.metadata[MANAGED]["active"], "Job managed launch changed.")
    record = read_record(paths, job.payload.budget, frame.context)
    require(
        isinstance(record, ControlRecord) and isinstance(record.payload, Budget),
        "Missing typed job budget.",
    )
    assert isinstance(record, ControlRecord) and isinstance(record.payload, Budget)
    budget = record.payload
    from .config import load_bootstrap_config

    # Deferred runtime import: both modules are initialized before this call.
    from .learning_policy import policy_from_metadata  # pylint: disable=cyclic-import

    policy = policy_from_metadata(
        load_bootstrap_config(paths.bootstrap_config_path).metadata
    )
    require(policy is not None, "Missing continuous limits.")
    assert policy is not None
    require(
        budget.launch == job.payload.managed_launch
        and budget.consumed_job_seconds < budget.max_job_seconds
        and budget.max_job_seconds == policy.worker_max_seconds
        and budget.limits_sha256 == object_digest(policy.to_metadata())
        and budget.reserved_output_bytes <= policy.max_workspace_mb * 2**20
        and budget.reserved_scratch_bytes <= policy.max_scratch_mb * 2**20,
        "Job allowance exhausted or retargeted.",
    )
    clock = read_ref(paths, budget.clock_evidence)
    # Deferred runtime import: both modules are initialized before this call.
    from .prepared_dataset import _remaining  # pylint: disable=cyclic-import

    _remaining(clock)
    require(
        clock["owner"]["boot"] == owner["boot"], "Original job clock predates boot."
    )
    from .pipeline_artifacts import load_pipeline_stage_claim
    from .prepared_dataset_intent import original_deadline

    original, _ = original_deadline(
        paths,
        policy.worker_max_seconds,
        load_pipeline_stage_claim(expected_claim),
        owner,
    )
    require(
        clock["deadline"] <= original + 0.1,
        "Job clock exceeds original dispatch or claim.",
    )


def apply_owned(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    boundary: transitions.Boundary,
    action: transitions.Transition,
) -> MorpionBootstrapRunState | transitions.NoWork:
    """Authenticate and propose one change; caller owns Growth and saves once.

    A proposed state may already contain the related accepted Dataset/Training
    reference. Its continuous revision must still equal the persisted revision.
    """
    persisted = load_bootstrap_run_state(paths.run_state_path)
    frame = load_frame(paths, state)
    require(
        persisted.metadata.get(KEY) == state.metadata.get(KEY),
        "Continuous state changed before owned apply.",
    )
    require(
        boundary.cycle == cycle(persisted) == cycle(state),
        "Unauthenticated current cycle.",
    )
    if isinstance(action, transitions.Offer):
        authenticate_source(paths, state, action.source)
        if action.sealed_head is not None:
            require(
                action.sealed_head == source(paths, state, state.generation),
                "Repair source is not current sealed head.",
            )
    if isinstance(
        action,
        (
            transitions.AdmitDataset,
            transitions.BindDataset,
            transitions.PublishReady,
            transitions.AdmitCohort,
            transitions.AcceptCohort,
            transitions.StartComparison,
        ),
    ):
        authenticate_job(paths, state, action.job)
    if isinstance(
        action,
        (
            transitions.AdmitDataset,
            transitions.AdmitCohort,
            transitions.StartComparison,
        ),
    ):
        # Deferred runtime import: both modules are initialized before this call.
        from .continuous_resources import authenticate_reservation  # pylint: disable=cyclic-import

        authenticate_reservation(paths, state, action.job)
    if isinstance(action, (transitions.BindDataset, transitions.AcceptCohort)):
        record = publication_record(state, action.job.payload.source.generation)
        key = (
            "accepted_dataset"
            if isinstance(action, transitions.BindDataset)
            else "accepted_training"
        )
        require(
            record.get(key) == action.accepted.to_metadata(),
            "Transition lacks joined scientific acceptance.",
        )
    # Deferred runtime import: both modules are initialized before this call.
    from .continuous_lifecycle import authenticate  # pylint: disable=cyclic-import

    authenticate(paths, state, action)
    if isinstance(action, transitions.AcceptCohort):
        require(
            action.remaining_debt == next_learning_debt(paths, state),
            "Cohort acceptance moved outstanding learning debt.",
        )
    records = slot_records(paths, frame)
    if isinstance(action, transitions.VerifyComparison):
        verified_reservation = action.reservation.payload
        from .continuous_contract import Reservation

        assert (
            isinstance(verified_reservation, Reservation)
            and verified_reservation.comparison_job is not None
        )
        job = records[verified_reservation.comparison_job]
        assert isinstance(job, JobRecord)
        authenticate_job(paths, state, job)
    if isinstance(action, transitions.BeginCutover):
        # Deferred runtime import: both modules are initialized before this call.
        from .promotion import request_bindings  # pylint: disable=cyclic-import

        request = read_ref(paths, action.request)
        require(
            all(
                request.get(k) == v
                for k, v in request_bindings(
                    paths, state, request["training_generation"]
                ).items()
            ),
            "Cutover request lost reservation proof.",
        )
    result = transitions.transition(frame, boundary, action, records)
    if isinstance(result, transitions.NoWork):
        return result
    for new_record in result.records:
        put_record(paths, new_record)
    updated = replace(
        state, metadata={**state.metadata, KEY: result.frame.state.to_metadata()}
    )
    if isinstance(action, transitions.PublishReady):
        from .generation_publication_io import PUBLICATION_KEY

        data = updated.metadata[PUBLICATION_KEY]
        generations = [
            {**r, "continuous_ready": record_ref(action.ready).to_metadata()}
            if r["generation"] == action.job.payload.source.generation
            else r
            for r in data["generations"]
        ]
        updated = replace(
            updated,
            metadata={
                **updated.metadata,
                PUBLICATION_KEY: {**data, "generations": generations},
            },
        )
    if result.displacement is not None:
        value = result.displacement
        updated = attach_disposition(
            paths,
            updated,
            Disposition(value.subject, value.reason, value.successor, value.debt),
        )
    return updated


def disposition_generation(
    paths: MorpionBootstrapPaths, frame: transitions.Frame, disposition: Disposition
) -> int:
    """Resolve the source once for both attachment and acceptance validation."""
    from .continuous_contract import Opportunity, ReadyRecord

    subject = disposition.subject
    if isinstance(subject, Source):
        generation = subject.generation
    else:
        record = read_record(paths, subject, frame.context)
        if isinstance(record, (JobRecord, ReadyRecord)):
            generation = record.payload.source.generation
        else:
            require(
                isinstance(record.payload, Opportunity),
                "Disposition subject has no source.",
            )
            assert isinstance(record.payload, Opportunity)
            generation = record.payload.generation
    return generation


def attach_disposition(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    disposition: Disposition,
) -> MorpionBootstrapRunState:
    """Attach mandatory proof to its source record, in the same owned save."""
    from .generation_publication_io import PUBLICATION_KEY

    frame = load_frame(paths, state)
    generation = disposition_generation(paths, frame, disposition)
    ref = put_record(paths, ControlRecord(frame.context, disposition)).to_metadata()
    publication = state.metadata[PUBLICATION_KEY]
    records = []
    for old in publication["generations"]:
        if old["generation"] == generation:
            refs = old.get("continuous_dispositions", [])
            old = {
                **old,
                "continuous_dispositions": refs if ref in refs else [*refs, ref],
            }
        records.append(old)
    return replace(
        state,
        metadata={
            **state.metadata,
            PUBLICATION_KEY: {**publication, "generations": records},
        },
    )


def next_learning_debt(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> Debt | None:
    """Roll acceptance to the oldest outstanding offer, including coalesced ancestors."""
    from .continuous_contract import ReadyRecord

    frame = load_frame(paths, state)
    sources = [frame.state.pending_source] if frame.state.pending_source else []
    records = slot_records(paths, frame)
    for ref in (frame.state.dataset_job, frame.state.ready):
        if ref is not None:
            record = records[ref]
            assert isinstance(record, (JobRecord, ReadyRecord))
            sources.append(record.payload.source)
    if not sources:
        return None
    ancestors = set(sources)
    # Receipts are ordered by source generation; reverse order follows a chain
    # from its newest still-outstanding successor back to the original offer.
    for record in reversed(publication_records(state.metadata)):
        for raw in reversed(record.get("continuous_dispositions", [])):
            envelope = read_record(paths, Ref.from_metadata(raw), frame.context)
            assert isinstance(envelope, ControlRecord) and isinstance(
                envelope.payload, Disposition
            )
            value = envelope.payload
            if (
                isinstance(value.subject, Source)
                and isinstance(value.successor, Source)
                and value.successor in ancestors
                and value.subject.generation > frame.state.last_consumed_generation
            ):
                ancestors.add(value.subject)
    oldest = min(ancestors, key=lambda s: (s.cycle, s.generation))
    return Debt(oldest.publication_id, oldest.cycle)


def growth_reason(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> str | None:
    """Resolve actual admitted input age as well as both durable debt clocks."""
    from .continuous_contract import ReadyRecord

    frame = load_frame(paths, state)
    records = slot_records(paths, frame)
    dataset = records.get(frame.state.dataset_job) if frame.state.dataset_job else None
    ready = records.get(frame.state.ready) if frame.state.ready else None
    assert dataset is None or isinstance(dataset, JobRecord)
    assert ready is None or isinstance(ready, ReadyRecord)
    refusal = transitions.growth_wait(
        frame,
        transitions.Boundary(frame.context, frame.state.revision, cycle(state)),
        dataset=dataset,
        ready=ready,
    )
    return None if refusal is None else refusal.reason


def reconcile_freshness(paths: MorpionBootstrapPaths) -> None:
    """Dispose only ineligible unclaimed inputs; repair from the sealed frozen head."""
    from .continuous_contract import ReadyRecord

    # Deferred runtime import: both modules are initialized before this call.
    from .continuous_dataset import boundary  # pylint: disable=cyclic-import
    from .generation_publication_io import growth_receipt_ownership
    from .run_state import save_bootstrap_run_state

    with growth_receipt_ownership(paths, wait=True):
        state = load_bootstrap_run_state(paths.run_state_path)
        if not enabled(state):
            return
        frame = load_frame(paths, state)
        if not frame.state.admission_open:
            return
        if frame.state.ready is not None:
            ready = read_record(paths, frame.state.ready, frame.context)
            assert isinstance(ready, ReadyRecord)
            value = ready.payload.source
            if (
                value.generation <= frame.state.last_consumed_generation
                or cycle(state) - value.cycle > 8
                or frame.state.incumbent_epoch - value.incumbent_epoch > 1
            ):
                updated = apply_owned(
                    paths, state, boundary(paths, state), transitions.DropReady(ready)
                )
                assert not isinstance(updated, transitions.NoWork)
                state = attach_disposition(
                    paths,
                    updated,
                    Disposition(
                        record_ref(ready),
                        "ineligible_for_training",
                        None,
                        frame.state.learning_debt,
                    ),
                )
                save_bootstrap_run_state(state, paths.run_state_path, durable=True)
        frame = load_frame(paths, state)
        if (
            growth_reason(paths, state) is not None
            and frame.state.dataset_job is None
            and frame.state.ready is None
        ):
            head = source(paths, state, state.generation)
            updated = apply_owned(
                paths, state, boundary(paths, state), transitions.Offer(head, head)
            )
            if not isinstance(updated, transitions.NoWork):
                save_bootstrap_run_state(updated, paths.run_state_path, durable=True)
