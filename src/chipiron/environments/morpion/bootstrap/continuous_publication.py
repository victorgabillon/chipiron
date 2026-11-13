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
) -> tuple[MorpionBootstrapRunState, dict[str, int] | None]:
    """Bind new source clocks before the scientific descriptor is sealed."""
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
            expected.workspace_epoch_sha256, expected.config_sha256
        )
        proposed = replace(
            proposed, metadata={**proposed.metadata, KEY: frame.state.to_metadata()}
        )
    else:
        frame = load_frame(paths, previous)
        require(
            proposed.metadata.get(KEY) == previous.metadata.get(KEY),
            "Publication changed continuous authority before fencing.",
        )
        require(
            proposed.cycle_index in (previous.cycle_index, previous.cycle_index + 1),
            "Nonsequential continuous cycle.",
        )
        if proposed.cycle_index > previous.cycle_index:
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
                    frame.context, frame.state.revision, previous.cycle_index
                ),
                dataset=dataset,
                ready=ready,
            )
            require(
                gate is None, "Growth would cross a continuous debt/freshness gate."
            )
        if activation is not None:
            # Cold activation owns resolution; enabling must wire the verified
            # reservation into the same proposed state before this hook.
            require(False, "Continuous activation resolution is not wired.")
    return proposed, {
        "cycle": proposed.cycle_index,
        "incumbent_epoch": frame.state.incumbent_epoch,
    }


def on_publication_owned(
    paths: MorpionBootstrapPaths,
    previous: MorpionBootstrapRunState | None,
    proposed: MorpionBootstrapRunState,
) -> MorpionBootstrapRunState:
    """Join an eligible ordinary offer to the caller's generation publication save."""
    if (
        not enabled(proposed)
        or previous is None
        or proposed.cycle_index == previous.cycle_index
    ):
        return proposed
    from .config import load_bootstrap_config
    from .continuous_contract import ReadyRecord
    from .learning_policy import policy_from_metadata

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
        transitions.Boundary(frame.context, frame.state.revision, proposed.cycle_index),
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
        value.cycle <= state.cycle_index
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
        else paths.pipeline_training_claim_path_for_generation(job.payload.source.generation)
    )
    require(job.payload.claim.path == paths.relative_to_work_dir(expected_claim), "Job claim path differs.")
    claim = read_ref(paths, job.payload.claim)
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
    from .learning_policy import policy_from_metadata

    policy = policy_from_metadata(load_bootstrap_config(paths.bootstrap_config_path).metadata)
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
    from .prepared_dataset import _remaining

    _remaining(clock)
    require(
        clock["owner"]["boot"] == owner["boot"], "Original job clock predates boot."
    )
    from .pipeline_artifacts import load_pipeline_stage_claim
    from .prepared_dataset_intent import original_deadline

    original, _ = original_deadline(paths, policy.worker_max_seconds, load_pipeline_stage_claim(expected_claim), owner)
    require(clock["deadline"] <= original + 0.1, "Job clock exceeds original dispatch or claim.")


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
        boundary.cycle == persisted.cycle_index == state.cycle_index,
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
    records = slot_records(paths, frame)
    result = transitions.transition(frame, boundary, action, records)
    if isinstance(result, transitions.NoWork):
        return result
    for new_record in result.records:
        put_record(paths, new_record)
    updated = replace(
        state, metadata={**state.metadata, KEY: result.frame.state.to_metadata()}
    )
    if result.displacement is not None:
        value = result.displacement
        updated = attach_disposition(
            paths,
            updated,
            Disposition(value.subject, value.reason, value.successor, value.debt),
        )
    return updated


def attach_disposition(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    disposition: Disposition,
) -> MorpionBootstrapRunState:
    """Attach mandatory proof to its source record, in the same owned save."""
    from .continuous_contract import Opportunity, ReadyRecord
    from .generation_publication_io import PUBLICATION_KEY

    frame = load_frame(paths, state)
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
