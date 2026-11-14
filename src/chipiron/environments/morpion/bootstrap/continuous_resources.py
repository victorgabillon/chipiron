"""Deduplicated byte reservations under the existing finite production limits."""

import shutil
from typing import TYPE_CHECKING, Any

from . import continuous_publication as publication
from . import prepared_dataset_storage as storage
from .continuous_contract import Budget, ControlRecord, JobRecord, Reservation, require
from .pipeline_memory import directory_disk_usage

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_paths import MorpionBootstrapPaths
    from .learning_policy import MorpionContinualLearningPolicy
    from .run_state import MorpionBootstrapRunState

COMMIT_HEADROOM = 2**20


def logical_bytes(path: Path) -> int:
    """Count existing payloads by stat only, including unaccepted private history."""
    return int(directory_disk_usage(path)["logical_bytes"]) if path.exists() else 0


def outstanding(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> tuple[int, int]:
    """Count each live job and operational Dataset once, minus materialized bytes."""
    frame = publication.load_frame(paths, state)
    records = publication.slot_records(paths, frame)
    refs = {frame.state.dataset_job, frame.state.cohort}
    if frame.state.reserved_promotion is not None:
        reserved = records[frame.state.reserved_promotion]
        assert isinstance(reserved.payload, Reservation)
        if reserved.payload.phase == "COMPARING":
            refs.add(reserved.payload.comparison_job)
    output = 0
    scratch = sum(
        max(0, r["reserved_bytes"] - logical_bytes(storage.directory(paths, key)))
        for key, r in storage.entries(state).items()
    )
    for ref in refs:
        if ref is None:
            continue
        job = records[ref]
        assert isinstance(job, JobRecord)
        control = publication.read_record(paths, job.payload.budget, frame.context)
        assert isinstance(control, ControlRecord) and isinstance(
            control.payload, Budget
        )
        budget = control.payload
        from .continuous_lifecycle import previous_admission

        if job.payload.stage in {"TRAIN", "EXTRACT"}:
            admission = previous_admission(paths, state, job)
            namespace = (
                "training_publications_v1"
                if job.payload.stage == "TRAIN"
                else "dataset_publications_v1"
            )
            materialized = sum(
                logical_bytes(
                    paths.work_dir / namespace / area / admission["attempt_id"]
                )
                for area in ("staging", "objects")
            )
            output += max(0, budget.reserved_output_bytes - materialized)
        elif job.payload.stage == "COMPARE":
            output += budget.reserved_output_bytes
        if job.payload.stage == "EXTRACT":
            scratch += budget.reserved_scratch_bytes
    return output, scratch


def reserve(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    policy: MorpionContinualLearningPolicy,
    *,
    dataset: bool,
    comparison: bool = False,
) -> tuple[int, int]:
    """Reserve remaining completion capacity before an authoritative admission."""
    from pathlib import Path

    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        _Budget,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation_reuse import (
        _limit,
        requested,
    )

    from .candidate_resources import available_memory_mb

    measured = logical_bytes(paths.work_dir)
    output, scratch = outstanding(paths, state)
    limits = _Budget.configured()
    prior = _limit() if requested() else 0
    # Native reservations include old/new coexistence and stable rows. COMPLETE
    # roots shrink to their measured inventory plus commit overhead in J4.1.
    prepared_reserved = sum(
        logical_bytes(storage.directory(paths, key)) for key in storage.entries(state)
    )
    scratch_cap = min(limits.scratch_bytes, policy.max_scratch_mb * 2**20)
    new_scratch = (
        max(0, scratch_cap - prior - prepared_reserved - scratch) if dataset else 0
    )
    free = (
        policy.max_workspace_mb * 2**20
        - measured
        - output
        - scratch
        - new_scratch
        - prior
        - limits.headroom_bytes
    )
    # Keep capacity for the other serial producer; this divides the existing cap,
    # not a change to the scientific recipe or the run-wide resource allowance.
    new_output = min(COMMIT_HEADROOM, free) if comparison else free // 2
    require(
        new_output > COMMIT_HEADROOM // 2
        and (not dataset or new_scratch > 2 * limits.chunk_bytes),
        "Continuous resource_backpressure: completion reservation cannot fit.",
    )
    require(
        shutil.disk_usage(paths.work_dir).free
        >= policy.min_free_disk_mb * 2**20
        + output
        + scratch
        + new_output
        + new_scratch
        + COMMIT_HEADROOM,
        "Continuous resource_backpressure: disk floor.",
    )
    require(
        available_memory_mb() >= policy.min_available_ram_mb,
        "Continuous resource_backpressure: RAM floor.",
    )
    require(
        not paths.work_dir.is_symlink() and isinstance(paths.work_dir, Path),
        "Unsafe reservation workspace.",
    )
    return new_output, new_scratch


def output_remaining(
    paths: MorpionBootstrapPaths, admission: dict[str, Any]
) -> int | None:
    """Retain the original output cap across retries and permanent attempt history."""
    from .continuous_contract import Ref
    from .generation_publication_io import publication_records
    from .run_state import load_bootstrap_run_state

    if "continuous_admission" not in admission:
        return None
    state = load_bootstrap_run_state(paths.run_state_path)
    frame = publication.load_frame(paths, state)
    job = publication.read_record(
        paths, Ref.from_metadata(admission["continuous_admission"]), frame.context
    )
    assert isinstance(job, JobRecord)
    control = publication.read_record(paths, job.payload.budget, frame.context)
    assert isinstance(control, ControlRecord) and isinstance(control.payload, Budget)
    namespace = (
        "training_publications_v1"
        if job.payload.stage == "TRAIN"
        else "dataset_publications_v1"
    )
    spent = 0
    for source in publication_records(state.metadata):
        if source["generation"] != job.payload.source.generation:
            continue
        for ref in source.get("continuous_admissions", []):
            previous = publication.read_ref(paths, Ref.from_metadata(ref))
            previous_job = publication.read_record(
                paths,
                Ref.from_metadata(previous["continuous_admission"]),
                frame.context,
            )
            assert isinstance(previous_job, JobRecord)
            if previous_job.payload.id == job.payload.id:
                spent += sum(
                    logical_bytes(
                        paths.work_dir / namespace / area / previous["attempt_id"]
                    )
                    for area in ("objects", "staging")
                )
    result = control.payload.reserved_output_bytes - spent - COMMIT_HEADROOM
    require(
        result > 0,
        "Continuous resource_backpressure: original output reservation exhausted.",
    )
    return result


def reserve_recovery(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    policy: MorpionContinualLearningPolicy,
    key: str,
) -> MorpionBootstrapRunState:
    """Reacquire bounded rebuild scratch before touching an unconsumed cache.

    COMPLETE roots retain only their compact inventory reservation. Rebuilding
    needs native staging again, bounded by the original producer allowance and
    the remaining aggregate capacity; it never renews the consumer's clock.
    Caller owns the artifact lease and Growth publication lock.
    """
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        _Budget,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation_reuse import (
        _limit,
        requested,
    )

    from .continuous_contract import Ref
    from .dataset_publication import _proposal

    data = storage.entries(state)
    record = data[key]
    admission = _proposal(paths, record["accepted"])["admission"]
    context = publication.context(paths, state)
    producer = publication.read_record(
        paths, Ref.from_metadata(admission["continuous_admission"]), context
    )
    assert isinstance(producer, JobRecord)
    original = publication.read_record(paths, producer.payload.budget, context)
    assert isinstance(original, ControlRecord) and isinstance(original.payload, Budget)
    limits = _Budget.configured()
    prior = _limit() if requested() else 0
    materialized = logical_bytes(storage.directory(paths, key))
    output, scratch = outstanding(paths, state)
    other_scratch = scratch - max(0, record["reserved_bytes"] - materialized)
    other_roots = sum(
        logical_bytes(storage.directory(paths, other)) for other in data if other != key
    )
    reserved = min(
        original.payload.reserved_scratch_bytes,
        min(limits.scratch_bytes, policy.max_scratch_mb * 2**20)
        - prior
        - other_roots
        - other_scratch,
        policy.max_workspace_mb * 2**20
        - logical_bytes(paths.work_dir)
        + materialized
        - output
        - other_scratch
        - prior
        - limits.headroom_bytes,
        shutil.disk_usage(paths.work_dir).free
        + materialized
        - output
        - other_scratch
        - prior
        - max(limits.headroom_bytes, policy.min_free_disk_mb * 2**20),
    )
    require(
        reserved > record["rows"]["bytes"] + 2 * limits.chunk_bytes + COMMIT_HEADROOM,
        "Continuous resource_backpressure: same-Dataset recovery cannot fit.",
    )
    data[key] = {**record, "reserved_bytes": reserved}
    return storage.updated(state, data)


def authenticate_reservation(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, job: JobRecord
) -> None:
    """Reject overcommitted caller-supplied budgets at the authoritative admission."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        _Budget,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation_reuse import (
        _limit,
        requested,
    )

    from .config import load_bootstrap_config
    from .learning_policy import policy_from_metadata

    policy = policy_from_metadata(
        load_bootstrap_config(paths.bootstrap_config_path).metadata
    )
    assert policy is not None
    control = publication.read_record(paths, job.payload.budget, job.context)
    assert isinstance(control, ControlRecord) and isinstance(control.payload, Budget)
    budget = control.payload
    output, scratch = outstanding(paths, state)
    prior = _limit() if requested() else 0
    total = (
        logical_bytes(paths.work_dir)
        + output
        + scratch
        + prior
        + budget.reserved_output_bytes
        + budget.reserved_scratch_bytes
        + _Budget.configured().headroom_bytes
    )
    require(
        total <= policy.max_workspace_mb * 2**20,
        "Continuous resource_backpressure: aggregate workspace reservation.",
    )
    require(
        shutil.disk_usage(paths.work_dir).free
        >= policy.min_free_disk_mb * 2**20
        + output
        + scratch
        + budget.reserved_output_bytes
        + budget.reserved_scratch_bytes,
        "Continuous resource_backpressure: aggregate disk reservation.",
    )
