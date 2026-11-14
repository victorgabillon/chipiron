"""Dataset-worker preparation and admitted Training borrowing of exact I5 inputs.

Acceptance stays in Dataset/Training publication. This bounded operational cache
has no source-selection or candidate-parent authority. All bulk IO and artifact
waits occur outside Growth ownership; owned paths never wait for artifact leases.
"""

import logging
import math
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from . import prepared_dataset_storage as storage
from .derived.provenance import atomic_json, inside
from .generation_publication_io import (
    copy_artifact,
    file_digest,
    generation_read_pin,
    growth_receipt_ownership,
    require,
    sync_artifact,
)
from .learning_worker import require_quiescent
from .managed_sessions import require_managed_execution
from .persistent_worker_state import monitored_identity
from .prepared_dataset_recipe import dataset_recipe
from .run_state import load_bootstrap_run_state, save_bootstrap_run_state
from .worker_result_io import publication_record

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        _PreparedInputs,
    )

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .continuous_contract import JobRecord, ReadyRecord
    from .run_state import MorpionBootstrapRunState


def _remaining(record: dict[str, Any]) -> float:
    """Reject expired/rebooted or inconsistent clocks rather than extending a job."""
    current = monitored_identity()
    require(
        record["owner"]["boot"] == current["boot"],
        "Preparation clock predates this boot",
    )
    end, wall = record["deadline"], record["utc_deadline"]
    require(
        all(
            isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (end, wall)
        ),
        "Invalid preparation deadline",
    )
    left = end - time.monotonic()
    require(
        abs(left - (wall - time.time())) < 5 and left > 0,
        "Preparation allowance exhausted or clock changed",
    )
    return float(end)


def _checked(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    key: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Fence every owned transition against admitted source, configuration and token."""
    require_managed_execution(paths, state)
    record = storage.entries(state).get(key)
    require(record == expected, "Prepared Dataset ownership replaced")
    assert record is not None
    scientific = publication_record(state, record["generation"])
    require(
        scientific.get("accepted_dataset") == record["accepted"]
        and storage.context(paths, state) == record["context"],
        "Prepared Dataset authority differs",
    )
    claim_path = paths.pipeline_dataset_claim_path_for_generation(record["generation"])
    if claim_path.exists():
        from .pipeline_artifacts import load_pipeline_stage_claim

        require(
            load_pipeline_stage_claim(claim_path).claim_id
            == record["dataset_claim_id"],
            "Dataset claim replaced during preparation",
        )
    return record


def _save(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    key: str,
    record: dict[str, Any],
    ready: ReadyRecord | None = None,
) -> None:
    data = storage.entries(state)
    data[key] = record
    proposed = storage.updated(state, data)
    from .continuous_publication import enabled as continuous_enabled

    if (
        continuous_enabled(state)
        and record["phase"] == "COMPLETE"
        and storage.entries(state)[key]["phase"] == "PREPARING"
    ):
        from .continuous_dataset import ready_owned

        require(ready is not None, "Missing verified Ready publication.")
        assert ready is not None
        proposed = ready_owned(paths, proposed, record, ready)
    save_bootstrap_run_state(proposed, paths.run_state_path, durable=True)


def prepare_accepted(
    paths: MorpionBootstrapPaths, args: MorpionBootstrapArgs, generation: int
) -> None:
    """Complete or recover only the already accepted Dataset's recorded intent."""
    if os.environ.get("CHIPIRON_SHARED_PREPARATION", "0") == "0":
        return
    from .continuous_lifecycle import recover_preparation

    recover_preparation(paths)
    state = load_bootstrap_run_state(paths.run_state_path)
    selected = [
        r for r in storage.entries(state).values() if r["generation"] == generation
    ]
    if not selected or selected[0]["phase"] in {"COMPLETE", "CONSUMING", "RETIRED"}:
        return
    record: dict[str, Any] = selected[0]
    key = record["accepted"]["descriptor_sha256"]
    recipe = dataset_recipe(args, storage.directory(paths, key) / "rows.jsonl")
    require(
        recipe is not None and recipe.identity() == record["recipe"],
        "Preparation recipe changed",
    )
    assert recipe is not None
    deadline = _remaining(record)
    scientific = publication_record(state, generation)
    with (
        generation_read_pin(paths.work_dir / scientific["tree_snapshot_path"]),
        storage.lease(paths, key, deadline=deadline) as root,
    ):
        with growth_receipt_ownership(paths, wait=True):
            state = load_bootstrap_run_state(paths.run_state_path)
            _checked(paths, state, key, record)
            current = monitored_identity()
            if current != record["owner"]:
                require_quiescent(record["owner"], current)
            record = {
                **record,
                "owner": current,
                "token": uuid.uuid4().hex,
                "phase": "PREPARING",
            }
            _save(paths, state, key, record)
        from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
            _Budget,
            prepare_shared_inputs,
        )

        from .dataset_publication import accepted_dataset_identity

        identity = accepted_dataset_identity(paths, state, generation)
        require(
            identity["reference"] == record["accepted"]
            and identity["source"] == record["source"],
            "Preparation source changed",
        )
        source = storage.verify_reference(paths, record["rows"])
        rows = root / "rows.jsonl"
        limits = _Budget.configured()
        assert args.continual_learning is not None
        floor = max(
            limits.headroom_bytes, args.continual_learning.min_free_disk_mb * 2**20
        )
        limits = replace(
            limits,
            scratch_bytes=record["reserved_bytes"] - record["rows"]["bytes"],
            headroom_bytes=max(
                limits.headroom_bytes,
                args.continual_learning.min_available_ram_mb * 2**20,
            ),
        )
        require(
            shutil.disk_usage(root).free >= floor + record["reserved_bytes"],
            "Prepared Dataset disk reservation refused",
        )
        from .pipeline_memory import directory_disk_usage

        # Reuse the monitor's stat-only accounting once before bulk work, outside
        # Growth ownership. Its ongoing run-wide guard still covers other workers.
        workspace_bytes = directory_disk_usage(paths.work_dir)["logical_bytes"]
        require(
            workspace_bytes + record["reserved_bytes"]
            <= args.continual_learning.max_workspace_mb * 2**20,
            "Prepared Dataset workspace reservation refused",
        )
        _remaining(record)
        if not rows.exists():
            temporary = root / "rows.pending"
            require(not temporary.is_symlink(), "Symlinked rows staging")
            if temporary.exists():
                temporary.unlink()
            copy_artifact(source, temporary)
            require(
                file_digest(temporary) == record["rows"]["sha256"],
                "Prepared row copy changed",
            )
            os.replace(temporary, rows)
            sync_artifact(rows, paths.work_dir)
        require(
            storage.reference(paths, rows)["sha256"] == record["rows"]["sha256"],
            "Stable prepared rows changed",
        )
        with prepare_shared_inputs(
            rows,
            fraction=recipe.fraction,
            seed=recipe.seed,
            max_rows=recipe.max_rows,
            row_chunk_size=recipe.row_chunk_size,
            budget=limits,
            deadline=deadline,
        ) as prepared:
            require(
                prepared is not None,
                "Prepared Dataset unexpectedly used legacy native fallback",
            )
            assert prepared is not None
            from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
                MorpionTrainingArgs,
            )

            prepared.prepare_split(
                MorpionTrainingArgs(
                    dataset_file=rows,
                    output_dir=root / "unused",
                    model_kind="linear",
                    stable_validation_split=True,
                    validation_fraction=recipe.fraction,
                    validation_seed=recipe.seed,
                ),
                recipe.max_rows,
            )
            native = prepared.directory
            value = {
                "schema": "operational_prepared_dataset_v1",
                **{k: record[k] for k in ("accepted", "source", "context", "recipe")},
                "rows": storage.reference(paths, rows),
                "native": storage.reference(paths, native / "complete.json"),
                "files": {
                    name: storage.reference(paths, native / name)
                    for name in sorted(storage.FILES)
                },
            }
        # No process-local producer lease survives this point.
        _remaining(record)
        atomic_json(root / "ready.json", value)
        ready = storage.reference(paths, root / "ready.json")
        finished = {**record, "phase": "COMPLETE", "ready": ready}
        storage.ready_descriptor(paths, finished)
        # Reserve bounded operation/descriptor rewrites as well as immutable data.
        actual_bytes = 2**20 + sum(
            p.stat().st_size for p in root.rglob("*") if p.is_file()
        )
        require(
            actual_bytes <= record["reserved_bytes"],
            "Prepared Dataset exceeded reservation",
        )
        from .continuous_contract import JobRecord
        from .continuous_dataset import ready_record
        from .continuous_publication import enabled, load_frame, read_record

        ready_evidence = None
        snapshot = load_bootstrap_run_state(paths.run_state_path)
        if enabled(snapshot):
            frame = load_frame(paths, snapshot)
            assert frame.state.dataset_job is not None
            producer = read_record(paths, frame.state.dataset_job, frame.context)
            assert isinstance(producer, JobRecord)
            ready_evidence = ready_record(paths, snapshot, finished, producer)
            from .continuous_inputs import preserve_recovery_metadata

            preserve_recovery_metadata(paths, ready_evidence)
        with growth_receipt_ownership(paths, wait=True):
            state = load_bootstrap_run_state(paths.run_state_path)
            _checked(paths, state, key, record)
            _save(
                paths,
                state,
                key,
                {**finished, "reserved_bytes": actual_bytes},
                ready_evidence,
            )


@contextmanager
def borrow(
    paths: MorpionBootstrapPaths, args: MorpionBootstrapArgs, admission: dict[str, Any]
) -> Iterator[tuple[Path, _PreparedInputs] | None]:
    """Borrow only the admission's complete input; keep lease through every reader."""
    if "continuous_admission" in admission:
        from .continuous_contract import JobRecord, ReadyRecord, Ref
        from .continuous_publication import load_frame, read_record

        current_state = load_bootstrap_run_state(paths.run_state_path)
        frame = load_frame(paths, current_state)
        job = read_record(
            paths, Ref.from_metadata(admission["continuous_admission"]), frame.context
        )
        require(
            isinstance(job, JobRecord) and job.payload.ready is not None,
            "Missing continuous reader job.",
        )
        assert isinstance(job, JobRecord) and job.payload.ready is not None
        ready = read_record(paths, job.payload.ready, frame.context)
        require(isinstance(ready, ReadyRecord), "Missing authoritative Ready.")
        assert isinstance(ready, ReadyRecord)
        with borrow_continuous(paths, args, job, ready) as borrowed:
            yield borrowed
        return
    key = admission["dataset"]["reference"]["descriptor_sha256"]
    recipe = dataset_recipe(args, storage.directory(paths, key) / "rows.jsonl")
    if recipe is None:
        yield None
        return
    state = load_bootstrap_run_state(paths.run_state_path)
    record = storage.entries(state).get(key)
    if record is None or record["phase"] == "RETIRED":
        # Private fallback must not silently coexist with unaccounted operational
        # tensors. Retire idle roots; a live builder/reader supplies backpressure.
        for other_key, other in storage.entries(state).items():
            require(
                other["phase"] in {"COMPLETE", "RETIRED", "SUSPENDED"}
                and retire(paths, other_key),
                "Private fallback blocked by live prepared input resources",
            )
        yield None
        return
    if record["recipe"] != recipe.identity():
        require(
            len(storage.entries(state)) == 1 and retire(paths, key),
            "Incompatible preparation needs quiescent resource release before private fallback",
        )
        yield None
        return
    if record["phase"] in {"PREPARING", "SUSPENDED"}:
        deadline = _remaining(record)
        while record["phase"] in {"PREPARING", "SUSPENDED"}:
            require(time.monotonic() < deadline, "Preparation wait allowance exhausted")
            # Wait on the artifact, not repeated full run-state reads during build.
            with storage.lease(paths, key, deadline=deadline):
                record = storage.entries(
                    load_bootstrap_run_state(paths.run_state_path)
                )[key]
            if record["phase"] not in {"PREPARING", "SUSPENDED"}:
                break
            from .persistent_worker_state import read_process_identity

            observed = read_process_identity(record["owner"]["pid"])
            if observed is None or observed[2] != record["owner"]["start_ticks"]:
                prepare_accepted(paths, args, record["generation"])
            else:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    else:
        deadline = time.monotonic()
    with storage.lease(paths, key, deadline=deadline) as root:
        with growth_receipt_ownership(paths, wait=True):
            state = load_bootstrap_run_state(paths.run_state_path)
            current = storage.entries(state).get(key)
            require(current is not None, "Prepared Dataset was retired")
            assert current is not None
            if current["phase"] == "CONSUMING":
                require_quiescent(current["owner"], monitored_identity())
            require(
                current["phase"] in {"COMPLETE", "CONSUMING"},
                "Prepared Dataset is not complete",
            )
            assert current is not None
            record = current
            _checked(paths, state, key, record)
            require(
                record["accepted"] == admission["dataset"]["reference"]
                and record["source"] == admission["dataset"]["source"],
                "Prepared admission differs",
            )
            from .pipeline_artifacts import load_pipeline_stage_claim

            claim = load_pipeline_stage_claim(
                paths.pipeline_training_claim_path_for_generation(
                    admission["generation"]
                )
            )
            require(
                claim.claim_id == admission["claim_id"],
                "Training claim replaced before borrowing",
            )
            consuming = {
                **record,
                "phase": "CONSUMING",
                "owner": monitored_identity(),
                "token": uuid.uuid4().hex,
            }
            _save(paths, state, key, consuming)
        entered = False
        try:
            # Missing/corrupt completion before consumption is never built in place.
            try:
                value = storage.ready_descriptor(paths, record)
            except OSError, ValueError:
                with growth_receipt_ownership(paths, wait=True):
                    state = load_bootstrap_run_state(paths.run_state_path)
                    _checked(paths, state, key, consuming)
                    require(
                        len(storage.entries(state)) == 1,
                        "Missing preparation needs resource release before private fallback",
                    )
                    _save(paths, state, key, {**record, "phase": "COMPLETE"})
                require(
                    _retire_leased(paths, key, root),
                    "Incomplete cache could not retire",
                )
                yield None
                return
            from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
                prepare_shared_inputs,
            )

            rows = inside(paths.work_dir, value["rows"]["path"])
            with prepare_shared_inputs(
                rows,
                fraction=recipe.fraction,
                seed=recipe.seed,
                max_rows=recipe.max_rows,
                row_chunk_size=recipe.row_chunk_size,
                deadline=time.monotonic(),
                existing_only=True,
            ) as prepared:
                require(prepared is not None, "Missing borrowed native inputs")
                assert prepared is not None
                entered = True
                yield rows, prepared
                storage.ready_descriptor(paths, record)
        finally:
            primary = sys.exc_info()[1]
            try:
                with growth_receipt_ownership(paths, wait=True):
                    state = load_bootstrap_run_state(paths.run_state_path)
                    # Late cleanup must not mutate another consumer's ownership.
                    if storage.entries(state).get(key) == consuming:
                        _save(
                            paths,
                            state,
                            key,
                            {**record, "phase": "COMPLETE" if entered else "SUSPENDED"},
                        )
            # Any secondary persistence failure must preserve an active reader failure.
            except Exception:  # pylint: disable=broad-exception-caught
                if primary is None:
                    raise
                LOGGER.exception(
                    "Prepared input cleanup failed; preserving primary failure"
                )


def prepare_continuous(
    paths: MorpionBootstrapPaths, args: MorpionBootstrapArgs, job: JobRecord
) -> ReadyRecord:
    """Complete only the authoritative admitted PREPARE, retaining its stable path."""
    from .continuous_contract import ReadyRecord, record_ref
    from .continuous_publication import authenticate_job, load_frame, read_record

    state = load_bootstrap_run_state(paths.run_state_path)
    frame = load_frame(paths, state)
    require(frame.state.dataset_job == record_ref(job), "Preparation job was replaced.")
    authenticate_job(paths, state, job)
    prepare_accepted(paths, args, job.payload.source.generation)
    state = load_bootstrap_run_state(paths.run_state_path)
    frame = load_frame(paths, state)
    require(frame.state.ready is not None, "Preparation did not commit Ready.")
    assert frame.state.ready is not None
    ready = read_record(paths, frame.state.ready, frame.context)
    require(
        isinstance(ready, ReadyRecord)
        and ready.payload.producer_job_id == job.payload.id,
        "Unexpected Ready producer.",
    )
    assert isinstance(ready, ReadyRecord)
    return ready


@contextmanager
def borrow_continuous(
    paths: MorpionBootstrapPaths,
    args: MorpionBootstrapArgs,
    job: JobRecord,
    ready: ReadyRecord,
) -> Iterator[tuple[Path, _PreparedInputs]]:
    """Retain stable and native leases for the admitted reader; never fall back."""
    from .continuous_contract import record_ref
    from .continuous_publication import authenticate_job, load_frame

    require(job.payload.ready == record_ref(ready), "Reader Ready binding changed.")
    state = load_bootstrap_run_state(paths.run_state_path)
    frame = load_frame(paths, state)
    if job.payload.stage == "COMPARE":
        from .continuous_contract import Reservation

        # Deferred runtime import: both modules are initialized before this call.
        from .promotion import reservation  # pylint: disable=cyclic-import

        reserved = reservation(paths, state)
        require(
            reserved is not None
            and isinstance(reserved.payload, Reservation)
            and reserved.payload.comparison_job == record_ref(job),
            "Reader is not the reserved comparison.",
        )
    else:
        require(
            frame.state.cohort == record_ref(job), "Reader is not the admitted cohort."
        )
    authenticate_job(paths, state, job)
    key = ready.payload.accepted_dataset.descriptor_sha256
    recipe = dataset_recipe(args, storage.directory(paths, key) / "rows.jsonl")
    require(recipe is not None, "Continuous preparation recipe unavailable.")
    assert recipe is not None
    with storage.lease(paths, key) as root:
        state = load_bootstrap_run_state(paths.run_state_path)
        record = storage.entries(state).get(key)
        require(
            record is not None and record["phase"] == "COMPLETE",
            "Continuous inputs unavailable.",
        )
        assert record is not None
        from .continuous_contract import Accepted, JobRecord
        from .continuous_dataset import admitted, ready_record

        # Reconstruct the producer envelope from the immutable Dataset admission.
        from .dataset_publication import _proposal

        proposal = _proposal(paths, ready.payload.accepted_dataset.to_metadata())
        extract = admitted(paths, state, proposal["admission"])
        producer = JobRecord(
            extract.context,
            replace(
                extract.payload,
                stage="PREPARE",
                accepted_dataset=Accepted.from_metadata(record["accepted"]),
            ),
        )
        from .continuous_inputs import recover_before_consumption

        recover_before_consumption(paths, args, job, ready, record, root)
        require(
            ready_record(paths, state, record, producer) == ready,
            "Consumed Ready bytes changed.",
        )
        require(
            record["recipe"] == recipe.identity(), "Continuous native recipe changed."
        )
        from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
            prepare_shared_inputs,
        )

        rows = root / "rows.jsonl"
        with prepare_shared_inputs(
            rows,
            fraction=recipe.fraction,
            seed=recipe.seed,
            max_rows=recipe.max_rows,
            row_chunk_size=recipe.row_chunk_size,
            deadline=time.monotonic(),
            existing_only=True,
        ) as prepared:
            require(prepared is not None, "Missing authoritative native inputs.")
            assert prepared is not None
            from .continuous_inputs import consumed_owned

            consumed_owned(paths, job, ready)
            yield rows, prepared
            require(
                ready_record(paths, state, record, producer) == ready,
                "Inputs changed after consumption.",
            )


def _retire_leased(paths: MorpionBootstrapPaths, key: str, root: Path) -> bool:
    """Retire after acquiring the stable artifact lease, without recursive locks."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        _ownership,
    )

    native_root = root / ".rows.jsonl.shared-preparation-v1"
    native_root.mkdir(exist_ok=True)
    with _ownership(native_root, deadline=time.monotonic()):
        with growth_receipt_ownership(paths, wait=True):
            state = load_bootstrap_run_state(paths.run_state_path)
            data = storage.entries(state)
            record = data.get(key)
            if record is None:
                return False
            from .continuous_publication import enabled as continuous_enabled
            from .continuous_publication import prepared_roots

            if continuous_enabled(state) and key in prepared_roots(paths, state):
                return False
            require(
                record["phase"] in {"COMPLETE", "RETIRED", "SUSPENDED"},
                "Live preparation cannot retire",
            )
            if continuous_enabled(state) and record["ready"] is not None:
                from .continuous_contract import ControlRecord, Ref, Retirement
                from .continuous_publication import load_frame, put_record
                from .generation_publication_io import PUBLICATION_KEY

                data_publication = state.metadata[PUBLICATION_KEY]
                scientific = publication_record(state, record["generation"])
                ready_ref = Ref.from_metadata(scientific["continuous_ready"])
                proof = put_record(
                    paths,
                    ControlRecord(
                        load_frame(paths, state).context,
                        Retirement(
                            ready_ref,
                            tuple(
                                paths.relative_to_work_dir(p)
                                for p in (
                                    root / "rows.jsonl",
                                    native_root / "entry/flat.pt",
                                    native_root / "entry/relational.pt",
                                )
                            ),
                            "unreferenced_operational_cache",
                        ),
                    ),
                )
                generations = [
                    {**r, "continuous_retirement": proof.to_metadata()}
                    if r["generation"] == record["generation"]
                    else r
                    for r in data_publication["generations"]
                ]
                state = replace(
                    state,
                    metadata={
                        **state.metadata,
                        PUBLICATION_KEY: {
                            **data_publication,
                            "generations": generations,
                        },
                    },
                )
            retired = {**record, "phase": "RETIRED"}
            atomic_json(root / "retired.json", retired)
            _save(paths, state, key, retired)
        # Keep manifest/split/descriptor proof and both stable lock inodes.
        for path in (
            root / "rows.jsonl",
            root / "rows.pending",
            native_root / "entry" / "flat.pt",
            native_root / "entry" / "relational.pt",
        ):
            if path.exists():
                require(
                    not path.is_symlink() and path.is_file(),
                    "Unsafe operational retirement",
                )
                path.unlink()
        sync_artifact(root, paths.work_dir)
        with growth_receipt_ownership(paths, wait=True):
            state = load_bootstrap_run_state(paths.run_state_path)
            data = storage.entries(state)
            require(data.get(key) == retired, "Retirement ownership replaced")
            del data[key]
            save_bootstrap_run_state(
                storage.updated(state, data), paths.run_state_path, durable=True
            )
        return True


def retire(paths: MorpionBootstrapPaths, key: str) -> bool:
    """Record retirement then remove only reconstructible payloads under exclusive leases."""
    try:
        with storage.lease(paths, key) as root:
            return _retire_leased(paths, key, root)
    except BlockingIOError, TimeoutError:
        return False


def reconcile(paths: MorpionBootstrapPaths, args: MorpionBootstrapArgs) -> None:
    """Service recorded accepted work only, without changing scientific discovery."""
    if os.environ.get("CHIPIRON_SHARED_PREPARATION", "0") == "0":
        return
    state = load_bootstrap_run_state(paths.run_state_path)
    for key, record in storage.entries(state).items():
        if record["phase"] in {"PREPARING", "SUSPENDED"}:
            prepare_accepted(paths, args, record["generation"])
        elif (
            (record["phase"] == "COMPLETE" and _continuous_unrooted(paths, state, key))
            or record["phase"] == "RETIRED"
            or (
                record["phase"] == "COMPLETE"
                and "accepted_training"
                in publication_record(state, record["generation"])
            )
        ):
            retire(paths, key)


def _continuous_unrooted(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState, key: str
) -> bool:
    from .continuous_publication import enabled, prepared_roots

    return enabled(state) and key not in prepared_roots(paths, state)
