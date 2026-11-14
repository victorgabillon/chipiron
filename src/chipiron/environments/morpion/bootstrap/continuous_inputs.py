"""Same-Dataset disposable recovery and durable first-consumption evidence."""

import shutil
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from . import continuous_publication as publication
from . import prepared_dataset_storage as storage
from .continuous_contract import JobRecord, ReadyRecord, Ref, record_ref, require
from .continuous_lifecycle import remaining
from .generation_publication_io import (
    PUBLICATION_KEY,
    growth_receipt_ownership,
    publication_records,
)
from .run_state import load_bootstrap_run_state, save_bootstrap_run_state

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState


def has_consumed(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    ready: ReadyRecord,
    job: JobRecord | None = None,
) -> bool:
    """Read authenticated compact evidence without requiring retired tensor payloads."""
    for record in publication_records(state.metadata):
        if record["generation"] == ready.payload.source.generation:
            for raw in record.get("continuous_consumed", []):
                proof = publication.read_ref(paths, Ref.from_metadata(raw))
                if proof["ready"] == record_ref(ready).to_metadata() and (
                    job is None or proof["job_id"] == job.payload.id
                ):
                    return True
    return False


def consumed_owned(
    paths: MorpionBootstrapPaths, job: JobRecord, ready: ReadyRecord
) -> None:
    """Commit before yielding a native reader; caller already holds artifact/native leases."""
    with growth_receipt_ownership(paths, wait=True):
        state = load_bootstrap_run_state(paths.run_state_path)
        publication.authenticate_job(paths, state, job)
        if has_consumed(paths, state, ready, job):
            return
        proof = publication.put_proof(
            paths,
            {
                "job_id": job.payload.id,
                "job": record_ref(job).to_metadata(),
                "ready": record_ref(ready).to_metadata(),
            },
        ).to_metadata()
        data = state.metadata[PUBLICATION_KEY]
        records = [
            {**r, "continuous_consumed": [*r.get("continuous_consumed", []), proof]}
            if r["generation"] == job.payload.source.generation
            else r
            for r in data["generations"]
        ]
        save_bootstrap_run_state(
            replace(
                state,
                metadata={
                    **state.metadata,
                    PUBLICATION_KEY: {**data, "generations": records},
                },
            ),
            paths.run_state_path,
            durable=True,
        )


def recover_before_consumption(
    paths: MorpionBootstrapPaths,
    args: MorpionBootstrapArgs,
    job: JobRecord,
    ready: ReadyRecord,
    record: dict[str, Any],
    root: Path,
) -> None:
    """Rebuild only disposable bytes at the same stable path, under the reader allowance."""
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        MorpionTrainingArgs,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.shared_preparation import (
        _Budget,
        _ownership,
        prepare_shared_inputs,
    )

    from .generation_publication_io import file_digest, sync_artifact
    from .prepared_dataset_recipe import dataset_recipe

    # Accepted rows are permanent scientific evidence, never a repair target.
    source = storage.verify_reference(paths, ready.payload.accepted_rows.to_metadata())
    state = load_bootstrap_run_state(paths.run_state_path)
    native = root / ".rows.jsonl.shared-preparation-v1"
    expected = (
        ready.payload.rows,
        ready.payload.native_complete,
        *(ref for _, ref in ready.payload.files),
    )
    invalid = False
    for ref in expected:
        path = paths.work_dir / ref.path
        require(
            not path.is_symlink() and not any(p.is_symlink() for p in path.parents),
            "Symlinked continuous input.",
        )
        if (
            not path.is_file()
            or path.stat().st_size != ref.bytes
            or file_digest(path) != ref.sha256
        ):
            invalid = True
    if not invalid:
        return
    consumed = has_consumed(paths, state, ready)
    # Record the failed disposable input before repair or refusal. Scientific
    # acceptance and the admitted job remain unchanged and pinned.
    from .continuous_contract import Disposition

    with growth_receipt_ownership(paths, wait=True):
        current = load_bootstrap_run_state(paths.run_state_path)
        publication.authenticate_job(paths, current, job)
        current = publication.attach_disposition(
            paths,
            current,
            Disposition(
                record_ref(job),
                "failed_incomplete",
                None if consumed else record_ref(ready),
                publication.load_frame(paths, current).state.learning_debt,
            ),
        )
        save_bootstrap_run_state(current, paths.run_state_path, durable=True)
    require(not consumed, "Corrupt input after consumption is not recoverable.")
    budget = remaining(paths, job)
    clock = publication.read_ref(paths, budget.clock_evidence)
    deadline = clock["deadline"]
    from .continuous_resources import COMMIT_HEADROOM, logical_bytes, reserve_recovery

    assert args.continual_learning is not None
    key = ready.payload.accepted_dataset.descriptor_sha256
    with growth_receipt_ownership(paths, wait=True):
        current = load_bootstrap_run_state(paths.run_state_path)
        publication.authenticate_job(paths, current, job)
        require(
            not has_consumed(paths, current, ready),
            "Corrupt input after consumption is not recoverable.",
        )
        current = reserve_recovery(paths, current, args.continual_learning, key)
        record = storage.entries(current)[key]
        save_bootstrap_run_state(current, paths.run_state_path, durable=True)
    native.mkdir(exist_ok=True)
    with _ownership(native, deadline=deadline):
        entry = native / "entry"
        if entry.exists():
            require(
                entry.is_dir()
                and not entry.is_symlink()
                and not any(p.is_symlink() for p in entry.rglob("*")),
                "Unsafe disposable native entry.",
            )
            shutil.rmtree(entry)
        rows = root / "rows.jsonl"
        if rows.exists():
            rows.unlink()
        shutil.copyfile(source, rows)
        sync_artifact(rows, paths.work_dir)
    recipe = dataset_recipe(args, rows)
    require(
        recipe is not None and recipe.identity() == record["recipe"],
        "Recovery recipe changed.",
    )
    assert recipe is not None
    limits = replace(
        _Budget.configured(),
        scratch_bytes=record["reserved_bytes"] - ready.payload.rows.bytes,
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
        require(prepared is not None, "Recovery cannot use private fallback.")
        assert prepared is not None
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
    for ref in expected:
        storage.verify_reference(paths, ref.to_metadata())
    remaining(paths, job)
    materialized = logical_bytes(root) + COMMIT_HEADROOM
    require(materialized <= record["reserved_bytes"], "Recovery exceeded reservation.")
    with growth_receipt_ownership(paths, wait=True):
        current = load_bootstrap_run_state(paths.run_state_path)
        publication.authenticate_job(paths, current, job)
        data = storage.entries(current)
        data[key] = {**data[key], "reserved_bytes": materialized}
        save_bootstrap_run_state(
            storage.updated(current, data), paths.run_state_path, durable=True
        )
