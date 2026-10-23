"""Fenced Dataset proposals; only the existing run-state replacement accepts one.

One reference is attached to the existing scientific publication entry. TTL claims
schedule attempts; they neither certify completion nor protect a reader's lifetime.
"""

from __future__ import annotations

import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.learning.tree_to_dataset import (
    iter_morpion_supervised_rows_from_path,
    load_morpion_supervised_rows_metadata,
)

from .derived.provenance import atomic_json, inside
from .generation_publication_io import (
    PUBLICATION_KEY,
    file_digest,
    generation_read_pin,
    publication_records,
    publish_directory_if_absent,
    read_object,
    require,
    sync_artifact,
    validate_capsule,
    validate_manifest_publication,
)
from .pipeline_artifacts import (
    MorpionPipelineGenerationManifest,
    load_pipeline_manifest,
    load_pipeline_stage_claim,
    pipeline_dataset_status_from_dict,
    pipeline_dataset_status_to_dict,
    pipeline_manifest_from_dict,
    pipeline_manifest_to_dict,
)
from .pipeline_claims import claim_pipeline_stage, release_pipeline_stage_claim
from .reevaluation_transaction import source_identity, transaction_state
from .run_state import load_bootstrap_run_state, save_bootstrap_run_state

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .pipeline_artifacts import MorpionPipelineDatasetStatusArtifact
    from .run_state import MorpionBootstrapRunState

KEY = "fenced_dataset"
ACCEPTED = "accepted_dataset"
NAMESPACE = "dataset_publications_v1"
CONTRACT = {"schema_version": 1}
OPTIONS = (
    "min_depth",
    "min_visit_count",
    "max_rows",
    "use_backed_up_value",
    "require_exact_or_terminal",
    "dataset_family_target_policy",
    "dataset_family_prediction_blend",
    "dataset_confidence_policy",
)


def enabled(state: MorpionBootstrapRunState) -> bool:
    """Require explicit fresh genesis; unsupported markers never become legacy mode."""
    if KEY not in state.metadata:
        require(
            not any(
                ACCEPTED in r
                for r in state.metadata.get(PUBLICATION_KEY, {}).get("generations", [])
            ),
            "Accepted Dataset references require their versioned protocol.",
        )
        return False
    require(
        state.metadata[KEY] == CONTRACT
        and isinstance(state.metadata[KEY].get("schema_version"), int)
        and not isinstance(state.metadata[KEY]["schema_version"], bool)
        and transaction_state(state) is not None,
        "Unsupported fenced Dataset protocol or missing workspace/epoch.",
    )
    return True


def _record(state: MorpionBootstrapRunState, generation: int) -> dict[str, Any]:
    records = [
        r for r in publication_records(state.metadata) if r["generation"] == generation
    ]
    require(len(records) == 1, "Dataset source is not a committed generation.")
    return records[0]


def _identity(state: MorpionBootstrapRunState) -> dict[str, str]:
    data = transaction_state(state)
    require(data is not None, "Dataset requires an authoritative workspace/epoch.")
    assert data is not None
    return {key: data[key] for key in ("workspace_id", "epoch_id")}


def _directory(
    paths: MorpionBootstrapPaths, attempt_id: str, *, private: bool = False
) -> Path:
    require(
        isinstance(attempt_id, str)
        and len(attempt_id) == 32
        and all(c in "0123456789abcdef" for c in attempt_id),
        "Malformed Dataset attempt identity.",
    )
    return (
        paths.work_dir / NAMESPACE / ("staging" if private else "objects") / attempt_id
    )


def _proposal(
    paths: MorpionBootstrapPaths, reference: dict[str, Any]
) -> dict[str, Any]:
    require(
        set(reference) == {"attempt_id", "descriptor_sha256"},
        "Malformed accepted Dataset reference.",
    )
    directory = _directory(paths, reference["attempt_id"])
    path = inside(
        paths.work_dir, paths.relative_to_work_dir(directory / "proposal.json")
    )
    require(
        file_digest(path) == reference["descriptor_sha256"],
        "Dataset descriptor changed.",
    )
    proposal = read_object(path)
    require(
        proposal["admission"]["attempt_id"] == reference["attempt_id"],
        "Dataset attempt identity mismatch.",
    )
    return proposal


def _validate_binding(
    state: MorpionBootstrapRunState, admission: dict[str, Any], record: dict[str, Any]
) -> None:
    require(enabled(state), "Fenced Dataset authority is required.")
    require(
        set(admission)
        == {
            "schema_version",
            "attempt_id",
            "identity",
            "source",
            "generation",
            "options",
            "claim_id",
            "source_model_pins",
            "source_config_sha256",
        }
        and isinstance(admission["options"], dict)
        and set(admission["options"]) == set(OPTIONS),
        "Incomplete Dataset admission.",
    )
    require(
        admission["schema_version"] == 1
        and isinstance(admission["schema_version"], int)
        and not isinstance(admission["schema_version"], bool),
        "Unsupported Dataset proposal.",
    )
    require(
        admission["identity"] == _identity(state), "Dataset workspace/epoch mismatch."
    )
    require(
        admission["generation"] == record["generation"]
        and admission["source"]
        == {
            key: record[key]
            for key in ("id", "descriptor_sha256", "scientific_identity")
        },
        "Dataset scientific source identity mismatch.",
    )


def _validate_rows(paths: MorpionBootstrapPaths, proposal: dict[str, Any]) -> Path:
    admission = proposal["admission"]
    directory = _directory(paths, admission["attempt_id"])
    rows = inside(paths.work_dir, paths.relative_to_work_dir(directory / "rows.jsonl"))
    require(
        rows.is_file()
        and rows.stat().st_size == proposal["rows_bytes"]
        and file_digest(rows) == proposal["rows_sha256"],
        "Accepted Dataset rows missing, incomplete or changed.",
    )
    return rows


def _completion(
    paths: MorpionBootstrapPaths, proposal: dict[str, Any]
) -> MorpionPipelineGenerationManifest:
    """Validate repair evidence before it can enter the authoritative commit."""
    require(
        set(proposal)
        == {"admission", "rows_sha256", "rows_bytes", "manifest", "status"},
        "Incomplete Dataset proposal.",
    )
    admission = proposal["admission"]
    completed = pipeline_manifest_from_dict(proposal["manifest"])
    status = pipeline_dataset_status_from_dict(proposal["status"])
    expected = paths.relative_to_work_dir(
        _directory(paths, admission["attempt_id"]) / "rows.jsonl"
    )
    require(
        completed.generation == status.generation == admission["generation"]
        and completed.rows_path == expected
        and completed.dataset_status == status.status == "done"
        and completed.metadata == status.metadata
        and completed.metadata["dataset_rows_bytes"] == proposal["rows_bytes"],
        "Dataset completion projection mismatch.",
    )
    return completed


def project_dataset(
    paths: MorpionBootstrapPaths,
    state: MorpionBootstrapRunState,
    generation: int,
    manifest: MorpionPipelineGenerationManifest | None,
    *,
    verify_rows: bool = False,
    record: dict[str, Any] | None = None,
) -> MorpionPipelineGenerationManifest:
    """Resolve advisory status from accepted evidence; selected consumers hash rows.

    Discovery reads only the small descriptor. It is not an archive integrity scan.
    Existing Training status remains advisory and is not made authoritative here.
    """
    record = _record(state, generation) if record is None else record
    reference = record.get(ACCEPTED)
    if reference is None:
        require(manifest is not None, "Missing unaccepted Dataset discovery manifest.")
        assert manifest is not None
        return replace(manifest, rows_path=None, dataset_status="not_started")
    proposal = _proposal(paths, reference)
    _validate_binding(state, proposal["admission"], record)
    require(
        proposal["admission"]["generation"] == generation,
        "Dataset generation mismatch.",
    )
    completed = _completion(paths, proposal)
    if verify_rows:
        _validate_rows(paths, proposal)
        sync_artifact(paths.run_state_path, paths.work_dir)
    return (
        completed
        if manifest is None
        else replace(
            manifest,
            rows_path=completed.rows_path,
            dataset_status="done",
            metadata={**manifest.metadata, **completed.metadata},
        )
    )


@dataclass(frozen=True)
class DatasetAttempt:
    """Admission captured before reading rows; staging is private to this identity."""

    admission: dict[str, Any]
    rows_path: Path


@contextmanager
def admit_dataset(
    paths: MorpionBootstrapPaths,
    args: MorpionBootstrapArgs,
    manifest: MorpionPipelineGenerationManifest,
    *,
    ttl_seconds: float,
    owner: str | None,
) -> Iterator[DatasetAttempt | MorpionPipelineGenerationManifest]:
    """Use Growth ownership only for admission/release, never during extraction.

    Caller must retain the existing scientific source pin through this context.
    A completed generation is immutable: retry returns it only for identical options.
    """
    from .generation_publication_io import growth_receipt_ownership

    generation = manifest.generation
    options = {key: getattr(args, key) for key in OPTIONS}
    claim_path = paths.pipeline_dataset_claim_path_for_generation(generation)
    result: DatasetAttempt | MorpionPipelineGenerationManifest
    with growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        require(enabled(state), "Dataset publication is not enabled.")
        record = _record(state, generation)
        if ACCEPTED in record:
            proposal = _proposal(paths, record[ACCEPTED])
            require(
                proposal["admission"]["options"] == options,
                "Conflicting Dataset configuration for accepted generation.",
            )
            result = project_dataset(
                paths, state, generation, manifest, verify_rows=True
            )
            _repair(paths, result, proposal)
            attempt = None
        else:
            descriptor = validate_capsule(paths.work_dir, record)
            validate_manifest_publication(paths.work_dir, manifest)
            claim = claim_pipeline_stage(
                generation=generation,
                stage="dataset",
                claim_path=claim_path,
                ttl_seconds=ttl_seconds,
                owner=owner,
            )
            identifier = uuid.uuid4().hex
            admission = {
                "schema_version": 1,
                "attempt_id": identifier,
                "identity": _identity(state),
                "source": source_identity(state, generation),
                "generation": generation,
                "options": options,
                "claim_id": claim.claim_id,
                "source_model_pins": descriptor["model_pins"],
                "source_config_sha256": descriptor["bootstrap_config_hash"],
            }
            staging = _directory(paths, identifier, private=True)
            staging.mkdir(parents=True, exist_ok=False)
            attempt = DatasetAttempt(admission, staging / "rows.jsonl")
            result = attempt
    try:
        yield result
    finally:
        if attempt is not None:
            # A replacement's claim must survive the old worker's late cleanup.
            try:
                with growth_receipt_ownership(paths):
                    if (
                        claim_path.is_file()
                        and load_pipeline_stage_claim(claim_path).claim_id
                        == attempt.admission["claim_id"]
                    ):
                        release_pipeline_stage_claim(
                            claim_path=claim_path,
                            claim_id=attempt.admission["claim_id"],
                        )
            except BlockingIOError:
                pass  # Advisory claim may expire; never obscure accepted state.


def publish_dataset(
    paths: MorpionBootstrapPaths,
    attempt: DatasetAttempt,
    manifest: MorpionPipelineGenerationManifest,
    status: MorpionPipelineDatasetStatusArtifact,
) -> dict[str, Any]:
    """Validate streaming rows, fsync, and publish one immutable create-only proposal."""
    rows = attempt.rows_path
    expected = manifest.metadata["dataset_rows"]
    require(
        sum(1 for _ in iter_morpion_supervised_rows_from_path(rows)) == expected,
        "Dataset output row count mismatch.",
    )
    metadata = load_morpion_supervised_rows_metadata(rows)
    require(
        metadata.get("num_rows") == expected
        and rows.stat().st_size == manifest.metadata["dataset_rows_bytes"],
        "Incomplete Dataset output.",
    )
    directory = _directory(paths, attempt.admission["attempt_id"])
    completed = replace(
        manifest, rows_path=paths.relative_to_work_dir(directory / "rows.jsonl")
    )
    proposal = {
        "admission": attempt.admission,
        "rows_sha256": file_digest(rows),
        "rows_bytes": rows.stat().st_size,
        "manifest": pipeline_manifest_to_dict(completed),
        "status": pipeline_dataset_status_to_dict(status),
    }
    atomic_json(rows.parent / "proposal.json", proposal)
    sync_artifact(rows.parent, paths.work_dir)
    directory.parent.mkdir(parents=True, exist_ok=True)
    publish_directory_if_absent(rows.parent, directory, paths.work_dir)
    return {
        "attempt_id": attempt.admission["attempt_id"],
        "descriptor_sha256": file_digest(directory / "proposal.json"),
    }


def _repair(
    paths: MorpionBootstrapPaths,
    manifest: MorpionPipelineGenerationManifest,
    proposal: dict[str, Any],
) -> None:
    """Rebuild projections only from a verified accepted reference, under ownership."""
    atomic_json(
        paths.pipeline_manifest_path_for_generation(manifest.generation),
        pipeline_manifest_to_dict(manifest),
    )
    atomic_json(
        paths.pipeline_dataset_status_path_for_generation(manifest.generation),
        proposal["status"],
    )


def accept_dataset(
    paths: MorpionBootstrapPaths, reference: dict[str, Any]
) -> MorpionPipelineGenerationManifest:
    """Recheck exact evidence then accept once in the sole authoritative run state."""
    from .checkpoint_patch_receipts import receipt_ledger
    from .generation_publication_io import growth_receipt_ownership

    with growth_receipt_ownership(paths), ExitStack() as pins:
        state = load_bootstrap_run_state(paths.run_state_path)
        receipt_ledger(state)
        proposal = _proposal(paths, reference)
        admission = proposal["admission"]
        generation = admission["generation"]
        record = _record(state, generation)
        _validate_binding(state, admission, record)
        completed = _completion(paths, proposal)
        require(
            (completed.tree_snapshot_path, completed.runtime_checkpoint_path)
            == (record["tree_snapshot_path"], record["checkpoint_path"]),
            "Dataset completion source mismatch.",
        )
        pins.enter_context(
            generation_read_pin(paths.work_dir / record["tree_snapshot_path"])
        )
        descriptor = validate_capsule(paths.work_dir, record)
        require(
            (admission["source_model_pins"], admission["source_config_sha256"])
            == (descriptor["model_pins"], descriptor["bootstrap_config_hash"]),
            "Dataset source model/configuration mismatch.",
        )
        rows = _validate_rows(paths, proposal)
        sync_artifact(rows.parent, paths.work_dir)
        previous = record.get(ACCEPTED)
        require(
            previous is None or previous == reference,
            "Conflicting accepted Dataset attempt.",
        )
        manifest_path = paths.pipeline_manifest_path_for_generation(generation)
        manifest = (
            load_pipeline_manifest(manifest_path) if manifest_path.exists() else None
        )
        if previous is None:
            claim = load_pipeline_stage_claim(
                paths.pipeline_dataset_claim_path_for_generation(generation)
            )
            require(
                claim.claim_id == admission["claim_id"]
                and claim.generation == generation
                and claim.stage == "dataset",
                "Stale Dataset claim was replaced.",
            )
            # Pending/extracting/failed projections cannot certify another result.
            # An older committed source remains eligible after Growth advances.
            require(
                manifest is not None
                and manifest.tree_snapshot_path == record["tree_snapshot_path"]
                and manifest.training_status in {"not_started", "failed"},
                "Dataset stage is not eligible.",
            )
            publication = state.metadata[PUBLICATION_KEY]
            state = replace(
                state,
                metadata={
                    **state.metadata,
                    PUBLICATION_KEY: {
                        **publication,
                        "generations": [
                            {**r, ACCEPTED: reference}
                            if r["generation"] == generation
                            else r
                            for r in publication["generations"]
                        ],
                    },
                },
            )
            # Only Dataset references change; the existing run-state serializer still rewrites its cumulative metadata.
            save_bootstrap_run_state(state, paths.run_state_path, durable=True)
        else:
            sync_artifact(paths.run_state_path, paths.work_dir)
        result = project_dataset(paths, state, generation, manifest)
        _repair(paths, result, proposal)
        return result
