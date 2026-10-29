"""Scientific-v2 adapter for the existing capsule transaction, never a new journal.

The first protocol deliberately supports only a new, aligned history: Growth G
is scientific G, starting at 1, with a fixed producer and evaluator/model epoch.
No historical v1 adoption, skipped export, producer epoch or implicit migration.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

from .generation_publication_io import (
    artifact_digest,
    copy_artifact,
    file_digest,
    object_digest,
    private_generation_validation,
    require,
)
from .scientific_validation_evidence import (
    canonical_validation_evidence,
    committed_canonical_validation,
    validator_identity,
)
from .sharded_training_reader import ShardedTrainingGenerationView, _scientific_manifest
from .sharded_training_v2 import _producer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def catalog_paths(path: Path, generation: int) -> list[Path]:
    """Require the exact consecutive v2 layout before cloning committed history."""
    root = path.parent.parent
    catalog = _scientific_manifest(root / "manifest.json", "catalog")
    mapping = {
        str(g): f"generation_{g:06d}/generation.json" for g in range(1, generation + 1)
    }
    require(
        isinstance(catalog.get("latest"), int)
        and not isinstance(catalog.get("latest"), bool)
        and catalog.get("latest") == generation
        and catalog.get("v2_generations") == mapping,
        "Scientific v2 requires consecutive aligned generations without skipped exports.",
    )
    require(
        path == root / mapping[str(generation)],
        "Scientific selected generation mismatch.",
    )
    return [root / relative for relative in mapping.values()]


def clone_scientific_catalog(source: Path, destination: Path) -> Path:
    """Link only validated committed prefix bytes; give the new catalog its own inode."""
    selected = _scientific_manifest(source, "generation")
    for path in catalog_paths(source, selected["generation"]):
        copy_artifact(path.parent, destination / path.parent.name, link=True)
    copy_artifact(source.parent.parent / "manifest.json", destination / "manifest.json")
    return destination / source.parent.name / source.name


def validate_scientific_export(
    path: Path,
    *,
    staging: Path,
    generation: int,
    producer: Mapping[str, str],
    checkpoint: Path,
    config_hash: str,
    work_dir: Path,
    parent: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the new complete dependency closure with the existing bounded index.

    Previous generations have been sealed and verified before cloning. The new
    descriptor must preserve their record declarations exactly. The temporary
    SQLite index verifies cumulative creation keys, canonical records, all current
    values and reciprocal DAG edges. It never enters consumer cache discovery.
    """
    require(path.is_relative_to(staging), "Scientific export escaped private attempt.")
    paths = catalog_paths(path, generation)
    header = _scientific_manifest(path, "generation")
    require(
        header["generation"] == generation, "Growth/scientific generation mismatch."
    )
    require(header["producer"] == _producer(producer), "Scientific producer changed.")
    require(
        producer.get("config_sha256") == config_hash,
        "Scientific producer/config identity mismatch.",
    )
    expected_records: list[dict[str, Any]] = []
    for g, manifest_path in enumerate(paths, 1):
        previous = _scientific_manifest(manifest_path, "generation")
        require(
            previous["generation"] == g and previous["producer"] == header["producer"],
            "Mixed scientific history identity.",
        )
        records = previous["record_shards"]
        require(
            len(records) == g and records[:-1] == expected_records,
            "Scientific immutable record prefix changed.",
        )
        require(
            records[-1]["generation"] == g
            and records[-1]["path"] == f"generation_{g:06d}/records.jsonl"
            and previous["update_shard"]["path"] == f"generation_{g:06d}/updates.jsonl",
            "Unsupported scientific shard layout.",
        )
        expected_records = records
    validator = validator_identity()
    with (
        committed_canonical_validation(
            work_dir=work_dir,
            parent=parent,
            manifest=path,
            header=header,
            validator=validator,
        ) as reused_from,
        tempfile.TemporaryDirectory(prefix=".validation-index-", dir=staging) as cache,
        private_generation_validation(staging),
        ShardedTrainingGenerationView(
            path, cache_dir=cache, allow_scientific_v2=True
        ) as view,
    ):
        require(not view.index_reused, "Publication validation requires a fresh index.")
        require(
            view.generation == generation, "Indexed scientific generation mismatch."
        )
        view.validate_source_identity()
        provenance = dict(view.provenance)
    identity = {
        **provenance,
        "growth_generation": generation,
        "generation_mapping": "growth-equals-scientific-from-1",
        "producer_sha256": object_digest(header["producer"]),
        "generation_descriptor_sha256": file_digest(path),
        "catalog_sha256": file_digest(path.parent.parent / "manifest.json"),
        "checkpoint_sha256": artifact_digest(checkpoint),
    }

    return identity, canonical_validation_evidence(
        identity=identity,
        records=expected_records,
        validator=validator,
        reused_from=reused_from,
    )


def validate_closed_export(
    path: Path,
    *,
    staging: Path,
    generation: int,
    producer: Mapping[str, str],
    checkpoint: Path,
    config_hash: str,
    work_dir: Path,
    parent: dict[str, Any] | None,
    validated: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt the closed layout to the same seal, inventory and commit protocol.

    A trusted writer can transfer its completed canonical checks in this process.
    Exact descriptor and byte hashes must still match; mutable rows/topology are
    always checked in a fresh existing index. Absent transfer means full decoding.
    No historical scientific descriptor or shard belongs to this closure.
    """
    del work_dir, parent  # Shared adapter call shape; no historical closure here.

    from contextlib import nullcontext

    from .scientific_validation_evidence import _validated_record_scope
    from .sharded_training_reader import read_closed_manifest

    require(
        path == staging / "scientific_snapshot/generation.json",
        "Closed export escaped its private layout.",
    )
    header = read_closed_manifest(path)
    require(
        header["generation"] == generation, "Growth/scientific generation mismatch."
    )
    require(header["producer"] == _producer(producer), "Scientific producer changed.")
    require(
        producer.get("config_sha256") == config_hash,
        "Scientific producer/config identity mismatch.",
    )
    require(
        {p.name for p in path.parent.iterdir()}
        == {
            "generation.json",
            header["record_shard"]["path"],
            header["update_shard"]["path"],
        },
        "Unexpected closed snapshot dependency.",
    )
    if validated:
        require(
            validated
            == {
                "manifest_sha256": file_digest(path),
                "producer": header["producer"],
                "record_shard": header["record_shard"],
            },
            "Closed writer validation identity changed.",
        )
    with (
        _validated_record_scope(path, header["producer"], [header["record_shard"]])
        if validated
        else nullcontext(),
        tempfile.TemporaryDirectory(prefix=".validation-index-", dir=staging) as cache,
        private_generation_validation(staging),
        ShardedTrainingGenerationView(
            path, cache_dir=cache, allow_closed_snapshot=True
        ) as view,
    ):
        require(not view.index_reused, "Publication validation requires a fresh index.")
        view.validate_source_identity()
        provenance = dict(view.provenance)
    identity = {
        **provenance,
        "growth_generation": generation,
        "generation_mapping": "growth-equals-scientific-from-1",
        "producer_sha256": object_digest(header["producer"]),
        "generation_descriptor_sha256": file_digest(path),
        "checkpoint_sha256": artifact_digest(checkpoint),
    }
    return identity, canonical_validation_evidence(
        identity=identity,
        records=[header["record_shard"]],
        validator=validator_identity(),
        reused_from=None,
    )
