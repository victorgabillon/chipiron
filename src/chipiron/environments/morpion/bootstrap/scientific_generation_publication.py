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
) -> dict[str, Any]:
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
    with (
        tempfile.TemporaryDirectory(prefix=".validation-index-", dir=staging) as cache,
        private_generation_validation(staging),
        ShardedTrainingGenerationView(
            path, cache_dir=cache, allow_scientific_v2=True
        ) as view,
    ):
        require(
            view.generation == generation, "Indexed scientific generation mismatch."
        )
        view.validate_source_identity()
        provenance = dict(view.provenance)
    return {
        **provenance,
        "growth_generation": generation,
        "generation_mapping": "growth-equals-scientific-from-1",
        "producer_sha256": object_digest(header["producer"]),
        "generation_descriptor_sha256": file_digest(path),
        "catalog_sha256": file_digest(path.parent.parent / "manifest.json"),
        "checkpoint_sha256": artifact_digest(checkpoint),
    }
