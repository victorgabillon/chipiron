"""Additive sharded Morpion training-export persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot
from anemone.training_export.builders import build_training_node_snapshot
from anemone.training_export.model import (
    TRAINING_TREE_SNAPSHOT_FORMAT_KIND,
    TRAINING_TREE_SNAPSHOT_FORMAT_VERSION,
)

MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND = "morpion_sharded_training_export"
MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION = 1


def _empty_metadata() -> dict[str, object]:
    """Return a typed empty metadata mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class MorpionShardedTrainingExportManifest:
    """Root manifest for one additive sharded training-export directory."""

    latest_generation: int
    generation_manifests: dict[str, str]
    node_index_path: str = "node_index.json"
    metadata: dict[str, object] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class MorpionShardedTrainingGenerationManifest:
    """Per-generation manifest referencing immutable and mutable shards."""

    generation: int
    root_node_id: str
    node_count: int
    new_node_count: int
    node_shard_path: str
    update_shard_path: str
    metadata: dict[str, object] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class MorpionShardedTrainingNodeRecord:
    """Immutable node data recorded only once at node creation generation."""

    node_id: str
    parent_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    depth: int
    creation_generation: int
    state_ref_payload: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class MorpionShardedTrainingNodeUpdate:
    """Mutable per-generation node data for one exported generation."""

    node_id: str
    order_index: int
    direct_value_scalar: float | None
    backed_up_value_scalar: float | None
    is_terminal: bool
    is_exact: bool
    over_event_label: str | None
    visit_count: int | None
    metadata: dict[str, object] = field(default_factory=_empty_metadata)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from anemone.training_export.builders import StateRefDumper, ValueScalarExtractor


def save_morpion_sharded_training_tree_from_live_nodes(
    *,
    nodes: Sequence[object],
    root_node_id: str,
    output_dir: str | Path,
    generation: int,
    state_ref_dumper: StateRefDumper,
    direct_value_extractor: ValueScalarExtractor | None = None,
    backed_up_value_extractor: ValueScalarExtractor | None = None,
) -> Path:
    """Persist one additive sharded training export from live ordered nodes."""
    root = Path(output_dir)
    node_shards_dir = root / "node_shards"
    update_shards_dir = root / "update_shards"
    node_shards_dir.mkdir(parents=True, exist_ok=True)
    update_shards_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.json"
    node_index_path = root / "node_index.json"
    manifest = _load_root_manifest(manifest_path)
    node_index = _load_node_index(node_index_path)

    node_records: list[MorpionShardedTrainingNodeRecord] = []
    node_updates: list[MorpionShardedTrainingNodeUpdate] = []
    for order_index, node in enumerate(nodes):
        update_snapshot = build_training_node_snapshot(
            node,
            state_ref_dumper=None,
            direct_value_extractor=direct_value_extractor,
            backed_up_value_extractor=backed_up_value_extractor,
        )
        node_updates.append(
            MorpionShardedTrainingNodeUpdate(
                node_id=update_snapshot.node_id,
                order_index=order_index,
                direct_value_scalar=update_snapshot.direct_value_scalar,
                backed_up_value_scalar=update_snapshot.backed_up_value_scalar,
                is_terminal=update_snapshot.is_terminal,
                is_exact=update_snapshot.is_exact,
                over_event_label=update_snapshot.over_event_label,
                visit_count=update_snapshot.visit_count,
                metadata=dict(update_snapshot.metadata),
            )
        )
        if update_snapshot.node_id in node_index:
            continue
        full_snapshot = build_training_node_snapshot(
            node,
            state_ref_dumper=state_ref_dumper,
            direct_value_extractor=direct_value_extractor,
            backed_up_value_extractor=backed_up_value_extractor,
        )
        node_records.append(
            MorpionShardedTrainingNodeRecord(
                node_id=full_snapshot.node_id,
                parent_ids=full_snapshot.parent_ids,
                child_ids=full_snapshot.child_ids,
                depth=full_snapshot.depth,
                creation_generation=generation,
                state_ref_payload=None
                if full_snapshot.state_ref_payload is None
                else dict(cast("dict[str, object]", full_snapshot.state_ref_payload)),
            )
        )
        node_index[full_snapshot.node_id] = generation

    node_shard_path = node_shards_dir / f"generation_{generation:06d}.json"
    update_shard_path = update_shards_dir / f"generation_{generation:06d}.json"
    _write_json(
        node_shard_path,
        {
            "format_kind": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
            "format_version": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
            "generation": generation,
            "nodes": [_node_record_to_dict(record) for record in node_records],
        },
    )
    _write_json(
        update_shard_path,
        {
            "format_kind": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
            "format_version": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
            "generation": generation,
            "updates": [_node_update_to_dict(update) for update in node_updates],
        },
    )

    generation_manifest = MorpionShardedTrainingGenerationManifest(
        generation=generation,
        root_node_id=root_node_id,
        node_count=len(node_updates),
        new_node_count=len(node_records),
        node_shard_path=node_shard_path.relative_to(root).as_posix(),
        update_shard_path=update_shard_path.relative_to(root).as_posix(),
    )
    generation_manifest_path = root / f"generation_{generation:06d}.json"
    _write_json(
        generation_manifest_path,
        _generation_manifest_to_dict(generation_manifest),
    )

    manifest.generation_manifests[str(generation)] = generation_manifest_path.name
    updated_manifest = MorpionShardedTrainingExportManifest(
        latest_generation=generation,
        generation_manifests=dict(manifest.generation_manifests),
        node_index_path=manifest.node_index_path,
        metadata=dict(manifest.metadata),
    )
    _write_json(manifest_path, _root_manifest_to_dict(updated_manifest))
    _write_json(
        node_index_path,
        {
            "format_kind": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
            "format_version": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
            "node_id_to_creation_generation": node_index,
        },
    )
    return generation_manifest_path


def load_morpion_sharded_training_tree_snapshot(
    generation_manifest_path: str | Path,
) -> TrainingTreeSnapshot:
    """Load one sharded export generation into a training-tree snapshot."""
    manifest_path = Path(generation_manifest_path)
    root = manifest_path.parent
    generation_manifest = _load_generation_manifest(manifest_path)
    root_manifest = _load_root_manifest(root / "manifest.json")

    node_records_by_id: dict[str, MorpionShardedTrainingNodeRecord] = {}
    for shard_generation in range(1, generation_manifest.generation + 1):
        relative_manifest_path = root_manifest.generation_manifests.get(str(shard_generation))
        if relative_manifest_path is None:
            continue
        shard_manifest = _load_generation_manifest(root / relative_manifest_path)
        node_shard_payload = _read_json(root / shard_manifest.node_shard_path)
        for raw_record in cast("list[object]", node_shard_payload.get("nodes", [])):
            record = _node_record_from_dict(raw_record)
            node_records_by_id[record.node_id] = record

    update_payload = _read_json(root / generation_manifest.update_shard_path)
    updates = [
        _node_update_from_dict(raw_update)
        for raw_update in cast("list[object]", update_payload.get("updates", []))
    ]
    ordered_nodes = sorted(updates, key=lambda update: update.order_index)
    snapshots = tuple(
        _merge_node_record_and_update(node_records_by_id[update.node_id], update)
        for update in ordered_nodes
    )
    return TrainingTreeSnapshot(
        root_node_id=generation_manifest.root_node_id,
        nodes=snapshots,
        metadata={
            "format_kind": TRAINING_TREE_SNAPSHOT_FORMAT_KIND,
            "format_version": TRAINING_TREE_SNAPSHOT_FORMAT_VERSION,
            "source_format_kind": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
            "source_format_version": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
            "source_generation": generation_manifest.generation,
        },
    )


def _merge_node_record_and_update(
    record: MorpionShardedTrainingNodeRecord,
    update: MorpionShardedTrainingNodeUpdate,
) -> TrainingNodeSnapshot:
    """Merge immutable and mutable shard rows into one training node snapshot."""
    return TrainingNodeSnapshot(
        node_id=record.node_id,
        parent_ids=record.parent_ids,
        child_ids=record.child_ids,
        depth=record.depth,
        state_ref_payload=None
        if record.state_ref_payload is None
        else dict(record.state_ref_payload),
        direct_value_scalar=update.direct_value_scalar,
        backed_up_value_scalar=update.backed_up_value_scalar,
        is_terminal=update.is_terminal,
        is_exact=update.is_exact,
        over_event_label=update.over_event_label,
        visit_count=update.visit_count,
        metadata=dict(update.metadata),
    )


def _root_manifest_to_dict(
    manifest: MorpionShardedTrainingExportManifest,
) -> dict[str, object]:
    """Serialize one root sharded-export manifest."""
    return {
        "format_kind": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
        "format_version": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
        "latest_generation": manifest.latest_generation,
        "generation_manifests": dict(manifest.generation_manifests),
        "node_index_path": manifest.node_index_path,
        "metadata": dict(manifest.metadata),
    }


def _generation_manifest_to_dict(
    manifest: MorpionShardedTrainingGenerationManifest,
) -> dict[str, object]:
    """Serialize one per-generation sharded-export manifest."""
    return {
        "format_kind": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
        "format_version": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
        "generation": manifest.generation,
        "root_node_id": manifest.root_node_id,
        "node_count": manifest.node_count,
        "new_node_count": manifest.new_node_count,
        "node_shard_path": manifest.node_shard_path,
        "update_shard_path": manifest.update_shard_path,
        "metadata": dict(manifest.metadata),
    }


def _node_record_to_dict(record: MorpionShardedTrainingNodeRecord) -> dict[str, object]:
    """Serialize one immutable node record."""
    return {
        "node_id": record.node_id,
        "parent_ids": list(record.parent_ids),
        "child_ids": list(record.child_ids),
        "depth": record.depth,
        "creation_generation": record.creation_generation,
        "state_ref_payload": None
        if record.state_ref_payload is None
        else dict(record.state_ref_payload),
    }


def _node_update_to_dict(update: MorpionShardedTrainingNodeUpdate) -> dict[str, object]:
    """Serialize one mutable node update."""
    return {
        "node_id": update.node_id,
        "order_index": update.order_index,
        "direct_value_scalar": update.direct_value_scalar,
        "backed_up_value_scalar": update.backed_up_value_scalar,
        "is_terminal": update.is_terminal,
        "is_exact": update.is_exact,
        "over_event_label": update.over_event_label,
        "visit_count": update.visit_count,
        "metadata": dict(update.metadata),
    }


def _node_record_from_dict(value: object) -> MorpionShardedTrainingNodeRecord:
    """Deserialize one immutable node record."""
    payload = cast("dict[str, object]", value)
    raw_state_ref_payload = payload.get("state_ref_payload")
    return MorpionShardedTrainingNodeRecord(
        node_id=str(payload["node_id"]),
        parent_ids=tuple(str(parent_id) for parent_id in cast("list[object]", payload.get("parent_ids", []))),
        child_ids=tuple(str(child_id) for child_id in cast("list[object]", payload.get("child_ids", []))),
        depth=int(payload.get("depth", 0)),
        creation_generation=int(payload.get("creation_generation", 0)),
        state_ref_payload=None
        if raw_state_ref_payload is None
        else dict(cast("dict[str, object]", raw_state_ref_payload)),
    )


def _node_update_from_dict(value: object) -> MorpionShardedTrainingNodeUpdate:
    """Deserialize one mutable node update."""
    payload = cast("dict[str, object]", value)
    return MorpionShardedTrainingNodeUpdate(
        node_id=str(payload["node_id"]),
        order_index=int(payload.get("order_index", 0)),
        direct_value_scalar=_optional_float(payload.get("direct_value_scalar")),
        backed_up_value_scalar=_optional_float(payload.get("backed_up_value_scalar")),
        is_terminal=bool(payload.get("is_terminal", False)),
        is_exact=bool(payload.get("is_exact", False)),
        over_event_label=_optional_str(payload.get("over_event_label")),
        visit_count=_optional_int(payload.get("visit_count")),
        metadata=dict(cast("dict[str, object]", payload.get("metadata", {}))),
    )


def _load_root_manifest(path: Path) -> MorpionShardedTrainingExportManifest:
    """Load the root sharded-export manifest or return an empty default."""
    if not path.is_file():
        return MorpionShardedTrainingExportManifest(
            latest_generation=0,
            generation_manifests={},
        )
    payload = _read_json(path)
    return MorpionShardedTrainingExportManifest(
        latest_generation=int(payload.get("latest_generation", 0)),
        generation_manifests=dict(
            cast("dict[str, str]", payload.get("generation_manifests", {}))
        ),
        node_index_path=str(payload.get("node_index_path", "node_index.json")),
        metadata=dict(cast("dict[str, object]", payload.get("metadata", {}))),
    )


def _load_generation_manifest(path: Path) -> MorpionShardedTrainingGenerationManifest:
    """Load one per-generation sharded-export manifest."""
    payload = _read_json(path)
    return MorpionShardedTrainingGenerationManifest(
        generation=int(payload["generation"]),
        root_node_id=str(payload["root_node_id"]),
        node_count=int(payload.get("node_count", 0)),
        new_node_count=int(payload.get("new_node_count", 0)),
        node_shard_path=str(payload["node_shard_path"]),
        update_shard_path=str(payload["update_shard_path"]),
        metadata=dict(cast("dict[str, object]", payload.get("metadata", {}))),
    )


def _load_node_index(path: Path) -> dict[str, int]:
    """Load the persisted node-id index or return an empty mapping."""
    if not path.is_file():
        return {}
    payload = _read_json(path)
    raw_mapping = cast(
        "dict[str, object]",
        payload.get("node_id_to_creation_generation", {}),
    )
    return {node_id: int(creation_generation) for node_id, creation_generation in raw_mapping.items()}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Persist one JSON artifact with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    """Load one JSON object payload from disk."""
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _optional_float(value: object) -> float | None:
    """Return one optional float payload field."""
    if value is None:
        return None
    return float(cast("int | float", value))


def _optional_int(value: object) -> int | None:
    """Return one optional int payload field."""
    if value is None:
        return None
    return int(cast("int | float", value))


def _optional_str(value: object) -> str | None:
    """Return one optional string payload field."""
    if value is None:
        return None
    return str(value)


__all__ = [
    "MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND",
    "MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION",
    "MorpionShardedTrainingExportManifest",
    "MorpionShardedTrainingGenerationManifest",
    "MorpionShardedTrainingNodeRecord",
    "MorpionShardedTrainingNodeUpdate",
    "load_morpion_sharded_training_tree_snapshot",
    "save_morpion_sharded_training_tree_from_live_nodes",
]