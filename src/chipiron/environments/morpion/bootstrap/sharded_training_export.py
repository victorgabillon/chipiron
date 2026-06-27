"""Additive sharded Morpion training-export persistence helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast

from anemone._best_effort import safe_getattr as _safe_getattr
from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    DeltaCheckpointStatePayload,
    checkpoint_payload_to_jsonable,
    payload_for_node_id_or_none,
)
from anemone.checkpoints.state_handles import CheckpointBackedStateHandle
from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot
from anemone.training_export.builders import build_training_node_snapshot
from anemone.training_export.model import (
    TRAINING_TREE_SNAPSHOT_FORMAT_KIND,
    TRAINING_TREE_SNAPSHOT_FORMAT_VERSION,
)

from .pipeline_memory import current_rss_mb, format_metric, log_pipeline_memory

MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND = "morpion_sharded_training_export"
MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION = 1
LOGGER = logging.getLogger(__name__)


def _invalid_int_like_json_field_error(field_name: str) -> TypeError:
    """Return the stable invalid int-like JSON-field error."""
    return TypeError(f"{field_name} must be int-like")


def _required_int(value: object, *, field_name: str) -> int:
    """Return one required int-like JSON field or raise."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise _invalid_int_like_json_field_error(field_name)
    return int(value)


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
class MorpionShardedTrainingExportStats:
    """Compact counters for one sharded training-export write."""

    generation: int
    node_count: int
    new_node_count: int
    rows_written: int = 0
    shards_written: int = 0
    bytes_written: int = 0
    row_build_s: float = 0.0
    json_encode_s: float = 0.0
    write_s: float = 0.0
    total_s: float = 0.0
    rss_before_mb: float | None = None
    rss_after_mb: float | None = None

    @property
    def reused_node_count(self) -> int:
        """Return how many nodes reused existing immutable state payloads."""
        return self.node_count - self.new_node_count


@dataclass(frozen=True, slots=True)
class _JsonWriteStats:
    """Small write timing payload for one JSON artifact."""

    bytes_written: int
    json_encode_s: float
    write_s: float


@dataclass(frozen=True, slots=True)
class MorpionShardedTrainingNodeRecord:
    """Immutable node data recorded only once at node creation generation.

    ``parent_ids`` are treated as stable ancestry links for the current sharded
    Morpion export. ``child_ids`` are intentionally excluded here because the
    live search can open new branches under an existing node across later
    generations.
    """

    node_id: str
    parent_ids: tuple[str, ...]
    depth: int
    creation_generation: int
    state_ref_payload: object | None


@dataclass(frozen=True, slots=True)
class MorpionShardedTrainingNodeUpdate:
    """Mutable per-generation node data for one exported generation.

    ``child_ids`` live here because opened-child topology can change between
    generations even when the node itself is old enough to reuse its immutable
    state payload.
    """

    node_id: str
    order_index: int
    child_ids: tuple[str, ...]
    direct_value_scalar: float | None
    backed_up_value_scalar: float | None
    is_terminal: bool
    is_exact: bool
    over_event_label: str | None
    visit_count: int | None
    metadata: dict[str, object] = field(default_factory=_empty_metadata)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from anemone.training_export.builders import (
        StateRefDumper,
        TrainingExportProfiler,
        ValueScalarExtractor,
    )


def save_morpion_sharded_training_tree_from_live_nodes(
    *,
    nodes: Sequence[object],
    root_node_id: str,
    output_dir: str | Path,
    generation: int,
    state_ref_dumper: StateRefDumper,
    direct_value_extractor: ValueScalarExtractor | None = None,
    backed_up_value_extractor: ValueScalarExtractor | None = None,
    profile: TrainingExportProfiler | None = None,
) -> tuple[Path, MorpionShardedTrainingExportStats]:
    """Persist one additive sharded training export from live ordered nodes."""
    total_started_at = perf_counter()
    rss_before_mb = current_rss_mb()
    root = Path(output_dir)
    log_pipeline_memory(
        stage="growth",
        generation=generation,
        event="before_sharded_snapshot_save",
        node_count=len(nodes),
    )
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
    row_build_started_at = perf_counter()
    for order_index, node in enumerate(nodes):
        node_id = _node_id_without_state(node)
        node_is_new = node_id is None or node_id not in node_index
        update_snapshot = build_training_node_snapshot(
            node,
            state_ref_dumper=None,
            direct_value_extractor=direct_value_extractor,
            backed_up_value_extractor=backed_up_value_extractor,
            _profile=profile,
        )
        node_updates.append(
            MorpionShardedTrainingNodeUpdate(
                node_id=update_snapshot.node_id,
                order_index=order_index,
                child_ids=update_snapshot.child_ids,
                direct_value_scalar=update_snapshot.direct_value_scalar,
                backed_up_value_scalar=update_snapshot.backed_up_value_scalar,
                is_terminal=update_snapshot.is_terminal,
                is_exact=update_snapshot.is_exact,
                over_event_label=update_snapshot.over_event_label,
                visit_count=update_snapshot.visit_count,
                metadata=dict(update_snapshot.metadata),
            )
        )
        if not node_is_new:
            continue
        state_ref_payload = _state_ref_payload_without_resolving(
            node,
            fallback_state_ref_dumper=state_ref_dumper,
            profile=profile,
        )
        node_records.append(
            MorpionShardedTrainingNodeRecord(
                node_id=update_snapshot.node_id,
                parent_ids=update_snapshot.parent_ids,
                depth=update_snapshot.depth,
                creation_generation=generation,
                state_ref_payload=None
                if state_ref_payload is None
                else checkpoint_payload_to_jsonable(state_ref_payload),
            )
        )
        node_index[update_snapshot.node_id] = generation
    row_build_s = perf_counter() - row_build_started_at

    node_shard_path = node_shards_dir / f"generation_{generation:06d}.json"
    update_shard_path = update_shards_dir / f"generation_{generation:06d}.json"
    json_write_stats: list[_JsonWriteStats] = []
    json_write_stats.append(
        _write_json(
            node_shard_path,
            {
                "format_kind": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
                "format_version": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
                "generation": generation,
                "nodes": [_node_record_to_dict(record) for record in node_records],
            },
        )
    )
    json_write_stats.append(
        _write_json(
            update_shard_path,
            {
                "format_kind": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
                "format_version": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
                "generation": generation,
                "updates": [_node_update_to_dict(update) for update in node_updates],
            },
        )
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
    json_write_stats.append(
        _write_json(
            generation_manifest_path,
            _generation_manifest_to_dict(generation_manifest),
        )
    )

    manifest.generation_manifests[str(generation)] = generation_manifest_path.name
    updated_manifest = MorpionShardedTrainingExportManifest(
        latest_generation=generation,
        generation_manifests=dict(manifest.generation_manifests),
        node_index_path=manifest.node_index_path,
        metadata=dict(manifest.metadata),
    )
    json_write_stats.append(
        _write_json(manifest_path, _root_manifest_to_dict(updated_manifest))
    )
    json_write_stats.append(
        _write_json(
            node_index_path,
            {
                "format_kind": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_KIND,
                "format_version": MORPION_SHARDED_TRAINING_EXPORT_FORMAT_VERSION,
                "node_id_to_creation_generation": node_index,
            },
        )
    )
    bytes_written = sum(item.bytes_written for item in json_write_stats)
    json_encode_s = sum(item.json_encode_s for item in json_write_stats)
    write_s = sum(item.write_s for item in json_write_stats)
    total_s = perf_counter() - total_started_at
    rss_after_mb = current_rss_mb()
    log_pipeline_memory(
        stage="growth",
        generation=generation,
        event="after_sharded_snapshot_save",
        node_count=len(node_updates),
        new_node_count=len(node_records),
    )
    stats = MorpionShardedTrainingExportStats(
        generation=generation,
        node_count=len(node_updates),
        new_node_count=len(node_records),
        rows_written=len(node_updates) + len(node_records),
        shards_written=len(json_write_stats),
        bytes_written=bytes_written,
        row_build_s=row_build_s,
        json_encode_s=json_encode_s,
        write_s=write_s,
        total_s=total_s,
        rss_before_mb=rss_before_mb,
        rss_after_mb=rss_after_mb,
    )
    _log_sharded_export_profile(stats)
    return generation_manifest_path, stats


def _node_id_without_state(node: object) -> str | None:
    """Return a node id without touching lazy state, or ``None`` if unavailable."""
    node_id = _safe_getattr(node, "node_id")
    if node_id is not None:
        return str(node_id)
    generic_id = _safe_getattr(node, "id")
    if generic_id is not None:
        return str(generic_id)
    return None


def _state_ref_payload_without_resolving(
    node: object,
    *,
    fallback_state_ref_dumper: StateRefDumper,
    profile: TrainingExportProfiler | None,
) -> object | None:
    """Return a Morpion state-ref payload without resolving checkpoint handles."""
    if profile is not None:
        profile.observe_state_handle(node)
    started_at = perf_counter()
    checkpoint_payload = _checkpoint_state_ref_payload_without_resolving(node)
    conversion_elapsed_s = perf_counter() - started_at
    if checkpoint_payload is not None:
        if profile is not None:
            profile.record_state_ref_conversion(conversion_elapsed_s)
        return checkpoint_payload

    state_started_at = perf_counter()
    state = _safe_getattr(node, "state")
    state_access_elapsed_s = perf_counter() - state_started_at
    if profile is not None:
        profile.record_state_access(
            state_access_elapsed_s,
            state_present=state is not None,
        )
    if state is None:
        return None

    conversion_started_at = perf_counter()
    state_ref_payload = fallback_state_ref_dumper(state)
    if profile is not None:
        profile.record_state_ref_conversion(perf_counter() - conversion_started_at)
    return state_ref_payload


def _checkpoint_state_ref_payload_without_resolving(node: object) -> object | None:
    """Return the training state-ref payload from a checkpoint handle, if possible."""
    handle = _raw_checkpoint_backed_state_handle(node)
    if handle is None:
        return None
    payload = handle.checkpoint_payload_for_reuse_or_none()
    if isinstance(payload, AnchorCheckpointStatePayload):
        return payload.anchor_ref
    if isinstance(payload, DeltaCheckpointStatePayload):
        return _morpion_anchor_ref_from_delta_chain(handle, payload)
    return None


def _raw_checkpoint_backed_state_handle(
    node: object,
) -> CheckpointBackedStateHandle | None:
    """Return a checkpoint-backed handle exposed by a node without resolving state."""
    handle = _safe_getattr(node, "state_handle")
    if isinstance(handle, CheckpointBackedStateHandle):
        return handle
    tree_node = _safe_getattr(node, "tree_node")
    if tree_node is None:
        return None
    tree_handle = _safe_getattr(tree_node, "state_handle")
    if isinstance(tree_handle, CheckpointBackedStateHandle):
        return tree_handle
    return None


def _morpion_anchor_ref_from_delta_chain(
    handle: CheckpointBackedStateHandle,
    payload: DeltaCheckpointStatePayload,
) -> object | None:
    """Flatten a Morpion checkpoint delta chain into the anchor training schema."""
    delta_refs: list[object] = []
    current_node_id = handle.node_id
    current_payload: object = payload
    seen_node_ids: set[int] = set()
    while isinstance(current_payload, DeltaCheckpointStatePayload):
        if current_node_id in seen_node_ids:
            return None
        seen_node_ids.add(current_node_id)
        delta_refs.append(current_payload.delta_ref)
        current_node_id = current_payload.state_parent_node_id
        current_payload = payload_for_node_id_or_none(
            handle.resolver,
            current_node_id,
        )
        if current_payload is None:
            return None

    if not isinstance(current_payload, AnchorCheckpointStatePayload):
        return None
    return _append_morpion_delta_refs_to_anchor_ref(
        current_payload.anchor_ref,
        tuple(reversed(delta_refs)),
    )


def _append_morpion_delta_refs_to_anchor_ref(
    anchor_ref: object,
    delta_refs: tuple[object, ...],
) -> object | None:
    """Return a compact Morpion anchor ref extended with checkpoint delta refs."""
    if not isinstance(anchor_ref, tuple | list) or len(anchor_ref) != 2:
        return None
    variant_code = anchor_ref[0]
    played_moves = anchor_ref[1]
    if not isinstance(variant_code, int) or isinstance(variant_code, bool):
        return None
    if not isinstance(played_moves, tuple | list):
        return None
    if any(not isinstance(move, int) or isinstance(move, bool) for move in played_moves):
        return None
    if any(not isinstance(move, int) or isinstance(move, bool) for move in delta_refs):
        return None
    return (variant_code, (*played_moves, *delta_refs))


def _log_sharded_export_profile(stats: MorpionShardedTrainingExportStats) -> None:
    """Emit one grep-friendly C6 profile line for sharded tree export."""
    LOGGER.info(
        "[tree-export-profile] generation=%s node_count=%s rows_written=%s "
        "new_node_count=%s reused_node_count=%s shards_written=%s bytes_written=%s",
        stats.generation,
        stats.node_count,
        stats.rows_written,
        stats.new_node_count,
        stats.reused_node_count,
        stats.shards_written,
        stats.bytes_written,
    )
    LOGGER.info(
        "[tree-export-timing] generation=%s row_build_s=%.6f json_encode_s=%.6f "
        "write_s=%.6f total_s=%.6f",
        stats.generation,
        stats.row_build_s,
        stats.json_encode_s,
        stats.write_s,
        stats.total_s,
    )
    rss_delta_mb = (
        None
        if stats.rss_before_mb is None or stats.rss_after_mb is None
        else stats.rss_after_mb - stats.rss_before_mb
    )
    LOGGER.info(
        "[tree-export-memory] generation=%s rss_before_mb=%s rss_after_mb=%s "
        "rss_delta_mb=%s",
        stats.generation,
        format_metric(stats.rss_before_mb),
        format_metric(stats.rss_after_mb),
        format_metric(rss_delta_mb),
    )


def load_morpion_sharded_training_tree_snapshot(
    generation_manifest_path: str | Path,
) -> TrainingTreeSnapshot:
    """Load one sharded export generation into a training-tree snapshot."""
    manifest_path = Path(generation_manifest_path)
    root = manifest_path.parent
    generation_manifest = _load_generation_manifest(manifest_path)
    log_pipeline_memory(
        stage="dataset",
        generation=generation_manifest.generation,
        event="before_sharded_snapshot_load",
        tree_snapshot_path=manifest_path,
    )
    root_manifest = _load_root_manifest(root / "manifest.json")

    node_records_by_id: dict[str, MorpionShardedTrainingNodeRecord] = {}
    for shard_generation in range(1, generation_manifest.generation + 1):
        relative_manifest_path = root_manifest.generation_manifests.get(
            str(shard_generation)
        )
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
    log_pipeline_memory(
        stage="dataset",
        generation=generation_manifest.generation,
        event="after_sharded_snapshot_load",
        node_count=len(snapshots),
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
    tree_value_scalar = update.backed_up_value_scalar
    effective_value_scalar = (
        tree_value_scalar
        if tree_value_scalar is not None
        else update.direct_value_scalar
    )
    effective_value_source = (
        "tree_value" if tree_value_scalar is not None else "direct_value"
    )
    target_value_scalar = (
        tree_value_scalar
        if tree_value_scalar is not None
        else update.direct_value_scalar
    )
    return TrainingNodeSnapshot(
        node_id=record.node_id,
        parent_ids=record.parent_ids,
        child_ids=update.child_ids,
        depth=record.depth,
        state_ref_payload=None
        if record.state_ref_payload is None
        else checkpoint_payload_to_jsonable(record.state_ref_payload),
        direct_value_scalar=update.direct_value_scalar,
        tree_value_scalar=tree_value_scalar,
        effective_value_scalar=effective_value_scalar,
        effective_value_source=effective_value_source,
        target_value_scalar=target_value_scalar,
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
        "depth": record.depth,
        "creation_generation": record.creation_generation,
        "state_ref_payload": None
        if record.state_ref_payload is None
        else checkpoint_payload_to_jsonable(record.state_ref_payload),
    }


def _node_update_to_dict(update: MorpionShardedTrainingNodeUpdate) -> dict[str, object]:
    """Serialize one mutable node update."""
    return {
        "node_id": update.node_id,
        "order_index": update.order_index,
        "child_ids": list(update.child_ids),
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
        parent_ids=tuple(
            str(parent_id)
            for parent_id in cast("list[object]", payload.get("parent_ids", []))
        ),
        depth=_required_int(payload.get("depth", 0), field_name="depth"),
        creation_generation=_required_int(
            payload.get("creation_generation", 0),
            field_name="creation_generation",
        ),
        state_ref_payload=None
        if raw_state_ref_payload is None
        else checkpoint_payload_to_jsonable(raw_state_ref_payload),
    )


def _node_update_from_dict(value: object) -> MorpionShardedTrainingNodeUpdate:
    """Deserialize one mutable node update."""
    payload = cast("dict[str, object]", value)
    return MorpionShardedTrainingNodeUpdate(
        node_id=str(payload["node_id"]),
        order_index=_required_int(
            payload.get("order_index", 0), field_name="order_index"
        ),
        child_ids=tuple(
            str(child_id)
            for child_id in cast("list[object]", payload.get("child_ids", []))
        ),
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
        latest_generation=_required_int(
            payload.get("latest_generation", 0),
            field_name="latest_generation",
        ),
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
        generation=_required_int(payload["generation"], field_name="generation"),
        root_node_id=str(payload["root_node_id"]),
        node_count=_required_int(payload.get("node_count", 0), field_name="node_count"),
        new_node_count=_required_int(
            payload.get("new_node_count", 0),
            field_name="new_node_count",
        ),
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
    return {
        node_id: _required_int(
            creation_generation,
            field_name="node_id_to_creation_generation",
        )
        for node_id, creation_generation in raw_mapping.items()
    }


def _write_json(path: Path, payload: dict[str, object]) -> _JsonWriteStats:
    """Persist one JSON artifact with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encode_started_at = perf_counter()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    json_encode_s = perf_counter() - encode_started_at
    write_started_at = perf_counter()
    path.write_text(text, encoding="utf-8")
    write_s = perf_counter() - write_started_at
    return _JsonWriteStats(
        bytes_written=path.stat().st_size,
        json_encode_s=json_encode_s,
        write_s=write_s,
    )


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
    "MorpionShardedTrainingExportStats",
    "MorpionShardedTrainingGenerationManifest",
    "MorpionShardedTrainingNodeRecord",
    "MorpionShardedTrainingNodeUpdate",
    "load_morpion_sharded_training_tree_snapshot",
    "save_morpion_sharded_training_tree_from_live_nodes",
]
