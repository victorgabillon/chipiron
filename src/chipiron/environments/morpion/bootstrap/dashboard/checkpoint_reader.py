"""Read persisted flat or split checkpoints for inspection without restoring search."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import zstandard
from anemone.checkpoints import (
    AlgorithmNodeCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    TreeCheckpointPayload,
    load_checkpoint_json_payload,
    load_sharded_search_checkpoint,
    read_sharded_checkpoint_manifest,
)
from dacite import Config, from_dict

from chipiron.environments.morpion.bootstrap.runtime.checkpoint_codec import (
    InvalidMorpionSearchCheckpointError,
    _normalize_algorithm_node_payload_for_dacite,
    load_morpion_search_checkpoint_payload,
)

if TYPE_CHECKING:
    from pathlib import Path


def checkpoint_exists(path: Path) -> bool:
    """Recognize a complete manifest-backed directory as well as a flat file."""
    return path.is_file() or (path.is_dir() and (path / "manifest.json").is_file())


def read_inspection_checkpoint(path: Path) -> SearchRuntimeCheckpointPayload:
    """Decode only persisted observations; never construct or advance a search runner."""
    if path.is_file():
        return load_morpion_search_checkpoint_payload(path)
    try:
        return _read_shards(path)
    except Exception as exc:
        raise InvalidMorpionSearchCheckpointError(
            path, f"invalid inspection shards: {exc}"
        ) from exc


def _read_shards(path: Path) -> SearchRuntimeCheckpointPayload:
    """Join the persisted split node records into the existing inspector DTOs."""
    manifest = read_sharded_checkpoint_manifest(path / "manifest.json")
    if any(shard.kind == "node_records" for shard in manifest.shards):
        return load_sharded_search_checkpoint(path)
    nodes: dict[int, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    seen: dict[str, set[int]] = {}
    for shard in manifest.shards:
        if shard.kind not in {
            "metadata",
            "node_shells",
            "state_payloads",
            "node_runtime",
        }:
            continue
        source = (path / shard.path).resolve()
        if not source.is_relative_to(path.resolve()):
            message = "Checkpoint shard escapes its directory."
            raise ValueError(message)
        if shard.kind == "metadata":
            raw_metadata, _stats = load_checkpoint_json_payload(source)
            if not isinstance(raw_metadata, dict):
                message = "Checkpoint metadata must be an object."
                raise ValueError(message)
            metadata = cast("dict[str, Any]", raw_metadata)
            continue
        ids = seen.setdefault(shard.kind, set())
        count = 0
        with (
            zstandard.open(source, "rt")
            if shard.encoding == "jsonl_zst"
            else source.open(encoding="utf-8") as stream
        ):
            for line in stream:
                row = json.loads(line)
                node_id = row["node_id"]
                if node_id in ids:
                    message = "Duplicate node in checkpoint shard."
                    raise ValueError(message)
                ids.add(node_id)
                nodes.setdefault(node_id, {}).update(row)
                count += 1
        if count != shard.record_count:
            message = "Checkpoint shard record count differs from its manifest."
            raise ValueError(message)
    if len(nodes) != manifest.total_node_count or any(
        seen.get(kind) != set(nodes)
        for kind in ("node_shells", "state_payloads", "node_runtime")
    ):
        message = "Checkpoint split shards have inconsistent node identities."
        raise ValueError(message)
    typed_nodes = [
        from_dict(
            AlgorithmNodeCheckpointPayload,
            cast("dict[str, Any]", _normalize_algorithm_node_payload_for_dacite(row)),
            config=Config(cast=[tuple], check_types=False),
        )
        for row in nodes.values()
    ]
    # Selector/RNG state is unnecessary for inspecting nodes. Nothing is resumed.
    return SearchRuntimeCheckpointPayload(
        format_version=metadata["format_version"],
        evaluator_version=metadata["evaluator_version"],
        rng_state=None,
        tree=TreeCheckpointPayload(
            root_node_id=metadata["root_node_id"], nodes=typed_nodes
        ),
    )
