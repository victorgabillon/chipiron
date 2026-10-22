"""Offline closed scientific snapshots; no run-state commit or pipeline routing.

The caller freezes the graph and never recycles node IDs for different states.
A committed source is pinned and verified for the entire copy. Only new states
are encoded; every mutable value and edge is regenerated and validated. The
runtime graph is caller-owned; auxiliary working memory is bounded by a row,
parser buffers and the existing SQLite cache (TEMP membership/topology on disk).
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from anemone.checkpoints import checkpoint_payload_to_jsonable
from anemone.node_evaluation.common.value_snapshot import NodeTargetSource
from anemone.training_export.builders import build_training_node_snapshot
from anemone.training_export.serialization import training_tree_snapshot_from_dict

from chipiron.environments.morpion.types import MorpionState

from .generation_publication_io import (
    assert_artifact_writable,
    committed_scientific_source,
    generation_read_pin,
    publish_directory_if_absent,
    sync_artifact,
)
from .scientific_validation_evidence import (
    _validated_record_scope,
    committed_source_validation,
)
from .sharded_training_reader import (
    ShardedTrainingGenerationView,
    iter_shard_rows,
    read_closed_manifest,
)
from .sharded_training_v2 import (
    _CODEC,
    ScientificGeneration,
    _descriptor,
    _encode,
    _integer,
    _producer,
    _require,
    _sha,
    _snapshot_dict,
    _validate_topology,
    _validate_update,
    _value_contract,
    _write,
    value_score,
)
from .sharded_training_v2_contract import (
    CLOSED_KIND,
    CLOSED_LAYOUT,
    CONTRACT,
    VERSION,
    closed_envelope,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def save_closed_scientific_snapshot(
    *,
    nodes: Sequence[object],
    root_node_id: str,
    destination: Path,
    generation: int,
    producer: Mapping[str, str],
    committed_source: Path | None = None,
    target_source: NodeTargetSource = NodeTargetSource.TREE_VALUE,
    metadata: Mapping[str, object] | None = None,
    created_at_unix_s: float | None = None,
    _validation_result: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create a new standalone directory atomically, without committing a run.

    No source means genesis (generation 1). Incremental copying requires the
    exact scientific head dependency of a committed v2 or closed capsule.
    The private result sink transfers completed byte checks to the existing
    publisher within this process; it is never a persistent cache or authority.
    Identical inputs produce identical payload bytes; a destination is never
    replaced. On interruption any private staging directory is non-authoritative.
    """
    started = perf_counter()
    identity = _producer(producer)
    _require(_integer(generation) and generation >= 1, "invalid generation")
    _require(
        isinstance(target_source, NodeTargetSource), "explicit target source required"
    )
    _require(
        committed_source is not None or generation == 1,
        "non-genesis requires committed source",
    )
    destination = destination.absolute()
    assert_artifact_writable(destination)
    _require(not destination.exists(), "snapshot destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_metadata: dict[str, object] = {
        "format_kind": "training_tree_snapshot",
        "format_version": 1,
        "default_target_source": target_source.value,
    }
    metadata = metadata or {}
    _require(
        all(
            key not in metadata or metadata[key] == value
            for key, value in snapshot_metadata.items()
        ),
        "metadata contradicts scientific contract",
    )
    snapshot_metadata.update(metadata or {})
    timings: dict[str, Any] = {
        "new_state_encodings": 0,
        "copied_records": 0,
        "state_encoding_s": 0.0,
        "state_copy_s": 0.0,
        "updates_s": 0.0,
    }
    staging = Path(tempfile.mkdtemp(prefix=".closed-pending-", dir=destination.parent))
    try:
        with ExitStack() as stack:
            # Disposable indexes only, never a new scientific proof cache.
            cache = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix=".closed-validation-", dir=destination.parent
                    )
                )
            )
            source = None
            connection = None
            verification = perf_counter()
            if committed_source is not None:
                committed_source = committed_source.resolve()
                stack.enter_context(generation_read_pin(committed_source))
                _require(
                    committed_scientific_source(committed_source) is not None,
                    "copy source must be an actual committed scientific dependency",
                )
                stack.enter_context(committed_source_validation(committed_source))
                source = stack.enter_context(
                    ShardedTrainingGenerationView(
                        committed_source,
                        cache_dir=cache / "source",
                        allow_scientific_v2=True,
                        allow_closed_snapshot=True,
                    )
                )
                _require(
                    source.provenance["producer"] == identity,
                    "source producer/codec identity changed",
                )
                _require(
                    generation == source.generation + 1,
                    "non-consecutive source generation",
                )
                connection = stack.enter_context(source.temporary_workspace())
                connection.execute(
                    "CREATE TEMP TABLE closed_current (id TEXT PRIMARY KEY)"
                )
            timings["source_verification_s"] = perf_counter() - verification
            records_digest = hashlib.sha256()
            updates_digest = hashlib.sha256()
            records_path = staging / "records.jsonl"
            with (
                records_path.open("wb") as records,
                (staging / "updates.jsonl").open("w", encoding="utf-8") as updates,
            ):

                def write_record(raw: bytes) -> None:
                    records.write(raw)
                    records_digest.update(raw)

                write_record(
                    _encode(closed_envelope("records", generation=generation)).encode()
                )
                update_header = _encode(
                    closed_envelope("updates", generation=generation)
                )
                updates.write(update_header)
                updates_digest.update(update_header.encode())
                for order, node in enumerate(nodes):
                    update_started = perf_counter()
                    snapshot = build_training_node_snapshot(
                        node,
                        state_ref_dumper=None,
                        direct_value_extractor=value_score,
                        tree_value_extractor=value_score,
                        effective_value_extractor=value_score,
                        target_value_extractor=value_score,
                        target_source=target_source,
                    )
                    row = _snapshot_dict(snapshot)
                    del row["state_ref_payload"]
                    row.update(
                        order_index=order,
                        value_contract=_value_contract(node, snapshot, target_source),
                    )
                    _validate_update(row)
                    encoded_update = _encode(row)
                    updates.write(encoded_update)
                    updates_digest.update(encoded_update.encode())
                    timings["updates_s"] += perf_counter() - update_started
                    if (
                        connection is not None
                        and connection.execute(
                            "SELECT 1 FROM records WHERE node_id=?", (snapshot.node_id,)
                        ).fetchone()
                    ):
                        connection.execute(
                            "INSERT INTO closed_current VALUES (?)", (snapshot.node_id,)
                        )
                        continue
                    canonical_started = perf_counter()
                    state = getattr(node, "state")  # noqa: B009
                    if isinstance(state, MorpionState):
                        state = state.to_atomheart_state()
                    ref = checkpoint_payload_to_jsonable(_CODEC.dump_state_ref(state))
                    _require(
                        ref is not None
                        and checkpoint_payload_to_jsonable(
                            _CODEC.dump_state_ref(_CODEC.load_state_ref(ref))
                        )
                        == ref,
                        "noncanonical new state reference",
                    )
                    write_record(
                        _encode({
                            "node_id": snapshot.node_id,
                            "creation_generation": generation,
                            "state_ref_payload": ref,
                        }).encode()
                    )
                    timings["new_state_encodings"] += 1
                    timings["state_encoding_s"] += perf_counter() - canonical_started
                if source is not None:
                    assert connection is not None
                    copying = perf_counter()
                    for node_id, raw in source.iter_record_bytes():
                        if connection.execute(
                            "SELECT 1 FROM closed_current WHERE id=?", (node_id,)
                        ).fetchone():
                            write_record(raw)
                            timings["copied_records"] += 1
                    _require(
                        timings["copied_records"]
                        == connection.execute(
                            "SELECT COUNT(*) FROM closed_current"
                        ).fetchone()[0],
                        "incomplete source copy",
                    )
                    connection.execute("DROP TABLE closed_current")
                    timings["state_copy_s"] = perf_counter() - copying
            validation = perf_counter()
            header = closed_envelope(
                "generation",
                generation=generation,
                snapshot={
                    "root_node_id": root_node_id,
                    "node_count": len(nodes),
                    "created_at_unix_s": created_at_unix_s,
                    "metadata": snapshot_metadata,
                },
                producer=identity,
                record_shard=_descriptor(records_path, staging, generation, len(nodes)),
                update_shard=_descriptor(
                    staging / "updates.jsonl", staging, generation, len(nodes)
                ),
            )
            _require(
                header["record_shard"]["sha256"] == records_digest.hexdigest(),
                "written records differ from validated bytes",
            )
            _require(
                header["update_shard"]["sha256"] == updates_digest.hexdigest(),
                "written updates differ from regenerated bytes",
            )
            manifest = staging / "generation.json"
            _write(manifest, header)
            # All records were either freshly codec-validated or copied byte for
            # byte from a pinned, freshly hashed, validated scientific source.
            # Mutable rows and topology still undergo the complete index checks.
            with (
                _validated_record_scope(manifest, identity, [header["record_shard"]]),
                ShardedTrainingGenerationView(
                    manifest, cache_dir=cache / "output", allow_closed_snapshot=True
                ) as view,
            ):
                _require(view.node_count() == len(nodes), "incomplete closed snapshot")
                view.validate_source_identity()
            if source is not None:
                source.validate_source_identity()
            timings["validation_hashing_s"] = perf_counter() - validation
            sync_artifact(staging, destination.parent)
            publish_directory_if_absent(staging, destination, destination.parent)
            if _validation_result is not None:
                _validation_result.update(
                    manifest_sha256=_sha(destination / "generation.json"),
                    producer=identity,
                    record_shard=header["record_shard"],
                )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    timings.update(
        total_s=perf_counter() - started,
        node_count=len(nodes),
        bytes_written=sum(p.stat().st_size for p in destination.iterdir()),
    )
    return destination / "generation.json", timings


def load_closed_scientific_snapshot(path: Path) -> ScientificGeneration:
    """Full independent oracle; never used by the indexed or writer paths."""
    with generation_read_pin(path):
        manifest_hash = _sha(path)
        header = read_closed_manifest(path)
        generation = header["generation"]
        records: dict[str, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        contracts: dict[str, dict[str, Any]] = {}
        for role, key in (("records", "record_shard"), ("updates", "update_shard")):
            descriptor = header[key]
            shard = path.parent / descriptor["path"]
            _require(_sha(shard) == descriptor["sha256"], "closed shard hash mismatch")
            count = 0
            for parsed in iter_shard_rows(
                shard, role, scientific_generation=generation, closed=True
            ):
                row = cast("dict[str, Any]", parsed)
                if role == "records":
                    _require(
                        set(row)
                        == {"node_id", "creation_generation", "state_ref_payload"},
                        "invalid immutable record",
                    )
                    identifier = row["node_id"]
                    _require(
                        isinstance(identifier, str)
                        and bool(identifier)
                        and identifier not in records,
                        "invalid/duplicate record",
                    )
                    _require(
                        _integer(row["creation_generation"])
                        and 1 <= row["creation_generation"] <= generation,
                        "invalid creation generation",
                    )
                    ref = row["state_ref_payload"]
                    _require(
                        ref is not None
                        and checkpoint_payload_to_jsonable(
                            _CODEC.dump_state_ref(_CODEC.load_state_ref(ref))
                        )
                        == ref,
                        "noncanonical state reference",
                    )
                    records[identifier] = row
                else:
                    _validate_update(row)
                    identifier = row["node_id"]
                    _require(
                        identifier in records and identifier not in contracts,
                        "missing/duplicate node",
                    )
                    _require(
                        row.pop("order_index") == len(rows), "invalid export order"
                    )
                    contracts[identifier] = row.pop("value_contract")
                    row["state_ref_payload"] = records[identifier]["state_ref_payload"]
                    rows.append(row)
                count += 1
            _require(
                count == descriptor["count"] and _sha(shard) == descriptor["sha256"],
                "closed shard changed/count mismatch",
            )
        info = header["snapshot"]
        _require(info["root_node_id"] in contracts, "missing root")
        _validate_topology({
            row["node_id"]: (row["parent_ids"], row["child_ids"]) for row in rows
        })
        _require(_sha(path) == manifest_hash, "closed manifest changed")
        snapshot = training_tree_snapshot_from_dict({**info, "nodes": rows})
        return ScientificGeneration(
            snapshot,
            contracts,
            {
                "source_format_kind": CLOSED_KIND,
                "source_format_version": VERSION,
                "scientific_contract": CONTRACT,
                "source_generation": generation,
                "producer": header["producer"],
                "storage_layout_version": CLOSED_LAYOUT,
            },
            {key: row["creation_generation"] for key, row in records.items()},
        )
