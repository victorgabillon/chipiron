"""Explicit, offline scientific v2 export; never selected by pipeline dispatch.

The v1 writer/loader remain untouched. This reference implementation uses a
different envelope and JSONL shards so an old loader cannot silently project
v2 through the lossy v1 merge. It is deliberately not a bounded Dataset reader.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from anemone.checkpoints import checkpoint_payload_to_jsonable, serialize_value
from anemone.node_evaluation.common.canonical_value import validate_value_semantics
from anemone.node_evaluation.common.value_snapshot import (
    NodeTargetSource,
    snapshot_node_values,
)
from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot
from anemone.training_export.builders import build_training_node_snapshot
from anemone.training_export.serialization import (
    training_tree_snapshot_from_dict,
    training_tree_snapshot_to_dict,
)
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec
from valanga.evaluations import Value

from chipiron.environments.morpion.types import MorpionState

from .generation_publication_io import assert_artifact_writable
from .pipeline_memory import current_rss_mb, log_pipeline_memory

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from anemone.node_evaluation.common.value_snapshot import NodeValueSnapshotAccess

KIND = "morpion_scientific_training_export"
VERSION = 2
CONTRACT = "anemone-values-canonical-morpion-1"
REFERENCE_CONTRACT = "atomheart-0.2.11-dump-state-ref"
_CODEC = MorpionStateCheckpointCodec()
_SCALARS = (
    "direct_value_scalar",
    "tree_value_scalar",
    "effective_value_scalar",
    "target_value_scalar",
    "backed_up_value_scalar",
)
_PACKAGES = ("algorhino-anemone", "atomheart", "valanga", "chipiron")


class ScientificExportError(ValueError):
    """A v2 source violates its explicit scientific or persistence contract."""


def _integer(value: object) -> bool:
    """Accept JSON integers, excluding bool and coercion from floats/strings."""
    return isinstance(value, int) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScientificExportError(message)


@dataclass(frozen=True, slots=True)
class ScientificGeneration:
    """Full reference result, including certainty not expressible in Anemone v1.

    ``snapshot`` preserves the authoritative flat fields. ``value_contracts``
    must accompany it for scientific consumers. Pipeline activation is absent.
    """

    snapshot: TrainingTreeSnapshot
    value_contracts: dict[str, dict[str, Any]]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScientificExportStats:
    """Writer cost, excluding input search construction and reference loading."""

    generation: int
    node_count: int
    new_node_count: int
    state_accesses: int
    canonicalization_s: float
    total_s: float
    bytes_written: int
    rss_before_mb: float | None
    rss_after_mb: float | None


def value_score(value: object | None) -> float | None:
    """Extract Valanga's untransformed score; never substitute another value."""
    if value is None:
        return None
    _require(isinstance(value, Value), "v2 requires typed Valanga Values")
    typed = cast("Value", value)
    validate_value_semantics(typed)
    score = float(typed.score)
    _require(math.isfinite(score), "non-finite score")
    return score


def _certainty(value: Value | None) -> str | None:
    if value is None:
        return None
    validate_value_semantics(value)
    return value.certainty.name


def _snapshot_dict(snapshot: TrainingNodeSnapshot) -> dict[str, Any]:
    payload = training_tree_snapshot_to_dict(
        TrainingTreeSnapshot(root_node_id=snapshot.node_id, nodes=(snapshot,))
    )
    return cast("list[dict[str, Any]]", payload["nodes"])[0]


def _value_contract(
    node: object, snapshot: TrainingNodeSnapshot, target_source: NodeTargetSource
) -> dict[str, Any]:
    evaluation = getattr(node, "tree_evaluation", None)
    _require(evaluation is not None, "v2 requires tree_evaluation")
    values = snapshot_node_values(cast("NodeValueSnapshotAccess", evaluation))
    opening = getattr(evaluation, "all_branches_generated", None)
    _require(isinstance(opening, bool), "opening completeness must be explicit")
    target = {
        NodeTargetSource.TREE_VALUE: values.tree_value,
        NodeTargetSource.DIRECT_VALUE: values.direct_value,
        NodeTargetSource.EFFECTIVE_VALUE: values.effective_value,
    }[target_source]
    associations = {
        "direct": values.direct_value,
        "tree": values.tree_value,
        "effective": values.effective_value,
        "target": target,
    }
    for name, value in associations.items():
        _require(
            getattr(snapshot, f"{name}_value_scalar") == value_score(value),
            f"non-quiescent or inconsistent {name} value",
        )
    _require(
        snapshot.effective_value_source == values.effective_value_source.value,
        "non-quiescent effective provenance",
    )
    return {
        "target_source": target_source.value,
        "values": {
            name: None
            if value is None
            else checkpoint_payload_to_jsonable(serialize_value(value))
            for name, value in associations.items()
        },
        "certainties": {
            name: _certainty(value) for name, value in associations.items()
        },
        "all_branches_generated": opening,
        # This is an observation, not a new Dataset policy or inferred certainty.
        "backup_exactness_compatible": not (
            snapshot.is_exact
            and values.tree_value is not None
            and _certainty(values.tree_value) == "ESTIMATE"
        ),
    }


def _envelope(role: str, **fields: Any) -> dict[str, Any]:
    return {
        "format_kind": KIND,
        "format_version": VERSION,
        "scientific_contract": CONTRACT,
        "role": role,
        **fields,
    }


def _encode(payload: object) -> str:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    )


def _write(path: Path, payload: object) -> None:
    path.write_text(_encode(payload), encoding="utf-8")


def _read(path: Path, role: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "object envelope required")
    data = cast("dict[str, Any]", payload)
    _require(
        data.get("format_kind") == KIND
        and _integer(data.get("format_version"))
        and data["format_version"] == VERSION
        and data.get("scientific_contract") == CONTRACT
        and data.get("role") == role,
        "unsupported v2 envelope/contract/role",
    )
    return data


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _path(root: Path, relative: object) -> Path:
    _require(isinstance(relative, str), "relative artifact path required")
    path = (root / cast("str", relative)).resolve()
    _require(path.is_relative_to(root.resolve()), "artifact escapes export root")
    return path


def _shard(
    root: Path, descriptor: dict[str, Any], role: str
) -> Iterator[dict[str, Any]]:
    path = _path(root, descriptor["path"])
    _require(_sha(path) == descriptor["sha256"], "source shard hash mismatch")
    with path.open(encoding="utf-8") as stream:
        first_line = stream.readline()
        _require(bool(first_line), "missing shard envelope")
        header = json.loads(first_line)
        _require(
            header == _envelope(role, generation=descriptor["generation"]),
            "unsupported or mixed shard version/generation",
        )
        count = 0
        for line in stream:
            row = json.loads(line)
            _require(isinstance(row, dict), "shard row must be an object")
            yield cast("dict[str, Any]", row)
            count += 1
    _require(count == descriptor["count"], "shard count mismatch")
    _require(_sha(path) == descriptor["sha256"], "source changed during read")


def _descriptor(path: Path, root: Path, generation: int, count: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "generation": generation,
        "count": count,
        "sha256": _sha(path),
    }


def _producer(identity: Mapping[str, str]) -> dict[str, Any]:
    _require(
        all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in identity.items()
        ),
        "producer identity must contain strings",
    )
    for key in ("revision", "evaluator", "config_sha256"):
        _require(bool(identity.get(key)), f"producer {key} required")
    _require(
        len(identity["config_sha256"]) == 64
        and all(
            character in "0123456789abcdef" for character in identity["config_sha256"]
        ),
        "config SHA256 required",
    )
    versions = {name: importlib.metadata.version(name) for name in _PACKAGES}
    _require(
        versions["atomheart"] == "0.2.11"
        and versions["algorhino-anemone"] == "0.2.26"
        and versions["valanga"] == "0.2.5",
        "unsupported scientific dependency contract",
    )
    return {
        "identity": dict(identity),
        "packages": versions,
        "api_sha256": {
            name.__module__: _sha(Path(inspect.getfile(name)))
            for name in (
                build_training_node_snapshot,
                snapshot_node_values,
                validate_value_semantics,
                MorpionStateCheckpointCodec,
            )
        },
        "state_ref_contract": REFERENCE_CONTRACT,
    }


def save_scientific_training_generation(
    *,
    nodes: Sequence[object],
    root_node_id: str,
    output_dir: str | Path,
    generation: int,
    producer: Mapping[str, str],
    target_source: NodeTargetSource = NodeTargetSource.TREE_VALUE,
    metadata: Mapping[str, object] | None = None,
    created_at_unix_s: float | None = None,
) -> tuple[Path, ScientificExportStats]:
    """Write immutable v2 generation files; publish root manifest atomically last.

    Only new node states are accessed/canonicalized. The caller must freeze the
    live graph during export and never recycle node IDs for different states.
    Generations start at 1 and are consecutive. Existing paths are never reused.
    """
    started = perf_counter()
    before = current_rss_mb()
    identity = _producer(producer)
    _require(
        isinstance(target_source, NodeTargetSource), "explicit target source required"
    )
    _require(_integer(generation) and generation >= 1, "invalid generation")
    root = Path(output_dir)
    assert_artifact_writable(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path, "catalog")
        _require(manifest["producer"] == identity, "producer identity changed")
        _require(generation == manifest["latest"] + 1, "non-consecutive generation")
        previous = _read(
            _path(root, manifest["v2_generations"][str(generation - 1)]), "generation"
        )
        index = dict(previous["creation_index"])
        record_shards = list(previous["record_shards"])
    else:
        _require(
            generation == 1 and not any(root.iterdir()),
            "v2 requires a new empty export directory",
        )
        manifest = _envelope("catalog", latest=0, v2_generations={}, producer=identity)
        index, record_shards = {}, []
    for key, expected in {
        "format_kind": "training_tree_snapshot",
        "format_version": 1,
        "default_target_source": target_source.value,
    }.items():
        _require(
            metadata is None or key not in metadata or metadata[key] == expected,
            "metadata contradicts scientific contract",
        )
    destination = root / f"generation_{generation:06d}"
    _require(
        not destination.exists(), "generation path exists; explicit recovery required"
    )
    staging = Path(tempfile.mkdtemp(prefix=".v2-pending-", dir=root))
    new_count = 0
    canonical_s = 0.0
    seen: set[str] = set()
    topology: dict[str, tuple[list[str], list[str]]] = {}
    log_pipeline_memory(
        stage="growth",
        generation=generation,
        event="before_scientific_v2_save",
        node_count=len(nodes),
    )
    try:
        with (
            (staging / "records.jsonl").open("w", encoding="utf-8") as records,
            (staging / "updates.jsonl").open("w", encoding="utf-8") as updates,
        ):
            records.write(_encode(_envelope("records", generation=generation)))
            updates.write(_encode(_envelope("updates", generation=generation)))
            for order, node in enumerate(nodes):
                snapshot = build_training_node_snapshot(
                    node,
                    state_ref_dumper=None,
                    direct_value_extractor=value_score,
                    tree_value_extractor=value_score,
                    effective_value_extractor=value_score,
                    target_value_extractor=value_score,
                    target_source=target_source,
                )
                node_id = snapshot.node_id
                _require(node_id not in seen, "duplicate node ID")
                seen.add(node_id)
                row = _snapshot_dict(snapshot)
                del row["state_ref_payload"]
                row.update(
                    order_index=order,
                    value_contract=_value_contract(node, snapshot, target_source),
                )
                _validate_update(row)
                topology[node_id] = (row["parent_ids"], row["child_ids"])
                updates.write(_encode(row))
                if node_id in index:
                    continue
                canonical_started = perf_counter()
                # Exact authoritative flat codec, one new node at a time. Never
                # construct a complete flat snapshot or materialize old nodes.
                state = getattr(node, "state")  # noqa: B009
                if isinstance(state, MorpionState):
                    state = state.to_atomheart_state()
                ref = checkpoint_payload_to_jsonable(_CODEC.dump_state_ref(state))
                canonical_s += perf_counter() - canonical_started
                records.write(
                    _encode({
                        "node_id": node_id,
                        "creation_generation": generation,
                        "state_ref_payload": ref,
                    })
                )
                index[node_id] = generation
                new_count += 1
        _require(root_node_id in seen, "root missing from generation")
        _validate_topology(topology)
        descriptor = _envelope(
            "generation",
            generation=generation,
            snapshot={
                "root_node_id": root_node_id,
                "node_count": len(nodes),
                "created_at_unix_s": created_at_unix_s,
                "metadata": {
                    "format_kind": "training_tree_snapshot",
                    "format_version": 1,
                    "default_target_source": target_source.value,
                    **dict(metadata or {}),
                },
            },
            producer=identity,
            creation_index=index,
            record_shards=record_shards,
        )
        # Final names enter descriptors before publication; hashes are of the
        # exact staged files. A failed publication leaves an unreferenced dir.
        for name, role, count in (
            ("records", "records", new_count),
            ("updates", "updates", len(nodes)),
        ):
            item = _descriptor(staging / f"{name}.jsonl", root, generation, count)
            item["path"] = f"{destination.name}/{name}.jsonl"
            if role == "records":
                record_shards.append(item)
            else:
                descriptor["update_shard"] = item
        _write(staging / "generation.json", descriptor)
        staging.rename(destination)
        manifest["v2_generations"][str(generation)] = (
            f"{destination.name}/generation.json"
        )
        manifest["latest"] = generation
        pending_manifest = root / ".manifest.v2.pending"
        _write(pending_manifest, manifest)
        os.replace(pending_manifest, manifest_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    written = sum(path.stat().st_size for path in destination.iterdir())
    written += manifest_path.stat().st_size
    stats = ScientificExportStats(
        generation,
        len(nodes),
        new_count,
        new_count,
        canonical_s,
        perf_counter() - started,
        written,
        before,
        current_rss_mb(),
    )
    log_pipeline_memory(
        stage="growth",
        generation=generation,
        event="after_scientific_v2_save",
        node_count=len(nodes),
        new_node_count=new_count,
    )
    return destination / "generation.json", stats


def _ids(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _validate_update(row: dict[str, Any]) -> None:
    # No permissive Anemone deserializer defaults before these checks: absent
    # values must not be confused with explicit nulls or guessed from aliases.
    required = {
        "node_id",
        "order_index",
        "parent_ids",
        "child_ids",
        "depth",
        "is_terminal",
        "is_exact",
        "over_event_label",
        "visit_count",
        "metadata",
        "effective_value_source",
        "value_contract",
        *_SCALARS,
    }
    _require(set(row) == required, "missing or unknown scientific update field")
    _require(
        isinstance(row["node_id"], str) and bool(row["node_id"]), "invalid node ID"
    )
    _require(_ids(row["parent_ids"]) and _ids(row["child_ids"]), "invalid topology")
    for key in ("depth", "order_index"):
        _require(_integer(row[key]) and row[key] >= 0, f"invalid {key}")
    _require(
        isinstance(row["is_terminal"], bool) and isinstance(row["is_exact"], bool),
        "invalid node flags",
    )
    _require(
        row["visit_count"] is None
        or (_integer(row["visit_count"]) and row["visit_count"] >= 0),
        "invalid visit count",
    )
    _require(
        row["over_event_label"] is None or isinstance(row["over_event_label"], str),
        "invalid over event",
    )
    _require(isinstance(row["metadata"], dict), "invalid metadata")
    for key in _SCALARS:
        value = row[key]
        _require(
            value is None
            or (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
            ),
            f"invalid {key}",
        )
    _require(
        row["backed_up_value_scalar"] == row["tree_value_scalar"],
        "tree/backup aliases disagree",
    )
    contract = row["value_contract"]
    _require(
        isinstance(contract, dict)
        and set(contract)
        == {
            "target_source",
            "values",
            "certainties",
            "all_branches_generated",
            "backup_exactness_compatible",
        },
        "invalid value contract",
    )
    _require(
        isinstance(contract["all_branches_generated"], bool)
        and isinstance(contract["backup_exactness_compatible"], bool),
        "invalid completeness/compatibility declaration",
    )
    certainties = contract["certainties"]
    _require(
        isinstance(certainties, dict)
        and set(certainties) == {"direct", "tree", "effective", "target"},
        "missing per-value certainty",
    )
    for name, certainty in certainties.items():
        _require(
            certainty in (None, "ESTIMATE", "FORCED", "TERMINAL"), "invalid certainty"
        )
        _require(
            (certainty is None) == (row[f"{name}_value_scalar"] is None),
            "score/certainty association lost",
        )
    payloads = contract["values"]
    _require(
        isinstance(payloads, dict) and set(payloads) == set(certainties),
        "missing value payload",
    )
    for name, payload in payloads.items():
        _require(
            (payload is None) == (certainties[name] is None),
            "value payload presence mismatch",
        )
        if payload is not None:
            _require(
                isinstance(payload, dict)
                and set(payload) == {"score", "certainty", "over_event", "line"},
                "invalid serialized Value",
            )
            _require(
                payload["score"] == row[f"{name}_value_scalar"]
                and payload["certainty"] == certainties[name],
                "serialized Value association mismatch",
            )
            event = payload["over_event"]
            _require(
                event is None
                or (
                    isinstance(event, dict)
                    and set(event) == {"outcome", "termination", "winner"}
                    and event["outcome"] in ("WIN", "LOSS", "DRAW")
                ),
                "invalid Value over event",
            )
            _require(
                not (payload["certainty"] == "TERMINAL" and event is None)
                and not (payload["certainty"] == "ESTIMATE" and event is not None),
                "invalid Value certainty/event",
            )
            _require(
                payload["line"] is None or isinstance(payload["line"], list),
                "invalid Value line",
            )
    source = row["effective_value_source"]
    _require(source in ("none", "direct_self", "tree_child"), "invalid provenance")
    selected = {"none": None, "direct_self": "direct", "tree_child": "tree"}[source]
    _require(
        (selected is None) == (row["effective_value_scalar"] is None),
        "effective source/value mismatch",
    )
    if selected is not None:
        _require(
            row["effective_value_scalar"] == row[f"{selected}_value_scalar"]
            and certainties["effective"] == certainties[selected]
            and payloads["effective"] == payloads[selected],
            "effective score/certainty/source mismatch",
        )
    target = contract["target_source"]
    _require(
        target in ("tree_value", "direct_value", "effective_value"),
        "invalid target source",
    )
    _require(
        row["target_value_scalar"] == row[f"{target}_scalar"]
        and certainties["target"] == certainties[target.removesuffix("_value")]
        and payloads["target"] == payloads[target.removesuffix("_value")],
        "target score/certainty/source mismatch",
    )
    expected = not (row["is_exact"] and certainties["tree"] == "ESTIMATE")
    _require(
        contract["backup_exactness_compatible"] == expected,
        "false certainty compatibility declaration",
    )


def _validate_topology(topology: dict[str, tuple[list[str], list[str]]]) -> None:
    """Require a closed reciprocal DAG, preserving ordered parallel child links."""
    for node_id, (parents, children) in topology.items():
        for parent in parents:
            _require(
                parent in topology and node_id in topology[parent][1],
                "missing/nonreciprocal parent",
            )
        for child in children:
            _require(
                child in topology and node_id in topology[child][0],
                "missing/nonreciprocal child",
            )
    incoming = {
        node_id: len(set(parents)) for node_id, (parents, _) in topology.items()
    }
    frontier = [node_id for node_id, count in incoming.items() if count == 0]
    count = 0
    while frontier:
        node_id = frontier.pop()
        count += 1
        for child in set(topology[node_id][1]):
            incoming[child] -= 1
            if incoming[child] == 0:
                frontier.append(child)
    _require(count == len(topology), "cyclic topology")


def load_scientific_training_generation(path: str | Path) -> ScientificGeneration:
    """Load v2 into memory for reference validation only; no legacy fallback."""
    manifest_path = Path(path)
    root = manifest_path.parent.parent
    manifest = _read(manifest_path, "generation")
    catalog = _read(root / "manifest.json", "catalog")
    generation = manifest["generation"]
    _require(_integer(generation) and generation >= 1, "invalid generation")
    _require(
        importlib.metadata.version("atomheart") == "0.2.11",
        "unsupported canonical decoder",
    )
    _require(
        catalog["v2_generations"].get(str(generation))
        == manifest_path.relative_to(root).as_posix(),
        "unpublished generation",
    )
    _require(manifest["producer"] == catalog["producer"], "mixed producer")
    records: dict[str, dict[str, Any]] = {}
    for descriptor in manifest["record_shards"]:
        _require(descriptor["generation"] <= generation, "future record shard")
        for record in _shard(root, descriptor, "records"):
            _require(
                set(record) == {"node_id", "creation_generation", "state_ref_payload"},
                "invalid record fields",
            )
            node_id = record["node_id"]
            _require(
                isinstance(node_id, str) and node_id not in records,
                "invalid or duplicate immutable record",
            )
            _require(
                record["creation_generation"]
                == descriptor["generation"]
                == manifest["creation_index"].get(node_id),
                "invalid creation generation",
            )
            _require(
                record["state_ref_payload"] is not None, "state ref cannot be null"
            )
            ref = record["state_ref_payload"]
            canonical = checkpoint_payload_to_jsonable(
                _CODEC.dump_state_ref(_CODEC.load_state_ref(ref))
            )
            _require(canonical == ref, "noncanonical state reference")
            records[node_id] = record
    _require(
        set(records) == set(manifest["creation_index"]), "incomplete creation index"
    )
    rows: list[dict[str, Any]] = []
    contracts: dict[str, dict[str, Any]] = {}
    descriptor = manifest["update_shard"]
    _require(descriptor["generation"] == generation, "mixed update generation")
    for row in _shard(root, descriptor, "updates"):
        _validate_update(row)
        node_id = row["node_id"]
        _require(
            node_id in records and node_id not in contracts, "missing/duplicate node"
        )
        _require(row.pop("order_index") == len(rows), "invalid export order")
        contracts[node_id] = row.pop("value_contract")
        row["state_ref_payload"] = records[node_id]["state_ref_payload"]
        rows.append(row)
    info = manifest["snapshot"]
    _require(len(rows) == info["node_count"], "node count mismatch")
    by_id = {row["node_id"]: row for row in rows}
    _require(info["root_node_id"] in by_id, "missing root")
    _validate_topology({
        key: (row["parent_ids"], row["child_ids"]) for key, row in by_id.items()
    })
    snapshot = training_tree_snapshot_from_dict({
        "root_node_id": info["root_node_id"],
        "nodes": rows,
        "created_at_unix_s": info["created_at_unix_s"],
        "metadata": info["metadata"],
    })
    return ScientificGeneration(
        snapshot,
        contracts,
        {
            "source_format_kind": KIND,
            "source_format_version": VERSION,
            "scientific_contract": CONTRACT,
            "source_generation": generation,
            "producer": manifest["producer"],
        },
    )
