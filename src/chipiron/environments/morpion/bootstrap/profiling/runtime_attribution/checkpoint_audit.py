"""Bounded on-disk checkpoint inspection; never construct a live runtime or snapshot."""

from __future__ import annotations

import io
import json
from collections import Counter
from typing import TYPE_CHECKING, Any

import zstandard

from chipiron.environments.morpion.bootstrap.derived.provenance import (
    inside,
    read_json,
    sha256,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def records(checkpoint: Path, shard: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Read one record at a time with an explicit maximum line allocation."""
    print(f"[checkpoint-audit] read {shard['kind']} {shard['path']}", flush=True)
    with inside(checkpoint, shard["path"]).open("rb") as raw:
        if shard["encoding"] not in {"jsonl_zst", "jsonl"}:
            message = f"Unsupported diagnostic shard encoding: {shard['encoding']}"
            raise ValueError(message)
        stream = (
            zstandard.ZstdDecompressor().stream_reader(raw)
            if shard["encoding"] == "jsonl_zst"
            else raw
        )
        with io.BufferedReader(stream) as buffered:
            while line := buffered.readline(4 * 1024**2 + 1):
                if len(line) > 4 * 1024**2:
                    message = "Checkpoint record exceeds diagnostic 4 MiB limit."
                    raise ValueError(message)
                yield json.loads(line)


def _empty_lazy_fields(evaluation: dict[str, Any]) -> dict[str, bool]:
    """Mirror released restore predicates; these are disk candidates, not live counts."""
    ordering = evaluation.get("decision_ordering") or {}
    pv = evaluation.get("principal_variation") or {}
    frontier = evaluation.get("branch_frontier") or {}
    backup = evaluation.get("backup_runtime") or {}
    return {
        "decision_ordering": not ordering.get("branch_ordering"),
        "principal_variation": not pv.get("best_branch_sequence")
        and pv.get("pv_version", 0) == 0
        and pv.get("cached_best_child_version") is None,
        "branch_frontier": not frontier.get("frontier_branches"),
        "backup_runtime": backup.get("best_branch") is None
        and backup.get("second_best_branch") is None
        and backup.get("exact_child_count", 0) == 0
        and backup.get("selected_child_pv_version") is None
        and not backup.get("is_initialized"),
    }


def inspect_checkpoint(checkpoint: Path, *, sample_nodes: int = 128) -> dict[str, Any]:
    """Audit layout and dense ids; sample runtime contents without decoding states."""
    if not 1 <= sample_nodes <= 512:
        message = "Checkpoint inspection sample must be between 1 and 512."
        raise ValueError(message)
    manifest = read_json(checkpoint / "manifest.json")
    count = manifest["total_node_count"]
    if not isinstance(count, int) or not 1 <= count <= 5_000_000:
        message = "Checkpoint node count exceeds the bounded diagnostic audit limit."
        raise ValueError(message)
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for shard in manifest["shards"]:
        inside(checkpoint, shard["path"])
        by_kind.setdefault(shard["kind"], []).append(shard)
    split = {
        "node_shells",
        "state_payloads",
        "node_runtime",
    } <= by_kind.keys() and "node_records" not in by_kind
    result: dict[str, Any] = {
        "manifest_sha256": sha256(checkpoint / "manifest.json"),
        "node_count": count,
        "branch_count": manifest.get("total_branch_count"),
        "split_layout": split,
        "shards": {
            k: {
                "count": len(v),
                "records": sum(s.get("record_count", 0) for s in v),
                "uncompressed_bytes": sum(s.get("uncompressed_bytes", 0) for s in v),
            }
            for k, v in by_kind.items()
        },
        "live_runtime_constructed": False,
    }
    if not split:
        message = "Restore-only diagnostics require the split sharded runtime format."
        raise ValueError(message)
    # A bytearray is at most 5 MiB, not two Python sets of all node ids.
    ids = bytearray(count)
    summaries = Counter[str]()
    payload_kinds = Counter[str]()
    records_seen = 0
    duplicates = invalid = 0
    for shard in by_kind["state_payloads"]:
        for record in records(checkpoint, shard):
            records_seen += 1
            node_id = record["node_id"]
            if not isinstance(node_id, int) or not 0 <= node_id < count:
                invalid += 1
            else:
                duplicates += bool(ids[node_id])
                ids[node_id] = 1
            payload = record["state_payload"]
            payload_kinds["anchor" if "anchor_ref" in payload else "delta"] += 1
            summary = payload.get("state_summary")
            summaries[
                "with_tag"
                if isinstance(summary, dict) and "tag" in summary
                else "other_or_missing"
            ] += 1
    result.update(
        dense_zero_based_ids=records_seen == count
        and not invalid
        and not duplicates
        and all(ids),
        payload_records=records_seen,
        payload_kinds=dict(payload_kinds),
        state_summaries=dict(summaries),
    )
    opened = bytearray(count)
    for shard in by_kind["node_shells"]:
        for record in records(checkpoint, shard):
            node_id = record["node_id"]
            if not isinstance(node_id, int) or not 0 <= node_id < count:
                message = "Node shell id is outside the manifest range."
                raise ValueError(message)
            opened[node_id] = bool(record["generated_all_branches"])
    runtime_sample: list[dict[str, Any]] = []
    runtime_counts = Counter[str]()
    for shard in by_kind["node_runtime"]:
        for record in records(checkpoint, shard):
            node_id = record["node_id"]
            if not isinstance(node_id, int) or not 0 <= node_id < count:
                message = "Node runtime id is outside the manifest range."
                raise ValueError(message)
            evaluation = record.get("evaluation") or {}
            runtime_counts["records"] += 1
            if (
                not opened[record["node_id"]]
                and evaluation.get("direct_value") is not None
                and evaluation.get("backed_up_value") is not None
            ):
                runtime_counts["partial_nodes_requiring_value_comparison"] += 1
            for name, empty in _empty_lazy_fields(evaluation).items():
                runtime_counts[f"{name}_empty_or_absent_on_disk"] += empty
                if evaluation.get(name) is not None:
                    runtime_counts[f"{name}_persisted"] += 1
            if len(runtime_sample) < sample_nodes:
                runtime_sample.append({
                    "evaluation_fields": sorted(evaluation),
                    "unopened_branches": len(record.get("unopened_branches", [])),
                    "children": len(record.get("linked_children", [])),
                })
    result["runtime_prefix_sample"] = runtime_sample
    result["runtime_record_counts"] = dict(runtime_counts)
    result["live_optimization_status"] = (
        "Requires restore-only observations; disk layout/dense ids alone do not prove lazy decoding or sparse live state."
    )
    return result
