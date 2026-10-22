"""Diagnose exact checkpoint differences without normalizing their fields away."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import zstandard

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from docs.research.rolling_snapshot.measure import PRODUCER, runner
from scripts.benchmark_morpion_scientific_reader import sha
from tests.environments.test_morpion_checkpoint_parity import _selection_signature


def rows(root: Path, kind: str) -> list[dict]:
    """Run rows for the isolated synthetic study."""
    result = []
    for item in json.loads((root / "manifest.json").read_text())["shards"]:
        if item["kind"] == kind:
            with zstandard.open(root / item["path"], "rt") as stream:
                result.extend(json.loads(line) for line in stream)
    return result


def checkpoint_differences(before: Path, after: Path) -> dict:
    """Run checkpoint differences for the isolated synthetic study."""
    a, b = rows(before, "node_shells"), rows(after, "node_shells")
    differences = [
        {
            "node_id": x["node_id"],
            "fields": {
                key: {"before": x.get(key), "after": y.get(key)}
                for key in x.keys() | y.keys()
                if x.get(key) != y.get(key)
            },
        }
        for x, y in zip(a, b, strict=True)
        if x != y
    ]
    assert all(
        set(x["fields"]) <= {"parent_node_id", "branch_from_parent"}
        for x in differences
    )
    runtime_before = rows(before, "node_runtime")
    runtime_after = rows(after, "node_runtime")
    assert runtime_before == runtime_after
    by_id = {node["node_id"]: node for node in runtime_before}
    for difference in differences:
        node = next(x for x in b if x["node_id"] == difference["node_id"])
        assert {
            "branch_key": node["branch_from_parent"],
            "child_node_id": node["node_id"],
        } in by_id[node["parent_node_id"]]["linked_children"]
    return {
        "exact_node_shell_parity": a == b,
        "differing_nodes": len(differences),
        "differences": differences,
        "all_linked_children_and_other_shell_fields_exact": True,
        "replacement_parent_edges_exist_in_original_graph": True,
    }


def continuation(root: Path) -> dict:
    """Run continuation for the isolated synthetic study."""
    root.mkdir(parents=True, exist_ok=False)
    continuous = runner()
    continuous.load_or_create(
        None, None, MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    )
    while continuous.current_tree_size() < 1000:
        continuous.grow(1)
    source = root / "checkpoint"
    continuous.save_checkpoint(source)
    restored = runner()
    restored.load_or_create(
        source, None, MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    )
    result = {"nodes": continuous.current_tree_size(), "steps": []}
    for step in range(4):
        if step:
            continuous.grow(1)
            restored.grow(1)
        outputs = []
        for name, instance in (("continuous", continuous), ("restored", restored)):
            export = root / f"{name}-{step}"
            instance.export_scientific_training_snapshot(
                export, generation=1, producer=PRODUCER
            )
            outputs.append({
                str(p.relative_to(export)): sha(p)
                for p in export.rglob("*")
                if p.is_file()
            })
        item = {
            "continuation_steps": step,
            "scientific_all_file_hashes_equal": outputs[0] == outputs[1],
            "scientific_file_hashes": outputs,
            "rng_equal": continuous._random_generator.getstate()
            == restored._random_generator.getstate(),
            "nodes_equal": continuous.current_tree_size()
            == restored.current_tree_size(),
        }
        if step:
            item["selector_report_equal"] = _selection_signature(
                continuous
            ) == _selection_signature(restored)
        result["steps"].append(item)
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    target = Path(sys.argv[1]).resolve()
    assert target.is_relative_to(
        Path(
            "/home/pompote/oldata/victor/morpion-scratch/rolling-snapshot-design-20261022"
        )
    )
    print(json.dumps(continuation(target), indent=2))
