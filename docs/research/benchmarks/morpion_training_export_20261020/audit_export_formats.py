"""Exact field audit; records differences rather than treating formats as equal."""

import argparse
import json
import logging
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from anemone.training_export.serialization import training_tree_snapshot_to_dict

from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    load_morpion_sharded_training_tree_snapshot,
    save_morpion_sharded_training_tree_from_live_nodes,
)

logging.disable(logging.CRITICAL)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
arguments = parser.parse_args()
assert arguments.root.resolve().is_relative_to(Path("/tmp"))
assert arguments.output.resolve().is_relative_to(Path("/tmp"))
assert not arguments.output.exists()
root = arguments.root
results = {}
for generation in [3, 4]:
    cold = next(root.glob("flat-*-cold"))
    sharded = next(root.glob("sharded-*-cold"))
    flat = json.loads(
        (cold / f"tree_exports/generation_{generation:06d}.json").read_text()
    )
    parsed = load_morpion_sharded_training_tree_snapshot(
        sharded / f"tree_exports_sharded/generation_{generation:06d}.json"
    )
    shard = json.loads(json.dumps(training_tree_snapshot_to_dict(parsed)))
    counts = Counter()
    differences = []
    assert [n["node_id"] for n in flat["nodes"]] == [
        n["node_id"] for n in shard["nodes"]
    ]
    for a, b in zip(flat["nodes"], shard["nodes"], strict=True):
        fields = {
            key: {"flat": a.get(key), "sharded": b.get(key)}
            for key in a.keys() | b.keys()
            if a.get(key) != b.get(key)
        }
        if fields:
            counts.update(fields.keys())
            differences.append(dict(node_id=a["node_id"], fields=fields))
    results[str(generation)] = {
        "different_node_count": len(differences),
        "field_counts": dict(counts),
        "tree_fields": {
            key: {"flat": flat.get(key), "sharded": shard.get(key)}
            for key in flat.keys() | shard.keys()
            if key != "nodes" and flat.get(key) != shard.get(key)
        },
        "differences": differences,
    }

out = (
    Path(tempfile.mkdtemp(prefix="morpion-export-topology-counterexample-")) / "shards"
)
assert not out.exists()


def node(name: Any, depth: Any, parents: Any = (), children: Any = ()) -> Any:
    """Build a tiny live node for the changing-parent correctness audit."""
    return SimpleNamespace(
        id=name,
        depth=depth,
        state={"opaque": name},
        direct_value=1.0,
        backed_up_value=2.0,
        is_terminal=False,
        is_exact=False,
        parent_ids=parents,
        child_ids=children,
        metadata={},
    )


r = node("root", 0, children=("a", "b"))
a = node("a", 1, ("root",), ("join",))
b = node("b", 1, ("root",))
join = node("join", 2, ("a",))
nodes = (r, a, b, join)
kwargs = dict(
    nodes=nodes,
    root_node_id="root",
    output_dir=out,
    state_ref_dumper=lambda state: state,
    direct_value_extractor=lambda value: value,
    backed_up_value_extractor=lambda value: value,
)
save_morpion_sharded_training_tree_from_live_nodes(generation=1, **kwargs)
b.child_ids = ("join",)
join.parent_ids = ("a", "b")
path, _ = save_morpion_sharded_training_tree_from_live_nodes(generation=2, **kwargs)
loaded = load_morpion_sharded_training_tree_snapshot(path)
results["parent_addition_counterexample"] = {
    "live_join_parents": list(join.parent_ids),
    "exported_join_parents": list(
        next(n for n in loaded.nodes if n.node_id == "join").parent_ids
    ),
    "exported_b_children": list(
        next(n for n in loaded.nodes if n.node_id == "b").child_ids
    ),
    "fixture": str(out),
}
arguments.output.write_text(json.dumps(results, indent=2))
print(
    json.dumps(
        {
            k: {f: v for f, v in d.items() if f != "differences"}
            for k, d in results.items()
        },
        indent=2,
    )
)
