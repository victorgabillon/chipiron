"""Validate measured synthetic artifacts after timing, without warming exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, get_args

from anemone.checkpoints import checkpoint_payload_to_jsonable, serialize_value
from anemone.node_evaluation.common.value_snapshot import snapshot_node_values
from anemone.training_export import load_training_tree_snapshot
from anemone.training_export.serialization import training_tree_snapshot_to_dict

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    load_morpion_sharded_training_tree_snapshot,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    load_scientific_training_generation,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)
from scripts.benchmark_morpion_sharded_v2 import runner


def sha(path: Path) -> str:
    """Hash the actual saved artifact bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(root: Path) -> dict[str, Any]:
    """Check complete snapshots, live Value payloads, repeat bytes and rows."""
    output = root / "row-validation"
    output.mkdir(exist_ok=False)
    result: dict[str, Any] = {"generations": [], "deterministic_exports": []}
    for label in ("1k", "5k"):
        first, second = (root / f"{label}-trial-{trial}" for trial in (1, 2))
        for fmt in ("flat", "v1", "v2"):
            left = {
                p.relative_to(first / fmt): sha(p)
                for p in (first / fmt).rglob("*")
                if p.is_file()
            }
            right = {
                p.relative_to(second / fmt): sha(p)
                for p in (second / fmt).rglob("*")
                if p.is_file()
            }
            assert left == right
            result["deterministic_exports"].append({
                "fixture": label,
                "format": fmt,
                "files": len(left),
                "all_bytes_equal": True,
            })
        live = runner()
        live.load_or_create(first / "checkpoint.json", None)
        for generation in (1, 2, 3):
            if generation == 3:
                live.grow(1)
            expected = load_training_tree_snapshot(
                first / "flat" / f"generation-{generation}.json"
            )
            actual = load_scientific_training_generation(
                first / "v2" / f"generation_{generation:06d}/generation.json"
            )
            flat_dict = training_tree_snapshot_to_dict(expected)
            assert training_tree_snapshot_to_dict(actual.snapshot) == flat_dict
            # This oracle uses the public live Value snapshot API directly,
            # independently of the new writer's value-contract helper.
            nodes = tuple(live.iter_profile_nodes())
            for node, flat_node in zip(nodes, expected.nodes, strict=True):
                evaluation: Any = getattr(node, "tree_evaluation")  # noqa: B009
                values = snapshot_node_values(evaluation)
                selected = {
                    "direct": values.direct_value,
                    "tree": values.tree_value,
                    "effective": values.effective_value,
                    "target": values.tree_value,
                }
                contract = actual.value_contracts[flat_node.node_id]
                assert contract["values"] == {
                    name: None
                    if value is None
                    else checkpoint_payload_to_jsonable(serialize_value(value))
                    for name, value in selected.items()
                }
                assert contract["certainties"] == {
                    name: None if value is None else value.certainty.name
                    for name, value in selected.items()
                }
                assert (
                    contract["all_branches_generated"]
                    == evaluation.all_branches_generated
                )
                assert contract["target_source"] == "tree_value"
            legacy = load_morpion_sharded_training_tree_snapshot(
                first / "v1" / f"generation_{generation:06d}.json"
            )
            differences: Counter[str] = Counter()
            legacy_nodes = training_tree_snapshot_to_dict(legacy)["nodes"]
            for a, b in zip(flat_dict["nodes"], legacy_nodes, strict=True):
                assert set(a) == set(b)
                differences.update(key for key in a if a[key] != b[key])
            row_evidence = []
            for policy in get_args(PvFamilyTargetPolicy):
                args = MorpionBootstrapArgs(
                    work_dir=output, dataset_family_target_policy=policy
                )
                left_rows = extract_rows_from_training_snapshot(
                    args=args, snapshot=expected, generation=generation
                )
                right_rows = extract_rows_from_training_snapshot(
                    args=args, snapshot=actual.snapshot, generation=generation
                )
                assert left_rows == right_rows
                paths = []
                for name, rows in (("flat", left_rows), ("v2", right_rows)):
                    path = output / f"{label}-g{generation}-{policy}-{name}.jsonl"
                    save_morpion_supervised_rows_streaming(
                        rows=rows.rows, metadata=rows.metadata, path=path
                    )
                    paths.append(path)
                assert paths[0].read_bytes() == paths[1].read_bytes()
                row_evidence.append({
                    "policy": policy,
                    "row_count": len(left_rows.rows),
                    "sha256": sha(paths[0]),
                    "all_fields_and_jsonl_bytes_equal": True,
                })
            entry = {
                "fixture": label,
                "generation": generation,
                "node_count": len(expected.nodes),
                "all_snapshot_fields_equal": True,
                "all_live_value_payloads_equal": True,
                "legacy_v1_different_field_counts": dict(differences),
                "rows": row_evidence,
            }
            result["generations"].append(entry)
            print(
                json.dumps({
                    key: value for key, value in entry.items() if key != "rows"
                }),
                flush=True,
            )
    return result


def main() -> None:
    """Validate only newly generated scratch exports, then save exact evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/tmp")):
        parser.error("only /tmp synthetic fixtures are allowed")
    logging.disable(logging.CRITICAL)
    result = validate(root)
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
