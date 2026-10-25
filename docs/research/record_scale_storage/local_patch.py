"""Fresh tiny legal search with a genuinely changing local estimate; offline only."""

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from anemone.checkpoints.sharded_io import _iter_jsonl_shard_records
from anemone.checkpoints.sharded_manifest import read_sharded_checkpoint_manifest

from chipiron.environments.morpion.bootstrap.control import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    sync_artifact,
)
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionReevaluationPatchRow,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    iter_shard_rows,
)
from docs.research.atomic_activation.measure import disk
from docs.research.record_scale_storage.measure import changes, files
from scripts.benchmark_morpion_scientific_reader import costs, delta
from tests.environments.morpion_patch_receipt_helpers import fixture_patch


def main() -> None:
    """Measure a local patch using a new legal synthetic graph."""
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=False)
    logging.disable(logging.CRITICAL)
    runner = AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            random_seed=4, runtime_checkpoint_format="sharded"
        )
    )
    runner.load_or_create(
        None,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(100000, reevaluation_blend_alpha=1.0),
    )
    runner.grow(3)
    timings = {}
    for mode in ("before", "after"):
        if mode == "after":
            runtime = runner._runtime
            leaves = [
                n
                for n in runtime._all_nodes_in_tree_order()
                if not n.tree_evaluation.has_exact_value()
                and not tuple(n.iter_child_nodes())
            ]
            node = max(leaves, key=lambda n: n.tree_evaluation.direct_value.score)
            old = node.tree_evaluation.direct_value.score
            proposal = replace(
                fixture_patch("record-storage-local"),
                rows=(
                    MorpionReevaluationPatchRow(
                        node_id=str(node.id),
                        direct_value=old + 1,
                        is_exact=False,
                        is_terminal=False,
                        metadata={"source": "active_model_reevaluation"},
                    ),
                ),
            )
            assert runner.apply_reevaluation_patch(proposal) == 1
            timings["patch"] = {
                "node": node.id,
                "old": old,
                "new": old + 1,
                "metrics": runner.last_reevaluation_patch_apply_metrics,
            }
        before, measurement = costs(), StageMeasurement()
        start = perf_counter()
        runner.save_checkpoint(root / mode / "checkpoint")
        sync_artifact(root / mode / "checkpoint", root)
        checkpoint_s = perf_counter() - start
        start = perf_counter()
        runner.export_closed_scientific_snapshot(
            root / mode / "science",
            generation=1,
            producer={
                "revision": "record-scale-resource-fixture",
                "evaluator": "existing-default-with-explicit-local-estimate",
                "config_sha256": "a" * 64,
            },
        )
        timings[mode] = {
            "checkpoint_s": checkpoint_s,
            "science_s": perf_counter() - start,
            "observation": measurement.finish(),
            "costs": delta(before),
            "disk": disk(root / mode),
        }
    runner.discard_uncommitted_runtime()
    parts = {}
    for mode in ("before", "after"):
        cp = root / mode / "checkpoint"
        manifest = read_sharded_checkpoint_manifest(cp / "manifest.json")
        parts[mode] = {
            kind: {
                str(r["node_id"]): r
                for s in manifest.shards
                if s.kind == kind
                for r in _iter_jsonl_shard_records(cp, s)
            }
            for kind in ("state_payloads", "node_shells", "node_runtime")
        }
        for kind in ("records", "updates"):
            parts[mode][kind] = {
                str(r["node_id"]): r
                for r in iter_shard_rows(
                    root / mode / "science" / (kind + ".jsonl"),
                    kind,
                    scientific_generation=1,
                    closed=True,
                )
            }
    result = {
        "timings": timings,
        "changes": {
            k: changes(parts["before"][k], parts["after"][k]) for k in parts["before"]
        },
        "nodes": len(parts["before"]["records"]),
        "files": files(root),
    }
    assert result["changes"]["records"]["changed_existing"] == 0
    assert result["changes"]["node_shells"]["changed_existing"] == 0
    assert result["changes"]["updates"]["changed_existing"] > 1
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["changes"]))


if __name__ == "__main__":
    main()
