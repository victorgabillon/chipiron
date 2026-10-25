"""Compare complete before/iterator exports using the existing scientific oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import get_args

import zstandard

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.reevaluation_transaction import (
    GROWTH_NUMERIC_POLICY,
)
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    MorpionActiveModelNodeReevaluationEvaluator,
    select_reevaluation_node_window,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.learning.target_confidence import (
    SAFE_FAMILY_POLICIES,
)
from docs.research.e2e_bootstrap.worker import arguments
from docs.research.scientific_index.compare import same


def main() -> None:
    """Keep all scientific bytes and complete JSONL, including metadata, exact."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    out = options.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in ("records", "updates"):
        with (
            zstandard.open(options.before / "science" / f"{name}.jsonl.zst", "rb") as a,
            zstandard.open(options.after / "science" / f"{name}.jsonl.zst", "rb") as b,
        ):
            digest = hashlib.sha256()
            while chunk := a.read(65536):
                assert chunk == b.read(len(chunk))
                digest.update(chunk)
            assert not b.read(1)
            hashes[name] = digest.hexdigest()
    assert (options.before / "science/generation.json").read_bytes() == (
        options.after / "science/generation.json"
    ).read_bytes()
    full = load_closed_scientific_snapshot(options.before / "science/generation.json")
    generation = full.provenance["source_generation"]
    rows = {}
    patches = []
    with ShardedTrainingGenerationView(
        options.after / "science/generation.json",
        cache_dir=out / "cache",
        allow_closed_snapshot=True,
    ) as view:
        assert tuple(view.iter_nodes_in_export_order()) == full.snapshot.nodes
        ids = tuple(n.node_id for n in full.snapshot.nodes)
        assert view.load_value_contracts(ids) == tuple(
            full.value_contracts[i] for i in ids
        )
        assert view.load_creation_generations(ids) == tuple(
            full.creation_generations[i] for i in ids
        )
        for confidence in ("legacy_v1", "selected_value_v2"):
            for policy in get_args(PvFamilyTargetPolicy):
                if (
                    confidence == "selected_value_v2"
                    and policy not in SAFE_FAMILY_POLICIES
                ):
                    continue
                args = replace(
                    arguments(options.workspace),
                    max_rows=128,
                    dataset_confidence_policy=confidence,
                    dataset_family_target_policy=policy,
                )
                expected = extract_rows_from_training_snapshot(
                    args=args,
                    snapshot=full.snapshot,
                    generation=generation,
                    value_contracts=full.value_contracts,
                    source_provenance=full.provenance,
                )
                with disk_dataset_context(IndexedDatasetSource(view), args) as context:
                    stream = context.stream(generation)
                    label = f"{confidence}-{policy}"
                    save_morpion_supervised_rows_streaming(
                        rows=expected.rows,
                        metadata=expected.metadata,
                        path=out / f"{label}-full.jsonl",
                    )
                    save_morpion_supervised_rows_streaming(
                        rows=stream.rows,
                        metadata=stream.metadata,
                        path=out / f"{label}-indexed.jsonl",
                    )
                    rows[label] = {
                        "rows": len(expected.rows),
                        "sha256": same(
                            out / f"{label}-full.jsonl", out / f"{label}-indexed.jsonl"
                        ),
                    }
        paths = MorpionBootstrapPaths.from_work_dir(options.workspace)
        model, physical = committed_evaluator(
            paths.work_dir, load_bootstrap_run_state(paths.run_state_path)
        )
        evaluator = MorpionActiveModelNodeReevaluationEvaluator(
            model_bundle_path=paths.work_dir / model["path"],
            sealed_model_bundle_path=physical,
            numeric_policy=GROWTH_NUMERIC_POLICY,
        )
        sorted_ids = tuple(sorted(ids))
        for cursor in (None, sorted_ids[-1], "missing"):
            selection = view.select_sorted_node_window(cursor, 257)
            assert selection == select_reevaluation_node_window(
                sorted_ids, start_cursor=cursor, max_nodes=257
            )
            expected = evaluator.evaluate_patch_rows(full.snapshot, selection[0])
            actual = evaluator.evaluate_node_window(view.load_nodes(selection[0]))
            assert expected == actual
            patches.append({
                "cursor": cursor,
                "next_cursor": selection[1],
                "wrapped": selection[2],
                "rows": [asdict(row) for row in actual],
            })
    (out / "parity.json").write_text(
        json.dumps(
            {"source": hashes, "nodes": len(ids), "policies": rows, "patches": patches},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
