"""Fresh-process synthetic comparisons using the existing measurement utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

from chipiron.environments.morpion.bootstrap import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
    save_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    _encode,
    _snapshot_dict,
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
from docs.research.durable_scientific_v2 import measure as instruments
from docs.research.rolling_snapshot.measure import measured, synthetic_nodes
from scripts.benchmark_morpion_scientific_reader import sha
from scripts.measure_morpion_generation_publication import disk_usage
from tests.environments.test_morpion_closed_scientific_snapshot import committed


def scientific_digest(nodes: Any, contracts: Any, origins: Any) -> str:
    digest = hashlib.sha256()
    for node in nodes:
        digest.update(
            _encode({
                "node": _snapshot_dict(node),
                "contract": contracts(node.node_id),
                "origin": origins(node.node_id),
            }).encode()
        )
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "initial",
            "repeat",
            "growth",
            "cold",
            "reuse",
            "reference",
            "dataset",
        ),
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--confidence", default="selected_value_v2")
    parser.add_argument("--policy", default="none")
    parser.add_argument(
        "--snapshot", default="repeat", choices=("repeat", "growth", "initial")
    )
    options = parser.parse_args()
    root = options.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "work").mkdir(exist_ok=True)
    instruments.options = SimpleNamespace(profile=False)
    logging.getLogger().setLevel(logging.WARNING)
    config_path = root / "inputs.json"
    result: dict[str, Any]
    if options.mode == "prepare":
        assert not config_path.exists()
        nodes = synthetic_nodes(options.count, options.offset)
        (source, producer), obs = measured(
            "prepare", root, lambda: committed(root / "work/run", nodes)
        )
        config = {
            "source": str(source),
            "producer": producer,
            "count": options.count,
            "offset": options.offset,
            "source_sha256": sha(source),
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        result = {"prepare": obs, **config}
    else:
        config = json.loads(config_path.read_text())
        path = root / f"work/{options.snapshot}/generation.json"
        if options.mode in ("initial", "repeat", "growth"):
            count = options.count + (
                max(1, options.count // 100) if options.mode == "growth" else 0
            )
            nodes = synthetic_nodes(count, options.offset)
            (path, stats), obs = measured(
                options.mode,
                root,
                lambda: save_closed_scientific_snapshot(
                    nodes=nodes,
                    root_node_id=str(nodes[0].id),
                    destination=root / f"work/{options.mode}",
                    generation=1 if options.mode == "initial" else 2,
                    producer=config["producer"],
                    committed_source=None
                    if options.mode == "initial"
                    else Path(config["source"]),
                ),
            )
            result = {
                "measurement": obs,
                "writer": stats,
                "disk": disk_usage(path.parent),
                "files": {
                    p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                    for p in path.parent.iterdir()
                },
            }
            if options.mode == "repeat":
                before = Path(config["source"]).parent / "updates.jsonl"
                after = path.parent / "updates.jsonl"
                with before.open("rb") as left, after.open("rb") as right:
                    next(left)
                    next(right)
                    for a, b in zip(left, right, strict=True):
                        assert a == b
                result["exact_original_update_rows"] = options.count
        elif options.mode in ("cold", "reuse"):

            def read() -> dict[str, Any]:
                with ShardedTrainingGenerationView(
                    path, cache_dir=root / "reader-cache", allow_closed_snapshot=True
                ) as view:
                    digest = scientific_digest(
                        view.iter_nodes_in_export_order(),
                        lambda key: view.load_value_contracts((key,))[0],
                        lambda key: view.load_creation_generations((key,))[0],
                    )
                    view.validate_source_identity()
                    return {
                        "scientific_sha256": digest,
                        "reused": view.index_reused,
                        "index_bytes": view.index_bytes,
                        "initialization": view.initialization_observation,
                        "nodes": view.node_count(),
                        "provenance": view.provenance,
                        "metadata": view.snapshot_metadata,
                    }

            value, obs = measured(options.mode, root, read)
            assert value["reused"] == (options.mode == "reuse")
            result = {"measurement": obs, **value}
        elif options.mode == "reference":
            reference, obs = measured(
                "reference", root, lambda: load_closed_scientific_snapshot(path)
            )
            digest = scientific_digest(
                reference.snapshot.nodes,
                reference.value_contracts.__getitem__,
                reference.creation_generations.__getitem__,
            )
            expected = json.loads(
                (
                    root
                    / (
                        "cold.json"
                        if options.snapshot == "repeat"
                        else f"{options.snapshot}-cold.json"
                    )
                ).read_text()
            )
            assert digest == expected["scientific_sha256"]
            assert (
                reference.provenance == expected["provenance"]
                and reference.snapshot.metadata == expected["metadata"]
            )
            result = {
                "measurement": obs,
                "scientific_sha256": digest,
                "exact_scientific_nodes": len(reference.snapshot.nodes),
            }
        else:
            policies = (
                sorted(SAFE_FAMILY_POLICIES)
                if options.policy == "all" and options.confidence == "selected_value_v2"
                else list(get_args(PvFamilyTargetPolicy))
                if options.policy == "all"
                else [options.policy]
            )
            oracle = load_closed_scientific_snapshot(path)
            comparisons = []
            with ShardedTrainingGenerationView(
                path, cache_dir=root / "reader-cache", allow_closed_snapshot=True
            ) as view:
                for policy in policies:
                    assert policy in get_args(PvFamilyTargetPolicy)
                    assert (
                        options.confidence == "legacy_v1"
                        or policy in SAFE_FAMILY_POLICIES
                    )
                    args = MorpionBootstrapArgs(
                        work_dir=root,
                        use_backed_up_value=False,
                        dataset_family_target_policy=policy,
                        dataset_confidence_policy=options.confidence,
                    )
                    expected = extract_rows_from_training_snapshot(
                        args=args,
                        snapshot=oracle.snapshot,
                        generation=2,
                        value_contracts=oracle.value_contracts,
                        source_provenance=oracle.provenance,
                    )
                    oracle_path = root / "oracle.jsonl"
                    indexed_path = root / "indexed.jsonl"
                    save_morpion_supervised_rows_streaming(
                        rows=expected.rows, metadata=expected.metadata, path=oracle_path
                    )

                    def extract() -> Any:
                        with disk_dataset_context(
                            IndexedDatasetSource(view), args
                        ) as context:
                            stream = context.stream(2)
                            assert stream.metadata == expected.metadata
                            save_morpion_supervised_rows_streaming(
                                rows=stream.rows,
                                metadata=stream.metadata,
                                path=indexed_path,
                            )

                    _, obs = measured("dataset", root, extract)
                    with (
                        oracle_path.open("rb") as left,
                        indexed_path.open("rb") as right,
                    ):
                        for a, b in zip(left, right, strict=True):
                            assert a == b
                    comparisons.append({
                        "measurement": obs,
                        "policy": policy,
                        "confidence": options.confidence,
                        "exact_jsonl_rows": len(expected.rows),
                        "sha256": sha(indexed_path),
                    })
                    print(
                        json.dumps({
                            "policy": policy,
                            "exact_rows": len(expected.rows),
                        }),
                        flush=True,
                    )
            result = {"comparisons": comparisons}
    suffix = (
        options.mode
        if options.mode != "dataset"
        else f"dataset-{options.confidence}-{options.policy}"
    )
    if options.mode in ("cold", "reuse", "reference") and options.snapshot != "repeat":
        suffix = options.snapshot + "-" + suffix
    (root / f"{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({
            "mode": suffix,
            "count": options.count,
            "result": str(root / f"{suffix}.json"),
        })
    )


if __name__ == "__main__":
    main()
