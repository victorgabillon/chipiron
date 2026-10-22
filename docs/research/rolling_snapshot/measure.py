"""Measurement-only rolling storage comparison; existing codecs, no new format.

Run prepare, checkpoint, scientific and oracle in separate serial processes.
All paths must be beneath the explicitly supplied newly generated scratch root.
Component copy/link timings are cost experiments, not published generations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import zstandard
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec
from atomheart.games.morpion.dynamics import MorpionDynamics

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    copy_artifact,
    sync_artifact,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
    AnemoneMorpionSearchRunnerArgs,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    load_scientific_training_generation,
    save_scientific_training_generation,
)
from docs.research.durable_scientific_v2 import measure as measurements
from scripts.benchmark_morpion_scientific_reader import encode, sha
from scripts.measure_morpion_generation_publication import disk_usage
from tests.environments import morpion_scientific_reader_helpers as fixtures

PRODUCER = {
    "revision": "rolling-design-synthetic-only",
    "evaluator": "controlled-fixture-or-default-runtime",
    "config_sha256": "2" * 64,
}


def measured(name: str, root: Path, operation: Any) -> tuple[Any, dict[str, Any]]:
    """Reuse existing profiling/RSS/I/O/fsync tools; count the incremental codec too."""
    (root / "work").mkdir(exist_ok=True)
    counters: dict[str, int] = {}
    with ExitStack() as stack:
        for method in ("dump_state_ref", "load_anchor_ref", "load_child_from_delta"):
            original = getattr(MorpionStateCheckpointCodec, method)

            def counted(
                *args: Any,
                _method: str = method,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                counters[_method] = counters.get(_method, 0) + 1
                return _original(*args, **kwargs)

            stack.enter_context(
                patch.object(MorpionStateCheckpointCodec, method, counted)
            )
        value, observation = measurements.measured(name, root, operation)
    observation["codec_calls"] = counters
    return value, observation


def checkpoint_digest(path: Path) -> dict[str, str]:
    """Compare every checkpoint shard record, including selector and RNG payloads."""
    records: dict[str, Any] = {}
    for shard in json.loads((path / "manifest.json").read_text())["shards"]:
        with zstandard.open(path / shard["path"], "rt") as stream:
            if shard["encoding"] == "jsonl_zst":
                records.setdefault(shard["kind"], []).extend(
                    json.loads(line) for line in stream
                )
            else:
                records[shard["kind"]] = json.load(stream)
    return {
        name: hashlib.sha256(encode(value)).hexdigest()
        for name, value in records.items()
    }


def runner() -> AnemoneMorpionSearchRunner:
    """Run runner for the isolated synthetic study."""
    return AnemoneMorpionSearchRunner(
        AnemoneMorpionSearchRunnerArgs(
            random_seed=4, runtime_checkpoint_format="sharded"
        )
    )


def save_checkpoint(instance: Any, path: Path, root: Path) -> None:
    """Run save checkpoint for the isolated synthetic study."""
    instance.save_checkpoint(path)
    sync_artifact(path, root)


def prepare(root: Path, count: int) -> dict[str, Any]:
    """Run prepare for the isolated synthetic study."""
    instance = runner()
    instance.load_or_create(
        None, None, MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    )
    while instance.current_tree_size() < count:
        instance.grow(1)
        assert instance.last_growth_steps > 0
    checkpoint = root / "checkpoint"
    _, observation = measured(
        "prepare-checkpoint", root, lambda: save_checkpoint(instance, checkpoint, root)
    )
    return {
        "requested_nodes": count,
        "nodes": instance.current_tree_size(),
        "branches": instance.current_tree_branch_count(),
        "save": observation,
        "checkpoint_disk": disk_usage(checkpoint),
        "checkpoint_digest": checkpoint_digest(checkpoint),
    }


def checkpoint(root: Path, source: Path) -> dict[str, Any]:
    """Run checkpoint for the isolated synthetic study."""
    instance = runner()
    _, restore = measured(
        "restore", root, lambda: instance.load_or_create(source, None)
    )
    saves = []
    expected = checkpoint_digest(source)
    for g in (1, 2, 3):
        path = root / f"checkpoint-{g}"
        _, observation = measured(
            f"save-{g}", root, lambda path=path: save_checkpoint(instance, path, root)
        )
        actual = checkpoint_digest(path)
        observation.update(
            disk=disk_usage(path),
            checkpoint_digest=actual,
            exact_checkpoint_parity=actual == expected,
        )
        if saves:
            assert actual == saves[0]["checkpoint_digest"], (
                "Repeated quiescent saves differ"
            )
        saves.append(observation)
    return {
        "restore": restore,
        "saves": saves,
        "nodes": instance.current_tree_size(),
        "branches": instance.current_tree_branch_count(),
    }


def synthetic_nodes(count: int, offset: int) -> list[Any]:
    """Run synthetic nodes for the isolated synthetic study."""
    nodes: list[Any] = []

    def capture(**kwargs: Any) -> Any:
        nodes.extend(kwargs["nodes"])
        return Path("unused"), None

    with patch.object(fixtures, "save_scientific_training_generation", capture):
        fixtures.generated_scientific_source(Path("unused"), count)
    if offset:
        codec, dynamics = MorpionStateCheckpointCodec(), MorpionDynamics()
        boards = [codec.load_state_ref((0, ()))]
        for _ in range(offset + count.bit_length()):
            actions = dynamics.all_legal_actions(boards[-1])
            assert actions, "Fixture history exceeded the legal greedy trajectory"
            boards.append(dynamics.step(boards[-1], actions[0]).next_state)
        for node in nodes:
            node.state = boards[offset + node.depth]
            node.depth += offset
    return nodes


def scientific(
    root: Path, source: Path | None, count: int, mode: str, offset: int
) -> dict[str, Any]:
    """Run scientific for the isolated synthetic study."""
    if source:
        instance = runner()
        instance.load_or_create(source, None)
        nodes = instance._require_runtime()._all_nodes_in_tree_order()
    else:
        nodes = synthetic_nodes(count, offset)
    observations = []
    for repeat in (1, 2, 3):
        export = root / (f"complete-{repeat}" if mode == "complete" else "shared")
        generation = 1 if mode == "complete" else repeat

        def save(export: Path = export, generation: int = generation) -> Any:
            result = save_scientific_training_generation(
                nodes=nodes,
                root_node_id=str(nodes[0].id),
                output_dir=export,
                generation=generation,
                producer=PRODUCER,
            )
            # Raw writer is not durable by itself: count actual file/directory barriers.
            sync_artifact(export / f"generation_{generation:06d}", export)
            sync_artifact(export / "manifest.json", export)
            return result

        (path, stats), observation = measured(f"scientific-{repeat}", root, save)
        descriptor = json.loads(path.read_text())
        record_paths = [export / d["path"] for d in descriptor["record_shards"]]
        update = export / descriptor["update_shard"]["path"]
        # These are copies/links of exact immutable inputs, not an alternative format.
        components = {}
        if mode == "shared":
            for link in (False, True):
                target = root / f"record-{'links' if link else 'copies'}-{repeat}"

                def materialize(
                    target: Path = target,
                    record_paths: list[Path] = record_paths,
                    link: bool = link,
                ) -> None:
                    target.mkdir()
                    for index, record in enumerate(record_paths):
                        copy_artifact(record, target / str(index), link=link)
                    sync_artifact(target, root)

                _, component = measured(
                    f"components-{repeat}-{link}", root, materialize
                )
                component["disk"] = disk_usage(target)
                component["exact_bytes"] = all(
                    sha(record) == sha(target / str(index))
                    for index, record in enumerate(record_paths)
                )
                assert component["exact_bytes"]
                components["link" if link else "copy"] = component
        observation.update(
            writer=asdict(stats),
            generation=generation,
            relative_source=str(path.relative_to(root)),
            immutable_bytes=sum(p.stat().st_size for p in record_paths),
            mutable_bytes=update.stat().st_size,
            descriptor_bytes=path.stat().st_size,
            catalog_bytes=(export / "manifest.json").stat().st_size,
            record_sha256=[sha(p) for p in record_paths],
            update_sha256=sha(update),
            components=components,
        )
        observations.append(observation)
    return {
        "nodes": len(nodes),
        "mode": mode,
        "history_offset": offset,
        "observations": observations,
        "disk": disk_usage(root),
    }


def oracle(root: Path, source: Path) -> dict[str, Any]:
    """Run oracle for the isolated synthetic study."""
    reference, observation = measured(
        "reference", root, lambda: load_scientific_training_generation(source)
    )

    def indexed() -> dict[str, Any]:
        with ShardedTrainingGenerationView(
            source, cache_dir=root / "cache", allow_scientific_v2=True
        ) as view:
            nodes = tuple(view.iter_nodes_in_export_order())
            assert nodes == tuple(reference.snapshot.nodes)
            assert view.load_value_contracts([node.node_id for node in nodes]) == tuple(
                reference.value_contracts[node.node_id] for node in nodes
            )
            assert view.provenance == reference.provenance
            return {
                "reused": view.index_reused,
                "nodes": len(nodes),
                "all_fields_equal": True,
            }

    cold, cold_time = measured("indexed-cold", root, indexed)
    reused, reuse_time = measured("indexed-reuse", root, indexed)
    return {
        "reference": observation,
        "cold": cold_time,
        "reuse": reuse_time,
        "cold_result": cold,
        "reuse_result": reused,
    }


def indexed_only(root: Path, source: Path) -> dict[str, Any]:
    """Run indexed only for the isolated synthetic study."""

    def consume() -> dict[str, Any]:
        with ShardedTrainingGenerationView(
            source,
            cache_dir=root.parent
            / (source.parent.parent.parent.name + "-reader-cache"),
            allow_scientific_v2=True,
        ) as view:
            # Exhaust the bounded cursor; never retain a complete snapshot.
            count = sum(1 for _ in view.iter_nodes_in_export_order())
            view.validate_source_identity()
            return {
                "nodes": count,
                "reused": view.index_reused,
                "source_identity": view.identity,
            }

    value, observation = measured("indexed-only", root, consume)
    return {**value, "observation": observation}


def main() -> None:
    """Run main for the isolated synthetic study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "checkpoint", "scientific", "oracle", "indexed"]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--nodes", type=int, choices=[1000, 5000, 10000], default=1000)
    parser.add_argument("--mode", choices=["complete", "shared"], default="complete")
    parser.add_argument("--offset", type=int, choices=[0, 20], default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert args.root.resolve().is_relative_to(
        Path(
            "/home/pompote/oldata/victor/morpion-scratch/rolling-snapshot-design-20261022"
        )
    )
    args.root.mkdir(parents=True, exist_ok=False)
    measurements.options = SimpleNamespace(profile=args.profile)
    logging.disable(logging.CRITICAL)
    if args.action == "prepare":
        result = prepare(args.root, args.nodes)
    elif args.action == "checkpoint":
        assert args.source is not None
        result = checkpoint(args.root, args.source)
    elif args.action == "indexed":
        assert args.source is not None
        result = indexed_only(args.root, args.source)
    elif args.action == "oracle":
        assert args.source is not None
        result = oracle(args.root, args.source)
    else:
        result = scientific(args.root, args.source, args.nodes, args.mode, args.offset)
    (args.root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({
            "action": args.action,
            "root": str(args.root),
            "nodes": result.get("nodes"),
            "complete": True,
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
