"""Serial synthetic publication probe using existing measurements and cProfile.

Run each invocation in a fresh process. This is a thin experiment driver, not a
profiling framework. No full scientific snapshot is used in measured paths.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import logging
import os
import pstats
import sqlite3
import stat
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

import ijson
import pytest
from anemone.node_evaluation.common import canonical_value

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    MorpionPipelineActiveModel,
    save_pipeline_active_model,
)
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    bootstrap_config_sha256,
    bootstrap_config_to_dict,
)
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
)
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.run_state import (
    initialize_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    save_scientific_training_generation,
)
from scripts.benchmark_morpion_scientific_reader import costs, delta, sha
from scripts.measure_morpion_generation_publication import disk_usage
from tests.environments import morpion_scientific_reader_helpers as fixture
from tests.environments.morpion_patch_receipt_helpers import (
    MODEL,
    ColdScalarRunner,
    fixture_patch,
    publish_fixture_patch,
    receipt_args,
)


def measured(name: str, root: Path, operation: Any) -> tuple[Any, dict[str, Any]]:
    """Retain existing RSS/CPU/I/O observations and standard profiler attribution."""
    from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

    profile = cProfile.Profile()
    counters = {
        "state_decodes": 0,
        "state_decode_seconds": 0.0,
        "creation_key_parsing_seconds": 0.0,
        "creation_key_count": 0,
        "creation_sql_seconds": 0.0,
        "creation_sql_calls": 0,
        "state_dumps": 0,
        "state_dump_seconds": 0.0,
        "file_fsync_count": 0,
        "directory_fsync_count": 0,
        "file_fsync_seconds": 0.0,
        "directory_fsync_seconds": 0.0,
    }
    original_decode = MorpionStateCheckpointCodec.load_state_ref
    original_dump = MorpionStateCheckpointCodec.dump_state_ref
    original_digest = hashlib.file_digest
    original_connect = sqlite3.connect
    original_kvitems = ijson.kvitems
    original_sync = os.fsync
    hashes: dict[str, dict[str, float | int]] = {}
    generation = 1 if name == "initial-publication" else int(name.rsplit("-", 1)[-1])

    def decode(codec: Any, ref: Any) -> Any:
        started = perf_counter()
        result = original_decode(codec, ref)
        counters["state_decodes"] += 1
        counters["state_decode_seconds"] += perf_counter() - started
        return result

    def dump(codec: Any, state: Any) -> Any:
        started = perf_counter()
        result = original_dump(codec, state)
        counters["state_dumps"] += 1
        counters["state_dump_seconds"] += perf_counter() - started
        return result

    def digest(handle: Any, algorithm: Any, **kwargs: Any) -> Any:
        path = Path(handle.name)
        role = path.name
        if role in {"records.jsonl", "updates.jsonl"}:
            source_generation = (
                int(path.parent.name.removeprefix("generation_"))
                if path.parent.name.startswith("generation_")
                else generation
            )
            role = (
                "historical-" if source_generation < generation else "current-"
            ) + role
        elif path.suffix == ".py" or path.suffix == ".so":
            role = "validator-code"
        elif "publication" in path.name:
            role = "capsule-seal"
        elif "checkpoint" in str(path):
            role = "checkpoint"
        else:
            role = "other-descriptor-model-config"
        started = perf_counter()
        result = original_digest(handle, algorithm, **kwargs)
        stats = hashes.setdefault(role, {"calls": 0, "bytes": 0, "seconds": 0.0})
        stats["calls"] += 1
        stats["bytes"] += path.stat().st_size
        stats["seconds"] += perf_counter() - started
        return result

    class CreationTimingConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> Any:
            if "v2_creation" not in sql:
                return super().execute(sql, parameters)
            started = perf_counter()
            try:
                return super().execute(sql, parameters)
            finally:
                counters["creation_sql_calls"] += 1
                counters["creation_sql_seconds"] += perf_counter() - started

    def connect(*args: Any, **kwargs: Any) -> Any:
        return original_connect(
            *args, **{**kwargs, "factory": CreationTimingConnection}
        )

    def keys(*args: Any, **kwargs: Any) -> Any:
        rows = original_kvitems(*args, **kwargs)
        if args[1] != "creation_index":
            yield from rows
            return
        while True:
            started = perf_counter()
            try:
                row = next(rows)
            except StopIteration:
                counters["creation_key_parsing_seconds"] += perf_counter() - started
                return
            counters["creation_key_count"] += 1
            counters["creation_key_parsing_seconds"] += perf_counter() - started
            yield row

    def sync(fd: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        started = perf_counter()
        original_sync(fd)
        counters[f"{kind}_fsync_count"] += 1
        counters[f"{kind}_fsync_seconds"] += perf_counter() - started

    stage, before = StageMeasurement(), costs()
    with (
        patch.object(MorpionStateCheckpointCodec, "load_state_ref", decode),
        patch.object(os, "fsync", sync),
        patch.object(MorpionStateCheckpointCodec, "dump_state_ref", dump),
        patch.object(hashlib, "file_digest", digest),
        patch.object(sqlite3, "connect", connect) if options.profile else nullcontext(),
        patch.object(ijson, "kvitems", keys) if options.profile else nullcontext(),
    ):
        if options.profile:
            profile.enable()
        result = operation()
        profile.disable()
    observation = stage.finish(costs=delta(before), counters=counters, hashes=hashes)
    if options.profile:
        profile.dump_stats(root / f"{name}.prof")
    stats = pstats.Stats(profile) if options.profile else None
    observation["profile"] = [
        {
            "file": filename,
            "line": line,
            "function": function,
            "primitive_calls": primitive,
            "calls": count,
            "own_seconds": own,
            "cumulative_seconds": cumulative,
        }
        for (filename, line, function), (
            primitive,
            count,
            own,
            cumulative,
            _callers,
        ) in (stats.stats.items() if stats is not None else [])
        if function
        in {
            "load_state_ref",
            "dump_state_ref",
            "validate_scientific_export",
            "validate_capsule",
            "_build_index",
            "_source_identity",
            "file_digest",
            "sync_directory",
            "publish_directory_if_absent",
        }
        or "fsync" in function
    ]
    observation["disk"] = disk_usage(root / "work")
    return result, observation


def publication(root: Path, count: int, mode: str) -> None:
    """Compare identical balanced legal replay populations, with zero new nodes in G2/3."""
    root.mkdir(parents=True, exist_ok=False)
    work = root / "work"
    paths = MorpionBootstrapPaths.from_work_dir(work)
    paths.ensure_directories()
    args = replace(
        receipt_args(work),
        training_export_mode="sharded",
        growth_skip_training_export=False,
    )
    config = bootstrap_config_from_args(args)
    atomic_json(paths.bootstrap_config_path, bootstrap_config_to_dict(config))
    producer = {
        "revision": "durable-scientific-v2-matched-synthetic",
        "evaluator": "declared-estimates-not-proofs",
        "config_sha256": bootstrap_config_sha256(config),
    }
    model = work / MODEL
    model.mkdir(parents=True)
    (model / "param.pt").write_bytes(b"synthetic-model-identity\x00")
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=1,
            evaluator_name="default",
            model_bundle_path=MODEL,
            updated_at_utc="2026-10-21T12:00:00Z",
        ),
        paths.pipeline_active_model_path,
    )
    nodes: list[Any] = []
    writer_stats = []

    def capture(**kwargs: Any) -> Any:
        nodes.extend(kwargs["nodes"])
        initial_profile = cProfile.Profile()
        if options.profile:
            initial_profile.enable()
        result = save_scientific_training_generation(**{**kwargs, "producer": producer})
        initial_profile.disable()
        if options.profile:
            initial_profile.dump_stats(root / "initial-writer.prof")
        writer_stats.append(asdict(result[1]))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            canonical_value, "make_forced_value", canonical_value.make_estimate_value
        )
        patch.setattr(fixture, "save_scientific_training_generation", capture)
        source = fixture.generated_scientific_source(
            work / "genesis-source", count, padding=4096
        )
    ids = [n.id for n in nodes]
    initial_values = {n.id: float(n.tree_evaluation.direct_value.score) for n in nodes}

    class Runner(ColdScalarRunner):
        def export_scientific_training_snapshot(
            self, output_dir: str | Path, *, generation: int, producer: dict[str, str]
        ) -> Path:
            assert self.values is not None
            for node in nodes:
                node.tree_evaluation.direct_value = canonical_value.make_estimate_value(
                    score=self.values[node.id]
                )
            path, stats = save_scientific_training_generation(
                nodes=nodes,
                root_node_id=ids[0],
                output_dir=output_dir,
                generation=generation,
                producer=producer,
            )
            writer_stats.append(asdict(stats))
            return path

    runner = Runner()
    runner.values = initial_values
    checkpoint = paths.runtime_checkpoint_path_for_generation(1)
    runner.save_checkpoint(checkpoint)
    state = replace(
        initialize_bootstrap_run_state(),
        generation=1,
        cycle_index=0,
        tree_size_at_last_save=count,
        last_save_unix_s=1e9,
        latest_runtime_checkpoint_path=paths.relative_to_work_dir(checkpoint),
        latest_tree_snapshot_path=paths.relative_to_work_dir(source),
        metadata={"bootstrap_config_hash": bootstrap_config_sha256(config)},
    )
    stages = []
    if mode == "durable":
        from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
            initialize_scientific_v2_publication,
        )

        state, stage = measured(
            "initial-publication",
            root,
            lambda: initialize_scientific_v2_publication(
                paths, state, producer=producer
            ),
        )
        stages.append({"generation": 1, **stage})
    for generation in (2, 3):
        patch = fixture_patch(f"resource-{generation}")
        patch = replace(
            patch,
            tree_generation=generation - 1,
            rows=tuple(
                replace(row, node_id=ids[i]) for i, row in enumerate(patch.rows[:2])
            ),
        )
        if mode == "durable":
            from chipiron.environments.morpion.bootstrap import (
                run_pipeline_growth_stage,
            )
            from chipiron.environments.morpion.bootstrap.generation_publication_io import (
                committed_scientific_source,
            )

            patch = replace(
                patch,
                metadata={
                    **patch.metadata,
                    "scientific_publication": committed_scientific_source(
                        work / state.latest_tree_snapshot_path
                    ),
                },
            )
            publish_fixture_patch(paths, patch)
            state, stage = measured(
                f"generation-{generation}",
                root,
                lambda: run_pipeline_growth_stage(
                    args,
                    Runner(),
                    checkpoint_bound_receipts=True,
                    durable_generation_publication=True,
                    scientific_v2_producer=producer,
                ),
            )
        else:

            def unmanaged(
                patch: Any = patch, generation: int = generation, state: Any = state
            ) -> Any:
                assert runner.values is not None
                runner.apply_reevaluation_patch(patch)
                checkpoint = paths.runtime_checkpoint_path_for_generation(generation)
                runner.save_checkpoint(checkpoint)
                export = runner.export_scientific_training_snapshot(
                    source.parent.parent, generation=generation, producer=producer
                )
                return replace(
                    state,
                    generation=generation,
                    latest_tree_snapshot_path=paths.relative_to_work_dir(export),
                    latest_runtime_checkpoint_path=paths.relative_to_work_dir(
                        checkpoint
                    ),
                )

            state, stage = measured(f"generation-{generation}", root, unmanaged)
        scientific_root = (work / state.latest_tree_snapshot_path).parent.parent
        stage["scientific_export_sha256"] = artifact_digest(scientific_root)
        stage["checkpoint_sha256"] = artifact_digest(
            work / state.latest_runtime_checkpoint_path
        )
        stage["historical_metadata_bytes"] = sum(
            p.stat().st_size for p in scientific_root.rglob("*.json")
        )
        stages.append({"generation": generation, **stage})
    result = {
        "mode": mode,
        "nodes": count,
        "producer": producer,
        "stages": stages,
        "writer_stats": writer_stats,
        "source": str(work / state.latest_tree_snapshot_path),
        "source_fingerprints": {
            p.relative_to(scientific_root).as_posix(): sha(p)
            for p in sorted(scientific_root.rglob("*"))
            if p.is_file()
        },
    }
    (root / "publication.json").write_text(json.dumps(result, indent=2) + "\n")


def reader(root: Path, label: str, dataset: bool) -> None:
    """Measure bounded cold/reused reading in a process without the writer's O(N) graph."""
    from chipiron.environments.morpion.bootstrap import MorpionBootstrapArgs
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

    source = Path(json.loads((root / "publication.json").read_text())["source"])
    stage, before = StageMeasurement(), costs()
    with ShardedTrainingGenerationView(
        source, allow_scientific_v2=True, cache_dir=root / f"{label}-cache"
    ) as view:
        init = stage.finish(
            costs=delta(before),
            reused=view.index_reused,
            index_bytes=view.index_path.stat().st_size,
            nodes=view.node_count(),
        )
        extraction = None
        output_hash = None
        if dataset:
            args = MorpionBootstrapArgs(
                work_dir=root / "work",
                dataset_confidence_policy="selected_value_v2",
                use_backed_up_value=False,
                dataset_family_target_policy="pv_blend_mean_prediction",
            )
            stage, before = StageMeasurement(), costs()
            with disk_dataset_context(IndexedDatasetSource(view), args) as context:
                stream = context.stream(3)
                stats = save_morpion_supervised_rows_streaming(
                    rows=stream.rows,
                    metadata=stream.metadata,
                    path=root / f"{label}.jsonl",
                )
            extraction = stage.finish(
                costs=delta(before),
                rows=stats.row_count,
                output_bytes=stats.bytes_written,
            )
            output_hash = sha(root / f"{label}.jsonl")
        view.validate_source_identity()
    (root / f"{label}-{'dataset' if dataset else 'cold'}.json").write_text(
        json.dumps(
            {
                "initialization": init,
                "dataset": extraction,
                "jsonl_sha256": output_hash,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["control", "durable", "cold", "dataset"], required=True
    )
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", default="after")
    parser.add_argument("--profile", action="store_true")
    options = parser.parse_args()
    if options.mode in ("control", "durable"):
        publication(options.root, options.nodes, options.mode)
    else:
        reader(options.root, options.label, options.mode == "dataset")
