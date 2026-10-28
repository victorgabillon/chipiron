"""Benchmark only generated temporary exports, serially in fresh processes.

Example: PYTHONPATH=src python scripts/benchmark_morpion_streaming_reevaluation.py
No existing experiment path is accepted. The real built-in adapter uses a tiny
constant-score model double (real training-state decoding, no trained bundle).
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator


def _write_array(path: Path, field: str, rows: Iterator[dict[str, object]]) -> None:
    """Generate fixtures incrementally, so fixture creation cannot inflate RSS."""
    with path.open("w", encoding="utf-8") as stream:
        stream.write('{"' + field + '":[')
        for index, row in enumerate(rows):
            if index:
                stream.write(",")
            json.dump(row, stream)
        stream.write("]}")


def _generate_fixture(root: Path, count: int, payload_bytes: int) -> Path:
    """Create an immutable export with compact real Morpion training references."""
    export = root / "tree_exports_sharded"
    export.mkdir()
    _write_array(
        export / "nodes.json",
        "nodes",
        (
            {
                "node_id": str(i),
                "parent_ids": [],
                "depth": 0,
                "creation_generation": 1,
                "state_ref_payload": [0, []],
            }
            for i in range(count)
        ),
    )
    _write_array(
        export / "updates.json",
        "updates",
        (
            {
                "node_id": str(i),
                "order_index": i,
                "child_ids": [],
                "direct_value_scalar": 1.25,
                "backed_up_value_scalar": None,
                "is_terminal": False,
                "is_exact": False,
                "metadata": {"synthetic_padding": "x" * payload_bytes},
            }
            for i in range(count)
        ),
    )
    manifest = export / "generation_000001.json"
    manifest.write_text(
        json.dumps({
            "generation": 1,
            "node_count": count,
            "new_node_count": count,
            "root_node_id": "0",
            "node_shard_path": "nodes.json",
            "update_shard_path": "updates.json",
        }),
        encoding="utf-8",
    )
    (export / "manifest.json").write_text(
        json.dumps({
            "latest_generation": 1,
            "generation_manifests": {"1": manifest.name},
        }),
        encoding="utf-8",
    )
    return manifest


class _ConstantScoreModel:
    """Synthetic inference keeps model/device cost independent of tree size."""

    def evaluate(self, state: object) -> float:
        """Return the same finite scalar after real state reconstruction."""
        del state
        return 7.25


def _child(root: Path, mode: str, window: int) -> dict[str, Any]:
    """Measure reader phases and one end-to-end worker in a fresh interpreter."""
    from chipiron.environments.morpion.bootstrap import reevaluation_worker as worker
    from chipiron.environments.morpion.bootstrap.bootstrap_args import (
        MorpionBootstrapArgs,
    )
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionPipelineActiveModel,
        MorpionPipelineGenerationManifest,
        save_pipeline_active_model,
        save_pipeline_manifest,
    )
    from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
        ShardedTrainingGenerationView,
        iter_shard_rows,
    )

    manifest = root / "tree_exports_sharded" / "generation_000001.json"
    source_bytes = sum(p.stat().st_size for p in manifest.parent.iterdir())
    if mode == "parser":
        measurement = StageMeasurement()
        tracemalloc.start()
        rows = sum(
            1 for _ in iter_shard_rows(manifest.parent / "updates.json", "updates")
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return measurement.finish(
            mode=mode,
            rows=rows,
            parser_python_peak_bytes=peak,
            source_bytes=source_bytes,
        )

    overall = StageMeasurement()
    init_measurement = StageMeasurement()
    snapshot = None
    view = None
    cache = root / "cache"
    if mode == "legacy":
        snapshot = worker.load_reevaluation_training_tree_snapshot(manifest)
        source_bytes_read = source_bytes
        index_bytes = 0
        index_status = "none"
        index_build_peak_disk_bytes = 0
        index_path = None
    else:
        view = ShardedTrainingGenerationView(manifest, cache_dir=cache)
        source_bytes_read = view.source_bytes_read
        index_bytes = view.index_bytes
        index_status = "reused" if view.index_reused else "built"
        index_build_peak_disk_bytes = view.index_build_peak_disk_bytes
        index_path = view.index_path
    initialization = init_measurement.finish(index_status=index_status)
    selection = StageMeasurement()
    if view is not None:
        ids, _, _ = view.select_sorted_node_window(None, window)
    else:
        assert snapshot is not None
        ids, _, _ = worker.select_reevaluation_node_window(
            tuple(sorted(node.node_id for node in snapshot.nodes)),
            start_cursor=None,
            max_nodes=window,
        )
    selection_observation = selection.finish()
    reconstruction = StageMeasurement()
    nodes = () if view is None else view.load_nodes(ids)
    reconstruction_observation = reconstruction.finish()
    if view is not None:
        view.close()
    adapter = worker.MorpionActiveModelNodeReevaluationEvaluator(
        root / "synthetic-bundle"
    )
    adapter._evaluator = _ConstantScoreModel()
    inference = StageMeasurement()
    rows = (
        adapter.evaluate_node_window(nodes)
        if snapshot is None
        else adapter.evaluate_patch_rows(snapshot, ids)
    )
    inference_observation = inference.finish(
        model_inference_s=adapter.last_inference_s,
        model_load_s=adapter.last_model_load_s,
    )
    assert len(rows) == len(ids)
    del snapshot, nodes, rows, view
    gc.collect()

    if mode == "cold":
        # Rebuild for the end-to-end cold worker too, inside generated scratch.
        assert index_path is not None
        index_path.unlink()
    paths = MorpionBootstrapPaths.from_work_dir(root)
    paths.ensure_directories()
    # This directory was generated by the parent, never an existing experiment.
    paths.pipeline_reevaluation_patch_path.unlink(missing_ok=True)
    paths.pipeline_reevaluation_cursor_path.unlink(missing_ok=True)
    save_pipeline_active_model(
        MorpionPipelineActiveModel(
            generation=1,
            evaluator_name="synthetic",
            model_bundle_path=str(root / "synthetic-bundle"),
            updated_at_utc="2026-01-01T00:00:00Z",
        ),
        paths.pipeline_active_model_path,
    )
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-01-01T00:00:00Z",
            tree_snapshot_path=str(manifest),
            dataset_status="not_started",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )
    # Patch only model construction; use the production built-in adapter path.
    worker.build_active_model_reevaluation_evaluator = lambda **_: adapter
    end_to_end = StageMeasurement()
    result = worker.run_morpion_reevaluation_worker_once(
        MorpionBootstrapArgs(
            work_dir=paths.work_dir, pipeline_mode="artifact_pipeline"
        ),
        max_nodes_per_patch=window,
        use_streaming_sharded_reader=mode != "legacy",
        reader_cache_dir=cache,
        patch_id="synthetic",
        now_unix_s=123456.0,
    )
    assert result.patch_written
    end_to_end_observation = end_to_end.finish()
    temporary_disk_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return overall.finish(
        mode=mode,
        selected_nodes=len(ids),
        source_bytes=source_bytes,
        source_bytes_read_reader=source_bytes_read,
        index_bytes=index_bytes,
        index_build_peak_disk_bytes=index_build_peak_disk_bytes,
        temporary_disk_bytes=temporary_disk_bytes,
        initialization=initialization,
        selection=selection_observation,
        reconstruction=reconstruction_observation,
        built_in=inference_observation,
        end_to_end=end_to_end_observation,
        nodes_per_s=len(ids) / cast("float", end_to_end_observation["elapsed_s"]),
    )


def main() -> None:
    """Run 1k/5k/10k fixtures sequentially, retaining only requested JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000])
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--payload-bytes", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--child", choices=["legacy", "cold", "reuse", "parser"], help=argparse.SUPPRESS
    )
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if args.fixture is None or os.environ.get(
            "CHIPIRON_SYNTHETIC_BENCHMARK"
        ) != str(args.fixture):
            parser.error("Child mode requires the parent's generated temporary fixture")
        print(json.dumps(_child(args.fixture, args.child, args.window)))
        return
    if min(args.sizes) < 1 or args.window < 1 or args.payload_bytes < 0:
        parser.error("Sizes/window must be positive and payload bytes nonnegative")
    report: dict[str, Any] = {
        "window": args.window,
        "payload_bytes": args.payload_bytes,
        "results": [],
    }
    for count in args.sizes:
        with tempfile.TemporaryDirectory(
            prefix="chipiron-synthetic-reevaluation-"
        ) as temporary:
            root = Path(temporary)
            _generate_fixture(root, count, args.payload_bytes)
            for mode in ("parser", "legacy", "cold", "reuse"):
                started = time.perf_counter()
                process = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        mode,
                        "--fixture",
                        str(root),
                        "--window",
                        str(args.window),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    env={**os.environ, "CHIPIRON_SYNTHETIC_BENCHMARK": str(root)},
                )
                if process.returncode:
                    raise RuntimeError(process.stderr)
                result = {
                    "total_nodes": count,
                    **json.loads(process.stdout),
                    "process_wall_s": time.perf_counter() - started,
                }
                report["results"].append(result)
                print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
