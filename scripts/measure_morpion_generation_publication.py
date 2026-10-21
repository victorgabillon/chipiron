"""Measure generated-fixture publication costs; never create a search runtime.

This driver uses the test scalar runner and actual scientific v1 exporter.
It is deliberately separate from real Growth, model training and worker CLIs.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import resource
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from chipiron.environments.morpion.bootstrap import (
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
)
from tests.environments.morpion_generation_publication_helpers import (
    PublicationRunner,
    run_publication,
    setup_publication_baseline,
)
from tests.environments.morpion_patch_receipt_helpers import (
    fixture_patch,
    publish_fixture_patch,
)


def process_io() -> dict[str, int]:
    """Read Linux process counters without interpreting them as device wear."""
    return {
        key: int(value)
        for line in Path("/proc/self/io").read_text().splitlines()
        for key, value in [line.split(":", 1)]
    }


def disk_usage(root: Path) -> dict[str, int]:
    """Report apparent bytes and allocated blocks, counting shared inodes once."""
    apparent = 0
    unique_size = 0
    allocated = 0
    seen: set[tuple[int, int]] = set()
    files = 0
    for path in [root, *root.rglob("*")]:
        stat = path.stat()
        if path.is_file():
            apparent += stat.st_size
            files += 1
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen.add(identity)
            allocated += stat.st_blocks * 512
            if path.is_file():
                unique_size += stat.st_size
    return {
        "apparent_file_bytes": apparent,
        "unique_file_bytes": unique_size,
        "allocated_bytes_unique_inodes": allocated,
        "file_names": files,
        "unique_inodes": len(seen),
    }


def fixture_measurement(mode: str, nodes: int, generations: int) -> dict[str, Any]:
    """Measure only generated data in an automatically removed temporary root."""
    logging.disable(logging.CRITICAL)
    values = {f"node-{i:05d}": float(i) for i in range(nodes - 2)}
    values.update({"root": 2.0, "leaf": 4.0})
    with tempfile.TemporaryDirectory(prefix="morpion-publication-cost-") as temporary:
        work_dir = Path(temporary)
        args, paths = setup_publication_baseline(
            work_dir, values=values, adopt_publication=mode == "durable"
        )
        baseline_disk = disk_usage(work_dir)
        baseline_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        before_io = process_io()
        started = perf_counter()
        state = load_bootstrap_run_state(paths.run_state_path)
        for generation in range(2, generations + 2):
            patch = publish_fixture_patch(paths, fixture_patch(f"cost-{generation}"))
            if mode == "durable":
                state = run_publication(args, PublicationRunner())
            else:
                runner = PublicationRunner()
                runner.values = dict(values)
                runner.apply_reevaluation_patch(patch)
                assert runner.values is not None
                values = dict(runner.values)
                # Match the original v1 eager-consumption/write sequence in this fixture.
                paths.pipeline_reevaluation_patch_path.unlink()
                checkpoint = paths.runtime_checkpoint_path_for_generation(generation)
                runner.save_checkpoint(checkpoint)
                snapshot = runner.export_sharded_training_tree_snapshot(
                    paths.sharded_tree_snapshot_dir, generation=generation
                )
                state = replace(
                    state,
                    generation=generation,
                    metadata={},
                    latest_runtime_checkpoint_path=paths.relative_to_work_dir(
                        checkpoint
                    ),
                    latest_tree_snapshot_path=paths.relative_to_work_dir(snapshot),
                )
                save_bootstrap_run_state(state, paths.run_state_path)
        elapsed = perf_counter() - started
        after_io = process_io()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert (
            state.latest_tree_snapshot_path is not None
            and state.latest_runtime_checkpoint_path is not None
        )
        return {
            "mode": mode,
            "nodes": nodes,
            "steady_generations": generations,
            "baseline_disk": baseline_disk,
            "final_disk": disk_usage(work_dir),
            "steady_process_io": {
                key: after_io[key] - value for key, value in before_io.items()
            },
            "baseline_peak_rss_mib": baseline_peak / 1024,
            "peak_rss_mib": peak / 1024,
            "peak_rss_growth_mib": (peak - baseline_peak) / 1024,
            "fixture_elapsed_seconds": elapsed,
            "scientific_export_sha256": artifact_digest(
                (work_dir / state.latest_tree_snapshot_path).parent
            ),
            "checkpoint_sha256": artifact_digest(
                work_dir / state.latest_runtime_checkpoint_path
            ),
        }


def main() -> None:
    """Run each measurement in a fresh process and retain exact output parity evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-mode", choices=["legacy", "durable"])
    args = parser.parse_args()
    if args.fixture_mode:
        print(
            json.dumps(
                fixture_measurement(args.fixture_mode, args.nodes[0], args.generations),
                sort_keys=True,
            )
        )
        return
    results = []
    for count in args.nodes:
        pair = []
        for mode in ["legacy", "durable"]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--fixture-mode",
                    mode,
                    "--nodes",
                    str(count),
                    "--generations",
                    str(args.generations),
                ],
                capture_output=True,
                text=True,
                timeout=180,
                check=True,
            )
            pair.append(json.loads(result.stdout))
        assert (
            pair[0]["scientific_export_sha256"] == pair[1]["scientific_export_sha256"]
        )
        assert pair[0]["checkpoint_sha256"] == pair[1]["checkpoint_sha256"]
        pair[1]["steady_wchar_ratio_vs_legacy"] = (
            pair[1]["steady_process_io"]["wchar"]
            / pair[0]["steady_process_io"]["wchar"]
        )
        results.extend(pair)
    payload = {
        "schema_version": 1,
        "base_sha": "7313ecfad36667ab2a8dd1bbd4b894aa682a7e9b",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scope": "Generated scalar checkpoints, real v1 exports, tiny synthetic model; no search or workers.",
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
