"""Matched history probe using the architecture review's existing fixture and tools.

Run serially in fresh processes with PYTHONPATH selecting the pinned implementation.
Adds representative Dataset/Training status files, never changes publication checks.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import logging
import os
import pstats
import tracemalloc
from pathlib import Path
from typing import Any
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_manifest,
    save_pipeline_dataset_status_file,
    save_pipeline_training_status_file,
)
from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
    load_available_pipeline_manifests,
)
from docs.research.durable_pipeline_architecture import probe
from scripts.benchmark_morpion_scientific_reader import costs, delta

PUBLICATION = "durable_generation_publication_v1"
RECEIPTS = "checkpoint_bound_reevaluation_v1"


def discover(root: Path) -> None:
    """Measure unchanged complete discovery separately from manifest allocations."""
    paths = MorpionBootstrapPaths.from_work_dir(root / "work")
    hashed = {"calls": 0, "bytes": 0}
    original = hashlib.file_digest

    def counted(handle: Any, *args: Any, **kwargs: Any) -> Any:
        hashed["calls"] += 1
        hashed["bytes"] += os.fstat(handle.fileno()).st_size
        return original(handle, *args, **kwargs)

    stage, before = StageMeasurement(), costs()
    profile = cProfile.Profile()
    with patch.object(hashlib, "file_digest", counted):
        profile.enable()
        manifests = load_available_pipeline_manifests(paths)
        profile.disable()
    g = max(manifests)
    observation = stage.finish(costs=delta(before), hashing=hashed)
    profile.dump_stats(root / f"discovery-{g}.prof")
    observation["calls"] = [
        {"function": name, "calls": calls, "inclusive_s": seconds}
        for (_, _, name), (_, calls, _, seconds, _) in pstats.Stats(
            profile
        ).stats.items()
        if name in {"validate_capsule", "read_object", "publication_records"}
    ]
    del manifests
    gc.collect()
    tracemalloc.start()
    manifests = load_available_pipeline_manifests(paths)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del manifests
    gc.collect()
    files = sorted(paths.pipeline_dir.glob("generation_*/manifest.json"))
    tracemalloc.start()
    manifests = [load_pipeline_manifest(path) for path in files]
    assert len(manifests) == g
    metadata_retained, metadata_peak = tracemalloc.get_traced_memory()
    allocations = sum(
        stat.count for stat in tracemalloc.take_snapshot().statistics("filename")
    )
    tracemalloc.stop()
    artifacts = files + sorted(paths.pipeline_dir.glob("generation_*/*status.json"))
    observation.update(
        generation=g,
        traced_discovery_retained_bytes=retained,
        traced_discovery_peak_bytes=peak,
        traced_manifest_retained_bytes=metadata_retained,
        traced_manifest_peak_bytes=metadata_peak,
        retained_manifest_allocations=allocations,
        artifacts={
            kind: {
                "count": len(
                    selected := [
                        p
                        for p in artifacts
                        if (p.name == "manifest.json") == (kind == "manifest")
                    ]
                ),
                "bytes": sum(p.stat().st_size for p in selected),
                "publication_entries": sum(
                    len(
                        json
                        .loads(p.read_bytes())["metadata"]
                        .get(PUBLICATION, {})
                        .get("generations", [])
                    )
                    for p in selected
                ),
                "receipt_entries": sum(
                    len(
                        json
                        .loads(p.read_bytes())["metadata"]
                        .get(RECEIPTS, {})
                        .get("receipts", [])
                    )
                    for p in selected
                ),
            }
            for kind in ("manifest", "status")
        },
        run_state_bytes=paths.run_state_path.stat().st_size,
        source_module=str(
            Path(
                __import__(
                    "chipiron.environments.morpion.bootstrap.pipeline_artifacts",
                    fromlist=["__file__"],
                ).__file__
            ).resolve()
        ),
    )
    (root / f"discovery-{g}.json").write_text(json.dumps(observation, indent=2) + "\n")
    print(
        json.dumps({
            "discovered": g,
            "hashing": hashed,
            "manifest_bytes": observation["artifacts"]["manifest"]["bytes"],
        }),
        flush=True,
    )


def generate(root: Path, last: int) -> None:
    """Reuse unchanged review generator; include status projections at every commit."""
    original = probe.measure.measured

    def with_status(name: str, directory: Path, operation: Any) -> Any:
        result, observation = original(name, directory, operation)
        paths = MorpionBootstrapPaths.from_work_dir(directory / "work")
        path = paths.pipeline_manifest_path_for_generation(result.generation)
        manifest = load_pipeline_manifest(path)
        for stage, save in (
            ("dataset", save_pipeline_dataset_status_file),
            ("training", save_pipeline_training_status_file),
        ):
            save(
                generation=result.generation,
                updated_at_utc=manifest.created_at_utc,
                metadata=manifest.metadata,
                path=path.parent / f"{stage}_status.json",
                **{f"{stage}_status": "not_started"},
            )
        return result, observation

    probe.__file__ = (
        __file__  # Child discovery uses this compatible counting entrypoint.
    )
    with patch.object(probe.measure, "measured", with_status):
        probe.generate(root, last)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--last", type=int, default=64)
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if args.discover:
        discover(args.root)
    else:
        generate(args.root, args.last)
