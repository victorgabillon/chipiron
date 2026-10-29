"""Review-only history probe; reuse the existing synthetic publication driver.

The sole fixture adaptation extends its literal (2, 3) loop. No production
validation, hashing, filesystem barrier or scientific policy is replaced.
"""

from __future__ import annotations

import argparse
import ast
import cProfile
import hashlib
import inspect
import json
import logging
import os
import pstats
import subprocess
import sys
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
    load_available_pipeline_manifests,
)
from docs.research.durable_scientific_v2 import measure
from scripts.benchmark_morpion_scientific_reader import costs, delta
from scripts.measure_morpion_generation_publication import disk_usage


def discover(root: Path) -> None:
    """Measure an actual discovery call in its own process, then retained RAM."""
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
        {
            "file": filename,
            "line": line,
            "function": name,
            "calls": calls,
            "inclusive_s": seconds,
        }
        for (filename, line, name), (_, calls, _, seconds, _) in pstats.Stats(
            profile
        ).stats.items()
        if name in {"validate_capsule", "read_object", "publication_records"}
    ]
    del manifests
    tracemalloc.start()
    manifests = load_available_pipeline_manifests(paths)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    observation.update(
        generation=g,
        traced_discovery_retained_bytes=retained,
        traced_discovery_peak_bytes=peak,
        manifest_publication_records=sum(
            len(m.metadata["durable_generation_publication_v1"]["generations"])
            for m in manifests.values()
        ),
        manifest_receipts=sum(
            len(m.metadata["checkpoint_bound_reevaluation_v1"]["receipts"])
            for m in manifests.values()
        ),
    )
    objects = paths.work_dir / "generation_publications_v2/objects"
    files = [p for p in objects.rglob("*") if p.is_file()]
    keys = {(p.stat().st_dev, p.stat().st_ino) for p in files}
    head = json.loads(paths.run_state_path.read_text())
    record = head["metadata"]["durable_generation_publication_v1"]["generations"][-1]
    export = (paths.work_dir / record["tree_snapshot_path"]).parent.parent
    descriptors = [
        json.loads(p.read_text()) for p in export.glob("generation_*/generation.json")
    ]
    observation["structure"] = {
        "capsule_file_paths": len(files),
        "unique_capsule_file_inodes": len(keys),
        "head_descriptor_record_entries": sum(
            len(d["record_shards"]) for d in descriptors
        ),
        "head_creation_index_entries": sum(
            len(d["creation_index"]) for d in descriptors
        ),
        "head_historical_metadata_bytes": sum(
            p.stat().st_size for p in export.rglob("*.json")
        ),
        "pipeline_manifest_bytes": sum(
            p.stat().st_size
            for p in paths.pipeline_dir.glob("generation_*/manifest.json")
        ),
        "run_state_bytes": paths.run_state_path.stat().st_size,
        "disk": disk_usage(paths.work_dir),
    }
    (root / f"discovery-{g}.json").write_text(json.dumps(observation, indent=2) + "\n")
    print(
        json.dumps({
            "discovery_generation": g,
            "hash_bytes": hashed["bytes"],
            "retained_bytes": retained,
        }),
        flush=True,
    )


def generate(root: Path, last: int) -> None:
    """Extend only the existing fixture generation loop and observe real calls."""
    measure.options = SimpleNamespace(profile=False)
    tree = ast.parse(inspect.getsource(measure.publication))
    replacements = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Tuple)
            and ast.literal_eval(node.iter) == (2, 3)
        ):
            node.iter = ast.parse(f"range(2, {last + 1})", mode="eval").body
            replacements += 1
    assert replacements == 1
    ast.fix_missing_locations(tree)
    namespace = dict(vars(measure))
    original = measure.measured

    def observed(
        name: str, directory: Path, operation: Any
    ) -> tuple[Any, dict[str, Any]]:
        result, observation = original(name, directory, operation)
        generation = result.generation
        if generation in {8, 16, 32, 64} or generation == last:
            (directory / f"publication-{generation}.json").write_text(
                json.dumps(observation, indent=2) + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--root",
                    str(directory),
                    "--discover",
                ],
                check=True,
            )
        print(
            json.dumps({
                "published": generation,
                "elapsed_s": observation.get("elapsed_s"),
            }),
            flush=True,
        )
        return result, observation

    namespace["measured"] = observed
    exec(compile(tree, inspect.getsourcefile(measure.publication), "exec"), namespace)
    namespace["publication"](root, 4, "durable")


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
