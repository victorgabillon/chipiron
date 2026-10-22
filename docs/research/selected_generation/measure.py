"""Reuse the four-node history probe; compare exact same files in serial processes."""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import logging
import os
import pstats
import subprocess
import sys
import tracemalloc
from pathlib import Path
from typing import Any
from unittest.mock import patch

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap import generation_publication_io as io
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.pipeline import manifests as selected_io
from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
    load_available_pipeline_manifests,
    select_next_claimable_dataset_generation,
)
from docs.research.lean_discovery import measure
from scripts.benchmark_morpion_scientific_reader import costs, delta

BASE = Path(
    os.environ.get(
        "MORPION_DISCOVERY_BASE",
        "/home/pompote/oldata/victor/morpion-scratch/lean-discovery-v2-20261021/worktree",
    )
)
CURRENT = Path(__file__).resolve().parents[3]


def operation(paths: MorpionBootstrapPaths, mode: str) -> dict:
    """Include selection plus the real selected-entry validation, not just JSON reads."""
    manifests = load_available_pipeline_manifests(paths)
    generation = select_next_claimable_dataset_generation(paths, manifests)
    assert generation == max(manifests)
    if mode == "before":
        assert (
            selected_io.load_generation_manifest(paths=paths, generation=generation)
            == manifests[generation]
        )
    elif mode == "audit":
        io.audit_generation_publications(paths.work_dir)
    else:
        with selected_io.pinned_generation_manifest(
            paths=paths, generation=generation
        ) as selected:
            assert selected == manifests[generation]
    return manifests


def discover(root: Path, mode: str) -> None:
    """Use cProfile, existing RSS/CPU/I/O tools, and a separate tracemalloc pass."""
    paths = MorpionBootstrapPaths.from_work_dir(root / "work")
    hashed = {"calls": 0, "bytes": 0}
    state_reads = 0
    old_digest, old_read = hashlib.file_digest, io.read_object
    hashed_paths = []

    def counted(handle: Any, *args: Any, **kwargs: Any) -> Any:
        hashed["calls"] += 1
        hashed["bytes"] += os.fstat(handle.fileno()).st_size
        hashed_paths.append(str(Path(handle.name).relative_to(paths.work_dir)))
        return old_digest(handle, *args, **kwargs)

    def read(path: Path) -> dict:
        nonlocal state_reads
        state_reads += int(path == paths.run_state_path)
        return old_read(path)

    stage, before = StageMeasurement(), costs()
    profile = cProfile.Profile()
    with (
        patch.object(hashlib, "file_digest", counted),
        patch.object(io, "read_object", read),
    ):
        profile.enable()
        manifests = operation(paths, mode)
        profile.disable()
    g = max(manifests)
    result = stage.finish(costs=delta(before), hashing=hashed)
    result.update(
        generation=g,
        mode=mode,
        state_reads=state_reads,
        validated_capsules=sum(
            calls
            for (_, _, name), (_, calls, _, _, _) in pstats.Stats(profile).stats.items()
            if name == "validate_capsule"
        ),
        hashed_paths=hashed_paths,
    )
    profile.dump_stats(root / f"{mode}-{g}.prof")
    del manifests
    gc.collect()
    tracemalloc.start()
    manifests = operation(paths, mode)
    retained, peak = tracemalloc.get_traced_memory()
    allocations = sum(
        s.count for s in tracemalloc.take_snapshot().statistics("filename")
    )
    tracemalloc.stop()
    result.update(
        retained_bytes=retained,
        peak_traced_bytes=peak,
        retained_allocations=allocations,
        source_module=str(Path(io.__file__).resolve()),
    )
    del manifests
    gc.collect()
    tracemalloc.start()
    manifests = [
        selected_io.load_pipeline_manifest(p)
        for p in sorted(paths.pipeline_dir.glob("generation_*/manifest.json"))
    ]
    assert len(manifests) == g
    metadata_retained, metadata_peak = tracemalloc.get_traced_memory()
    metadata_allocations = sum(
        s.count for s in tracemalloc.take_snapshot().statistics("filename")
    )
    tracemalloc.stop()
    result.update(
        metadata_retained_bytes=metadata_retained,
        metadata_peak_bytes=metadata_peak,
        metadata_allocations=metadata_allocations,
    )
    state = json.loads(paths.run_state_path.read_bytes())
    record = state["metadata"][io.PUBLICATION_KEY]["generations"][-1]
    capsule = paths.work_dir / record["descriptor_path"]
    inventory = json.loads(capsule.read_bytes())["files"]
    result["selected_inventory_files"] = len(inventory)
    result["selected_identity"] = record["scientific_identity"]
    result["run_state_sha256"] = hashlib.sha256(
        paths.run_state_path.read_bytes()
    ).hexdigest()
    if mode == "after":
        expected = {
            str((capsule.parent / e["path"]).relative_to(paths.work_dir))
            for e in inventory
        }
        assert expected <= set(
            hashed_paths
        )  # Includes old records/updates/model/checkpoint.
        assert result["validated_capsules"] == result["state_reads"] == 1
    (root / f"{mode}-{g}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"generation": g, "mode": mode, "hashing": hashed}), flush=True)


def compare(root: Path) -> None:
    """Only interpreter source changes; each process sees identical committed files."""
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE, text=True
        ).strip()
        == "fa9955a2b9fc31ac43a97ed358affda801029a3b"
    )
    for mode, source in [("before", BASE), ("after", CURRENT), ("audit", CURRENT)]:
        env = {**os.environ, "PYTHONPATH": f"{source / 'src'}:{CURRENT}"}
        subprocess.run(
            [sys.executable, __file__, "--root", str(root), "--mode", mode],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--last", type=int, default=64)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--mode", choices=["before", "after", "audit"])
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if args.mode:
        discover(args.root, args.mode)
    elif args.discover:
        compare(args.root)
    else:
        # Existing generator launches --discover children at G8/16/32/64.
        measure.__file__ = __file__
        measure.generate(args.root, args.last)
