"""Fresh-process retry/head measurements using existing costs, cProfile and hash I/O."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import logging
import pstats
import statistics
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import chipiron
from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap import (
    generation_publication_io as publication,
)
from chipiron.environments.morpion.bootstrap.model_activation import activate_model
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from docs.research.atomic_activation.measure import disk
from scripts.benchmark_morpion_scientific_reader import costs, delta


def main() -> None:
    """Report both uninstrumented elapsed time and a separate exact I/O trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("retry", "head"))
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--request", type=Path)
    options = parser.parse_args()
    logging.disable(logging.CRITICAL)
    root = options.workspace.resolve()
    paths = MorpionBootstrapPaths.from_work_dir(root)
    state = load_bootstrap_run_state(paths.run_state_path)
    authority = paths.run_state_path.read_bytes()
    records = publication.publication_records(state.metadata)
    head = records[-1]
    request = (
        None if options.request is None else json.loads(options.request.read_text())
    )
    assert options.operation != "retry" or request is not None

    def operation() -> None:
        if options.operation == "retry":
            assert activate_model(paths, request) == state
        else:
            model, _ = publication.committed_evaluator(root, state)
            assert (
                model["sha256"]
                == head["scientific_identity"]["producer"]["identity"]["evaluator"]
            )
        assert paths.run_state_path.read_bytes() == authority

    samples = []
    before = costs()
    for _ in range(5):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    elapsed_costs = delta(before, costs())
    hashed_files = []
    descriptors = []
    original_hash = hashlib.file_digest
    original_read = publication.read_object

    def hash_file(stream: Any, *args: Any, **kwargs: Any) -> Any:
        path = Path(stream.name)
        hashed_files.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
        })
        return original_hash(stream, *args, **kwargs)

    def read_object(path: Path) -> dict[str, Any]:
        if path.name == ".generation_publication_v3.json":
            descriptors.append(str(path.relative_to(root)))
        return original_read(path)

    profile = cProfile.Profile()
    with (
        patch.object(hashlib, "file_digest", hash_file),
        patch.object(publication, "read_object", read_object),
    ):
        profile.runcall(operation)
    profile_path = options.output.with_suffix(".prof")
    profile.dump_stats(profile_path)
    with options.output.with_suffix(".profile.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(35)
    current_descriptor = head["descriptor_path"]
    result = {
        "operation": options.operation,
        "implementation": str(Path(chipiron.__file__).resolve()),
        "generation": state.generation,
        "authority_sha256": hashlib.sha256(authority).hexdigest(),
        "authority_bytes": len(authority),
        "authority_unchanged": True,
        "uninstrumented_seconds": samples,
        "median_seconds": statistics.median(samples),
        "five_call_costs": elapsed_costs,
        "workspace_disk": disk(root),
        "profile_file_hash_calls": len(hashed_files),
        "profile_file_hash_bytes": sum(item["bytes"] for item in hashed_files),
        "profile_hashed_files": hashed_files,
        "profile_descriptor_reads": descriptors,
        "historical_descriptors_opened": sorted(
            set(descriptors) - {current_descriptor}
        ),
        "historical_payload_files_hashed": [
            item
            for item in hashed_files
            if "/objects/" in item["path"]
            and "/" + head["id"] + "/" not in item["path"]
            and not item["path"].endswith(".generation_publication_v3.json")
        ],
    }
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({
            k: v
            for k, v in result.items()
            if k
            in {
                "operation",
                "median_seconds",
                "profile_file_hash_calls",
                "profile_file_hash_bytes",
                "authority_bytes",
            }
        })
    )


if __name__ == "__main__":
    main()
