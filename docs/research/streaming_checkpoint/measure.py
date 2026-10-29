"""Matched Chipiron saves, cold restores and exact offline comparisons.

Measurements use existing StageMeasurement, process counters and allocation tools.
Full payloads are permitted only in the explicit old-save and offline oracles.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import logging
import subprocess
import threading
import tracemalloc
from contextlib import ExitStack
from pathlib import Path
from types import MethodType
from typing import Any
from unittest.mock import patch

import torch
from anemone.checkpoints import (
    AlgorithmNodeCheckpointPayload,
    build,
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
    write_sharded_search_checkpoint,
)

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.pipeline_memory import current_rss_mb
from chipiron.environments.morpion.bootstrap.runtime import runner as module
from docs.research.atomic_activation.measure import disk
from docs.research.model_activation.probe import graph
from docs.research.rolling_snapshot.measure import checkpoint_digest
from scripts.benchmark_morpion_scientific_reader import costs, delta

BASE = "d744b4b7d2947d9e8fe20de02e9bdc3266cbb401"
PRODUCER = {
    "revision": "streaming-checkpoint-fixture",
    "evaluator": "default",
    "config_sha256": "0" * 64,
}


def hashes(root: Path) -> dict[str, str]:
    """Fingerprint every byte, including the manifest."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def old_save(instance: Any, batch: int) -> None:
    """Execute the exact frozen caller with only physical shard sizing overridden."""
    source = subprocess.check_output(
        [
            "git",
            "show",
            BASE + ":src/chipiron/environments/morpion/bootstrap/runtime/runner.py",
        ],
        text=True,
    )
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AnemoneMorpionSearchRunner"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "save_checkpoint"
    )
    namespace = dict(vars(module))
    namespace["write_sharded_search_checkpoint"] = lambda *a, **k: (
        write_sharded_search_checkpoint(*a, **k, node_count_per_shard=batch)
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<frozen-save>", "exec"),
        namespace,
    )
    instance.save_checkpoint = MethodType(namespace["save_checkpoint"], instance)


def restored(path: Path) -> Any:
    """Cold runtime construction through the supported production path."""
    live = module.AnemoneMorpionSearchRunner(
        module.AnemoneMorpionSearchRunnerArgs(
            random_seed=999, runtime_checkpoint_format="sharded"
        )
    )
    live.load_or_create(
        path, None, MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
    )
    return live


def main() -> None:
    """Run one fresh-process measurement or a deliberately separate full oracle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=["save", "restore", "validate", "compare", "science"]
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--other", type=Path)
    parser.add_argument("--mode", choices=["old", "stream"], default="stream")
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.other is not None:
        args.other = args.other.resolve()
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    checkpoint = args.root / "checkpoint"
    imported_rss = current_rss_mb()
    if args.operation == "compare":
        assert args.other is not None
        a, b = checkpoint, args.other / "checkpoint"
        exact_files = hashes(a) == hashes(b)
        assert checkpoint_digest(a) == checkpoint_digest(b)
        x, y = restored(a), restored(b)
        science = []
        for index, live in enumerate((x, y)):
            destination = args.root / ("science-" + str(index) + "-" + args.other.name)
            live.export_closed_scientific_snapshot(
                destination, generation=1, producer=PRODUCER
            )
            science.append(hashes(destination))
        assert science[0] == science[1]
        for _ in range(3):
            assert build_search_checkpoint_payload(
                x._runtime, state_codec=x._state_codec
            ) == build_search_checkpoint_payload(y._runtime, state_codec=y._state_codec)
            x.grow(1)
            y.grow(1)
        assert build_search_checkpoint_payload(
            x._runtime, state_codec=x._state_codec
        ) == build_search_checkpoint_payload(y._runtime, state_codec=y._state_codec)
        result = {
            "byte_identical": exact_files,
            "complete_payload_equal": True,
            "scientific_file_hashes": science,
            "paired_continuations": 3,
        }
        label = "compare-" + args.other.name
    else:
        if args.operation == "save":
            args.root.mkdir(parents=True, exist_ok=False)
            live = graph(args.nodes)
            module.CHECKPOINT_NODES_PER_SHARD = args.batch
            # Compile/retain the frozen caller in BOTH processes: otherwise its
            # AST allocation warms only the old process's allocator before RSS.
            old_save(live, args.batch)
            frozen_save = live.save_checkpoint
            if args.mode == "stream":
                live.save_checkpoint = MethodType(
                    module.AnemoneMorpionSearchRunner.save_checkpoint, live
                )
        elif args.operation == "science":
            live = restored(checkpoint)
        gc.collect()
        before = costs()
        measurement = StageMeasurement()
        peak = [current_rss_mb()]
        stop = threading.Event()

        def sample() -> None:
            while not stop.wait(0.002):
                peak[0] = max(peak[0], current_rss_mb())

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        selector: dict[str, int] = {}
        original = build._build_selector_state_payload

        def observe(search: Any) -> Any:
            start, previous_peak = tracemalloc.get_traced_memory()
            value = original(search)
            end, _ = tracemalloc.get_traced_memory()
            selector.update(
                node_phase_peak_bytes=previous_peak, retained_payload_bytes=end - start
            )
            return value

        if args.trace:
            tracemalloc.start()
        counts = {"active": 0, "peak": 0}
        original_init = AlgorithmNodeCheckpointPayload.__init__

        def initialize(instance: Any, *pos: Any, **kw: Any) -> None:
            original_init(instance, *pos, **kw)
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])

        def destroy(_instance: Any) -> None:
            counts["active"] -= 1

        with ExitStack() as stack:
            if args.trace:
                stack.enter_context(
                    patch.object(build, "_build_selector_state_payload", observe)
                )
                stack.enter_context(
                    patch.object(AlgorithmNodeCheckpointPayload, "__init__", initialize)
                )
                stack.enter_context(
                    patch.object(
                        AlgorithmNodeCheckpointPayload, "__del__", destroy, create=True
                    )
                )
            if args.operation == "save":
                live.save_checkpoint(checkpoint)
            elif args.operation == "restore":
                live = restored(checkpoint)
            elif args.operation == "science":
                live.export_closed_scientific_snapshot(
                    args.root / "measured-science", generation=1, producer=PRODUCER
                )
            else:
                value = checkpoint_digest(checkpoint)
                validated = restored(checkpoint)
                validated.validate_model_refresh_certificates()
        traces = tracemalloc.get_traced_memory() if args.trace else None
        if args.trace:
            tracemalloc.stop()
        peak[0] = max(peak[0], current_rss_mb())
        result = {**delta(before), **measurement.finish()}
        stop.set()
        sampler.join()
        result.update(
            operation=args.operation,
            mode=args.mode,
            batch=args.batch,
            trace=args.trace,
            imported_rss_mb=imported_rss,
            sampled_peak_rss_mb=peak[0],
            sampled_extra_rss_mb=peak[0] - measurement.rss_before_mb,
            traced_bytes=traces,
            selector=selector,
            typed_node_payload_peak=counts["peak"] if args.trace else None,
            disk=disk(checkpoint),
            files=hashes(checkpoint),
            manifest_bytes=(checkpoint / "manifest.json").stat().st_size,
        )
        if args.operation == "save":
            assert frozen_save is not None
            result["nodes"] = live.current_tree_size()
            result["caller_metrics"] = live._latest_checkpoint_metrics
            result["symmetric_frozen_method_setup"] = True
        if args.operation == "validate":
            result["payload_sha256"] = hashlib.sha256(
                json.dumps(
                    checkpoint_payload_to_jsonable(value),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        label = args.operation
    (args.root / (label + ".json")).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
