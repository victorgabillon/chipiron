"""Fresh-process activation measurements using existing resource/profile tools."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import logging
import os
import pstats
import time
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Thread
from typing import Any
from unittest.mock import patch

import pytest
import torch

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    committed_evaluator,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    _runner,
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from scripts.benchmark_morpion_scientific_reader import costs, delta
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard


def disk(root: Path) -> dict[str, int]:
    """Separate logical bytes from inode-aware actual allocation."""
    from chipiron.environments.morpion.bootstrap.pipeline_memory import (
        directory_disk_usage,
    )

    return directory_disk_usage(root)


def main() -> None:
    """Run setup and activation in separate interpreters, always serial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("setup", "activate", "recover"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    options = parser.parse_args()
    options.root = options.root.resolve()
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    paths = MorpionBootstrapPaths.from_work_dir(options.root)
    before = costs()
    measurement = StageMeasurement()
    profile = cProfile.Profile()
    boundaries: dict[str, Any] = {}
    temporary_peak = {"logical_bytes": 0, "allocated_bytes": 0, "samples": 0}
    stop = Event()

    def observe_disk() -> None:
        while not stop.is_set():
            try:
                current = disk(options.root / "generation_publications_v3" / "staging")
            except FileNotFoundError:  # Atomic rename can race a read-only sample.
                continue
            for key in ("logical_bytes", "allocated_bytes"):
                temporary_peak[key] = max(temporary_peak[key], current[key])
            temporary_peak["samples"] += 1
            stop.wait(0.1)

    if options.operation == "setup":
        with pytest.MonkeyPatch.context() as monkeypatch:
            local_leaderboard(monkeypatch, options.root)
            setup_activation(options.root, count=options.count)
        state = load_bootstrap_run_state(paths.run_state_path)
        request = prepare_model_activation_request(paths, training_generation=1)
        (options.root / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    elif options.operation == "activate":
        request = json.loads((options.root / "request.json").read_text())
        sampler = Thread(target=observe_disk)
        sampler.start()
        try:
            with ExitStack() as stack:
                if options.profile:
                    from anemone.node_evaluation.tree.node_tree_evaluation import (
                        NodeTreeEvaluationState,
                    )
                    from anemone.node_selector.composed.composed_node_selector import (
                        ComposedNodeSelector,
                    )
                    from anemone.tree_exploration import TreeExploration

                    from chipiron.environments.morpion.bootstrap import (
                        scientific_generation_publication,
                    )
                    from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
                        GrowthPatchReceiptSession,
                    )
                    from chipiron.environments.morpion.bootstrap.runtime import (
                        runner as runner_module,
                    )
                    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
                        MorpionRegressor,
                    )
                    from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
                        MorpionFeatureTensorConverter,
                    )

                    # Explicit wall clocks complement cProfile: parent cumulative
                    # totals can be non-additive with this Python/Torch toolchain.
                    def observe(owner: Any, name: str, label: str) -> None:
                        original_method = getattr(owner, name)

                        def timed(*args: Any, **kwargs: Any) -> Any:
                            started = time.perf_counter()
                            try:
                                return original_method(*args, **kwargs)
                            finally:
                                row = boundaries.setdefault(
                                    label, {"calls": 0, "wall_s": 0.0}
                                )
                                row["calls"] += 1
                                row["wall_s"] += time.perf_counter() - started

                        stack.enter_context(patch.object(owner, name, timed))

                    for owner, name, label in (
                        (hashlib, "file_digest", "file_hashing"),
                        (os, "fsync", "fsync"),
                        (
                            runner_module.AnemoneMorpionSearchRunner,
                            "load_or_create",
                            "cold_restore",
                        ),
                        (
                            runner_module.AnemoneMorpionSearchRunner,
                            "validate_model_refresh_certificates",
                            "certificate_preflight",
                        ),
                        (TreeExploration, "refresh_with_evaluator", "anemone_refresh"),
                        (
                            MorpionFeatureTensorConverter,
                            "state_to_model_input_tensors",
                            "feature_preparation",
                        ),
                        (MorpionRegressor, "forward", "model_forward"),
                        (NodeTreeEvaluationState, "backup_from_children", "backup"),
                        (
                            ComposedNodeSelector,
                            "refresh_state_for_checkpoint",
                            "selector",
                        ),
                        (
                            runner_module.AnemoneMorpionSearchRunner,
                            "save_checkpoint",
                            "checkpoint",
                        ),
                        (
                            GrowthPatchReceiptSession,
                            "export_scientific",
                            "scientific_snapshot",
                        ),
                        (GrowthPatchReceiptSession, "commit", "seal_and_commit"),
                        (
                            scientific_generation_publication,
                            "validate_closed_export",
                            "output_validation",
                        ),
                    ):
                        observe(owner, name, label)

                    original = runner_module.load_morpion_evaluator_from_model_bundle

                    def load(path: Path) -> Any:
                        started = time.perf_counter()
                        try:
                            return original(path)
                        finally:
                            key = (
                                "candidate_model_load_s"
                                if "staging" in path.parts
                                else "committed_model_load_s"
                            )
                            boundaries[key] = (
                                boundaries.get(key, 0.0) + time.perf_counter() - started
                            )

                    stack.enter_context(
                        patch.object(
                            runner_module,
                            "load_morpion_evaluator_from_model_bundle",
                            load,
                        )
                    )
                    profile.enable()
                state = activate_model(paths, request)
                if options.profile:
                    profile.disable()
                    profile.dump_stats(
                        str(options.root.parent / f"{options.root.name}.prof")
                    )
        finally:
            stop.set()
            sampler.join()
    else:
        state = load_bootstrap_run_state(paths.run_state_path)
        runner, effective = _runner(paths, state)
        _, physical = committed_evaluator(paths.work_dir, state)
        runner.load_or_create(
            paths.work_dir / state.latest_runtime_checkpoint_path, physical, effective
        )
        runner.validate_model_refresh_certificates()
        runner.discard_uncommitted_runtime()
    result: dict[str, Any] = {
        **delta(before),
        **measurement.finish(),
        "operation": options.operation,
        "profiled": options.profile,
        "generation": state.generation,
        "nodes": state.tree_size_at_last_save,
        "disk": disk(options.root),
        "checkpoint_digest": artifact_digest(
            options.root / state.latest_runtime_checkpoint_path
        ),
        "head": publication_records(state.metadata)[-1],
        "temporary_staging_sampled_peak": temporary_peak,
        "sample_interval_s": 0.1,
        "boundaries": boundaries,
    }
    if options.profile:
        selected = {
            "activate_model",
            "_request",
            "_certificates",
            "load_or_create",
            "refresh_model_estimates",
            "refresh_with_evaluator",
            "evaluate_nodes",
            "backup_from_children",
            "refresh_state_for_checkpoint",
            "save_checkpoint",
            "export_scientific",
            "validate_closed_export",
            "publish",
            "commit",
            "publish_manifest",
            "file_digest",
            "artifact_digest",
            "sync_artifact",
            "save_bootstrap_run_state",
            "validate_model_refresh_certificates",
            "extract_morpion_features",
            "state_to_model_input_tensors",
            "validate_capsule",
            "forward",
        }
        result["profile"] = [
            {
                "file": Path(file).name,
                "line": line,
                "function": function,
                "calls": calls,
                "self_s": own,
                "cumulative_s": total,
            }
            for (file, line, function), (_, calls, own, total, _) in pstats.Stats(
                profile
            ).stats.items()
            if function in selected
        ]
    destination = options.root.parent / f"{options.root.name}-{options.operation}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
