"""Offline next-decision witness from an already audited immutable checkpoint.

Run separate processes on the pre-activation and activated generations. Nothing
is published. This uses the existing checkpoint loader and actual runtime step.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    committed_evaluator,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.model_activation import _runner
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.runtime.runner import _runtime_can_step
from docs.research.e2e_bootstrap.checks import digest, payload


def main() -> None:
    """Load one historical research boundary and discard one private graph on exit."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("generation", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=8)
    options = parser.parse_args()
    assert 0 < options.steps <= 32
    paths = MorpionBootstrapPaths.from_work_dir(options.workspace)
    authority = paths.run_state_path.read_bytes()
    current = load_bootstrap_run_state(paths.run_state_path)
    records = publication_records(current.metadata)
    selected = next(
        i for i, r in enumerate(records) if r["generation"] == options.generation
    )
    record = records[selected]
    state = replace(
        current,
        generation=record["generation"],
        latest_runtime_checkpoint_path=record["checkpoint_path"],
        latest_tree_snapshot_path=record["tree_snapshot_path"],
        metadata={
            **current.metadata,
            PUBLICATION_KEY: {
                **current.metadata[PUBLICATION_KEY],
                "generations": records[: selected + 1],
                "head_id": record["id"],
            },
        },
    )
    model, physical = committed_evaluator(paths.work_dir, state)
    runner, _ = _runner(paths, state)
    with torch.random.fork_rng(devices=[]):
        runner._runtime = runner._load_runtime_from_checkpoint(
            paths.work_dir / record["checkpoint_path"],
            search_args=runner._args.search_args,
            model_bundle_path=physical,
        )
    try:
        runner.validate_model_refresh_certificates()
        checkpoint = payload(runner)
        witness = {
            "generation": record["generation"],
            "model": model,
            "rng_sha256": digest(checkpoint["rng_state"]),
            "rollout_rng_sha256": digest(checkpoint["rollout_rng_state"]),
            "initial_nodes": runner.current_tree_size(),
            "steps": [],
        }
        for _ in range(options.steps):
            if (
                runner._runtime.tree.root_node.tree_evaluation.has_exact_value()
                or not _runtime_can_step(runner._runtime)
            ):
                break
            report = runner._runtime.step()
            witness["steps"].append({
                "selected_node_id": report.selected_node_id,
                "selected_depth": report.selected_depth,
                "nodes_added": report.nodes_added,
            })
        assert paths.run_state_path.read_bytes() == authority
        with options.output.open("x") as stream:
            json.dump(witness, stream, indent=2)
    finally:
        runner.discard_uncommitted_runtime()


if __name__ == "__main__":
    main()
