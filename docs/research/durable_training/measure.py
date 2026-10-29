"""Tiny serial Training protocol probe using existing measurements and cProfile."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap import training_publication as publication
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    file_digest,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.worker_result_io import publication_record
from docs.research.durable_scientific_v2 import measure
from scripts.measure_morpion_generation_publication import disk_usage
from tests.environments.morpion_durable_training_helpers import setup_training
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard


def main(root: Path, *, restart: bool) -> None:
    """Measure a new two-row/two-model run or fresh-process accepted recovery."""
    root = root.resolve()
    logging.disable(logging.CRITICAL)
    root.mkdir(parents=True, exist_ok=True)
    measure.options = SimpleNamespace(profile=True)
    observations: dict[str, Any] = {}
    if restart:
        from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
            MorpionBootstrapPaths,
        )

        paths = MorpionBootstrapPaths.from_work_dir(root / "work")
        state = load_bootstrap_run_state(paths.run_state_path)
        reference = publication_record(state, 1)[publication.ACCEPTED]
        _, observations["restart"] = measure.measured(
            "restart", root, lambda: publication.accept_training(paths, reference)
        )
        (root / "restart.json").write_text(json.dumps(observations, indent=2) + "\n")
        return
    with pytest.MonkeyPatch.context() as patch:
        local_leaderboard(patch, root)
        args, paths, _ = setup_training(root / "work")
        before = paths.run_state_path.stat().st_size
        dataset = publication.dataset.project_dataset(
            paths, load_bootstrap_run_state(paths.run_state_path), 1, None
        )
        source_hash = file_digest(paths.work_dir / dataset.rows_path)
        publish, accept = publication.publish_training, publication.accept_training

        def timed_publish(*a: Any, **kw: Any) -> Any:
            result, observations["publication"] = measure.measured(
                "publication", root, lambda: publish(*a, **kw)
            )
            return result

        def timed_accept(*a: Any, **kw: Any) -> Any:
            result, observations["acceptance"] = measure.measured(
                "acceptance", root, lambda: accept(*a, **kw)
            )
            return result

        patch.setattr(publication, "publish_training", timed_publish)
        patch.setattr(publication, "accept_training", timed_accept)
        torch.manual_seed(91)
        from chipiron.environments.morpion.bootstrap.performance import StageMeasurement

        stage = StageMeasurement(cuda_device="cpu")
        result = run_pipeline_training_stage(args, generation=1)
        observations["stage"] = stage.finish()
        observations["run_state_bytes"] = {
            "before": before,
            "after": paths.run_state_path.stat().st_size,
        }
        observations["rows_sha256"] = source_hash
        scientific = paths.work_dir / dataset.tree_snapshot_path
        observations["scientific_source_hashes"] = {
            p.name: file_digest(p) for p in scientific.parent.iterdir() if p.is_file()
        }
        observations["models"] = {
            name: file_digest(paths.work_dir / bundle / "param.pt")
            for name, bundle in result.model_bundle_paths.items()
        }
        observations["disk"] = disk_usage(paths.work_dir)
        state = load_bootstrap_run_state(paths.run_state_path)
        reference = publication_record(state, 1)[publication.ACCEPTED]
        directory = publication._directory(paths, reference["attempt_id"])
        observations["accepted_output_disk"] = disk_usage(directory)
        observations["descriptor_bytes"] = (directory / "proposal.json").stat().st_size
        observations["transient_input_retained"] = (directory / "input").exists()
    (root / "measures.json").write_text(json.dumps(observations, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--restart", action="store_true")
    options = parser.parse_args()
    main(options.root, restart=options.restart)
