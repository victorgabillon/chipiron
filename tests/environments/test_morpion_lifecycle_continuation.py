"""Installed, bounded real continuation checks for the user-run validation command."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
import torch
from anemone.checkpoints.rng_state import rollout_random_generator

from chipiron.environments.morpion.bootstrap import lifecycle
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    publication_records,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_checkpoint_parity import _selection_signature
from tests.environments.test_morpion_committed_evaluator_restore import fingerprint
from tests.environments.test_morpion_continual_publication import grow, setup_continual

if TYPE_CHECKING:
    from pathlib import Path


def equal_state(left: Any, right: Any) -> None:
    """Compare actual tensors and Adam slots, not serialized provenance UUIDs."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_state(a, b)
    else:
        assert left == right


def inert_supervisor(path: Path) -> Path:
    """Real owned shell/monitor receipt, not a fabricated session or experiment."""
    path.write_text("#!/bin/bash\nsleep 1\n")
    return path


@pytest.mark.exhaustive
def test_installed_normal_fresh_resume_and_zero_write_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the actual installed CLI; its brief inert supervisor does no scientific work."""
    monkeypatch.delenv("MORPION_PERSISTENT_WORKERS", raising=False)
    root = tmp_path / "normal"
    script = inert_supervisor(tmp_path / "supervisor.sh")
    prefix = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "chipiron.environments.morpion.bootstrap.operator_cli",
        "start",
        "--work-dir",
        str(root),
    ]
    result = subprocess.run(
        [
            *prefix,
            "--cluster-script",
            str(script),
            "--",
            "--training-device",
            "cpu",
            "--max-tree-nodes",
            "1000",
            "--max-elapsed-seconds",
            "30",
            "--min-available-ram-mb",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((root / "run_state.json").read_text())
    state["metadata"].pop("managed_sessions_v1", None)
    usage = json.loads((root / "resource_monitor/usage.json").read_text())
    assert usage["elapsed_seconds"] > 0
    result = subprocess.run(prefix, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    after = json.loads((root / "resource_monitor/usage.json").read_text())
    assert after["elapsed_seconds"] > usage["elapsed_seconds"]
    after_state = json.loads((root / "run_state.json").read_text())
    after_state["metadata"].pop("managed_sessions_v1", None)
    assert after_state == state
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = subprocess.run(
        [*prefix, "--mode", "fresh"], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 2
    assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


@pytest.mark.exhaustive
def test_committed_prefix_resume_keeps_every_candidate_adam_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three real accepted rounds; native entry between prefix/suffix never resets losers.

    The scientific stages are small direct fixtures. Native monitor sessions own
    only the inert shell; their timings are not claimed as fixture training usage.
    UUID/path/source receipt bytes are not cross-workspace equality assertions.
    """
    monkeypatch.delenv("MORPION_PERSISTENT_WORKERS", raising=False)
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", "1")
    local_leaderboard(monkeypatch, tmp_path)
    cases = {}
    for label in ["continuous", "resumed"]:
        base = tmp_path / label
        args, paths = setup_continual(base / "work", memory_floor_mb=1)
        script = inert_supervisor(base / "supervisor.sh")
        assert lifecycle.execute_start(paths.work_dir, "resume", (), script) == 0
        for number in range(3):
            state = grow(args)
            run_dataset(args, generation=state.generation)
            accepted = run_pipeline_training_stage(args, generation=state.generation)
            assert (
                run_pipeline_training_stage(args, generation=state.generation)
                == accepted
            )
            if label == "resumed" and number == 0:
                # An unaccepted private attempt is deliberately not a candidate parent.
                private = (
                    paths.work_dir / "training_publications_v1/staging" / ("f" * 32)
                )
                private.mkdir(parents=True, exist_ok=True)
                (private / "unfinished").write_text("not accepted")
                before = paths.run_state_path.read_bytes()
                plan = lifecycle.plan_start(paths.work_dir, cluster_script=script)
                assert (
                    plan["action"] == "resume" and len(plan["candidate_parents"]) == 2
                )
                assert (
                    lifecycle.execute_start(paths.work_dir, "resume", (), script) == 0
                )
                assert paths.run_state_path.read_bytes() == before
        state = load_bootstrap_run_state(paths.run_state_path)
        reference = publication_records(state.metadata)[-1][training.ACCEPTED]
        proposal = training._proposal(paths, reference)
        candidates = {}
        for name, model in proposal["models"].items():
            bundle = paths.work_dir / model["path"]
            manifest = json.loads((bundle / "morpion_manifest.json").read_text())[
                "metadata"
            ]
            candidates[name] = {
                "weights": torch.load(
                    bundle / "param.pt", map_location="cpu", weights_only=True
                ),
                "adam": torch.load(
                    bundle / "optimizer.pt", map_location="cpu", weights_only=True
                ),
                "seed": manifest["continuation"]["seed"],
                "steps": manifest["continuation"]["final_optimizer_step_count"],
                "train": manifest["num_train_samples"],
                "validation": manifest["num_validation_samples"],
            }
        from chipiron.environments.morpion.bootstrap.model_activation import _runner

        runner, effective = _runner(paths, state)
        try:
            assert state.latest_runtime_checkpoint_path is not None
            runner.load_or_create(
                paths.work_dir / state.latest_runtime_checkpoint_path, None, effective
            )
            runner.grow(1)
            growth = (
                fingerprint(runner),
                _selection_signature(runner),
                repr(runner._random_generator.getstate()),
                repr(rollout_random_generator(runner._runtime).getstate()),
            )
        finally:
            runner.discard_uncommitted_runtime()
        cases[label] = (
            candidates,
            proposal["status"]["selected_evaluator_name"],
            proposal["status"]["metadata"]["continual_selection"][
                "validation_population"
            ],
            growth,
        )
    equal_state(cases["continuous"], cases["resumed"])
    assert (
        len(cases["resumed"][0]) == 2
    )  # Both winner and losing candidate retain state.
    assert not torch.cuda.is_initialized()
