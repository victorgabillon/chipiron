"""Real process death inside activation; no fault injection in production code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    publication_records,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    _runner,
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard

PRE = [
    "before_load",
    "model_load",
    "inference",
    "backup",
    "selector",
    "checkpoint",
    "stream_partial_shard",
    "stream_before_manifest",
    "stream_complete",
    "science",
    "validation",
    "hash_validation",
    "file_fsync",
    "directory_fsync",
    "before_rename",
    "after_rename",
    "before_commit",
    "partial_commit",
]
POST = ["after_commit", "commit_directory_fsync", "discovery", "projection"]


def crash_child(root: Path, phase: str) -> None:
    """Install process-local seams and terminate without Python cleanup."""
    from anemone.node_evaluation.tree.node_tree_evaluation import (
        NodeTreeEvaluationState,
    )
    from anemone.node_selector.composed.composed_node_selector import (
        ComposedNodeSelector,
    )

    import chipiron.environments.morpion.bootstrap.closed_scientific_snapshot as closed
    import chipiron.environments.morpion.bootstrap.generation_publication as publication
    from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
        GrowthPatchReceiptSession,
    )
    from chipiron.environments.morpion.players.evaluators.neural_evaluator import (
        MorpionRegressorMasterEvaluator,
    )

    armed = phase == "recovery"
    committed = False
    from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module

    original_refresh = runner_module.AnemoneMorpionSearchRunner.refresh_model_estimates

    def refresh(*args: Any, **kwargs: Any) -> Any:
        nonlocal armed
        armed = True
        return original_refresh(*args, **kwargs)

    runner_module.AnemoneMorpionSearchRunner.refresh_model_estimates = refresh
    original_load = runner_module.load_morpion_evaluator_from_model_bundle

    def load(*args: Any, **kwargs: Any) -> Any:
        nonlocal armed
        if armed and phase == "before_load":
            os._exit(73)
        return original_load(*args, **kwargs)

    runner_module.load_morpion_evaluator_from_model_bundle = load

    def after(owner: Any, name: str, boundary: str) -> None:
        original = getattr(owner, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if armed and phase == boundary:
                os._exit(73)
            return result

        setattr(owner, name, wrapped)

    after(MorpionRegressorMasterEvaluator, "evaluate_batch_items", "inference")
    import torch

    after(torch, "load", "model_load")
    after(publication, "file_digest", "hash_validation")
    after(NodeTreeEvaluationState, "backup_from_children", "backup")
    after(ComposedNodeSelector, "refresh_state_for_checkpoint", "selector")
    after(GrowthPatchReceiptSession, "publish_manifest", "discovery")
    after(GrowthPatchReceiptSession, "publish_manifest", "recovery")
    # Exercise the actual streaming writer inside the existing private transaction.
    from anemone.checkpoints import sharded_encode

    original_manifest = sharded_encode.write_sharded_checkpoint_manifest

    def manifest(*args: Any, **kwargs: Any) -> Any:
        if armed and phase == "stream_before_manifest":
            os._exit(73)
        return original_manifest(*args, **kwargs)

    sharded_encode.write_sharded_checkpoint_manifest = manifest
    after(
        runner_module, "write_sharded_search_checkpoint_from_search", "stream_complete"
    )
    from chipiron.environments.morpion.bootstrap import pipeline_artifacts

    after(pipeline_artifacts, "save_pipeline_active_model", "projection")
    from chipiron.environments.morpion.bootstrap import (
        scientific_generation_publication,
    )

    after(scientific_generation_publication, "validate_closed_export", "validation")
    original_write = Path.write_bytes

    def write(path: Path, contents: Any) -> Any:
        if (
            armed
            and phase == "stream_partial_shard"
            and "checkpoint" in path.parts
            and path.name.endswith(".jsonl.zst")
        ):
            with path.open("wb") as stream:
                stream.write(contents[: len(contents) // 2])
                stream.flush()
                os._exit(73)
        result = original_write(path, contents)
        if (
            armed
            and phase == "checkpoint"
            and "checkpoint" in path.parts
            and "staging" in path.parts
        ):
            os._exit(73)
        return result

    Path.write_bytes = write
    original_encode = closed._encode

    def encode(row: Any) -> Any:
        if (
            armed
            and phase == "science"
            and isinstance(row, dict)
            and "value_contract" in row
        ):
            os._exit(73)
        return original_encode(row)

    closed._encode = encode
    original_fsync = os.fsync

    def fsync(fd: int) -> Any:
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if armed and (
            (phase == "file_fsync" and path.name == "records.jsonl")
            or (phase == "directory_fsync" and path.name == "scientific_snapshot")
            or (phase == "commit_directory_fsync" and committed and path == root)
        ):
            os._exit(73)
        return original_fsync(fd)

    os.fsync = fsync
    original_rename = publication.publish_directory_if_absent

    def rename(*args: Any, **kwargs: Any) -> Any:
        if armed and phase == "before_rename":
            os._exit(73)
        result = original_rename(*args, **kwargs)
        if armed and phase == "after_rename":
            os._exit(73)
        return result

    publication.publish_directory_if_absent = rename
    original_replace = os.replace

    def replace(source: Any, target: Any) -> Any:
        nonlocal committed
        authority = Path(target) == root / "run_state.json"
        if armed and authority and phase == "before_commit":
            os._exit(73)
        result = original_replace(source, target)
        if armed and authority:
            committed = True
            if phase == "after_commit":
                os._exit(73)
        return result

    os.replace = replace
    original_dump = json.dump

    def dump(value: Any, stream: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            armed
            and phase == "partial_commit"
            and isinstance(value, dict)
            and "cycle_index" in value
        ):
            stream.write('{"generation":')
            stream.flush()
            os._exit(73)
        return original_dump(value, stream, *args, **kwargs)

    json.dump = dump
    activate_model(
        MorpionBootstrapPaths.from_work_dir(root),
        json.loads((root / "request.json").read_text()),
    )


def run_child(root: Path, phase: str) -> subprocess.CompletedProcess[str]:
    """Fresh interpreter imports the same source/package selected by its parent."""
    import chipiron

    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; from tests.environments.test_morpion_model_activation_crashes import crash_child; crash_child(Path(sys.argv[1]), sys.argv[2])",
            str(root),
            phase,
        ],
        env={
            **os.environ,
            "PYTHONPATH": f"{Path(chipiron.__file__).parent.parent}:{Path.cwd()}",
        },
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )


@pytest.mark.parametrize("phase", PRE + POST)
def test_process_death_is_one_complete_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Every boundary restores M1 or M2, then exact retry completes/repairs once."""
    local_leaderboard(monkeypatch, tmp_path)
    _, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    (paths.work_dir / "request.json").write_text(json.dumps(request))
    before = paths.run_state_path.read_bytes()
    child = run_child(paths.work_dir, phase)
    assert child.returncode == 73, child.stdout[-2000:] + child.stderr[-2000:]
    recovered = load_bootstrap_run_state(paths.run_state_path)
    expected = request["previous_model"] if phase in PRE else request["candidate"]
    assert committed_evaluator(paths.work_dir, recovered)[0] == expected
    if phase in PRE:
        assert paths.run_state_path.read_bytes() == before
    head = publication_records(recovered.metadata)[-1]
    validate_capsule(paths.work_dir, head)
    result = activate_model(paths, request)
    assert result.generation == 2
    assert committed_evaluator(paths.work_dir, result)[0] == request["candidate"]
    runner, effective = _runner(paths, result)
    runner.load_or_create(
        paths.work_dir / result.latest_runtime_checkpoint_path, None, effective
    )
    runner.validate_model_refresh_certificates()
    runner.discard_uncommitted_runtime()


def test_death_after_m1_drain_requires_new_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted patch is not replayed and the original optimistic head is stale."""
    from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
        receipt_ledger,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_reevaluation_patch,
    )
    from tests.environments.test_morpion_model_activation_edges import propose

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    (paths.work_dir / "request.json").write_text(json.dumps(request))
    assert propose(args).patch_written
    patch = load_reevaluation_patch(paths.pipeline_reevaluation_patch_path)
    child = run_child(paths.work_dir, "before_load")
    assert child.returncode == 73, child.stderr
    drained = load_bootstrap_run_state(paths.run_state_path)
    assert drained.generation == 2
    assert patch.patch_id in receipt_ledger(drained)
    assert committed_evaluator(paths.work_dir, drained)[0] == request["previous_model"]
    with pytest.raises(ValueError, match="Stale"):
        activate_model(paths, request)
    renewed = prepare_model_activation_request(paths, training_generation=1)
    after = activate_model(paths, renewed)
    assert after.generation == 3
    assert receipt_ledger(after) == receipt_ledger(drained)


def test_producer_restart_after_committed_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Death while loading the committed M2 adapter leaves producer progress intact."""
    import chipiron
    from tests.environments.test_morpion_model_activation_edges import propose

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    after = activate_model(paths, request)
    paths.pipeline_active_model_path.unlink()
    code = """
import os, sys
from pathlib import Path
from chipiron.environments.morpion.bootstrap import reevaluation_worker as w
from tests.environments.morpion_durable_v2_helpers import scientific_args
original = w.load_morpion_evaluator_from_model_bundle
def die(*a, **k):
    original(*a, **k)
    os._exit(73)
w.load_morpion_evaluator_from_model_bundle = die
w.run_morpion_reevaluation_worker_once(scientific_args(Path(sys.argv[1])),
    use_streaming_sharded_reader=True, allow_scientific_v2=True, max_nodes_per_patch=3)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(paths.work_dir)],
        env={
            **os.environ,
            "PYTHONPATH": f"{Path(chipiron.__file__).parent.parent}:{Path.cwd()}",
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert child.returncode == 73, child.stderr
    assert load_bootstrap_run_state(paths.run_state_path) == after
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert propose(args).patch_written


def test_death_during_recovery_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated interruption of advisory repair cannot create another transition."""
    local_leaderboard(monkeypatch, tmp_path)
    _, paths, _ = setup_activation(tmp_path / "workspace")
    request = prepare_model_activation_request(paths, training_generation=1)
    (paths.work_dir / "request.json").write_text(json.dumps(request))
    after = activate_model(paths, request)
    authority = paths.run_state_path.read_bytes()
    for _ in range(2):
        paths.pipeline_active_model_path.unlink()
        child = run_child(paths.work_dir, "recovery")
        assert child.returncode == 73, child.stderr
        assert paths.run_state_path.read_bytes() == authority
    assert activate_model(paths, request) == after
